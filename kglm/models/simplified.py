import logging
import math
from typing import Any, Dict, List, Optional

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, masked_log_softmax, \
    sequence_cross_entropy_with_logits
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.data import AliasDatabase
from kglm.modules import embedded_dropout, LockedDropout, WeightDrop
from kglm.training.metrics import Ppl

logger = logging.getLogger(__name__)


@Model.register('simplified')
class AliasCopynet(Model):
    """
    Oracle alias copynet language model.

    Parameters
    ----------
    vocab : ``Vocabulary``
        The model vocabulary.
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed tokens.
    encoder : ``Seq2SeqEncoder``
        Used to encode the sequence of token embeddings.
    embedding_dim : ``int``
        The dimension of entity / length embeddings. Should match the encoder output size.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 entity_embedder: TextFieldEmbedder,
                 alias_encoder: Seq2SeqEncoder,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.4,
                 dropouth: float = 0.3,
                 dropouti: float = 0.65,
                 dropoute: float = 0.1,
                 wdrop: float = 0.5,
                 alpha: float = 2.0,
                 beta: float = 1.0,
                 tie_weights: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(AliasCopynet, self).__init__(vocab)

        # Model architecture - Note: we need to extract the `Embedding` layers from the
        # `TokenEmbedders` to apply dropout later on.
        # pylint: disable=protected-access
        self._token_embedder = token_embedder._token_embedders['tokens']
        self._entity_embedder = entity_embedder._token_embedders['entity_ids']
        self._alias_encoder = alias_encoder
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._tie_weights = tie_weights

        # Dropout
        self._locked_dropout = LockedDropout()
        self._dropout = dropout
        self._dropouth = dropouth
        self._dropouti = dropouti
        self._dropoute = dropoute
        self._wdrop = wdrop

        # Regularization strength
        self._alpha = alpha
        self._beta = beta

        # RNN Encoders. TODO: Experiment with seperate encoder for aliases.
        entity_embedding_dim = entity_embedder.get_output_dim()
        token_embedding_dim = entity_embedder.get_output_dim()
        assert entity_embedding_dim == token_embedding_dim
        embedding_dim = token_embedding_dim

        rnns: List[torch.nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                input_size = token_embedding_dim
            else:
                input_size = hidden_size
            if (i == num_layers - 1) and tie_weights:
                output_size = token_embedding_dim
            else:
                output_size = hidden_size
            rnns.append(torch.nn.LSTM(input_size, output_size, batch_first=True))
        rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in rnns]
        self.rnns = torch.nn.ModuleList(rnns)

        # Various linear transformations.
        self._fc_mention = torch.nn.Linear(
            in_features=embedding_dim,
            out_features=2)

        self._fc_entity = torch.nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim)

        self._fc_condense = torch.nn.Linear(
            in_features=2 * embedding_dim,
            out_features=embedding_dim)

        self._fc_generate = torch.nn.Linear(
            in_features=embedding_dim,
            out_features=vocab.get_vocab_size('tokens'))

        self._fc_copy = torch.nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim)

        if tie_weights:
            self._fc_generate.weight = self._token_embedder.weight

        self._state: Optional[Dict[str, Any]]= None

        # Metrics
        # self._avg_mention_loss = Average()
        # self._avg_entity_loss = Average()
        # self._avg_vocab_loss = Average()
        self._unk_index = vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self._unk_penalty = math.log(vocab.get_vocab_size('tokens_unk'))
        self._ppl = Ppl()
        self._upp = Ppl()
        self._kg_ppl = Ppl()  # Knowledge-graph ppl
        self._bg_ppl = Ppl()  # Background ppl

        initializer(self)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                source: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                reset: torch.Tensor,
                entity_ids: torch.Tensor = None,
                shortlist: Dict[str, torch.Tensor] = None,
                shortlist_inds: torch.Tensor = None,
                alias_copy_inds: torch.Tensor = None,
                alias_tokens: torch.Tensor = None,
                alias_inds: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        alias_tokens = alias_tokens['tokens']
        # Inds have fixed size and don't get truncated on split so truncate
        # now.
        alias_inds = alias_inds[:, :, :alias_tokens.shape[2]]

        # Reset the model if needed
        if reset.any() and (self._state is not None):
            for layer in range(self._num_layers):
                h, c = self._state['layer_%i' % layer]
                h[:, reset, :] = torch.zeros_like(h[:, reset, :])
                c[:, reset, :] = torch.zeros_like(c[:, reset, :])
                self._state['layer_%i' % layer] = (h, c)

        if entity_ids is not None:
            output_dict = self._forward_loop(
                source=source,
                target=target,
                alias_copy_inds=alias_copy_inds,
                alias_tokens=alias_tokens,
                alias_inds=alias_inds)
        else:
            # TODO: Figure out what we want here - probably to do some king of inference on
            # entities / mention types.
            output_dict = {}

        return output_dict

    def _mention_loss(self,
                      encoded: torch.Tensor,
                      targets: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for predicting whether or not the the next token will be part of an
        entity mention.
        """
        logits = self._fc_mention(encoded)
        mention_loss = sequence_cross_entropy_with_logits(logits, targets, mask,
                                                          average='token')
        return mention_loss

    def _entity_loss(self,
                     encoded: torch.Tensor,
                     targets: torch.Tensor,
                     mask: torch.Tensor,
                     shortlist_embeddings: torch.Tensor,
                     shortlist_mask: torch.Tensor) -> torch.Tensor:
        # Logits are computed using a bilinear form that measures the similarity between the
        # projected hidden state and the embeddings of entities in the shortlist
        projected = self._fc_entity(encoded)
        projected = self._locked_dropout(projected, self._dropout)
        logits = torch.bmm(projected, shortlist_embeddings.transpose(1, 2))

        # There are technically two masks that need to be accounted for: a class-wise mask which
        # specifies which logits to ignore in the class dimension, and a token-wise mask (e.g.
        # `mask`) which avoids measuring loss for predictions on non-mention tokens. In practice,
        # we only need the class-wise mask since the non-mention tokens cannot be associated with a
        # valid target.
        batch_size = encoded.shape[0]
        entity_loss = 0.0
        for i in range(batch_size):
            entity_loss += F.cross_entropy(
                input=logits[i],
                target=targets[i],
                weight=shortlist_mask[i].float(),
                reduction='sum')
        entity_loss = entity_loss / (mask.float().sum() + 1e-13)
        return entity_loss

    def _copy_scores(self,
                     encoded: torch.Tensor,
                     alias_tokens: torch.Tensor) -> torch.Tensor:
        # Begin by flattening the tokens so that they fit the expected shape of a
        # ``Seq2SeqEncoder``.
        batch_size, sequence_length, alias_length = alias_tokens.shape
        flattened = alias_tokens.view(-1, alias_length)
        copy_mask = flattened != 0
        if copy_mask.sum() == 0:
            return encoded.new_zeros((batch_size, sequence_length, alias_length),
                                     dtype=torch.float32)

        # Embed and encode the alias tokens.
        embedded = self._token_embedder(flattened)
        mask = flattened.gt(0)
        encoded_aliases = self._alias_encoder(embedded, mask)

        # Equation 8 in the CopyNet paper recommends applying the additional step.
        projected = torch.tanh(self._fc_copy(encoded_aliases))
        projected = self._locked_dropout(projected, self._dropout)

        # This part gets a little funky - we need to make sure that the first dimension in
        # `projected` and `hidden` is batch_size x sequence_length.
        encoded = encoded.view(batch_size * sequence_length, 1, -1)
        projected = projected.view(batch_size * sequence_length, -1, alias_length)
        copy_scores = torch.bmm(encoded, projected).squeeze()
        copy_scores = copy_scores.view(batch_size, sequence_length, -1).contiguous()

        return copy_scores

    def _vocab_loss(self,
                    generate_scores: torch.Tensor,
                    copy_scores: torch.Tensor,
                    target_tokens: torch.Tensor,
                    target_alias_indices: torch.Tensor,
                    mask: torch.Tensor,
                    alias_indices: torch.Tensor,
                    alias_tokens: torch.Tensor):
        mention_mask = target_alias_indices.gt(0)
        batch_size, sequence_length, vocab_size = generate_scores.shape
        copy_sequence_length = copy_scores.shape[-1]

        # Flat sequences make life **much** easier.
        flattened_targets = target_tokens.view(batch_size * sequence_length, 1)
        flattened_mask = mask.view(-1, 1).byte()
        alias_mask = alias_indices.view(batch_size, sequence_length, -1).gt(0)

        # The log-probability distribution is then given by taking the masked log softmax.
        generate_log_probs = masked_log_softmax(generate_scores,
                                                torch.ones_like(generate_scores))
        copy_log_probs = masked_log_softmax(copy_scores, alias_mask)

        # GENERATE LOSS ###
        # The generated token loss is a simple cross-entropy calculation, we can just gather
        # the log probabilties...
        flattened_log_probs = generate_log_probs.view(batch_size * sequence_length, -1)
        generate_log_probs_source_vocab = flattened_log_probs.gather(1, flattened_targets)
        # ...except we need to ignore the contribution of UNK tokens that are copied (only when
        # computing the loss). To do that we create a mask which is 1 only if the token is not a
        # copied UNK (or padding).
        unks = target_tokens.eq(self._unk_index).view(-1, 1)
        copied = target_alias_indices.gt(0).view(-1, 1)
        generate_mask = ~copied & flattened_mask
        # Since we are in log-space we apply the mask by addition.
        generate_log_probs_extended_vocab = generate_log_probs_source_vocab + (generate_mask.float() + 1e-45).log()

        # COPY LOSS ###
        copy_log_probs = copy_log_probs.view(batch_size * sequence_length, -1)
        # When computing the loss we need to get the log probability of **only** the copied tokens.
        alias_indices = alias_indices.view(batch_size * sequence_length, -1)
        target_alias_indices = target_alias_indices.view(-1, 1)
        copy_mask = alias_indices.eq(target_alias_indices) & flattened_mask & target_alias_indices.gt(0)
        copy_log_probs_extended_vocab = copy_log_probs + (copy_mask.float() + 1e-45).log()
        # When computing perplexity, we do so with respect to the source vocabulary. Evaluating the
        # contribution of the copy mechanism using the probabilities above is then incorrect for
        # UNK tokens since we are only adding the likelihood of the copied term - we really want the
        # sum of likelihoods of all words that would be mapped to <UNK>. We compute this below.
        alias_tokens = alias_tokens.view(batch_size * sequence_length, -1)
        copy_mask = alias_tokens.eq(flattened_targets) & flattened_mask
        copy_log_probs_source_vocab = copy_log_probs + (copy_mask.float() + 1e-45).log()

        # COMBINED LOSS ###
        # The final loss term is computed using our log probs computed w.r.t to the entire
        # vocabulary.

        kg_mask = (mention_mask & mask.byte()).view(-1)
        bg_mask = (~mention_mask & mask.byte()).view(-1)
        mask = mask.byte().view(-1)

        combined_log_probs_extended_vocab = torch.cat((generate_log_probs_extended_vocab,
                                                       copy_log_probs_extended_vocab),
                                                      dim=1)
        combined_log_probs_extended_vocab = torch.logsumexp(combined_log_probs_extended_vocab,
                                                            dim=1)
        vocab_loss = -combined_log_probs_extended_vocab[mask].sum() / (mask.float().sum() + 1e-13)

        # PERPLEXITY ###
        # Our perplexity terms are computed using the log probs computed w.r.t the source
        # vocabulary.
        combined_log_probs_source_vocab = torch.cat((generate_log_probs_extended_vocab,
                                                     copy_log_probs_source_vocab),
                                                    dim=1)
        combined_log_probs_source_vocab = torch.logsumexp(combined_log_probs_source_vocab,
                                                          dim=1)

        penalty = self._unk_penalty * unks.float().sum()


        self._ppl(-combined_log_probs_source_vocab[mask].sum(), mask.float().sum() + 1e-13)
        self._upp(-combined_log_probs_source_vocab[mask].sum() + penalty, mask.float().sum() + 1e-13)
        if kg_mask.any():
            self._kg_ppl(-combined_log_probs_source_vocab[kg_mask].sum(), kg_mask.float().sum() + 1e-13)
        if bg_mask.any():
            self._bg_ppl(-combined_log_probs_source_vocab[bg_mask].sum(), bg_mask.float().sum() + 1e-13)

        return vocab_loss

    def _forward_loop(self,
                      source: Dict[str, torch.Tensor],
                      target: Dict[str, torch.Tensor],
                      alias_copy_inds: torch.Tensor,
                      alias_tokens: torch.Tensor,
                      alias_inds: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Get the token mask and unwrap the target tokens.
        target_mask = get_text_field_mask(target)
        target = target['tokens']

        # Embed source tokens.
        source = source['tokens']
        source_embeddings = embedded_dropout(
            embed=self._token_embedder,
            words=source,
            dropout=self._dropoute if self.training else 0)
        source_embeddings = self._locked_dropout(source_embeddings, self._dropouti)

        # Encode source tokens.
        current_input = source_embeddings
        hidden_states = []
        for layer, rnn in enumerate(self.rnns):
            # Retrieve previous hidden state for layer.
            if self._state is not None:
                prev_hidden = self._state['layer_%i' % layer]
            else:
                prev_hidden = None
            # Forward-pass.
            output, hidden = rnn(current_input, prev_hidden)
            output = output.contiguous()
            # Update hidden state for layer.
            hidden = tuple(h.detach() for h in hidden)
            hidden_states.append(hidden)
            # Apply dropout.
            if layer == self._num_layers - 1:
                dropped_output = self._locked_dropout(output, self._dropout)
            else:
                dropped_output = self._locked_dropout(output, self._dropouth)
            current_input = dropped_output
        encoded = current_input
        self._state = {'layer_%i' % i: h for i, h in enumerate(hidden_states)}

        # Predict generation-mode scores. Start by concatenating predicted entity embeddings with
        # the encoder output - then feed through a linear layer.
        generate_scores = self._fc_generate(encoded)

        # Predict copy-mode scores.
        # alias_tokens, alias_inds = alias_database.lookup(entity_ids)
        copy_scores = self._copy_scores(encoded, alias_tokens)

        # Combine scores to get vocab loss
        vocab_loss = self._vocab_loss(generate_scores,
                                      copy_scores,
                                      target,
                                      alias_copy_inds,
                                      target_mask,
                                      alias_inds,
                                      alias_tokens)

        # Compute total loss
        loss = vocab_loss

        # Activation regularization
        if self._alpha:
            loss = loss + self._alpha * dropped_output.pow(2).mean()
        # Temporal activation regularization (slowness)
        if self._beta:
            loss = loss + self._beta * (output[:, 1:] - output[:, :-1]).pow(2).mean()

        return {'loss': loss}

    @overrides
    def train(self, mode=True):
        # TODO: This is a temporary hack to ensure that the internal state resets when the model
        # switches from training to evaluation. The complication arises from potentially differing
        # batch sizes (e.g. the `reset` tensor will not be the right size). In future
        # implementations this should be handled more robustly.
        super().train(mode)
        self._state = None

    @overrides
    def eval(self):
        # TODO: See train.
        super().eval()
        self._state = None

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'ppl': self._ppl.get_metric(reset),
            'upp': self._upp.get_metric(reset),
            'kg_ppl': self._kg_ppl.get_metric(reset),
            'bg_ppl': self._bg_ppl.get_metric(reset)
        }
