import logging
from typing import Any, Dict, List

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, masked_log_softmax
from allennlp.training.metrics import Average
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.common.typing import StateDict
from kglm.data import AliasDatabase

logger = logging.getLogger(__name__)


LOG0 = torch.tensor(1e-45).log()  # pylint: disable=not-callable


@Model.register('kglm')
class Kglm(Model):
    """
    Knowledge graph language model.

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
                 encoder: Seq2SeqEncoder,
                 token_embedder: TextFieldEmbedder,
                 entity_embedder: TextFieldEmbedder,
                 tie_weights: bool,
                 dropout_rate: float,
                 variational_dropout_rate: float) -> None:
        super(Kglm, self).__init__(vocab)

        assert entity_embedder.get_output_dim() == token_embedder.get_output_dim()

        self._token_embedder = token_embedder
        self._entity_embedder = entity_embedder
        self._encoder = encoder
        self._tie_weights = tie_weights
        self._variational_dropout_rate = variational_dropout_rate
        self._dropout_rate = dropout_rate
        self._unk_index = vocab.get_token_index(DEFAULT_OOV_TOKEN)

        embedding_dim = self._token_embedder.get_output_dim()
        hidden_dim = self._encoder.get_output_dim()

        # Merity et al 2017, section 4.5 suggests that the LSTM hidden layer be larger than the
        # embeddings. We'll use these `Linear` layers to project them.
        self._embedding_to_hidden = torch.nn.Linear(in_features=embedding_dim,
                                                    out_features=hidden_dim)
        self._hidden_to_embedding = torch.nn.Linear(in_features=hidden_dim,
                                                    out_features=embedding_dim)

        self._variational_dropout = InputVariationalDropout(variational_dropout_rate)
        self._dropout = torch.nn.Dropout(dropout_rate)

        self._mention_type_projection = torch.nn.Linear(in_features=embedding_dim,
                                                        out_features=2)
        self._entity_projection = torch.nn.Linear(in_features=embedding_dim,
                                                  out_features=embedding_dim)

        self._condense_projection = torch.nn.Linear(in_features=2 * embedding_dim,
                                                    out_features=embedding_dim)
        self._generate_mode_projection = torch.nn.Linear(in_features=embedding_dim,
                                                         out_features=vocab.get_vocab_size('tokens'))
        self._copy_mode_projection = torch.nn.Linear(in_features=embedding_dim,
                                                     out_features=embedding_dim)

        if tie_weights:
            self._generate_mode_projection.weight = self._token_embedder._token_embedders['tokens'].weight  # pylint: disable=W0212

        self._state: StateDict = None

        # Metrics
        self._avg_mention_loss = Average()
        self._avg_entity_loss = Average()
        self._avg_vocab_loss = Average()
        self._avg_mention_vocab_loss = Average()
        self._avg_non_mention_vocab_loss = Average()

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Dict[str, torch.Tensor],
                reset: bool,
                metadata: List[Dict[str, Any]],
                entity_identifiers: torch.Tensor = None,
                shortlist: Dict[str, torch.Tensor] = None,
                shortlist_indices: torch.Tensor = None,
                alias_copy_indices: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # Tensorize the alias_database - this will only perform the operation once.
        alias_database = metadata[0]['alias_database']
        alias_database.tensorize(vocab=self.vocab)

        # Reset the model if needed
        if reset:
            self.reset_states()

        if entity_identifiers is not None:
            output_dict = self._forward_loop(tokens=tokens,
                                             alias_database=alias_database,
                                             entity_identifiers=entity_identifiers,
                                             shortlist=shortlist,
                                             shortlist_indices=shortlist_indices,
                                             alias_copy_indices=alias_copy_indices)

        else:
            output_dict = {}

        return output_dict

    def _mention_loss(self,
                      hidden: torch.Tensor,
                      targets: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss term for predicting whether or not the the next token will be part of
        an entity mention.
        """
        logits = self._mention_type_projection(hidden)
        mention_loss = F.cross_entropy(logits.view(-1, 2), targets.view(-1), reduction='none')
        mention_loss *= mask.view(-1).float()
        mention_loss = mention_loss.sum() / (mask.float().sum() + 1e-13)
        self._avg_mention_loss(mention_loss)  # Update metric
        return mention_loss

    def _entity_loss(self,
                     hidden: torch.Tensor,
                     targets: torch.Tensor,
                     mask: torch.Tensor,
                     shortlist_embeddings: torch.Tensor,
                     shortlist_mask: torch.Tensor) -> torch.Tensor:
        # Logits are computed using a bilinear form that measures the similarity between the
        # projected hidden state and the embeddings of entities in the shortlist
        projected = self._entity_projection(hidden)
        logits = torch.bmm(projected, shortlist_embeddings.transpose(1, 2))

        # There are technically two masks that need to be accounted for: a class-wise mask which
        # specifies which logits to ignore in the class dimension, and a token-wise mask (e.g.
        # `mask`) which avoids measuring loss for predictions on non-mention tokens. In practice,
        # we only need the class-wise mask since the non-mention tokens cannot be associated with a
        # valid target.
        batch_size = hidden.shape[0]
        entity_loss = 0.0
        for i in range(batch_size):
            entity_loss += F.cross_entropy(logits[i], targets[i], shortlist_mask[i].float(), reduction='sum')
        entity_loss = entity_loss / (mask.float().sum() + 1e-13)
        self._avg_entity_loss(entity_loss)
        return entity_loss

    def _copy_scores(self,
                     hidden: torch.Tensor,
                     alias_tokens: torch.Tensor) -> torch.Tensor:
        # Begin by flattening the tokens so that they fit the expected shape of a
        # ``Seq2SeqEncoder``.
        batch_size, sequence_length, num_aliases, alias_length = alias_tokens.shape
        flattened = alias_tokens.view(-1, alias_length)
        copy_mask = flattened != 0
        if copy_mask.sum() == 0:
            return hidden.new_zeros(batch_size, sequence_length, 1, dtype=torch.float32)

        # Next we run through standard pipeline
        embedded = self._token_embedder({'tokens': flattened})  # UGLY
        expanded = self._embedding_to_hidden(embedded)
        encoded = self._encoder(expanded, copy_mask)
        contracted = self._hidden_to_embedding(encoded)

        # Equation 8 in the CopyNet paper recommends applying the additional step.
        projected = torch.tanh(self._copy_mode_projection(contracted))

        # This part gets a little funky - we need to make sure that the first dimension in
        # `projected` and `hidden` is batch_size x sequence_length.
        hidden = hidden.view(batch_size * sequence_length, 1, -1)
        projected = projected.view(batch_size * sequence_length, -1, num_aliases * alias_length)
        copy_scores = torch.bmm(hidden, projected).squeeze()
        copy_scores = copy_scores.view(batch_size, sequence_length, -1).contiguous()

        return copy_scores

    def _vocab_loss(self,
                    generate_scores: torch.Tensor,
                    copy_scores: torch.Tensor,
                    target_tokens: torch.Tensor,
                    target_copy_indices: torch.Tensor,
                    mask: torch.Tensor,
                    alias_indices: torch.Tensor):

        batch_size, sequence_length, vocab_size = generate_scores.shape
        copy_sequence_length = copy_scores.shape[-1]

        # We create a mask to ensure that padding alias tokens are omitted from the softmax.
        alias_mask = alias_indices.view(batch_size, sequence_length, -1).gt(0)
        score_mask = mask.new_ones(batch_size, sequence_length, vocab_size + copy_sequence_length)
        score_mask[:, :, vocab_size:] = alias_mask

        # Next we concatenate the score tensors together in order to compute the log probabilities.
        concatenated_scores = torch.cat((generate_scores, copy_scores), dim=-1)
        log_probs = masked_log_softmax(concatenated_scores, score_mask)

        # The generated token loss is a simple cross-entropy calculation, we can just gather
        # the log probabilties...
        flattened_log_probs = log_probs.view(batch_size * sequence_length, -1)
        flattened_targets = target_tokens.view(batch_size * sequence_length, 1)
        generate_log_probs = flattened_log_probs.gather(1, flattened_targets).squeeze()
        # ...except we need to omit any <UNK> terms that correspond to copy tokens.
        unk_targets = target_tokens.eq(self._unk_index)
        copied_targets = target_copy_indices.gt(0)
        ignore_mask = (unk_targets * copied_targets)
        generate_mask = (1 - mask.byte()) * ignore_mask
        generate_log_probs[generate_mask.view(-1)] = LOG0

        # The copied token loss requires adding up all of the relevant token log probabilities.
        # We'll use a for loop to keep things simple for now.
        flattened_copy_mask = copied_targets.view(-1)
        flattened_alias_indices = alias_indices.view(batch_size * sequence_length, -1)
        flattened_target_copy_indices = target_copy_indices.view(-1)
        copy_log_probs = torch.zeros_like(generate_log_probs)
        for i in range(batch_size * sequence_length):
            selection_mask = flattened_alias_indices[i].eq(flattened_target_copy_indices[i])
            if selection_mask.sum() > 0 and flattened_copy_mask[i] != 0:
                selected_log_probs = flattened_log_probs[i, vocab_size:].masked_select(selection_mask)
                total_log_prob = torch.logsumexp(selected_log_probs, dim=0)
                copy_log_probs[i] = total_log_prob
            else:
                copy_log_probs[i] = LOG0

        combined_log_probs = torch.stack((generate_log_probs, copy_log_probs), dim=1)
        combined_log_probs = torch.logsumexp(combined_log_probs, dim=1)
        vocab_loss = -combined_log_probs.sum() / (mask.float().sum() + 1e-13)
        self._avg_vocab_loss(vocab_loss)

        mention_vocab_loss = -combined_log_probs[flattened_copy_mask].sum() / (flattened_copy_mask.float().sum() + 1e-13)
        non_mention_vocab_loss = -combined_log_probs[1 - flattened_copy_mask].sum() / (mask.float().sum() - flattened_copy_mask.float().sum() + 1e-13)
        self._avg_mention_vocab_loss(mention_vocab_loss)
        self._avg_non_mention_vocab_loss(non_mention_vocab_loss)

        return vocab_loss

    def _forward_loop(self,
                      tokens: Dict[str, torch.Tensor],
                      alias_database: AliasDatabase,
                      entity_identifiers: Dict[str, torch.Tensor],
                      shortlist: Dict[str, torch.Tensor],
                      shortlist_indices: torch.Tensor,
                      alias_copy_indices: torch.Tensor) -> Dict[str, torch.Tensor]:

        if self._state is not None:
            tokens = {field: torch.cat((self._state['prev_tokens'][field], tokens[field]), dim=1)
                      for field in tokens}
            entity_identifiers = {field: torch.cat((self._state['prev_entity_identifiers'][field],
                                                    entity_identifiers[field]), dim=1)
                                  for field in entity_identifiers}
            shortlist_indices = torch.cat((self._state['prev_shortlist_indices'], shortlist_indices), dim=1)
            alias_copy_indices = torch.cat((self._state['prev_alias_copy_indices'], alias_copy_indices), dim=1)

        # Get the token mask and target tensors
        token_mask = get_text_field_mask(tokens)
        target_mask = token_mask[:, 1:].contiguous()
        target_tokens = tokens['tokens'][:, 1:].contiguous()
        target_copy_indices = alias_copy_indices[:, 1:].contiguous()

        # Embed and encode the source tokens
        source_mask = token_mask[:, :-1].contiguous()
        source_embeddings = self._token_embedder(tokens)[:, :-1].contiguous()
        source_embeddings = self._variational_dropout(source_embeddings)
        expanded = self._embedding_to_hidden(source_embeddings)
        hidden = self._encoder(expanded, source_mask)
        hidden = self._hidden_to_embedding(hidden)

        # Embed entities
        entity_mask = get_text_field_mask(entity_identifiers)
        entity_embeddings = self._entity_embedder(entity_identifiers)
        entity_embeddings = self._variational_dropout(entity_embeddings)

        # Embed entity shortlist
        shortlist_mask = get_text_field_mask(shortlist)
        shortlist_embeddings = self._entity_embedder(shortlist)
        shortlist_embeddings = self._variational_dropout(shortlist_embeddings)

        # Predict whether or not the next token will be an entity mention.
        target_mentions = entity_mask[:, 1:].contiguous()
        mention_loss = self._mention_loss(hidden, target_mentions, target_mask)

        # Predict which entity (among those in the supplied shortlist) is going to be
        # mentioned.
        target_shortlist_indices = shortlist_indices[:, 1:].contiguous()
        entity_loss = self._entity_loss(hidden,
                                        target_shortlist_indices,
                                        target_mask,
                                        shortlist_embeddings,
                                        shortlist_mask)

        # Predict generation-mode scores. Start by concatenating predicted entity embeddings with
        # the encoder output - then feed through a linear layer.
        target_embeddings = entity_embeddings[:, 1:].contiguous()
        concatenated = torch.cat((hidden, target_embeddings), dim=-1)
        condensed = self._condense_projection(concatenated)
        generate_scores = self._generate_mode_projection(condensed)

        # Predict copy-mode scores.
        target_entity_identifiers = entity_identifiers['entity_ids'][:, 1:].contiguous()
        alias_tokens, alias_indices = alias_database.lookup(target_entity_identifiers)
        copy_scores = self._copy_scores(hidden, alias_tokens)

        # Combine scores to get vocab loss
        vocab_loss = self._vocab_loss(generate_scores,
                                      copy_scores,
                                      target_tokens,
                                      target_copy_indices,
                                      target_mask,
                                      alias_indices)

        # Compute total loss
        loss = mention_loss + entity_loss + vocab_loss

        # Update state
        self._state = {
                'prev_tokens': {field: tokens[field][:, -1].unsqueeze(1).detach() for field in tokens},
                'prev_entity_identifiers': {field: entity_identifiers[field][:, -1].unsqueeze(1).detach()
                                            for field in entity_identifiers},
                'prev_shortlist_indices': shortlist_indices[:, -1].unsqueeze(1).detach(),
                'prev_alias_copy_indices': alias_copy_indices[:, -1].unsqueeze(1).detach()
        }

        return {'loss': loss}

    def reset_states(self) -> None:
        """Resets the model's internals. Should be called at the start of a new batch."""
        self._encoder.reset_states()
        self._state = None

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'mention_loss': self._avg_mention_loss.get_metric(reset).item(),
                'entity_loss': self._avg_entity_loss.get_metric(reset).item(),
                'vocab_loss': self._avg_vocab_loss.get_metric(reset).item(),
                'mention_vocab_loss': self._avg_mention_vocab_loss.get_metric(reset).item(),
                'non_mention_vocab_loss': self._avg_non_mention_vocab_loss.get_metric(reset).item()
        }
