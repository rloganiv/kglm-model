import logging
import math
from typing import Any, Dict, List, Optional

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import (
    get_text_field_mask, masked_log_softmax, sequence_cross_entropy_with_logits)
from allennlp.training.metrics import Average, CategoricalAccuracy, F1Measure, SequenceAccuracy
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.data import AliasDatabase
from kglm.modules import (
    embedded_dropout, LockedDropout, WeightDrop, KnowledgeGraphLookup, RecentEntities)
from kglm.training.metrics import Ppl

logger = logging.getLogger(__name__)


@Model.register('no-story')
class NoStory(Model):
    """
    Knowledge graph language model - generative story

    Parameters
    ----------
    vocab : ``Vocabulary``
        The model vocabulary.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 entity_embedder: TextFieldEmbedder,
                 alias_encoder: Seq2SeqEncoder,
                 use_shortlist: bool,
                 hidden_size: int,
                 num_layers: int,
                 cutoff: int = 30,
                 tie_weights: bool = False,
                 dropout: float = 0.4,
                 dropouth: float = 0.3,
                 dropouti: float = 0.65,
                 dropoute: float = 0.1,
                 wdrop: float = 0.5,
                 alpha: float = 2.0,
                 beta: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(NoStory, self).__init__(vocab)

        # We extract the `Embedding` layers from the `TokenEmbedders` to apply dropout later on.
        # pylint: disable=protected-access
        self._token_embedder = token_embedder._token_embedders['tokens']
        self._entity_embedder = entity_embedder._token_embedders['entity_ids']
        self._alias_encoder = alias_encoder
        self._recent_entities = RecentEntities(cutoff=cutoff)
        self._use_shortlist = use_shortlist
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._cutoff = cutoff
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

        # RNN Encoders.
        entity_embedding_dim = entity_embedder.get_output_dim()
        token_embedding_dim = token_embedder.get_output_dim()
        self.entity_embedding_dim = entity_embedding_dim
        self.token_embedding_dim = token_embedding_dim

        rnns: List[torch.nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                input_size = token_embedding_dim
            else:
                input_size = hidden_size
            if (i == num_layers - 1):
                output_size = token_embedding_dim + entity_embedding_dim
            else:
                output_size = hidden_size
            rnns.append(torch.nn.LSTM(input_size, output_size, batch_first=True))
        rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in rnns]
        self.rnns = torch.nn.ModuleList(rnns)

        # Various linear transformations.
        self._fc_mention_type = torch.nn.Linear(
            in_features=token_embedding_dim,
            out_features=2)

        if not use_shortlist:
            self._fc_new_entity = torch.nn.Linear(
                in_features=entity_embedding_dim,
                out_features=vocab.get_vocab_size('entity_ids'))

            if tie_weights:
                self._fc_new_entity.weight = self._entity_embedder.weight

        self._fc_condense = torch.nn.Linear(
            in_features=token_embedding_dim + entity_embedding_dim,
            out_features=token_embedding_dim)

        self._fc_generate = torch.nn.Linear(
            in_features=token_embedding_dim,
            out_features=vocab.get_vocab_size('tokens'))

        self._fc_copy = torch.nn.Linear(
            in_features=token_embedding_dim,
            out_features=token_embedding_dim)

        if tie_weights:
            self._fc_generate.weight = self._token_embedder.weight

        self._state: Optional[Dict[str, Any]] = None

        # Metrics
        self._unk_index = vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self._unk_penalty = math.log(vocab.get_vocab_size('tokens_unk'))
        self._ppl = Ppl()
        self._upp = Ppl()
        self._kg_ppl = Ppl()  # Knowledge-graph ppl
        self._bg_ppl = Ppl()  # Background ppl
        self._avg_mention_type_loss = Average()
        self._avg_new_entity_loss = Average()
        self._avg_vocab_loss = Average()
        self._new_mention_f1 =  F1Measure(positive_label=1)
        self._new_entity_accuracy = CategoricalAccuracy()
        self._new_entity_accuracy20 = CategoricalAccuracy(top_k=20)

        initializer(self)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                source: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                reset: torch.Tensor,
                metadata: List[Dict[str, Any]],
                mention_type: torch.Tensor = None,
                raw_entity_ids: Dict[str, torch.Tensor] = None,
                entity_ids: Dict[str, torch.Tensor] = None,
                parent_ids: Dict[str, torch.Tensor] = None,
                relations: Dict[str, torch.Tensor] = None,
                shortlist: Dict[str, torch.Tensor] = None,
                shortlist_inds: torch.Tensor = None,
                alias_copy_inds: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # Tensorize the alias_database - this will only perform the operation once.
        alias_database = metadata[0]['alias_database']
        alias_database.tensorize(vocab=self.vocab)

        # Reset the model if needed
        if reset.any() and (self._state is not None):
            for layer in range(self._num_layers):
                h, c = self._state['layer_%i' % layer]
                h[:, reset, :] = torch.zeros_like(h[:, reset, :])
                c[:, reset, :] = torch.zeros_like(c[:, reset, :])
                self._state['layer_%i' % layer] = (h, c)
        self._recent_entities.reset(reset)

        if entity_ids is not None:
            output_dict = self._forward_loop(
                source=source,
                target=target,
                alias_database=alias_database,
                mention_type=mention_type,
                raw_entity_ids=raw_entity_ids,
                entity_ids=entity_ids,
                parent_ids=parent_ids,
                relations=relations,
                shortlist=shortlist,
                shortlist_inds=shortlist_inds,
                alias_copy_inds=alias_copy_inds)
        else:
            # TODO: Figure out what we want here - probably to do some king of inference on
            # entities / mention types.
            output_dict = {}

        return output_dict

    def _encode_source(self, source: Dict[str, torch.Tensor]) -> torch.Tensor:

        # Extract and embed source tokens.
        source_embeddings = embedded_dropout(
            embed=self._token_embedder,
            words=source,
            dropout=self._dropoute if self.training else 0)
        source_embeddings = self._locked_dropout(source_embeddings, self._dropouti)

        # Encode.
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

        alpha_loss = dropped_output.pow(2).mean()
        beta_loss = (output[:, 1:] - output[:, :-1]).pow(2).mean()

        # Update state.
        self._state = {'layer_%i' % i: h for i, h in enumerate(hidden_states)}

        return encoded, alpha_loss, beta_loss

    def _mention_type_loss(self,
                           encoded: torch.Tensor,
                           mention_type: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for predicting whether or not the the next token will be part of an
        entity mention.
        """
        logits = self._fc_mention_type(encoded)
        mention_type_loss = sequence_cross_entropy_with_logits(logits, mention_type, mask,
                                                               average='token')
        # if not self.training:
        self._new_mention_f1(predictions=logits,
                             gold_labels=mention_type,
                             mask=mask)

        return mention_type_loss

    def _new_entity_loss(self,
                         encoded: torch.Tensor,
                         target_inds: torch.Tensor,
                         shortlist: torch.Tensor,
                         target_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ==========
        target_inds : ``torch.Tensor``
            Either the shortlist inds if using shortlist, otherwise the target entity ids.
        """
        if self._use_shortlist:

            # First we embed the shortlist entries
            shortlist_mask = get_text_field_mask(shortlist)
            shortlist_embeddings = embedded_dropout(
                embed=self._entity_embedder,
                words=shortlist['entity_ids'],
                dropout=self._dropoute if self.training else 0)

            # Logits are computed using the inner product that between the predicted entity embedding
            # and the embeddings of entities in the shortlist
            encodings = self._locked_dropout(encoded, self._dropout)
            logits = torch.bmm(encodings, shortlist_embeddings.transpose(1, 2))

            # Take masked softmax to get log probabilties and gather the targets.
            log_probs = masked_log_softmax(logits, shortlist_mask)
            target_log_probs = torch.gather(log_probs, -1, target_inds.unsqueeze(-1)).squeeze(-1)

            # If not generating a new mention, the action is deterministic - so the loss is 0 for these tokens.
            mask = ~target_inds.eq(0)
            target_log_probs[~mask] = 0

            # if not self.training:
            self._new_entity_accuracy(predictions=log_probs[mask],
                                      gold_labels=target_inds[mask])
            self._new_entity_accuracy20(predictions=log_probs[mask],
                                        gold_labels=target_inds[mask])

            # Return the token-wise average loss
            return -target_log_probs.sum() / (target_mask.sum() + 1e-13)

        else:
            logits = self._fc_new_entity(encoded)
            log_probs = F.log_softmax(logits, dim=-1)

            num_categories = log_probs.shape[-1]
            flat_log_probs = log_probs.view(-1, num_categories)
            flat_target_inds = target_inds.view(-1)
            target_log_probs = torch.gather(flat_log_probs, -1, flat_target_inds.unsqueeze(-1)).squeeze(-1)

            mask = ~flat_target_inds.eq(0)
            target_log_probs[~mask] = 0

            self._new_entity_accuracy(predictions=flat_log_probs[mask],
                                      gold_labels=flat_target_inds[mask])
            self._new_entity_accuracy20(predictions=flat_log_probs[mask],
                                        gold_labels=flat_target_inds[mask])

            return -target_log_probs.sum() / (target_mask.sum() + 1e-13)

    def _parent_log_probs(self,
                          encoded_head: torch.Tensor,
                          entity_ids: torch.Tensor,
                          parent_ids: torch.Tensor) -> torch.Tensor:
        # Lookup recent entities (which are candidates for parents) and get their embeddings.
        candidate_ids, candidate_mask = self._recent_entities(entity_ids)
        logger.debug('Candidate ids shape: %s', candidate_ids.shape)
        candidate_embeddings = embedded_dropout(self._entity_embedder,
                                                words=candidate_ids,
                                                dropout=self._dropoute if self.training else 0)

        # Logits are computed using a general bilinear form that measures the similarity between
        # the projected hidden state and the embeddings of candidate entities
        encoded = self._locked_dropout(encoded_head, self._dropout)
        selection_logits = torch.bmm(encoded, candidate_embeddings.transpose(1, 2))

        # Get log probabilities using masked softmax (need to double check mask works properly).

        # shape: (batch_size, sequence_length, num_candidates)
        log_probs = masked_log_softmax(selection_logits, candidate_mask)

        # Now for the tricky part. We need to convert the parent ids to a mask that selects the
        # relevant probabilities from log_probs. To do this we need to align the candidates with
        # the parent ids, which can be achieved by an element-wise equality comparison. We also
        # need to ensure that null parents are not selected.

        # shape: (batch_size, sequence_length, num_parents, 1)
        _parent_ids = parent_ids.unsqueeze(-1)

        batch_size, num_candidates = candidate_ids.shape
        # shape: (batch_size, 1, 1, num_candidates)
        _candidate_ids = candidate_ids.view(batch_size, 1, 1, num_candidates)

        # shape: (batch_size, sequence_length, num_parents, num_candidates)
        is_parent = _parent_ids.eq(_candidate_ids)
        # shape: (batch_size, 1, 1, num_candidates)
        non_null = ~_candidate_ids.eq(0)

        # Since multiplication is addition in log-space, we can apply mask by adding its log (+
        # some small constant for numerical stability).
        mask = is_parent & non_null
        masked_log_probs = log_probs.unsqueeze(2) + (mask.float() + 1e-45).log()
        logger.debug('Masked log probs shape: %s', masked_log_probs.shape)

        # Lastly, we need to get rid of the num_candidates dimension. The easy way to do this would
        # be to marginalize it out. However, since our data is sparse (the last two dims are
        # essentially a delta function) this would add a lot of unneccesary terms to the computation graph.
        # To get around this we are going to try to use a gather.
        _, index = torch.max(mask, dim=-1, keepdim=True)
        target_log_probs = torch.gather(masked_log_probs, dim=-1, index=index).squeeze(-1)

        return target_log_probs

    def _generate_scores(self,
                         encoded: torch.Tensor,
                         entity_ids: torch.Tensor) -> torch.Tensor:
        entity_embeddings = embedded_dropout(embed=self._entity_embedder,
                                             words=entity_ids,
                                             dropout=self._dropoute if self.training else 0)
        concatenated = torch.cat((encoded, entity_embeddings), dim=-1)
        condensed = self._fc_condense(concatenated)
        return self._fc_generate(condensed)

    def _copy_scores(self,
                     encoded: torch.Tensor,
                     alias_tokens: torch.Tensor) -> torch.Tensor:
        # Begin by flattening the tokens so that they fit the expected shape of a
        # ``Seq2SeqEncoder``.
        batch_size, sequence_length, num_aliases, alias_length = alias_tokens.shape
        flattened = alias_tokens.view(-1, alias_length)
        copy_mask = flattened != 0
        if copy_mask.sum() == 0:
            return encoded.new_zeros((batch_size, sequence_length, num_aliases * alias_length),
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
        projected = projected.view(batch_size * sequence_length, -1, num_aliases * alias_length)
        copy_scores = torch.bmm(encoded, projected).squeeze()
        copy_scores = copy_scores.view(batch_size, sequence_length, -1).contiguous()
        logger.debug('Copy scores shape: %s', copy_scores.shape)

        return copy_scores

    def _vocab_loss(self,
                    generate_scores: torch.Tensor,
                    copy_scores: torch.Tensor,
                    target_tokens: torch.Tensor,
                    target_alias_indices: torch.Tensor,
                    mask: torch.Tensor,
                    alias_indices: torch.Tensor,
                    mention_mask: torch.Tensor):
        batch_size, sequence_length, vocab_size = generate_scores.shape
        copy_sequence_length = copy_scores.shape[-1]

        # Flat sequences make life **much** easier.
        flattened_targets = target_tokens.view(batch_size * sequence_length, 1)
        flattened_mask = mask.view(-1, 1).byte()

        # In order to obtain proper log probabilities we create a mask to omit padding alias tokens
        # from the calculation.
        alias_mask = alias_indices.view(batch_size, sequence_length, -1).gt(0)
        score_mask = mask.new_ones(batch_size, sequence_length, vocab_size + copy_sequence_length)
        score_mask[:, :, vocab_size:] = alias_mask

        # The log-probability distribution is then given by taking the masked log softmax.
        concatenated_scores = torch.cat((generate_scores, copy_scores), dim=-1)
        log_probs = masked_log_softmax(concatenated_scores, score_mask)

        # GENERATE LOSS ###
        # The generated token loss is a simple cross-entropy calculation, we can just gather
        # the log probabilties...
        flattened_log_probs = log_probs.view(batch_size * sequence_length, -1)
        generate_log_probs_source_vocab = flattened_log_probs.gather(1, flattened_targets)
        # ...except we need to ignore the contribution of UNK tokens that are copied (only when
        # computing the loss). To do that we create a mask which is 1 only if the token is not a
        # copied UNK (or padding).
        unks = target_tokens.eq(self._unk_index).view(-1, 1)
        copied = target_alias_indices.gt(0).view(-1, 1)
        generate_mask = ~(unks & copied) & flattened_mask
        # Since we are in log-space we apply the mask by addition.
        generate_log_probs_extended_vocab = generate_log_probs_source_vocab + (generate_mask.float() + 1e-45).log()

        # COPY LOSS ###
        copy_log_probs = flattened_log_probs[:, vocab_size:]
        # When computing the loss we need to get the log probability of **only** the copied tokens.
        alias_indices = alias_indices.view(batch_size * sequence_length, -1)
        target_alias_indices = target_alias_indices.view(-1, 1)
        copy_mask = alias_indices.eq(target_alias_indices) & flattened_mask & target_alias_indices.gt(0)
        copy_log_probs = copy_log_probs + (copy_mask.float() + 1e-45).log()

        # COMBINED LOSS ###
        # The final loss term is computed using our log probs computed w.r.t to the entire
        # vocabulary.
        combined_log_probs_extended_vocab = torch.cat((generate_log_probs_extended_vocab,
                                                       copy_log_probs),
                                                      dim=1)
        combined_log_probs_extended_vocab = torch.logsumexp(combined_log_probs_extended_vocab,
                                                            dim=1)
        flattened_mask = flattened_mask.squeeze()
        # Zero out padding loss
        combined_log_probs_extended_vocab = combined_log_probs_extended_vocab * flattened_mask.float()
        vocab_loss = -combined_log_probs_extended_vocab.sum() / (mask.sum() + 1e-13)

        # Unknown penalty - only applies to non-copied unks
        true_unks = unks.squeeze() & ~copied.squeeze() & flattened_mask
        penalized_log_probs = combined_log_probs_extended_vocab - self._unk_penalty * true_unks.float()
        penalized_log_probs[~flattened_mask] = 0
        penalized_vocab_loss = -penalized_log_probs.sum() / (mask.sum() + 1e-13)

        # PERPLEXITY ###
        # Our perplexity terms are computed using the log probs computed w.r.t the source
        # vocabulary.
        combined_log_probs_source_vocab = torch.cat((generate_log_probs_source_vocab,
                                                     copy_log_probs),
                                                    dim=1)
        combined_log_probs_source_vocab = torch.logsumexp(combined_log_probs_source_vocab,
                                                          dim=1)

        # For UPP we penalize **only** p(UNK); not the copy probabilities!
        penalized_log_probs_source_vocab = generate_log_probs_source_vocab - self._unk_penalty * unks.float()
        penalized_log_probs_source_vocab = torch.cat((penalized_log_probs_source_vocab,
                                                      copy_log_probs),
                                                     dim=1)
        penalized_log_probs_source_vocab = torch.logsumexp(penalized_log_probs_source_vocab,
                                                           dim=1)

        kg_mask = (mention_mask * mask.byte()).view(-1)
        bg_mask = ((1 - mention_mask) * mask.byte()).view(-1)
        mask = (kg_mask | bg_mask)

        self._ppl(-combined_log_probs_source_vocab[mask].sum(), mask.float().sum() + 1e-13)
        self._upp(-penalized_log_probs_source_vocab[mask].sum(), mask.float().sum() + 1e-13)
        if kg_mask.any():
            self._kg_ppl(-combined_log_probs_source_vocab[kg_mask].sum(), kg_mask.float().sum() + 1e-13)
        if bg_mask.any():
            self._bg_ppl(-combined_log_probs_source_vocab[bg_mask].sum(), bg_mask.float().sum() + 1e-13)

        return vocab_loss, penalized_vocab_loss

    def _forward_loop(self,
                      source: Dict[str, torch.Tensor],
                      target: Dict[str, torch.Tensor],
                      alias_database: AliasDatabase,
                      mention_type: torch.Tensor,
                      raw_entity_ids: Dict[str, torch.Tensor],
                      entity_ids: Dict[str, torch.Tensor],
                      parent_ids: Dict[str, torch.Tensor],
                      relations: Dict[str, torch.Tensor],
                      shortlist: Dict[str, torch.Tensor],
                      shortlist_inds: torch.Tensor,
                      alias_copy_inds: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get the token mask and extract indexed text fields.
        # shape: (batch_size, sequence_length)
        target_mask = get_text_field_mask(target)
        source = source['tokens']
        target = target['tokens']
        raw_entity_ids = raw_entity_ids['raw_entity_ids']
        entity_ids = entity_ids['entity_ids']

        logger.debug('Source & Target shape: %s', source.shape)
        logger.debug('Entity ids shape: %s', entity_ids.shape)
        logger.debug('Shortlist shape: %s', shortlist['entity_ids'].shape)
        # Embed source tokens.
        # shape: (batch_size, sequence_length, embedding_dim)
        encoded, alpha_loss, beta_loss = self._encode_source(source)
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim]
        encoded_token, encoded_head = encoded.split(splits, dim=-1)

        # Predict whether or not the next token will be an entity mention, and if so which type.
        mention_type = mention_type.gt(0).long() #  Map 1, 2 -> 1
        mention_type_loss = self._mention_type_loss(encoded_token, mention_type, target_mask)
        self._avg_mention_type_loss(float(mention_type_loss))

        # For new mentions, predict which entity (among those in the supplied shortlist) will be
        # mentioned.
        if self._use_shortlist:
            new_entity_loss = self._new_entity_loss(encoded_head,
                                                    shortlist_inds,
                                                    shortlist,
                                                    target_mask)
        else:
            new_entity_loss = self._new_entity_loss(encoded_head,
                                                    entity_ids,
                                                    None,
                                                    target_mask)

        self._avg_new_entity_loss(float(new_entity_loss))

        # Predict generation-mode scores. Note: these are W.R.T to entity_ids since we need the embedding.
        generate_scores = self._generate_scores(encoded_token, entity_ids)

        # Predict copy-mode scores. Note: these are W.R.T raw_entity_ids since we need to look up aliases.
        alias_tokens, alias_inds = alias_database.lookup(raw_entity_ids)
        copy_scores = self._copy_scores(encoded_token, alias_tokens)

        # Combine scores to get vocab loss
        vocab_loss, penalized_vocab_loss = self._vocab_loss(generate_scores,
                                                            copy_scores,
                                                            target,
                                                            alias_copy_inds,
                                                            target_mask,
                                                            alias_inds,
                                                            entity_ids.gt(0))
        self._avg_vocab_loss(float(vocab_loss))

        # Compute total loss. Also compute logp (needed for importance sampling evaluation).
        loss = vocab_loss + mention_type_loss + new_entity_loss
        logp = -(vocab_loss + mention_type_loss + new_entity_loss) * target_mask.sum()
        penalized_logp = -(penalized_vocab_loss + mention_type_loss + new_entity_loss) * target_mask.sum()

        # Activation regularization
        if self._alpha:
            loss = loss + self._alpha * alpha_loss
        # Temporal activation regularization (slowness)
        if self._beta:
            loss = loss + self._beta * beta_loss

        return {'loss': loss, 'logp': logp, 'penalized_logp': penalized_logp}

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
        out = {
            'ppl': self._ppl.get_metric(reset),
            'upp': self._upp.get_metric(reset),
            'kg_ppl': self._kg_ppl.get_metric(reset),
            'bg_ppl': self._bg_ppl.get_metric(reset),
            'type': self._avg_mention_type_loss.get_metric(reset),
            'new': self._avg_new_entity_loss.get_metric(reset),
            'vocab': self._avg_vocab_loss.get_metric(reset),
        }
        # if not self.training:
        p, r, f = self._new_mention_f1.get_metric(reset)
        out['new_p'] = p
        out['new_r'] = r
        out['new_f1'] = f
        out['new_ent_acc'] = self._new_entity_accuracy.get_metric(reset)
        out['new_ent_acc_20'] = self._new_entity_accuracy20.get_metric(reset)
        return out
