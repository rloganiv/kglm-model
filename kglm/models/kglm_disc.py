import logging
import math
from typing import Any, Dict, List, Optional

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import (get_text_field_mask, masked_log_softmax, masked_softmax,
    sequence_cross_entropy_with_logits)
from allennlp.training.metrics import Average, CategoricalAccuracy, F1Measure, SequenceAccuracy
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.data import AliasDatabase
from kglm.modules import (embedded_dropout, LockedDropout, WeightDrop, KnowledgeGraphLookup,
    RecentEntities)
from kglm.nn.util import nested_enumerate, parallel_sample
from kglm.training.metrics import Ppl

logger = logging.getLogger(__name__)


@Model.register('kglm-disc')
class KglmDisc(Model):
    """
    Knowledge graph language model discriminator (for importance sampling).

    Parameters
    ----------
    vocab : ``Vocabulary``
        The model vocabulary.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 entity_embedder: TextFieldEmbedder,
                 relation_embedder: TextFieldEmbedder,
                 knowledge_graph_path: str,
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
        super(KglmDisc, self).__init__(vocab)

        # We extract the `Embedding` layers from the `TokenEmbedders` to apply dropout later on.
        # pylint: disable=protected-access
        self._token_embedder = token_embedder._token_embedders['tokens']
        self._entity_embedder = entity_embedder._token_embedders['entity_ids']
        self._relation_embedder = relation_embedder._token_embedders['relations']
        self._recent_entities = RecentEntities(cutoff=cutoff)
        self._knowledge_graph_lookup = KnowledgeGraphLookup(knowledge_graph_path, vocab=vocab)
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
                output_size = token_embedding_dim + 2 * entity_embedding_dim
            else:
                output_size = hidden_size
            rnns.append(torch.nn.LSTM(input_size, output_size, batch_first=True))
        rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in rnns]
        self.rnns = torch.nn.ModuleList(rnns)

        # Various linear transformations.
        self._fc_mention_type = torch.nn.Linear(
            in_features=token_embedding_dim,
            out_features=4)

        if not use_shortlist:
            self._fc_new_entity = torch.nn.Linear(
                in_features=entity_embedding_dim,
                out_features=vocab.get_vocab_size('entity_ids'))

            if tie_weights:
                self._fc_new_entity.weight = self._entity_embedder.weight

        self._state: Optional[Dict[str, Any]] = None

        # Metrics
        self._unk_index = vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self._unk_penalty = math.log(vocab.get_vocab_size('tokens_unk'))
        self._avg_mention_type_loss = Average()
        self._avg_new_entity_loss = Average()
        self._avg_knowledge_graph_entity_loss = Average()
        self._new_mention_f1 =  F1Measure(positive_label=1)
        self._kg_mention_f1 = F1Measure(positive_label=2)
        self._new_entity_accuracy = CategoricalAccuracy()
        self._new_entity_accuracy20 = CategoricalAccuracy(top_k=20)
        self._parent_ppl = Ppl()
        self._relation_ppl = Ppl()

        initializer(self)

    def sample(self,
               source: Dict[str, torch.Tensor],
               target: Dict[str, torch.Tensor],
               reset: torch.Tensor,
               metadata: Dict[str, Any],
               alias_copy_inds: torch.Tensor,
               shortlist: Dict[str, torch.Tensor] = None,
               **kwargs) -> Dict[str, Any]:  # **kwargs intended to eat the other fields if they are provided.
        """
        Sampling annotations for the generative model. Note that unlike forward, this function
        expects inputs from a **generative** dataset reader, not a **discriminative** one.
        """

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

        logp = 0.0

        mask = get_text_field_mask(target).byte()
        # We encode the target tokens (**not** source) since the discriminitative model makes
        # predictions on the current token, but the generative model expects labels for the
        # **next** (e.g. target) token!
        encoded, *_ = self._encode_source(target['tokens'])
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim] * 2
        encoded_token, encoded_head, encoded_relation = encoded.split(splits, dim=-1)

        # Compute new mention logits
        mention_logits = self._fc_mention_type(encoded_token)
        mention_probs = F.softmax(mention_logits, dim=-1)
        mention_type = parallel_sample(mention_probs)
        mention_logp = mention_probs.gather(-1, mention_type.unsqueeze(-1)).log()
        mention_logp[~mask] = 0
        mention_logp = mention_logp.sum()

        # Compute entity logits
        new_entity_mask = mention_type.eq(1)
        new_entity_logits = self._new_entity_logits(encoded_head + encoded_relation, shortlist)
        if self._use_shortlist:
            # If using shortlist, then samples are indexed w.r.t the shortlist and entity_ids must be looked up
            shortlist_mask = get_text_field_mask(shortlist)
            new_entity_probs = masked_softmax(new_entity_logits, shortlist_mask)
            shortlist_inds = torch.zeros_like(mention_type)
            # Some sequences may be full of padding in which case the shortlist
            # is empty
            not_just_padding = shortlist_mask.byte().any(-1)
            shortlist_inds[not_just_padding] = parallel_sample(new_entity_probs[not_just_padding])
            shortlist_inds[~new_entity_mask] = 0
            _new_entity_logp = new_entity_probs.gather(-1, shortlist_inds.unsqueeze(-1)).log()
            new_entity_samples = shortlist['entity_ids'].gather(1, shortlist_inds)
        else:
            new_entity_logits = new_entity_logits
            # If not using shortlist, then samples are indexed w.r.t to the global vocab
            new_entity_probs = F.softmax(new_entity_logits, dim=-1)
            new_entity_samples = parallel_sample(new_entity_probs)
            _new_entity_logp = new_entity_probs.gather(-1, new_entity_samples.unsqueeze(-1)).log()
            shortlist_inds = None
        # Zero out masked tokens and non-new entity predictions
        _new_entity_logp[~mask] = 0
        _new_entity_logp[~new_entity_mask] = 0
        new_entity_logp = _new_entity_logp.sum()

        # Start filling in the entity ids
        entity_ids = torch.zeros_like(target['tokens'])
        entity_ids[new_entity_mask] = new_entity_samples[new_entity_mask]

        # ...UGH... we also need the raw ids - remapping time
        raw_entity_ids = torch.zeros_like(target['tokens'])
        for *index, entity_id in nested_enumerate(entity_ids.tolist()):
            token = self.vocab.get_token_from_index(entity_id, 'entity_ids')
            raw_entity_id = self.vocab.get_token_index(token, 'raw_entity_ids')
            raw_entity_ids[tuple(index)] = raw_entity_id

        # Derived mentions need to be computed sequentially.
        parent_ids = torch.zeros_like(target['tokens']).unsqueeze(-1)
        derived_entity_mask = mention_type.eq(2)
        derived_entity_logp = 0.0

        sequence_length = target['tokens'].shape[1]
        for i in range(sequence_length):

            current_mask = derived_entity_mask[:, i] & mask[:, i]

            ## SAMPLE PARENTS ##

            # Update recent entities with **current** entity only
            current_entity_id = entity_ids[:, i].unsqueeze(1)
            candidate_ids, candidate_mask = self._recent_entities(current_entity_id)

            # If no mentions are derived, there is no point continuing after entities have been updated.
            if not current_mask.any():
                continue

            # Otherwise we proceed
            candidate_embeddings = self._entity_embedder(candidate_ids)

            # Compute logits w.r.t **current** hidden state only
            current_head_encoding = encoded_head[:, i].unsqueeze(1)
            selection_logits = torch.bmm(current_head_encoding, candidate_embeddings.transpose(1, 2))
            selection_probs = masked_softmax(selection_logits, candidate_mask)

            # Only sample if the is at least one viable candidate (e.g. if a sampling distribution
            # has no probability mass we cannot sample from it). Return zero as the parent for
            # non-viable distributions.
            viable_candidate_mask = candidate_mask.any(-1).squeeze()
            _parent_ids = torch.zeros_like(current_entity_id)
            parent_logp = torch.zeros_like(current_entity_id, dtype=torch.float32)
            if viable_candidate_mask.any():
                viable_candidate_ids = candidate_ids[viable_candidate_mask]
                viable_candidate_probs = selection_probs[viable_candidate_mask]
                viable_parent_samples = parallel_sample(viable_candidate_probs)
                viable_logp = viable_candidate_probs.gather(-1, viable_parent_samples.unsqueeze(-1)).log()
                viable_parent_ids = viable_candidate_ids.gather(-1, viable_parent_samples)
                _parent_ids[viable_candidate_mask] = viable_parent_ids
                parent_logp[viable_candidate_mask] = viable_logp.squeeze(-1)

            parent_ids[current_mask, i] = _parent_ids[current_mask]  # TODO: Double-check
            derived_entity_logp += parent_logp[current_mask].sum()

            ## SAMPLE RELATIONS ##

            # Lookup sampled parent ids in the knowledge graph
            indices, parent_ids_list, relations_list, tail_ids_list = self._knowledge_graph_lookup(_parent_ids)
            relation_embeddings = [self._relation_embedder(r) for r in relations_list]

            # Sample tail ids
            current_relation_encoding = encoded_relation[:, i].unsqueeze(1)
            _raw_tail_ids = torch.zeros_like(_parent_ids).squeeze(-1)
            _tail_ids = torch.zeros_like(_parent_ids).squeeze(-1)
            for index, relation_embedding, tail_id_lookup in zip(indices, relation_embeddings, tail_ids_list):
                # Compute the score for each relation w.r.t the current encoding. NOTE: In the loss
                # code index has a slice. We don't need that here since there is always a
                # **single** parent.
                logits = torch.mv(relation_embedding, current_relation_encoding[index])
                # Convert to probability
                tail_probs = F.softmax(logits, dim=-1)
                # Sample
                tail_sample = torch.multinomial(tail_probs, 1)
                # Get logp. Ignoring the current_mask here is **super** dodgy, but since we forced
                # null parents to zero we shouldn't be accumulating probabilities for unused predictions.
                tail_logp = tail_probs.gather(-1, tail_sample).log()
                derived_entity_logp += tail_logp.sum()  # Sum is redundant, just need it to make logp a scalar

                # Map back to raw id
                raw_tail_id = tail_id_lookup[tail_sample]
                # Convert raw id to id
                tail_id_string = self.vocab.get_token_from_index(raw_tail_id.item(), 'raw_entity_ids')
                tail_id = self.vocab.get_token_index(tail_id_string, 'entity_ids')

                _raw_tail_ids[index[:-1]] = raw_tail_id
                _tail_ids[index[:-1]] = tail_id

            raw_entity_ids[current_mask, i] = _raw_tail_ids[current_mask]  # TODO: Double-check
            entity_ids[current_mask, i] = _tail_ids[current_mask]  # TODO: Double-check

            self._recent_entities.insert(_tail_ids, current_mask)

            ## CONTINUE MENTIONS ##
            continue_mask = mention_type[:, i].eq(3) & mask[:, i]
            if not current_mask.any() or i==0:
                continue
            raw_entity_ids[continue_mask, i] = raw_entity_ids[continue_mask, i-1]
            entity_ids[continue_mask, i] = entity_ids[continue_mask, i-1]
            entity_ids[continue_mask, i] = entity_ids[continue_mask, i-1]
            parent_ids[continue_mask, i] = parent_ids[continue_mask, i-1]
            if self._use_shortlist:
                shortlist_inds[continue_mask, i] = shortlist_inds[continue_mask, i-1]
            alias_copy_inds[continue_mask, i] = alias_copy_inds[continue_mask, i-1]

        # Lastly, because entities won't always match the true entity ids, we need to zero out any alias copy ids that won't be valid.
        true_raw_entity_ids = kwargs['raw_entity_ids']['raw_entity_ids']
        invalid_id_mask = ~true_raw_entity_ids.eq(raw_entity_ids)
        alias_copy_inds[invalid_id_mask] = 0

        # Pass denotes fields that are passed directly from input to output.
        sample = {
            'source': source,  # Pass
            'target': target,  # Pass
            'reset': reset,  # Pass
            'metadata': metadata,  # Pass
            'mention_type': mention_type,
            'raw_entity_ids': {'raw_entity_ids': raw_entity_ids},
            'entity_ids': {'entity_ids': entity_ids},
            'parent_ids': {'entity_ids': parent_ids},
            'relations': {'relations': None},  # We aren't using them - eventually should remove entirely
            'shortlist': shortlist,  # Pass
            'shortlist_inds': shortlist_inds,
            'alias_copy_inds': alias_copy_inds
        }
        logp = mention_logp + new_entity_logp + derived_entity_logp
        return {'sample': sample, 'logp': logp}

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                source: Dict[str, torch.Tensor],
                reset: torch.Tensor,
                metadata: List[Dict[str, Any]],
                mention_type: torch.Tensor = None,
                raw_entity_ids: Dict[str, torch.Tensor] = None,
                entity_ids: Dict[str, torch.Tensor] = None,
                parent_ids: Dict[str, torch.Tensor] = None,
                relations: Dict[str, torch.Tensor] = None,
                shortlist: Dict[str, torch.Tensor] = None,
                shortlist_inds: torch.Tensor = None) -> Dict[str, torch.Tensor]:

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
                alias_database=alias_database,
                mention_type=mention_type,
                raw_entity_ids=raw_entity_ids,
                entity_ids=entity_ids,
                parent_ids=parent_ids,
                relations=relations,
                shortlist=shortlist,
                shortlist_inds=shortlist_inds)
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
        self._kg_mention_f1(predictions=logits,
                            gold_labels=mention_type,
                            mask=mask)

        return mention_type_loss

    def _new_entity_logits(self,
                           encoded: torch.Tensor,
                           shortlist: torch.Tensor) -> torch.Tensor:
        if self._use_shortlist:
            # Embed the shortlist entries
            shortlist_embeddings = embedded_dropout(
                embed=self._entity_embedder,
                words=shortlist['entity_ids'],
                dropout=self._dropoute if self.training else 0)
            # Compute logits using inner product between the predicted entity embedding and the
            # embeddings of entities in the shortlist
            encodings = self._locked_dropout(encoded, self._dropout)
            logits = torch.bmm(encodings, shortlist_embeddings.transpose(1, 2))
        else:
            logits = self._fc_new_entity(encoded)
        return logits

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
        logits = self._new_entity_logits(encoded, shortlist)
        if self._use_shortlist:
            # Take masked softmax to get log probabilties and gather the targets.
            shortlist_mask = get_text_field_mask(shortlist)
            log_probs = masked_log_softmax(logits, shortlist_mask)
        else:
            logits = logits
            log_probs = F.log_softmax(logits, dim=-1)
            num_categories = log_probs.shape[-1]
            log_probs = log_probs.view(-1, num_categories)
            target_inds = target_inds.view(-1)
        target_log_probs = torch.gather(log_probs, -1, target_inds.unsqueeze(-1)).squeeze(-1)

        mask = ~target_inds.eq(0)
        target_log_probs[~mask] = 0

        if mask.any():
            self._new_entity_accuracy(predictions=log_probs[mask],
                                    gold_labels=target_inds[mask])
            self._new_entity_accuracy20(predictions=log_probs[mask],
                                        gold_labels=target_inds[mask])

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

    def _relation_log_probs(self,
                            encoded_relation: torch.Tensor,
                            raw_entity_ids: torch.Tensor,
                            parent_ids: torch.Tensor) -> torch.Tensor:

        # Lookup edges out of parents
        indices, parent_ids_list, relations_list, tail_ids_list = self._knowledge_graph_lookup(parent_ids)

        # Embed relations
        relation_embeddings = [self._relation_embedder(r) for r in relations_list]

        # Logits are computed using a general bilinear form that measures the similarity between
        # the projected hidden state and the embeddings of relations
        encoded = self._locked_dropout(encoded_relation, self._dropout)

        # This is a little funky, but to avoid massive amounts of padding we are going to just
        # iterate over the relation and tail_id vectors one-by-one.
        # shape: (batch_size, sequence_length, num_parents, num_relations)
        target_log_probs = encoded.new_empty(*parent_ids.shape).fill_(math.log(1e-45))
        for index, parent_id, relation_embedding, tail_id in zip(indices, parent_ids_list, relation_embeddings, tail_ids_list):
            # First we compute the score for each relation w.r.t the current encoding, and convert
            # the scores to log-probabilities
            logits = torch.mv(relation_embedding, encoded[index[:-1]])
            logger.debug('Relation logits shape: %s', logits.shape)
            log_probs = F.log_softmax(logits, dim=-1)

            # Next we gather the log probs for edges with the correct tail entity and sum them up
            target_id = raw_entity_ids[index[:-1]]
            mask = tail_id.eq(target_id)
            relevant_log_probs = log_probs.masked_select(tail_id.eq(target_id))
            target_log_prob = torch.logsumexp(relevant_log_probs, dim=0)
            target_log_probs[index] = target_log_prob

        return target_log_probs

    def _knowledge_graph_entity_loss(self,
                                     encoded_head: torch.Tensor,
                                     encoded_relation: torch.Tensor,
                                     raw_entity_ids: torch.Tensor,
                                     entity_ids: torch.Tensor,
                                     parent_ids: torch.Tensor,
                                     target_mask: torch.Tensor) -> torch.Tensor:
        # First get the log probabilities of the parents and relations that lead to the current
        # entity.
        parent_log_probs = self._parent_log_probs(encoded_head, entity_ids, parent_ids)
        relation_log_probs = self._relation_log_probs(encoded_relation, raw_entity_ids, parent_ids)
        # Next take their product + marginalize
        combined_log_probs = parent_log_probs + relation_log_probs
        target_log_probs = torch.logsumexp(combined_log_probs, dim=-1)
        # Zero out any non-kg predictions
        mask = ~parent_ids.eq(0).all(dim=-1)
        target_log_probs = target_log_probs * mask.float()
        # If validating, measure ppl of the predictions:
        # if not self.training:
        self._parent_ppl(-torch.logsumexp(parent_log_probs, dim=-1)[mask].sum(), mask.float().sum())
        self._relation_ppl(-torch.logsumexp(relation_log_probs, dim=-1)[mask].sum(), mask.float().sum())
        # Lastly return the tokenwise average loss
        return -target_log_probs.sum() / (target_mask.sum() + 1e-13)

    def _forward_loop(self,
                      source: Dict[str, torch.Tensor],
                      alias_database: AliasDatabase,
                      mention_type: torch.Tensor,
                      raw_entity_ids: Dict[str, torch.Tensor],
                      entity_ids: Dict[str, torch.Tensor],
                      parent_ids: Dict[str, torch.Tensor],
                      relations: Dict[str, torch.Tensor],
                      shortlist: Dict[str, torch.Tensor],
                      shortlist_inds: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get the token mask and extract indexed text fields.
        # shape: (batch_size, sequence_length)
        target_mask = get_text_field_mask(source)
        source = source['tokens']
        raw_entity_ids = raw_entity_ids['raw_entity_ids']
        entity_ids = entity_ids['entity_ids']
        parent_ids = parent_ids['entity_ids']
        relations = relations['relations']

        logger.debug('Source & Target shape: %s', source.shape)
        logger.debug('Entity ids shape: %s', entity_ids.shape)
        logger.debug('Relations & Parent ids shape: %s', relations.shape)
        logger.debug('Shortlist shape: %s', shortlist['entity_ids'].shape)
        # Embed source tokens.
        # shape: (batch_size, sequence_length, embedding_dim)
        encoded, alpha_loss, beta_loss = self._encode_source(source)
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim] * 2
        encoded_token, encoded_head, encoded_relation = encoded.split(splits, dim=-1)

        # Predict whether or not the next token will be an entity mention, and if so which type.
        mention_type_loss = self._mention_type_loss(encoded_token, mention_type, target_mask)
        self._avg_mention_type_loss(float(mention_type_loss))

        # For new mentions, predict which entity (among those in the supplied shortlist) will be
        # mentioned.
        if self._use_shortlist:
            new_entity_loss = self._new_entity_loss(encoded_head + encoded_relation,
                                                    shortlist_inds,
                                                    shortlist,
                                                    target_mask)
        else:
            new_entity_loss = self._new_entity_loss(encoded_head + encoded_relation,
                                                    entity_ids,
                                                    None,
                                                    target_mask)

        self._avg_new_entity_loss(float(new_entity_loss))

        # For derived mentions, first predict which parent(s) to expand...
        knowledge_graph_entity_loss = self._knowledge_graph_entity_loss(encoded_head,
                                                                        encoded_relation,
                                                                        raw_entity_ids,
                                                                        entity_ids,
                                                                        parent_ids,
                                                                        target_mask)
        self._avg_knowledge_graph_entity_loss(float(knowledge_graph_entity_loss))

        # Compute total loss
        loss = mention_type_loss + new_entity_loss + knowledge_graph_entity_loss

        # Activation regularization
        if self._alpha:
            loss = loss + self._alpha * alpha_loss
        # Temporal activation regularization (slowness)
        if self._beta:
            loss = loss + self._beta * beta_loss

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
        out =  {
            'type': self._avg_mention_type_loss.get_metric(reset),
            'new': self._avg_new_entity_loss.get_metric(reset),
            'kg': self._avg_knowledge_graph_entity_loss.get_metric(reset),
        }
        # if not self.training:
        p, r, f = self._new_mention_f1.get_metric(reset)
        out['new_p'] = p
        out['new_r'] = r
        out['new_f1'] = f
        p, r, f = self._kg_mention_f1.get_metric(reset)
        out['kg_p'] = p
        out['kg_r'] = r
        out['kg_f1'] = f
        out['new_ent_acc'] = self._new_entity_accuracy.get_metric(reset)
        out['new_ent_acc_20'] = self._new_entity_accuracy20.get_metric(reset)
        out['parent_ppl'] = self._parent_ppl.get_metric(reset)
        out['relation_ppl'] = self._relation_ppl.get_metric(reset)
        return out
