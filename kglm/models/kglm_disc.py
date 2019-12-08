import logging
from copy import deepcopy
from collections import namedtuple
import math
from typing import Any, Dict, List, Optional, Tuple

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import (get_text_field_mask, masked_log_softmax, masked_softmax,
    sequence_cross_entropy_with_logits)
from allennlp.training.metrics import Average, CategoricalAccuracy, F1Measure, SequenceAccuracy
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.data import AliasDatabase
from kglm.modules import (embedded_dropout, LockedDropout, WeightDroppedLstm, KnowledgeGraphLookup,
    RecentEntities)
from kglm.nn.util import nested_enumerate, parallel_sample
from kglm.training.metrics import Ppl

logger = logging.getLogger(__name__)


# Decoding from the KGLM discriminator requires ensuring that:
#   * New mentions cannot be of recently mentioned entities.
#   * Related mentions must be related to a recently mentioned entity.
#   * Ongoing mention cannot continue non-mentions.
# The following structure tracks this information when performing beam search.
KglmBeamState = namedtuple('KglmBeamState', ['recent_entities', 'ongoing'])


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
        rnn_output_dim = token_embedding_dim + 2 * entity_embedding_dim
        self._rnn = WeightDroppedLstm(num_layers=num_layers,
                                      input_embedding_dim=self.token_embedding_dim,
                                      hidden_size=self._hidden_size,
                                      output_embedding_dim=rnn_output_dim,
                                      dropout=self._wdrop)

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
        self.reset_states(reset)

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
        self.reset_states(reset)

        mask = get_text_field_mask(target).byte()
        batch_size =  mask.shape[0]
        # We encode the target tokens (**not** source) since the discriminitative model makes
        # predictions on the current token, but the generative model expects labels for the
        # **next** (e.g. target) token!
        encoded, *_ = self._encode_source(target['tokens'])
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim] * 2
        encoded_token, encoded_head, encoded_relation = encoded.split(splits, dim=-1)

        # logp = 0.0
        logp = encoded.new_zeros(batch_size)

        # Compute new mention logits
        mention_logits = self._fc_mention_type(encoded_token)
        mention_probs = F.softmax(mention_logits, dim=-1)
        mention_type = parallel_sample(mention_probs)
        _mention_logp = mention_probs.gather(-1, mention_type.unsqueeze(-1)).log()
        _mention_logp[~mask] = 0
        mention_logp = _mention_logp.view(batch_size, -1).sum(-1)

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
            new_entity_logits[:,:,:4] = -1e32  # A new entity mustn't be padding, unknown, or a literal
            # If not using shortlist, then samples are indexed w.r.t to the global vocab
            new_entity_probs = F.softmax(new_entity_logits, dim=-1)
            new_entity_samples = parallel_sample(new_entity_probs)
            _new_entity_logp = new_entity_probs.gather(-1, new_entity_samples.unsqueeze(-1)).log()
            shortlist_inds = None
        # Zero out masked tokens and non-new entity predictions
        _new_entity_logp[~mask] = 0
        _new_entity_logp[~new_entity_mask] = 0
        new_entity_logp = _new_entity_logp.view(batch_size, -1).sum(-1)

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
        derived_entity_logp = torch.zeros_like(new_entity_logp)

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
            derived_entity_logp[current_mask] += parent_logp[current_mask].squeeze(-1)

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
                derived_entity_logp[index[:-1]] += tail_logp.sum()  # Sum is redundant, just need it to make logp a scalar

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

    def get_raw_entity_ids(self, entity_ids: torch.LongTensor) ->  torch.LongTensor:
        raw_entity_ids = torch.zeros_like(entity_ids)
        for *index, entity_id in nested_enumerate(entity_ids.tolist()):
            token = self.vocab.get_token_from_index(entity_id, 'entity_ids')
            raw_entity_id = self.vocab.get_token_index(token, 'raw_entity_ids')
            raw_entity_ids[tuple(index)] = raw_entity_id
        return raw_entity_ids

    def get_entity_ids(self, raw_entity_ids: torch.LongTensor) ->  torch.LongTensor:
        entity_ids = torch.zeros_like(raw_entity_ids)
        for *index, raw_entity_id in nested_enumerate(raw_entity_ids.tolist()):
            token = self.vocab.get_token_from_index(raw_entity_id, 'raw_entity_ids')
            entity_id = self.vocab.get_token_index(token, 'entity_ids')
            entity_ids[tuple(index)] = entity_id
        return entity_ids

    def _encode_source(self, source: Dict[str, torch.Tensor]) -> torch.Tensor:

        # Extract, embed and encode source tokens.
        source_embeddings = embedded_dropout(
            embed=self._token_embedder,
            words=source,
            dropout=self._dropoute if self.training else 0)
        source_embeddings = self._locked_dropout(source_embeddings, self._dropouti)
        encoded_raw = self._rnn(source_embeddings)
        encoded = self._locked_dropout(encoded_raw)

        alpha_loss = encoded.pow(2).mean()
        beta_loss = (encoded_raw[:, 1:] - encoded_raw[:, :-1]).pow(2).mean()

        return encoded, alpha_loss, beta_loss

    def _mention_type_loss(self,
                           encoded: torch.Tensor,
                           mention_type: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for predicting whether or not the the next token will be part of an
        entity mention.
        """
    def _mention_type_loss(self,
                           encoded: torch.Tensor,
                           mention_type: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for predicting whether or not the the next token will be part of an
        entity mention.
        """
        logits = self._fc_mention_type(encoded)
        mention_logp = F.log_softmax(logits, -1)
        mention_loss = -mention_logp.gather(-1, mention_type.unsqueeze(-1)).squeeze()
        mention_loss = mention_loss * mask.float()
        # mention_loss = sequence_cross_entropy_with_logits(logits, mention_type, mask,
        #                                                   average='token')

        # if not self.training:
        self._new_mention_f1(predictions=logits,
                             gold_labels=mention_type,
                             mask=mask)
        self._kg_mention_f1(predictions=logits,
                            gold_labels=mention_type,
                            mask=mask)

        return mention_loss.sum(-1)

    def _new_entity_logits(self,
                           encoded: torch.Tensor,
                           shortlist: torch.Tensor = None) -> torch.Tensor:
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
            log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs.gather(-1, target_inds.unsqueeze(-1)).squeeze(-1)
        loss = loss * target_mask.float()

        if target_mask.any():
            self._new_entity_accuracy(predictions=log_probs[target_mask],
                                      gold_labels=target_inds[target_mask])
            self._new_entity_accuracy20(predictions=log_probs[target_mask],
                                        gold_labels=target_inds[target_mask])

        return loss.sum(-1) # / (target_mask.sum() + 1e-13)

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
        return -target_log_probs.sum(-1) # / (target_mask.sum() + 1e-13)

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
        self._avg_mention_type_loss(float(mention_type_loss.sum()/target_mask.sum()))

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
        logger.debug('new_entity_loss: %s', new_entity_loss)

        self._avg_new_entity_loss(float(new_entity_loss.sum()/target_mask.sum()))

        # For derived mentions, first predict which parent(s) to expand...
        knowledge_graph_entity_loss = self._knowledge_graph_entity_loss(encoded_head,
                                                                        encoded_relation,
                                                                        raw_entity_ids,
                                                                        entity_ids,
                                                                        parent_ids,
                                                                        target_mask)
        self._avg_knowledge_graph_entity_loss(float(knowledge_graph_entity_loss.sum()/target_mask.sum()))

        # Compute total loss
        loss = (mention_type_loss + new_entity_loss + knowledge_graph_entity_loss).sum() / target_mask.sum()

        # Activation regularization
        if self._alpha:
            loss = loss + self._alpha * alpha_loss
        # Temporal activation regularization (slowness)
        if self._beta:
            loss = loss + self._beta * beta_loss

        return {'loss': loss}

    def _next_mention_type_logp(self, next_mention_type_logits, beam_states):
        """
        Computes log probabilities of mention type for next token, .e.g, adjusts logits to prevent ongoing non-mentions.
        Intended for use when performing beam search.

        Parameters
        ==========
        next_mention_type_logits: torch.FloatTensor
            Tensor of shape (batch_size, num_mention_types) containing next mention type logits.
        beam_states: List[KglmBeamState]
            List of previous beam states.

        Returns
        =======
        next_mention_type_logp:
            Tensor of shape (batch_size, beam_width, num_mention_types) containing next mention type log probabilities.
        """
        beam_width = len(beam_states)

        # Tile the mention_logits, and apply penalty to non-ongoing mentions
        out = next_mention_type_logits.unsqueeze(1).repeat(1, beam_width, 1)
        for i, beam_state in enumerate(beam_states):
            out[~beam_state.ongoing, i, -1] = -1e32

        return F.log_softmax(out, dim=-1)

    def  _next_new_entity_logp(self, next_new_entity_logits, beam_states):
        """
        Computes log probabilities of new entity mentions.
        Intended for use when performing beam search.

        Parameters
        ==========
        next_new_entity_logits: torch.FloatTensor
            Tensor of shape (batch_size, num_entities) containing next new entity logits.
        beam_states: List[KglmBeamState]
            List of previous beam states.

        Returns
        =======
        next_new_entity_logp:
            Tensor of shape (batch_size, beam_width, num_mention_types) containing next new entity log probabilities.
        """
        beam_width = len(beam_states)
        # Tile the mention_logits, and apply penalty to non-ongoing mentions
        out = next_new_entity_logits.unsqueeze(1).repeat(1, beam_width, 1)
        for j, beam_state in enumerate(beam_states):
            self._recent_entities.load_beam_state(beam_state.recent_entities)
            for i, recent_ids in enumerate(self._recent_entities._remaining):
                for recent_id in recent_ids:
                    out[i, j, recent_id] = -1e32
        return F.log_softmax(out, dim=-1)

    def _next_related_entity_logp(self, next_encoded_head, next_encoded_relation, beam_states):
        """
        Computes log probabilities of related entity mentions.
        Intended for use when performing beam search.

        Parameters
        ==========
        next_encoded_head: torch.FloatTensor
            Tensor of shape (batch_size, embedding_dim) of the head encodings.
        next_encoded_relation: torch.FloatTensor
            Tensor of shape (batch_size, embedding_dim) of the relation encodings.
        beam_states: List[KglmBeamState]
            List of previous beam states.

        Returns
        =======
        logp:
            Tensor of shape (batch_size, beam_width, num_candidates) containing the log
            probability of the parent/relation combination.
        And a dictionary containing the annotation data.
            parent_ids:
                Tensor of shape (batch_size, beam_width, num_candidates)
            relation_ids:
                Tensor of shape (batch_size, beam_width, num_candidates)
            raw_entity_ids:
                Tensor of shape (batch_size, beam_width, num_candidates)
        """
        batch_size = next_encoded_head.size(0)
        beam_width = len(beam_states)
        logp_arr = np.empty((batch_size, beam_width), dtype=object)
        parent_ids_arr = np.empty((batch_size, beam_width), dtype=object)
        relations_arr = np.empty((batch_size, beam_width), dtype=object)
        raw_entity_ids_arr = np.empty((batch_size, beam_width),  dtype=object)
        for j, beam_state in enumerate(beam_states):
            # Get the set of candidate parents from the RecentEntities module.
            # Since we are only considering candidates for a single timestep we can get the parents
            # directly from the RecentEntities._remaining dictionaries' keys.
            self._recent_entities.load_beam_state(beam_state.recent_entities)
            for i, candidate_ids in enumerate(self._recent_entities._remaining):
                # Cast candidate ids to a tensor, lookup embeddings, and compute score.
                candidate_ids = torch.LongTensor(list(candidate_ids.keys()),
                                                 device=next_encoded_head.device)
                candidate_embeddings = self._entity_embedder(candidate_ids)
                candidate_logits = torch.mv(candidate_embeddings, next_encoded_head[i])
                candidate_logp = F.log_softmax(candidate_logits)

                # Lookup relations
                _, s, r, o = self._knowledge_graph_lookup(candidate_ids)
                relation_embeddings_list = [self._relation_embedder(_r) for _r in r]

                # Stop early if node is isolated
                if not s:
                    logp_arr[i, j] = torch.FloatTensor([], device=next_encoded_head.device)
                    parent_ids_arr[i, j] = torch.LongTensor([], device=next_encoded_head.device)
                    relations_arr[i, j] = torch.LongTensor([], device=next_encoded_head.device)
                    raw_entity_ids_arr[i, j] = torch.LongTensor([], device=next_encoded_head.device)
                    continue

                # Otherwise compute relation probabilities for each parent and combine
                temp_logp = []
                temp_parent_ids = []
                temp_relations = []
                temp_raw_entity_ids = []
                for idx, relation_embeddings in enumerate(relation_embeddings_list):
                    num_relations = relation_embeddings.size(0)
                    relation_logits = torch.mv(relation_embeddings, next_encoded_relation[i])
                    relation_logp = F.log_softmax(relation_logits)
                    temp_logp.append(candidate_logp[idx] + relation_logp)
                    temp_parent_ids.append(s[idx].repeat(num_relations))
                    temp_relations.append(r[idx])
                    temp_raw_entity_ids.append(o[idx])
                logp_arr[i, j] = torch.cat(temp_logp)
                parent_ids_arr[i, j] = torch.cat(temp_parent_ids)
                relations_arr[i, j] = torch.cat(temp_relations)
                raw_entity_ids_arr[i, j] = torch.cat(temp_raw_entity_ids)

        num_candidates = max(t.size(0) for t in logp_arr.flatten())
        logp = next_encoded_head.new_full((batch_size, beam_width, num_candidates), -1e32)
        parent_ids = next_encoded_head.new_zeros((batch_size, beam_width, num_candidates), dtype=torch.int64)
        relations = next_encoded_head.new_zeros((batch_size, beam_width, num_candidates), dtype=torch.int64)
        raw_entity_ids = next_encoded_head.new_zeros((batch_size, beam_width, num_candidates), dtype=torch.int64)
        for i in range(batch_size):
            for j in range(beam_width):
                size = logp_arr[i][j].size(0)
                logp[i, j, :size] = logp_arr[i][j]
                parent_ids[i, j, :size] = parent_ids_arr[i][j]
                relations[i, j, :size] = relations_arr[i][j]
                raw_entity_ids[i ,j, :size] = raw_entity_ids_arr[i][j]

        annotations = {
            'parent_ids': parent_ids,
            'relations': relations,
            'raw_entity_ids': raw_entity_ids
        }

        return logp, annotations

    def _top_k_annotations(self,
                           next_mention_type_logp,
                           next_new_entity_logp,
                           next_related_entity_logp,
                           related_entity_annotations,
                           output,
                           k):
        """
        Aggregate log probabilities and return top-k results.

        Don't be intimidated by the amount of code - almost all of it relates to various
        bookkeeping tasks to get the annotations.
        """
        # === Bookkeeping ====
        # Need to get all of the relevant sizes
        batch_size, beam_width, n_new = next_new_entity_logp.size()
        n_related = next_related_entity_logp.size(-1)

        # Derive the length of the full tensor: # new + # related + ongoing + unrelated
        length = n_new + n_related + 2
        total_logp = next_mention_type_logp.new_empty(batch_size, beam_width, length)

        # For clarity, name the slices
        new_slice = slice(0, n_new)
        related_slice = slice(n_new, n_new + n_related)
        ongoing_slice = -2
        null_slice = -1

        # === Annotation lookups ===
        mention_type_lookup = torch.zeros_like(total_logp, dtype=torch.int64)
        parent_id_lookup = torch.zeros_like(total_logp, dtype=torch.int64)
        relation_lookup = torch.zeros_like(total_logp, dtype=torch.int64)
        raw_entity_id_lookup = torch.zeros_like(total_logp, dtype=torch.int64)
        entity_id_lookup = torch.zeros_like(total_logp, dtype=torch.int64)

        # Mention type
        mention_type_lookup[:, :, new_slice] = 1
        mention_type_lookup[:, :, related_slice] = 2
        mention_type_lookup[:, :, ongoing_slice] = 3
        mention_type_lookup[:, :, null_slice] = 0

        # New
        id_range = torch.arange(n_new, device=entity_id_lookup.device).view(1, 1, n_new)
        entity_id_lookup[:, :, new_slice] = id_range
        raw_entity_id_lookup[:, :, new_slice] = self.get_raw_entity_ids(id_range)

        # Related
        parent_id_lookup[:, :, related_slice] = related_entity_annotations['parent_ids']
        relation_lookup[:, :, related_slice] = related_entity_annotations['relations']
        raw_entity_id_lookup[:, :, related_slice] = related_entity_annotations['raw_entity_ids']
        entity_id_lookup[:, :, related_slice] = self.get_entity_ids(related_entity_annotations['raw_entity_ids'])

        # Ongoing
        if output is not None:
            parent_id_lookup[:, :, ongoing_slice] =  output['parent_ids']
            relation_lookup[:, :, ongoing_slice] = output['relations']
            entity_id_lookup[:, :, ongoing_slice] =  output['entity_ids']
            raw_entity_id_lookup[:, :, ongoing_slice] = output['raw_entity_ids']

        # === Logp ===

        # Set the mention probabilities
        total_logp[:, :, new_slice] = next_mention_type_logp[:, :, 1].unsqueeze(-1)
        total_logp[:, :, related_slice] = next_mention_type_logp[:, :, 2].unsqueeze(-1)
        total_logp[:, :, ongoing_slice] = next_mention_type_logp[:, :, 3]
        total_logp[:, :, null_slice] = next_mention_type_logp[:, :, 0]

        # Add the entity probabilities
        total_logp[:, :, new_slice] += next_new_entity_logp
        total_logp[:, :, related_slice] += next_related_entity_logp

        # If available add the previous beam probabilities
        if output is not None:
            total_logp += output['logp'].unsqueeze(-1)

        # Get the top-k outputs
        top_logp, top_indices = total_logp.view(batch_size, -1).topk(k, dim=-1)
        output = {
            'logp': top_logp,
            'backpointers': top_indices // length,
            'mention_types':  mention_type_lookup.view(batch_size, -1).gather(-1, top_indices),
            'parent_ids':  parent_id_lookup.view(batch_size, -1).gather(-1, top_indices),
            'relations':  relation_lookup.view(batch_size, -1).gather(-1, top_indices),
            'entity_ids':  entity_id_lookup.view(batch_size, -1).gather(-1, top_indices),
            'raw_entity_ids':  raw_entity_id_lookup.view(batch_size, -1).gather(-1, top_indices)
        }
        return output

    def _update_beam_states(self, output, beam_states):
        """
        Ensure that the correct recent entities modules and ongoing flags are properly taken from
        the last step and updated using the current predicted outputs.
        """
        new_beam_states = []
        backpointers = output['backpointers']
        batch_size, beam_width =  backpointers.size()
        # To facilitate indexing with the backpointers, we'll store the RecentEntities' _remaining
        # dicts in a numpy array.
        remaining_dicts = np.empty((batch_size, len(beam_states)), dtype=object)
        for j, beam_state in enumerate(beam_states):
            self._recent_entities.load_beam_state(beam_state.recent_entities)
            for i in range(batch_size):
                remaining_dicts[i, j] = self._recent_entities._remaining[i]

        for i in range(beam_width):
            # Everything but null mention types can be ongoing in next step.
            ongoing = output['mention_types'][:, i] != 0

            # Trace backpointers to retrieve correct recent entities dicts, and update using the
            # current output.
            bp = backpointers[:, i].cpu().numpy()
            remaining = remaining_dicts[np.arange(batch_size), bp].tolist()
            self._recent_entities.load_beam_state({'remaining': remaining})
            self._recent_entities(output['entity_ids'][:, i].unsqueeze(-1))

            # Add beam states
            new_beam_states.append(
                KglmBeamState(recent_entities=self._recent_entities.beam_state(),
                              ongoing=ongoing)
            )

        return new_beam_states

    def _to_raw_entity_tokens(self, x):
        """
        Returns the raw entity id strings for a nested list of raw entity ids
        """
        if isinstance(x, list):
            return [self._to_raw_entity_tokens(i) for i in x]
        elif isinstance(x, int):
            return self.vocab.get_token_from_index(x, 'raw_entity_ids')
        else:
            return ValueError('Expecting a nested list of raw entity ids')

    def _trace_backpointers(self,
                            source,
                            target,
                            reset,
                            metadata,
                            k,
                            predictions):
        """
        Traces backpointers to collect the top-k annotations.
        """
        batch_size, seq_length = source['tokens'].shape
        alias_database = metadata[0]['alias_database']

        new_source = {key: value.unsqueeze(1).repeat(1, k, 1).view(batch_size * k, -1) for key, value in source.items()}
        new_target = {key: value.unsqueeze(1).repeat(1, k, 1).view(batch_size * k, -1) for key, value in target.items()}
        new_reset = reset.unsqueeze(1).repeat(1, k).view(batch_size * k)
        new_metadata = [metadata[i] for i in range(batch_size) for _ in range(k)]

        mention_types = []
        parent_ids = []
        relations = []
        raw_entity_ids = []
        entity_ids = []

        backpointer = None

        for prediction in reversed(predictions):
            if backpointer is None:
                mention_types.append(prediction['mention_types'])
                parent_ids.append(prediction['parent_ids'])
                relations.append(prediction['relations'])
                raw_entity_ids.append(prediction['raw_entity_ids'])
                entity_ids.append(prediction['entity_ids'])
            else:
                mention_types.append(prediction['mention_types'].gather(1, backpointer))
                parent_ids.append(prediction['parent_ids'].gather(1, backpointer))
                relations.append(prediction['relations'].gather(1, backpointer))
                raw_entity_ids.append(prediction['raw_entity_ids'].gather(1, backpointer))
                entity_ids.append(prediction['entity_ids'].gather(1, backpointer))
            if backpointer is None:
                backpointer = prediction['backpointers']
            else:
                backpointer = prediction['backpointers'].gather(1, backpointer)

        mention_types = torch.stack(mention_types[::-1], dim=-1).view(batch_size * k, -1)
        parent_ids = torch.stack(parent_ids[::-1], dim=-1).view(batch_size * k, -1)
        relations = torch.stack(relations[::-1], dim=-1).view(batch_size * k, -1)
        raw_entity_ids = torch.stack(raw_entity_ids[::-1], dim=-1).view(batch_size * k, -1)
        entity_ids = torch.stack(entity_ids[::-1], dim=-1).view(batch_size * k, -1)

        # One final bit of complexity - we need to get copy indices.
        raw_entity_tokens = self._to_raw_entity_tokens(raw_entity_ids.tolist())
        target_tokens = [x['target_tokens'] for x in new_metadata]
        alias_copy_inds_list = alias_database.nested_token_to_uid(raw_entity_tokens, target_tokens)
        alias_copy_inds = torch.tensor(alias_copy_inds_list, device=mention_types.device)

        return {
            'source': new_source,
            'target': new_target,
            'reset': new_reset,
            'metadata': new_metadata,
            'mention_types': mention_types,
            'parent_ids': parent_ids,
            'relations': relations,
            'raw_entity_ids': raw_entity_ids,
            'entity_ids': entity_ids,
            'alias_copy_inds': alias_copy_inds
        }

    def beam_search(self,
                    source: Dict[str, torch.Tensor],
                    target: Dict[str, torch.Tensor],
                    reset: torch.ByteTensor,
                    metadata: Dict[str, Any],
                    k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain the top-k (approximately) most likely predictions from the model using beam
        search. Unlike typical beam search all of the beam states are returned instead of just
        the most likely.

        The returned candidates are intended to be marginalized over to obtain an upper bound for
        the token-level perplexity of the EntityNLM.

        Parameters
        ==========
        source : ``Dict[str, torch.Tensor]``
            A tensor of shape ``(batch_size, sequence_length)`` containing the sequence of
            tokens.
        reset : ``torch.ByteTensor``
            Whether or not to reset the model's state. This should be done at the start of each
            new sequence.
        metadata : ``Dict[str, Any]``
            Assorted metadata. Should contain the alias database, as well as the token strings (needed to retrieve copy indices).
        k : ``int``
            Number of predictions to return.

        Returns
        =======
        predictions : ``torch.Tensor``
            A tensor of shape ``(batch_size * k, sequence_length)`` containing the top-k
            predictions.
        logp : ``torch.Tensor``
            The log-probabilities of each prediction. WARNING: These are returned purely for
            diagnostic purposes and should not be factored in the the perplexity calculation.
        """
        # We want the output fields to be properly aligned for the generative model, which makes
        # predictions for the **target** tokens! Hence, we feed them as the input (instead of the
        # source tokens).
        batch_size, sequence_length = target['tokens'].shape

        # Reset the model's internal state.
        if not reset.all():
            raise RuntimeError('Detecting that not all states are being `reset` (e.g., that input '
                               'sequences have been split). Cannot predict top-K annotations in '
                               'this setting!')
        self.reset_states(reset)

        # The following tensors can be computed using only the encoder:
        #   * The 3-headed encodings.
        #   * The (unconstrained) mention type logits.
        #   * The (unconstrained) new entity logits.
        # Although we can compute the mention type and new entity logits, we will need to compute
        # the log-probabilities during decoding due to the following constraints:
        #   * `mention_type` = CONTINUE only if the previous token type was a new or ongoing mention.
        #   * `new_entity` cannot be in recent entities.
        encoded, *_ = self._encode_source(target['tokens'])
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim] * 2
        encoded_token, encoded_head, encoded_relation = encoded.split(splits, dim=-1)
        mention_type_logits = self._fc_mention_type(encoded_token)
        new_entity_logits = self._new_entity_logits(encoded_head + encoded_relation)

        # Beam search logic
        predictions:  List[Dict[str, torch.Tensor]] = []
        beam_states = [KglmBeamState(recent_entities=self._recent_entities.beam_state(),
                                     ongoing=torch.zeros_like(reset))]
        output = None

        for timestep in range(sequence_length):
            # Get log probabilities of all next states
            next_mention_type_logp = self._next_mention_type_logp(mention_type_logits[:, timestep],
                                                                  beam_states)
            next_new_entity_logp = self._next_new_entity_logp(new_entity_logits[:, timestep],
                                                              beam_states)
            next_related_entity_logp, related_entity_annotations = self._next_related_entity_logp(
                encoded_head[:, timestep],
                encoded_relation[:, timestep],
                beam_states)

            output = self._top_k_annotations(next_mention_type_logp,
                                             next_new_entity_logp,
                                             next_related_entity_logp,
                                             related_entity_annotations,
                                             output,
                                             k)
            beam_states = self._update_beam_states(output, beam_states)
            predictions.append(output)

        annotation = self._trace_backpointers(source, target, reset, metadata, k, predictions)

        return annotation

    @overrides
    def train(self, mode=True):
        # This is a hack to ensure that the internal state resets when the model switches from
        # training to evaluation. The complication arises from potentially differing batch sizes
        # (e.g. the `reset` tensor will not be the right size).
        super().train(mode)
        self._rnn.reset()

    @overrides
    def eval(self):
        # See train.
        super().eval()
        self._rnn.reset()

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

    def reset_states(self, reset):
        self._rnn.reset(reset)
        self._recent_entities.reset(reset)
