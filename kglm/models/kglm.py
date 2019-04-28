from collections import defaultdict
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
from kglm.modules import (
    embedded_dropout, LockedDropout, WeightDrop, KnowledgeGraphLookup, RecentEntities)
from kglm.nn.util import nested_enumerate, parallel_sample
from kglm.training.metrics import Ppl

logger = logging.getLogger(__name__)


@Model.register('kglm')
class Kglm(Model):
    """
    Knowledge graph language model.

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
                 alias_encoder: Seq2SeqEncoder,
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
        super(Kglm, self).__init__(vocab)

        # We extract the `Embedding` layers from the `TokenEmbedders` to apply dropout later on.
        # pylint: disable=protected-access
        self._token_embedder = token_embedder._token_embedders['tokens']
        self._entity_embedder = entity_embedder._token_embedders['entity_ids']
        self._relation_embedder = relation_embedder._token_embedders['relations']
        self._alias_encoder = alias_encoder
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
        self._avg_knowledge_graph_entity_loss = Average()
        self._avg_vocab_loss = Average()
        self._new_mention_f1 = F1Measure(positive_label=1)
        self._kg_mention_f1 = F1Measure(positive_label=2)
        self._new_entity_accuracy = CategoricalAccuracy()
        self._new_entity_accuracy20 = CategoricalAccuracy(top_k=20)
        self._parent_ppl = Ppl()
        self._relation_ppl = Ppl()

        initializer(self)

    def predict_mention_type(self, encoded_token: torch.FloatTensor) -> torch.LongTensor:
        # Predict mention type
        mention_type_logits = self._fc_mention_type(encoded_token)
        mention_type_probs = F.softmax(mention_type_logits, dim=-1)
        mention_type = parallel_sample(mention_type_probs)
        return mention_type

    def predict_new_entities(self,
                             encoded: torch.FloatTensor,
                             mask: torch.ByteTensor,
                             shortlist: Dict[str, torch.LongTensor] = None) -> torch.LongTensor:
        # Compute entity logits
        new_entity_logits = self._new_entity_logits(encoded, shortlist)

        # If using shortlist, then samples are indexed w.r.t the shortlist and entity_ids must be
        # looked up
        if self._use_shortlist:
            shortlist_mask = get_text_field_mask(shortlist)
            if shortlist_mask.shape[-1] == 0:
                new_entity_ids = shortlist_mask.new_zeros(encoded.shape[:-1])
                shortlist_inds = shortlist_mask.new_zeros(encoded.shape[:-1])
            else:
                new_entity_probs = masked_softmax(new_entity_logits, shortlist_mask)
                shortlist_inds = torch.zeros_like(mask, dtype=torch.int64)
                # Some sequences may be full of padding in which case the shortlist is empty
                not_just_padding = shortlist_mask.byte().any(-1)
                shortlist_inds[not_just_padding] = parallel_sample(new_entity_probs[not_just_padding])
                shortlist_inds[~mask] = 0
                new_entity_ids = shortlist['entity_ids'].gather(1, shortlist_inds)
                # Zero out any non-new entities.
                new_entity_ids[~mask] = 0
        # If not using shortlist, then samples are indexed w.r.t to the global vocab
        else:
            new_entity_probs = F.softmax(new_entity_logits, dim=-1)
            new_entity_ids = parallel_sample(new_entity_probs)
            shortlist_inds = None
            # Zero out any non-new entities.
            new_entity_ids[~mask] = 0

        return new_entity_ids, shortlist_inds

    def get_raw_entity_ids(self, entity_ids: torch.LongTensor) ->  torch.LongTensor:
        raw_entity_ids = torch.zeros_like(entity_ids)
        for *index, entity_id in nested_enumerate(entity_ids.tolist()):
            token = self.vocab.get_token_from_index(entity_id, 'entity_ids')
            raw_entity_id = self.vocab.get_token_index(token, 'raw_entity_ids')
            raw_entity_ids[tuple(index)] = raw_entity_id
        return raw_entity_ids

    def predict_knowledge_graph_entities(self,
                                         encoded_head: torch.FloatTensor,
                                         encoded_relation: torch.FloatTensor,
                                         new_entity_ids: torch.LongTensor,
                                         mask: torch.ByteTensor) -> torch.LongTensor:
        # Initialize outputs
        parent_ids = torch.zeros_like(new_entity_ids)
        relations = torch.zeros_like(new_entity_ids)
        derived_entity_ids = torch.zeros_like(new_entity_ids)
        derived_raw_entity_ids = torch.zeros_like(new_entity_ids)

        # We can't make predictions in parallel since current predictions affect future predictions
        # through the recent entity list.
        sequence_length = mask.shape[1]

        for i in range(sequence_length):

            # Update recent entity list with any observed new entities.
            current_entity_id = new_entity_ids[:, i].unsqueeze(1)
            candidate_ids, candidate_mask = self._recent_entities(current_entity_id)

            # Mask indicates whether to predict at this timestep
            current_mask = mask[:, i]
            if not current_mask.any():
                continue

            # Initialize outputs
            current_parent_ids = torch.zeros_like(current_entity_id)
            current_relations = torch.zeros_like(current_entity_id)
            current_raw_tail_ids = torch.zeros_like(current_entity_id)
            current_tail_ids = torch.zeros_like(current_entity_id)

            ### SAMPLE PARENT IDS ###
            current_encoded_head = encoded_head[:, i].unsqueeze(1)
            candidate_embeddings = self._entity_embedder(candidate_ids)
            selection_logits = torch.bmm(current_encoded_head, candidate_embeddings.transpose(1, 2))
            selection_probs = masked_softmax(selection_logits, candidate_mask)
            # Only sample if the is at least one viable candidate. Return zero for non-viable
            # distributions.
            viable_candidate_mask = candidate_mask.any(-1).squeeze()
            if viable_candidate_mask.any():
                viable_candidate_ids = candidate_ids[viable_candidate_mask]
                viable_candidate_probs = selection_probs[viable_candidate_mask]
                viable_parent_samples = parallel_sample(viable_candidate_probs)
                viable_parent_ids = viable_candidate_ids.gather(-1, viable_parent_samples)
                current_parent_ids[viable_candidate_mask] = viable_parent_ids
            parent_ids[current_mask, i] = current_parent_ids[current_mask].squeeze(-1)

            ## SAMPLE RELATIONS ##
            current_encoded_relation = encoded_relation[:, i].unsqueeze(1)
            # Look up parents in the knowledge graph
            indices, _, relations_list, tail_ids_list = self._knowledge_graph_lookup(current_parent_ids)
            relation_embeddings = [self._relation_embedder(r) for r in relations_list]

            for index, relation_lookup, relation_embedding, tail_id_lookup in \
                    zip(indices, relations_list, relation_embeddings, tail_ids_list):

                logits = torch.mv(relation_embedding, current_encoded_relation[index])
                tail_probs = F.softmax(logits, dim=-1)
                tail_sample = torch.multinomial(tail_probs, 1)
                # Lookup relation and tail's raw entity id
                relation = relation_lookup[tail_sample]
                raw_tail_id = tail_id_lookup[tail_sample]
                # Convert raw entity id to entity id
                tail_id_string = self.vocab.get_token_from_index(raw_tail_id.item(), 'raw_entity_ids')
                tail_id = self.vocab.get_token_index(tail_id_string, 'entity_ids')
                # Update list
                current_relations[index] = relation
                current_raw_tail_ids[index] = raw_tail_id
                current_tail_ids[index] = tail_id

            # Update outputs
            relations[current_mask, i] = current_relations[current_mask].squeeze(-1)
            derived_raw_entity_ids[current_mask, i] = current_raw_tail_ids[current_mask].squeeze(-1)
            derived_entity_ids[current_mask, i] = current_tail_ids[current_mask].squeeze(-1)

            # Add any predicted tail entities to recent entities
            self._recent_entities.insert(current_tail_ids, current_mask)

        return parent_ids, relations, derived_entity_ids, derived_raw_entity_ids

    def sample(self,
               source: Dict[str, torch.LongTensor],
               reset: torch.ByteTensor,
               metadata: List[Dict[str, Any]],
               emit_tokens: bool,
               shortlist: Dict[str, torch.LongTensor] = None,
               **kwargs):
        # Tensorize the alias_database - this will only perform the operation once.
        alias_database = metadata[0]['alias_database']
        alias_database.tensorize(vocab=self.vocab)

        # Reset
        if reset.any() and (self._state is not None):
            for layer in range(self._num_layers):
                h, c = self._state['layer_%i' % layer]
                h[:, reset, :] = torch.zeros_like(h[:, reset, :])
                c[:, reset, :] = torch.zeros_like(c[:, reset, :])
                self._state['layer_%i' % layer] = (h, c)
        self._recent_entities.reset(reset)

        # Get source tokens
        source_tokens = source['tokens']

        # Embed source tokens.
        # shape: (batch_size, sequence_length, embedding_dim)
        encoded, *_ = self._encode_source(source_tokens)
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim] * 2
        encoded_token, encoded_head, encoded_relation = encoded.split(splits, dim=-1)

        # Predict mention type
        mention_type = self.predict_mention_type(encoded_token)
        logger.debug('Passes mention type')

        # Predict new entities
        new_entity_ids, shortlist_inds = self.predict_new_entities(encoded_head + encoded_relation,
                                                                   mask=mention_type.eq(1),
                                                                   shortlist=shortlist)
        logger.debug('Passes new ent')

        # Get their raw ids
        new_raw_entity_ids = self.get_raw_entity_ids(new_entity_ids)
        logger.debug('Passes raw')

        # Predict parents and relations
        parent_ids, relations, derived_entity_ids, derived_raw_entity_ids = self.predict_knowledge_graph_entities(
            encoded_head,
            encoded_relation,
            new_entity_ids=new_entity_ids,
            mask=mention_type.eq(2))
        logger.debug('Passes parents')

        # Combine new and derived entity annotations
        entity_ids = new_entity_ids
        entity_ids[mention_type.eq(2)] = derived_entity_ids[mention_type.eq(2)]
        raw_entity_ids = new_raw_entity_ids
        raw_entity_ids[mention_type.eq(2)] = derived_raw_entity_ids[mention_type.eq(2)]

        out_dict = {
            'source': source,
            'reset': reset,
            'metadata': metadata,
            'mention_type': mention_type,
            'raw_entity_ids': {'raw_entity_ids': raw_entity_ids},
            'entity_ids': {'entity_ids': entity_ids},
            'parent_ids': {'entity_ids': parent_ids.unsqueeze(-1)},
            'relations': {'relations': relations},
            'shortlist': shortlist,
            'shortlist_inds': shortlist_inds
        }
        if 'target' in kwargs:
            out_dict['target'] = kwargs['target']

        if emit_tokens:

            # Predict generation-mode scores.
            generate_scores = self._generate_scores(encoded_token, entity_ids)

            # Predict copy-mode scores.
            alias_tokens, alias_inds = alias_database.lookup(raw_entity_ids)
            copy_scores = self._copy_scores(encoded_token, alias_tokens)

            # Vocab stuff
            vocab_size = self.vocab.get_vocab_size('tokens')
            generate_scores = self._generate_scores(encoded_token, entity_ids)
            alias_tokens, alias_indices = alias_database.lookup(raw_entity_ids)
            copy_scores = self._copy_scores(encoded_token, alias_tokens)
            copy_sequence_length = copy_scores.shape[-1]
            concatenated_scores = torch.cat((generate_scores, copy_scores), dim=-1)

            # In order to obtain proper log probabilities we create a mask to omit padding alias tokens
            # from the calculation.
            batch_size = source_tokens.shape[0]
            score_mask = torch.ones_like(concatenated_scores)
            alias_mask = alias_tokens.gt(0).view(batch_size, 1, -1)
            score_mask[:, :, vocab_size:] = alias_mask

            # The log-probability distribution is then given by taking the masked log softmax.
            target_probs = masked_softmax(concatenated_scores, score_mask)
            out_dict['target_probs'] = target_probs.detach()
            out_dict['alias_indices'] = alias_indices.view(batch_size, -1)

        return out_dict

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                source: Dict[str, torch.Tensor],
                reset: torch.Tensor,
                metadata: List[Dict[str, Any]],
                target: Dict[str, torch.Tensor] = None,
                mention_type: torch.Tensor = None,
                raw_entity_ids: Dict[str, torch.Tensor] = None,
                entity_ids: Dict[str, torch.Tensor] = None,
                parent_ids: Dict[str, torch.Tensor] = None,
                relations: Dict[str, torch.Tensor] = None,
                shortlist: Dict[str, torch.Tensor] = None,
                shortlist_inds: torch.Tensor = None,
                alias_copy_inds: torch.Tensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:

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

        if target is not None:
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
                alias_copy_inds=alias_copy_inds,
                **kwargs)

        else:
            output_dict = self._greedy_decode(
                source=source,
                alias_database=alias_database,
                shortlist=shortlist)


        return output_dict

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
        target_mask = get_text_field_mask(source)
        source = source['tokens']
        raw_entity_ids = raw_entity_ids['raw_entity_ids']
        entity_ids = entity_ids['entity_ids']
        parent_ids = parent_ids['entity_ids']
        if target is not None:
            target = target['tokens']

        # Embed source tokens.
        # shape: (batch_size, sequence_length, embedding_dim)
        encoded, alpha_loss, beta_loss = self._encode_source(source)
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim] * 2
        encoded_token, encoded_head, encoded_relation = encoded.split(splits, dim=-1)

        # Predict whether or not the next token will be an entity mention, and if so which type.
        mention_type_loss = self._mention_type_loss(encoded_token, mention_type, target_mask)
        self._avg_mention_type_loss(float(mention_type_loss))
        logger.debug('mention type loss: %0.4f', mention_type_loss)

        # For new mentions, predict which entity (among those in the supplied shortlist) will be
        # mentioned.
        if self._use_shortlist:
            new_entity_loss = self._new_entity_loss(encoded_head + encoded_relation,
                                                    shortlist_inds,
                                                    entity_ids,
                                                    shortlist,
                                                    target_mask)
        else:
            new_entity_loss = self._new_entity_loss(encoded_head + encoded_relation,
                                                    entity_ids,
                                                    entity_ids,
                                                    None,
                                                    target_mask)

        self._avg_new_entity_loss(float(new_entity_loss))
        logger.debug('new entity loss: %0.4f', new_entity_loss)

        # For derived mentions, first predict which parent(s) to expand...
        knowledge_graph_entity_loss = self._knowledge_graph_entity_loss(encoded_head,
                                                                        encoded_relation,
                                                                        raw_entity_ids,
                                                                        entity_ids,
                                                                        parent_ids,
                                                                        target_mask)
        self._avg_knowledge_graph_entity_loss(float(knowledge_graph_entity_loss))
        logger.debug('kg entity loss: %0.4f', knowledge_graph_entity_loss)

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

        # Compute total loss. Also compute logp (needed for importance sampling evaluation).
        loss = vocab_loss + mention_type_loss + new_entity_loss + knowledge_graph_entity_loss
        logp = -(vocab_loss + mention_type_loss + new_entity_loss + knowledge_graph_entity_loss) * target_mask.sum()
        penalized_logp = -(penalized_vocab_loss + mention_type_loss + new_entity_loss + knowledge_graph_entity_loss) * target_mask.sum()

        # Activation regularization
        if self._alpha:
            loss = loss + self._alpha * alpha_loss
        # Temporal activation regularization (slowness)
        if self._beta:
            loss = loss + self._beta * beta_loss

        return {'loss': loss, 'logp': logp, 'penalized_logp': penalized_logp}


    def _greedy_decode(self,
                       source: Dict[str, torch.Tensor],
                       alias_database: AliasDatabase,
                       shortlist: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        # Get the token mask and extract indexed text fields.
        # shape: (batch_size, sequence_length)
        mask = get_text_field_mask(source)
        source = source['tokens']

        # Embed source tokens.
        # shape: (batch_size, sequence_length, embedding_dim)
        encoded, alpha_loss, beta_loss = self._encode_source(source)
        splits = [self.token_embedding_dim] + [self.entity_embedding_dim] * 2
        encoded_token, encoded_head, encoded_relation = encoded.split(splits, dim=-1)

        # Determine mention type
        mention_type_logits = self._fc_mention_type(encoded_token)
        _, mention_type = torch.max(mention_type_logits, dim=-1)

        # Compute entity logits
        new_entity_mask = mention_type.eq(1)
        new_entity_logits = self._new_entity_logits(encoded_head + encoded_relation, shortlist)
        if self._use_shortlist:
            # If using shortlist, then samples are indexed w.r.t the shortlist and entity_ids must be looked up
            shortlist_mask = get_text_field_mask(shortlist)
            new_entity_probs = masked_softmax(new_entity_logits, shortlist_mask)
            _, shortlist_inds = torch.max(new_entity_probs, dim=-1)
            shortlist_inds[~new_entity_mask] = 0
            new_entity_samples = shortlist['entity_ids'].gather(1, shortlist_inds)
        else:
            # If not using shortlist, then samples are indexed w.r.t to the global vocab
            new_entity_probs = F.softmax(new_entity_logits, dim=-1)
            _, new_entity_samples = torch.max(new_entity_probs, dim=-1)

        # Start filling in the entity ids
        entity_ids = torch.zeros_like(source)
        entity_ids[new_entity_mask] = new_entity_samples[new_entity_mask]

        # ...UGH... we also need the raw ids - remapping time
        raw_entity_ids = torch.zeros_like(source)
        for *index, entity_id in nested_enumerate(entity_ids.tolist()):
            token = self.vocab.get_token_from_index(entity_id, 'entity_ids')
            raw_entity_id = self.vocab.get_token_index(token, 'raw_entity_ids')
            raw_entity_ids[tuple(index)] = raw_entity_id

        # Derived mentions.
        derived_entity_mask = mention_type.eq(2)
        if derived_entity_mask.any():

            # Parent selection
            candidate_ids, candidate_mask = self._recent_entities(entity_ids)
            candidate_embeddings = self._entity_embedder(candidate_ids)
            selection_logits = torch.bmm(encoded_head, candidate_embeddings.transpose(1, 2))
            selection_probs = masked_softmax(selection_logits, candidate_mask)
            _, selection_ids = torch.max(selection_probs, dim=-1)
            parent_ids = candidate_ids.gather(-1, selection_ids)

            # Relation selection
            indices, parent_ids_list, relations_list, tail_ids_list = self._knowledge_graph_lookup(parent_ids)
            relation_embeddings = [self._relation_embedder(r) for r in relations_list]

            raw_tail_ids = torch.zeros_like(parent_ids)
            tail_ids = torch.zeros_like(parent_ids)
            for index, relation_embedding, tail_id_lookup in zip(indices, relation_embeddings, tail_ids_list):
                logits = torch.mv(relation_embedding, encoded_relation[index])
                _, selected_relation = torch.max(logits, dim=-1)
                raw_tail_id = tail_id_lookup[selected_relation]

                tail_id_string = self.vocab.get_token_from_index(raw_tail_id.item(), 'raw_entity_ids')
                tail_id = self.vocab.get_token_index(tail_id_string, 'entity_ids')

                raw_tail_ids[index[:-1]] = raw_tail_id
                tail_ids[index[:-1]] = tail_id

            # TODO: Mask
            raw_entity_ids = raw_tail_ids
            entity_ids = tail_ids

        # Vocab stuff
        vocab_size = self.vocab.get_vocab_size('tokens')
        generate_scores = self._generate_scores(encoded_token, entity_ids)
        alias_tokens, alias_indices = alias_database.lookup(raw_entity_ids)
        copy_scores = self._copy_scores(encoded_token, alias_tokens)
        copy_sequence_length = copy_scores.shape[-1]
        concatenated_scores = torch.cat((generate_scores, copy_scores), dim=-1)

        # In order to obtain proper log probabilities we create a mask to omit padding alias tokens
        # from the calculation.
        score_mask = torch.ones_like(concatenated_scores)
        alias_mask = alias_tokens.gt(0).view(1, 1, -1)
        score_mask[:, :, vocab_size:] = alias_mask

        # The log-probability distribution is then given by taking the masked log softmax.
        word_probs = masked_softmax(concatenated_scores, score_mask)

        word_probs = word_probs.squeeze()
        raw_entity_id = raw_entity_ids.squeeze().item()
        alias_indices = alias_indices.view(-1)

        return {
            'word_probs': word_probs,
            'alias_indices': alias_indices,
            'raw_entity_id': raw_entity_id,
            'alias_database': alias_database}

    def decode(self, output: Dict[str, Any]):

        if 'word_probs' in output and 'raw_entity_id' in output:
            word_probs = output['word_probs']
            alias_indices = output['alias_indices']
            raw_entity_id = output['raw_entity_id']
            alias_database = output['alias_database']

            # Seperate generate from copy probs
            vocab_size = self.vocab.get_vocab_size('tokens')
            generate_probs = word_probs[:vocab_size]
            copy_probs = word_probs[vocab_size:]

            # We'll use a dict to manage accumulating probabilities
            word_probs = defaultdict(float)
            for i, prob in enumerate(generate_probs.tolist()):
                word = self.vocab.get_token_from_index(i, 'tokens')
                word_probs[word] = prob

            # For the copy tokens, we first need to make a reverse lookup.
            entity = self.vocab.get_token_from_index(raw_entity_id, 'raw_entity_ids')
            try:
                id_map = alias_database._id_map_lookup[entity]
                reverse_id_map = {i: x for x, i in id_map.items()}
            except KeyError:
                id_map = dict()
                reverse_id_map = dict()

            # Now we accumulate probabilities
            for idx, prob in zip(alias_indices.tolist(), copy_probs.tolist()):
                if idx == 0:
                    continue
                else:
                    word = reverse_id_map[idx]
                    word_probs[word] += prob

            # Lastly sort words by prob
            pairs = list(word_probs.items())
            pairs.sort(key=lambda x: x[1], reverse=True)
            words, probs = zip(*pairs)
            return {'words': [words[:1000]], 'probs': [probs[:1000]]}

        return output

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
        mention_loss = sequence_cross_entropy_with_logits(logits, mention_type, mask,
                                                          average='token')


        # if not self.training:
        self._new_mention_f1(predictions=logits,
                             gold_labels=mention_type,
                             mask=mask)
        self._kg_mention_f1(predictions=logits,
                            gold_labels=mention_type,
                            mask=mask)

        return mention_loss

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
                         entity_ids: torch.Tensor,
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
        target_log_probs = torch.gather(log_probs, -1, target_inds.unsqueeze(-1)).squeeze(-1)
        target_log_probs = target_log_probs * target_mask.float()
        # Also don't predict on non-mentions
        mentions = ~entity_ids.eq(0)
        target_log_probs = target_log_probs * mentions.float()

        # self._new_entity_accuracy(predictions=log_probs[mask],
        #                           gold_labels=target_inds[mask])
        # self._new_entity_accuracy20(predictions=log_probs[mask],
        #                             gold_labels=target_inds[mask])

        return -target_log_probs.sum() / (target_mask.sum() + 1e-13)

    def _parent_log_probs(self,
                          encoded_head: torch.Tensor,
                          entity_ids: torch.Tensor,
                          parent_ids: torch.Tensor) -> torch.Tensor:
        # Lookup recent entities (which are candidates for parents) and get their embeddings.
        candidate_ids, candidate_mask = self._recent_entities(entity_ids)
        # logger.debug('Candidate ids shape: %s', candidate_ids.shape)
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
        # logger.debug('Masked log probs shape: %s', masked_log_probs.shape)

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
            # logger.debug('Relation logits shape: %s', logits.shape)
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

    def _vocab_logp(self,
                    generate_scores: torch.Tensor,
                    copy_scores: torch.Tensor,
                    alias_indices: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, vocab_size = generate_scores.shape
        copy_sequence_length = copy_scores.shape[-1]

        # In order to obtain proper log probabilities we create a mask to omit padding alias tokens
        # from the calculation.
        alias_mask = alias_indices.view(batch_size, sequence_length, -1).gt(0)
        score_mask = alias_mask.new_ones(batch_size, sequence_length, vocab_size + copy_sequence_length)
        score_mask[:, :, vocab_size:] = alias_mask

        # The log-probability distribution is then given by taking the masked log softmax.
        concatenated_scores = torch.cat((generate_scores, copy_scores), dim=-1)
        log_probs = masked_log_softmax(concatenated_scores, score_mask)

        return log_probs

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

        log_probs = self._vocab_logp(generate_scores, copy_scores, alias_indices)

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
        combined_log_probs_extended_vocab[~flattened_mask] = 0

        # For UPP we penalize **only** p(UNK); not the copy probabilities!
        penalized_log_probs_source_vocab = generate_log_probs_source_vocab - self._unk_penalty * unks.float()
        penalized_log_probs_source_vocab = torch.cat((penalized_log_probs_source_vocab,
                                                      copy_log_probs),
                                                     dim=1)
        penalized_log_probs_source_vocab = torch.logsumexp(penalized_log_probs_source_vocab,
                                                           dim=1)
        combined_log_probs_extended_vocab[~flattened_mask] = 0

        kg_mask = (mention_mask & mask.byte()).view(-1)
        bg_mask = ((~mention_mask) & mask.byte()).view(-1)
        mask = (kg_mask | bg_mask)

        self._ppl(-combined_log_probs_source_vocab[mask].sum(), mask.float().sum() + 1e-13)
        self._upp(-penalized_log_probs_source_vocab[mask].sum(), mask.float().sum() + 1e-13)
        if kg_mask.any():
            self._kg_ppl(-combined_log_probs_source_vocab[kg_mask].sum(), kg_mask.float().sum() + 1e-13)
        if bg_mask.any():
            self._bg_ppl(-combined_log_probs_source_vocab[bg_mask].sum(), bg_mask.float().sum() + 1e-13)

        return vocab_loss, penalized_vocab_loss

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
            'ppl': self._ppl.get_metric(reset),
            'upp': self._upp.get_metric(reset),
            'kg_ppl': self._kg_ppl.get_metric(reset),
            'bg_ppl': self._bg_ppl.get_metric(reset),
            'type': self._avg_mention_type_loss.get_metric(reset),
            'new': self._avg_new_entity_loss.get_metric(reset),
            'kg': self._avg_knowledge_graph_entity_loss.get_metric(reset),
            'vocab': self._avg_vocab_loss.get_metric(reset),
        }
        # if not self.training:
        p, r, f  = self._new_mention_f1.get_metric(reset)
        out['new_p'] = p
        out['new_r'] = r
        out['new_f1'] = f
        p, r, f  = self._kg_mention_f1.get_metric(reset)
        out['kg_p'] = p
        out['kg_r'] = r
        out['kg_f1'] = f
        out['new_ent_acc'] = self._new_entity_accuracy.get_metric(reset)
        out['new_ent_acc_20'] = self._new_entity_accuracy20.get_metric(reset)
        out['parent_ppl'] = self._parent_ppl.get_metric(reset)
        out['relation_ppl'] = self._relation_ppl.get_metric(reset)
        return out

