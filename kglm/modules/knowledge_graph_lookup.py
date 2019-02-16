import logging
import pickle
from typing import List, Tuple

from allennlp.data.vocabulary import Vocabulary
from tqdm import tqdm
import torch

from kglm.nn.util import nested_enumerate

logger = logging.getLogger(__name__)


class KnowledgeGraphLookup(object):
    def __init__(self,
                 kg_path: str,
                 vocab: Vocabulary) -> None:
        self._kg_path = kg_path
        self._vocab = vocab
        self._relations, self._tail_ids = self.load_kg(kg_path)

    def load_kg(self, kg_path: str) -> Tuple[List[torch.LongTensor], List[torch.LongTensor]]:
        logger.info('Loading knowledge graph from: %s', kg_path)
        with open(kg_path, 'rb') as f:
            knowledge_graph = pickle.load(f)

        entity_idx_to_token = self._vocab.get_index_to_token_vocabulary('entity_ids')
        all_relations: List[torch.Tensor] = []
        all_tail_ids: List[torch.Tensor] = []
        for i in tqdm(range(len(entity_idx_to_token))):
            entity_id = entity_idx_to_token[i]
            try:
                # Get the relation and tail id tokens
                relation_tokens, tail_id_tokens = zip(*knowledge_graph[entity_id])
                # Index tokens
                relations = [self._vocab.get_token_index(t, 'relations') for t in relation_tokens]
                tail_ids = [self._vocab.get_token_index(t, 'entity_ids') for t in tail_id_tokens]
                # Convert to tensors
                relations = torch.LongTensor(relations)
                tail_ids = torch.LongTensor(tail_ids)
            except KeyError:
                relations = torch.LongTensor([])
                tail_ids = torch.LongTensor([])
            all_relations.append(relations)
            all_tail_ids.append(tail_ids)
        return all_relations, all_tail_ids

    def __call__(self,
                 parent_ids: torch.LongTensor) -> torch.Tensor:
        """Returns all relations of the form:

            (parent_id, relation, tail_id)

        from the knowledge graph.

        Parameters
        -----------
        parent_ids : ``torch.LongTensor``
            A tensor of arbitrary shape `(N, *)` where `*` means any number of additional
            dimensions. Each element is an id in the knowledge graph.

        Returns
        -------
        A tuple `(relations, tail_ids)` containing the following elements:
        relations : ``torch.LongTensor``
            A tensor of shape `(N, *, K)` corresponding to the input shape, with an
            additional dimension whose size `K` is the largest number of relations to
            a parent in the input. Each element is a relation id.
        tail_ids : ``torch.LongTensor``
            A tensor of shape `(N, *, K)` containing the corresponding tail ids.
        """
        # Collect the information to load into the output tensors.
        K = 0
        indices: List[Tuple[int, ...]] = []
        relations_list: List[torch.LongTensor] = []
        tail_ids_list: List[torch.LongTensor] = []
        for *inds, parent_id in nested_enumerate(parent_ids):
            # Retrieve data
            relations = self._relations[parent_id]
            tail_ids = self._tail_ids[parent_id]
            # Update output size
            K = max(K, len(relations))
            # Add to lists
            indices.append(inds)
            relations_list.append(self._relations[parent_id])
            tail_ids_list.append(self._tail_ids[parent_id])

        # Construct the output tensors
        relations = parent_ids.new_zeros(size=(*parent_ids.shape, K))
        tail_ids = parent_ids.new_zeros(size=(*parent_ids.shape, K))
        for inds, _relations, _tail_ids in zip(indices, relations_list, tail_ids_list):
            index = (*inds, slice(None, len(_relations)))
            relations[index] = _relations
            tail_ids[index] = _tail_ids

        return relations, tail_ids
