import pickle

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.vocabulary import Vocabulary
import torch

from kglm.modules import KnowledgeGraphLookup


class KnowledgeGraphLookupTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # We assume our knowledge graphs are given in the form of pickled dictionaries, we'll
        # create a simple one here for testing purposes.
        self.temp_knowledge_graph = {
            'E1': [['R1', 'E2'], ['R2', 'E3']],
            'E2': [['R1', 'E3']],
        }
        self.path = self.TEST_DIR / 'relation.pkl'
        with open(self.path, 'wb') as f:
            pickle.dump(self.temp_knowledge_graph, f)

        # We need a vocab to track the unique entity ids and relations
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace('E1', 'raw_entity_ids')
        self.vocab.add_token_to_namespace('E2', 'raw_entity_ids')
        self.vocab.add_token_to_namespace('E3', 'raw_entity_ids')
        self.vocab.add_token_to_namespace('E1', 'entity_ids')
        self.vocab.add_token_to_namespace('E2', 'entity_ids')
        self.vocab.add_token_to_namespace('E3', 'entity_ids')
        self.vocab.add_token_to_namespace('R1', 'relations')
        self.vocab.add_token_to_namespace('R2', 'relations')

        # Lastly we create the knowledge graph lookup
        self.knowledge_graph_lookup = KnowledgeGraphLookup(self.path, self.vocab)

    def test_lists_are_correct(self):
        # The lookup converts the data in the knowledge graph into lists of tensors.
        # Here we check that these lists contain the expected information.
        relations = self.knowledge_graph_lookup._relations
        tail_ids = self.knowledge_graph_lookup._tail_ids

        # We'll check that the information is correct for entity 'E1'. We'll start by building the
        # expected tensors from our inputs...
        expected_relations, expected_tail_ids = zip(*self.temp_knowledge_graph['E1'])
        expected_relations = [self.vocab.get_token_index(t, 'relations') for t in expected_relations]
        expected_tail_ids = [self.vocab.get_token_index(t, 'raw_entity_ids') for t in expected_tail_ids]
        expected_relations = torch.LongTensor(expected_relations)
        expected_tail_ids = torch.LongTensor(expected_tail_ids)
        # ...then checking whether the corresponding elements in the lists are correct.
        index = self.vocab.get_token_index('E1', 'entity_ids')
        assert relations[index].equal(expected_relations)
        assert tail_ids[index].equal(expected_tail_ids)

    def test_lookup(self):
        # Check that the output of the lookup matches our expectations.
        parent_ids = [
            self.vocab.get_token_index('E1', 'entity_ids'),
            self.vocab.get_token_index('E2', 'entity_ids'),
            self.vocab.get_token_index('E3', 'entity_ids')  # Should work, even though E3 not in the KG
        ]
        parent_ids = torch.LongTensor(parent_ids)
        indices, _, relations, tail_ids = self.knowledge_graph_lookup(parent_ids)

        # Lookup indices of tokens expected to be in the output
        e2 = self.vocab.get_token_index('E2', 'raw_entity_ids')
        e3 = self.vocab.get_token_index('E3', 'raw_entity_ids')
        r1 = self.vocab.get_token_index('R1', 'relations')
        r2 = self.vocab.get_token_index('R2', 'relations')

        # Expected outputs (these are directly transcribed from the KG)
        expected_indices = [(0,), (1,)]
        expected_relations = [
            torch.LongTensor([r1, r2]),
            torch.LongTensor([r1])
        ]
        expected_tail_ids = [
            torch.LongTensor([e2, e3]),
            torch.LongTensor([e3])
        ]

        # Check expectations are met
        assert indices == expected_indices
        for observed, expected in zip(relations, expected_relations):
            assert observed.equal(expected)
        for observed, expected in zip(tail_ids, expected_tail_ids):
            assert observed.equal(expected)
