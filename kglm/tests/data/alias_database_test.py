from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
import numpy as np

from kglm.data import AliasDatabase


class AliasDatabaseTest(AllenNlpTestCase):
    # pylint: disable=protected-access,no-self-use

    def setUp(self):
        self.token_lookup = {
                'Entity1': [['Robert', 'Logan'], ['Robby']]
        }
        self.id_map_lookup = {
                'Entity1': {'Robert': 1, 'Logan': 2, 'Robby': 3}
        }
        self.id_array_lookup = {
                'Entity1': np.array([[1, 2], [3, 0]], dtype=int)
        }
        token_indexer = SingleIdTokenIndexer()
        entity_indexer = SingleIdTokenIndexer(namespace='entity_ids')
        text_field = TextField([Token(t) for t in ['Robby', 'is', 'a', 'nickname', 'for', 'Robert']],
                               {'tokens': token_indexer})
        entity_field = TextField([Token(t) for t in ['Entity1', '', '', '', '', 'Entity1']],
                                 {'entity_ids': entity_indexer})
        self.instance = Instance({
                'tokens': text_field,
                'entity_identifiers': entity_field
        })
        self.dataset = Batch([self.instance])
        self.vocab = Vocabulary.from_instances(self.dataset)
        self.dataset.index_instances(self.vocab)
        super(AliasDatabaseTest, self).setUp()

    def test_load(self):
        # Test that the load function has the expected behavior
        alias_database = AliasDatabase.load('kglm/tests/fixtures/mini.alias.pkl')
        test_entity = 'Q156216'  # Benton County

        # Check that aliases are tokenized properly
        expected_tokenized_aliases = [
                ['Benton', 'County'],
                ['Benton', 'County', ',', 'Washington']
        ]
        assert alias_database._token_lookup[test_entity] == expected_tokenized_aliases

        # Check that the id map has 4 unique tokens
        assert len(alias_database._id_map_lookup[test_entity]) == 4

        # Check that the first token in each alias has the same local id
        test_id_array = alias_database._id_array_lookup[test_entity]
        assert test_id_array[0, 0] == test_id_array[1, 0]

    def test_token_to_uid(self):
        alias_database = AliasDatabase(token_lookup=self.token_lookup,
                                       id_map_lookup=self.id_map_lookup,
                                       id_array_lookup=self.id_array_lookup)
        assert alias_database.token_to_uid('Entity1', 'Robert') == 1
        assert alias_database.token_to_uid('Entity1', 'Nelson') == 0

    def test_tensorize_and_lookup(self):
        # Tensor fields should be empty when ``AliasDatabase``` is created
        alias_database = AliasDatabase(token_lookup=self.token_lookup,
                                       id_map_lookup=self.id_map_lookup,
                                       id_array_lookup=self.id_array_lookup)
        assert not alias_database.is_tensorized

        # But should exist after ``AliasDatabase`` is tensorized
        alias_database.tensorize(self.vocab)
        return

        assert alias_database.is_tensorized
        assert alias_database._global_id_lookup != []
        assert alias_database._local_id_lookup != []

        tensor_dict = self.dataset.as_tensor_dict()
        entity_ids = tensor_dict['entity_identifiers']['entity_ids']
        global_tensor, local_tensor = alias_database.lookup(entity_ids)

        # The first two dimensions should match the batch_size and sequence length of the index.
        # The next dimensions should be the max number of aliases of all entities (in this case 2
        # for 'Robert Logan', and 'Robby') and the max length of the aliases (again 2 because
        # 'Robert Logan' is two tokens).
        assert global_tensor.shape == (1, 6, 2, 2)
        assert local_tensor.shape == (1, 6, 2, 2)

        # Check that the global ids match the vocabulary indices
        assert global_tensor[0, 0, 0, 0] == self.vocab.get_token_index('Robert', namespace='tokens')
        assert global_tensor[0, 1, 0, 0] == 0  # Padding since not an alias

        assert local_tensor[0, 0, 0, 0] == 1
        assert local_tensor[0, 1, 0, 0] == 0  # Padding since not an alias
