from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.token_embedders import Embedding
import torch

from kglm.modules.recent_entities import RecentEntities


class RecentEntitiesTest(AllenNlpTestCase):
    # pylint: disable=protected-access
    def setUp(self):
        self.entity_embedder = Embedding(5, 10)
        self.cutoff = 2
        self.recent_entities = RecentEntities(entity_embedder=None,
                                              cutoff=self.cutoff)
        super().setUp()

    def test_get_all_ids(self):
        # shape: (batch_size, seq_len, max_parents)
        entity_ids_t0 = torch.tensor([
            [[1, 2], [3, 0], [4, 0]],
            [[1, 0], [0, 0], [2, 0]]
        ])
        # ``RecentEntities.reset()`` must always be called before other operations.
        reset = torch.ones(entity_ids_t0.shape[0], dtype=torch.uint8)
        self.recent_entities.reset(reset)
        # ``RecentEntities._get_all_ids()`` will get all of the unique ids in each batch element.
        # We set ``sorted=True`` to ensure output is deterministic.
        all_ids = self.recent_entities._get_candidates(entity_ids_t0, sorted=True)
        expected = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 0, 0]])
        assert all_ids.equal(expected)

    def test_update_state(self):
        # shape: (batch_size, seq_len, max_parents)
        entity_ids_t0 = torch.tensor([
            [[1, 2], [3, 0], [4, 0]],
            [[1, 0], [0, 0], [2, 0]]
        ])
        # ``RecentEntities.reset()`` must always be called before other operations.
        reset = torch.ones(entity_ids_t0.shape[0], dtype=torch.uint8)
        self.recent_entities.reset(reset)
        # We update the state
        self.recent_entities._update_state(entity_ids_t0)
