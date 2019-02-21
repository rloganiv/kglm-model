from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.token_embedders import Embedding
import torch

from kglm.modules.recent_entities import RecentEntities


class RecentEntitiesTest(AllenNlpTestCase):
    # pylint: disable=protected-access
    def setUp(self):
        self.entity_embedder = Embedding(5, 10)
        self.cutoff = 2
        self.recent_entities = RecentEntities(cutoff=self.cutoff)
        super().setUp()

    def test_get_candidates(self):
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
        all_ids = self.recent_entities._get_candidates(entity_ids_t0)
        expected = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 0, 0]])
        assert all_ids.equal(expected)

    def test_mask(self):
        # Checks that the ``__call__``` method behaves as expected. We'll work with a single
        # sequence of two entities to keep life simple.
        entity_ids_t0 = torch.tensor([
            [[1, 2], [3, 0], [4, 0]],
            [[1, 0], [0, 0], [2, 0]]
        ])
        reset = torch.ones(entity_ids_t0.shape[0], dtype=torch.uint8)
        self.recent_entities.reset(reset)

        candidate_ids, candidate_mask = self.recent_entities(entity_ids_t0)
        # We know that the candidate ids are given in the order in the last test,
        # here we check that the mask looks correct for entities in the first batch.
        # The mask for entities 1 and 2 in the first sequence should be the same
        assert candidate_mask[0, :, 1].equal(candidate_mask[0,:,2])
        # The mask for entity 3 should be 1 for the 3rd timestep - it is not recent when it is first observed
        expected_3 = torch.tensor([0, 0, 1], dtype=torch.uint8)
        assert candidate_mask[0, :, 3].equal(expected_3)
        # The mask for entity 4 should only all zeros - it will not be recent until the next batch
        expected_4 = torch.tensor([0, 0, 0], dtype=torch.uint8)
        assert candidate_mask[0, :, 4].equal(expected_4)

        # Let's check that the remainders are correct
        # Entity 3 should have a remainder of 1, Entity 4 should have a remainder of 2, and
        # everything else should be filtered out.
        assert self.recent_entities._remaining[0] == {3: 1, 4: 2}

        # If we run on the same data again, the remainder for entity 4 should activate the mask for
        # the first and second timestep.
        candidate_ids, candidate_mask = self.recent_entities(entity_ids_t0)
        expected_4 = torch.tensor([1, 1, 0], dtype=torch.uint8)
        assert candidate_mask[0, :, 4].equal(expected_4)

        # But this should not happen if we reset the first sequence
        reset = torch.tensor([1, 0], dtype=torch.uint8)
        self.recent_entities.reset(reset)
        candidate_ids, candidate_mask = self.recent_entities(entity_ids_t0)
        expected_4 = torch.tensor([0, 0, 0], dtype=torch.uint8)
        assert candidate_mask[0, :, 4].equal(expected_4)
