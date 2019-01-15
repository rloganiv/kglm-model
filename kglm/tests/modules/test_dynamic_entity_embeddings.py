from unittest import TestCase

from allennlp.common import Params
from allennlp.nn.initializers import Initializer, InitializerApplicator
import torch

from kglm.modules import DynamicEmbedding

# pylint: disable=W0212,C0103


class TestDynamicEmbedding(TestCase):

    def test_initialization_and_reset(self):
        embedding_dim = 4
        max_embeddings = 10

        dynamic_embedding = DynamicEmbedding(embedding_dim, max_embeddings)

        self.assertEqual(dynamic_embedding._initial_embedding.shape, (embedding_dim,))
        self.assertEqual(dynamic_embedding._embedding_projection.in_features, embedding_dim)
        self.assertEqual(dynamic_embedding._embedding_projection.out_features, embedding_dim)
        self.assertEqual(dynamic_embedding._embedding_projection.in_features, embedding_dim)
        self.assertEqual(dynamic_embedding._embedding_projection.out_features, embedding_dim)
        self.assertIsNone(dynamic_embedding.embeddings)
        self.assertIsNone(dynamic_embedding.num_embeddings)
        self.assertIsNone(dynamic_embedding.last_seen)

        batch_size = 2
        dynamic_embedding.reset_states(batch_size)

        self.assertIsNotNone(dynamic_embedding.embeddings)
        self.assertIsNotNone(dynamic_embedding.num_embeddings)
        self.assertIsNotNone(dynamic_embedding.last_seen)
        self.assertEqual(dynamic_embedding.embeddings.shape,
                         (batch_size, max_embeddings, embedding_dim))
        self.assertEqual(dynamic_embedding.num_embeddings.shape,
                         (batch_size,))
        self.assertEqual(dynamic_embedding.last_seen.shape,
                         (batch_size, max_embeddings))

    def test_add_embedding(self):
        embedding_dim = 4
        max_embeddings = 10
        batch_size = 2

        dynamic_embedding = DynamicEmbedding(embedding_dim, max_embeddings)
        dynamic_embedding.reset_states(batch_size)

        timestep = 1
        mask = torch.tensor([0, 1], dtype=torch.uint8)  # pylint: disable=E1102
        dynamic_embedding.add_embeddings(timestep, mask)

        # Check new embeddings[0,1] is zero and [1,1] is non-zero
        embedding_0 = dynamic_embedding.embeddings[0, 1]
        embedding_1 = dynamic_embedding.embeddings[1, 1]
        zero = torch.zeros_like(embedding_0)
        self.assertTrue(torch.allclose(embedding_0, zero))
        self.assertFalse(torch.allclose(embedding_1, zero))

        # Check last seen is correct
        self.assertEqual(dynamic_embedding.last_seen[1, 1], 1)

        # Check gradient propagates to initial embedding
        dynamic_embedding.embeddings.sum().backward()
        self.assertIsNotNone(dynamic_embedding._initial_embedding.grad)

    def test_update_embedding(self):
        embedding_dim = 4
        max_embeddings = 10
        batch_size = 1

        dynamic_embedding = DynamicEmbedding(embedding_dim, max_embeddings)
        dynamic_embedding.reset_states(batch_size)

        hidden = torch.randn((batch_size, embedding_dim), requires_grad=True)
        update_indices = torch.tensor([0])  # pylint: disable=E1102
        timestep = 1

        # Check embedding changes on update
        original = dynamic_embedding.embeddings[0, 0].clone()
        dynamic_embedding.update_embeddings(hidden, update_indices, timestep)
        updated = dynamic_embedding.embeddings[0, 0]
        self.assertFalse(torch.allclose(original, updated))

        # Check last seen is correct
        self.assertEqual(dynamic_embedding.last_seen[0, 0], 1)

        # Check gradient propagates to initial embedding and hidden
        updated.sum().backward()
        self.assertIsNotNone(dynamic_embedding._initial_embedding.grad)
        self.assertIsNotNone(hidden.grad)

    # def test_forward(self):
    #     embedding_dim = 4
    #     max_embeddings = 10
    #     batch_size = 2

    #     dynamic_embedding = DynamicEmbedding(embedding_dim, max_embeddings, initializer)
    #     dynamic_embedding.reset(batch_size)
    #     dynamic_embedding.add_embeddings(0)
    #     dynamic_embedding.add_embeddings(0)

    #     hidden = torch.randn((batch_size, embedding_dim), requires_grad=True)
    #     target = torch.tensor([0, 2])  # pylint: disable=E1102
    #     dynamic_embedding(hidden, target)
