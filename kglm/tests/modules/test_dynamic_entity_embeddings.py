from unittest import TestCase

import torch

from kglm.modules import DynamicEmbedding

# pylint: disable=invalid-name


class TestDynEnt(TestCase):
    def setUp(self):
        self.dim = 4
        self.r = torch.randn(self.dim, requires_grad=True)
        self.dyn_ent = DynamicEmbedding(self.r, self.dim)

    def test_add_entity(self):
        self.dyn_ent._add_entity(self.r)
        self.assertEqual(len(self.dyn_ent.entity_embeddings), 2)

    def test_update_entity(self):
        initial = self.dyn_ent.entity_embeddings[0].clone()
        h = torch.randn(self.dim, requires_grad=True)
        self.dyn_ent._update_entity(h, 0)
        updated = self.dyn_ent.entity_embeddings[0]
        self.assertFalse(torch.allclose(initial, updated))

    def test_computation_graph(self):
        # Gradients should propagate back to r and h after update.
        h = torch.randn(self.dim, requires_grad=True)
        self.dyn_ent._update_entity(h, 0)
        updated = self.dyn_ent.entity_embeddings[0]
        updated.sum().backward(retain_graph=True)
        self.assertIsNotNone(self.r.grad)
        self.assertIsNotNone(h.grad)

        # Gradients should not exist after detach.
        self.dyn_ent.detach()
        self.assertFalse(self.dyn_ent.entity_embeddings[0].requires_grad)
