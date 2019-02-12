from allennlp.common.testing import AllenNlpTestCase
import torch

from kglm.modules.locked_dropout import LockedDropout


class LockedDropoutTest(AllenNlpTestCase):
    def setUp(self):
        self.dropout_rate = 0.75
        self.model = LockedDropout()
        super().setUp()

    def test_sequence_elements_droppped_properly(self):
        self.model.train()
        x = torch.randn(10, 10, 10)
        x_prime = self.model(x, self.dropout_rate)

        # Tensors should differ
        assert not(x.equal(x_prime))

        # And zero elements should be the same across rows
        row_0 = x_prime.eq(0)[:, 0, :]
        row_1 = x_prime.eq(0)[:, 1, :]
        assert row_0.equal(row_1)

    def test_sequence_elements_not_dropped_during_eval(self):
        self.model.eval()
        x = torch.randn(10, 10, 10)
        x_prime = self.model(x, self.dropout_rate)
        assert x.equal(x_prime)
