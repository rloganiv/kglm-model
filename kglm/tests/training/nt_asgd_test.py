from allennlp.common.testing import AllenNlpTestCase
import torch

from kglm.training.nt_asgd import NTASGDOptimizer, NTASGDScheduler


class NTASGDOptimizerTest(AllenNlpTestCase):
    def setUp(self):
        self.dim = 10
        self.model = torch.nn.Linear(10, 10)
        self.optim = NTASGDOptimizer(self.model.parameters(),
                                     lr=30.0,
                                     weight_decay=1.2e-6)
        super().setUp()

    def test_trigger(self):
        # Active optimizer should be SGD before triggering
        assert self.optim.active_optimizer == self.optim._sgd
        assert not self.optim.triggered
        # Active optimizer should be ASGD after triggering
        self.optim.trigger()
        assert self.optim.active_optimizer == self.optim._asgd
        assert self.optim.triggered

    def test_awd_lstm_magic_trick(self):
        # Here we verify we can replicate the confusing trick done in:
        #   github.com/salesforce/awd-lstm-lm/main.py 244-260

        # We need to be in asgd mode.
        self.optim.trigger()

        # Perform a couple iterations of "training".
        for _ in range(3):
            self.optim.zero_grad()
            x = torch.randn(1, 10)
            y_hat = self.model(x)
            y_true = torch.randn(1, 10)
            loss = (y_hat - y_true).pow(2).mean()
            loss.backward()
            self.optim.step()

        # Now for the trick: assign the model parameters to the asgd average during evaluation.
        tmp = {}
        for prm in self.model.parameters():
            tmp[prm] = prm.data.clone()
            prm.data = self.optim.active_optimizer.state[prm]['ax'].clone()

        # HERE IS WHERE WE WOULD EVALUATE

        # Once we're done evaluating we reset the params for training
        for prm in self.model.parameters():
            prm.data = tmp[prm].clone()


class NTASGDSchedulerTest(AllenNlpTestCase):
    def setUp(self):
        self.dim = 10
        self.model = torch.nn.Linear(10, 10)
        self.optim = NTASGDOptimizer(self.model.parameters(),
                                     lr=30.0,
                                     weight_decay=1.2e-6)
        self.scheduler = NTASGDScheduler(optimizer=self.optim,
                                         non_monotone_interval=3)
        super().setUp()

    def test_scheduler_does_not_trigger_early(self):
        assert not self.optim.triggered
        self.scheduler.step(0.0, 0)
        self.scheduler.step(0.1, 1)
        self.scheduler.step(0.2, 2)
        assert not self.optim.triggered

    def test_scheduler_does_not_trigger_if_always_improving(self):
        assert not self.optim.triggered
        self.scheduler.step(10, 0)
        self.scheduler.step(9, 1)
        self.scheduler.step(8, 2)
        self.scheduler.step(7, 3)
        self.scheduler.step(6, 4)
        self.scheduler.step(5, 5)
        self.scheduler.step(4, 6)
        assert not self.optim.triggered

    def test_scheduler_does_trigger_when_expected(self):
        assert not self.optim.triggered
        self.scheduler.step(10, 0)
        self.scheduler.step(9, 1)
        self.scheduler.step(8, 2)
        self.scheduler.step(7, 3)
        self.scheduler.step(11, 4)
        assert self.optim.triggered
