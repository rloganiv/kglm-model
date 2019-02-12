"""
Hacky implementation of the NT-ASGD optimization strategy from:
    TODO: Paste arXiv link.
"""
import logging
from typing import Iterable, List

from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
import torch

logger = logging.getLogger(__name__)


@Optimizer.register('nt-asgd')
class NTASGDOptimizer(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float,
                 weight_decay: float = 0,
                 triggered: bool = False) -> None:
        params = list(params)
        self._triggered = triggered
        self._sgd = torch.optim.SGD(params,
                                    lr=lr,
                                    weight_decay=weight_decay)
        self._asgd = torch.optim.ASGD(params,
                                      lr=lr,
                                      t0=0,
                                      lambd=0.0,
                                      weight_decay=weight_decay)
        if triggered:
            self._active_optimizer = self._asgd
        else:
            self._active_optimizer = self._sgd
        defaults = dict(lr=lr,
                        weight_decay=weight_decay,
                        triggered=triggered)
        super(NTASGDOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        self._active_optimizer.step(closure)

    def trigger(self):
        logger.info('Triggering ASGD')
        self._active_optimizer = self._asgd
        self._triggered = True

    @property
    def triggered(self):
        return self._triggered

# Note: This technically does not change the learning rate, but I really don't
# want to have to make changes to the Trainer to accept this as a general
# Scheduler when its usage is exactly the same.
@LearningRateScheduler.register('nt-asgd')
class NTASGDScheduler(LearningRateScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 non_monotone_interval: int,
                 mode: str = 'min',
                 last_epoch: int = -1) -> None:
        if not isinstance(optimizer, NTASGDOptimizer):
            raise ConfigurationError('You must use an NTASDGOptimizer in '
                                     'order to use an NTASGDScheduler.')
        if mode not in ['min', 'max']:
            raise ConfigurationError('Mode can either be "min" or "max"')

        self.optimizer = optimizer
        self.non_monotone_interval = non_monotone_interval
        self.mode = mode
        self.last_epoch = last_epoch

        self.history: List[float] = []

    def step(self, metric: float = None, epoch: int = None) -> None:
        logger.debug('Optimizer: %s', self.optimizer._active_optimizer)
        logger.debug('Metric: %0.4f', metric)

        # Don't need to do anything if we've already switched from SGD to
        # ASGD.
        if self.optimizer.triggered:
            return
        # Otherwise check if it is time to trigger
        if epoch <= self.non_monotone_interval:
            self.history.append(metric)
            return
        if self.mode == 'min':
            best = min(self.history[:-self.non_monotone_interval])
            worse_off = metric > best
        elif self.mode == 'max':
            best = max(self.history[:-self.non_monotone_interval])
            worse_off = metric < best
        if worse_off:
            self.optimizer.trigger()
        self.history.append(metric)
