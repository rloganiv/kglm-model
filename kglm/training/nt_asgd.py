"""
Hacky implementation of the NT-ASGD optimization strategy from:
    TODO: Paste arXiv link.
"""
from copy import deepcopy
from itertools import chain
import logging
from typing import Iterable, List

from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
import torch

logger = logging.getLogger(__name__)


@Optimizer.register('nt-asgd')
class NTASGDOptimizer:
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

    ### Things that only NTASGDOptimizer does ###
    def trigger(self):
        logger.info('Triggering ASGD')
        self._triggered = True

    @property
    def triggered(self):
        return self._triggered

    @property
    def active_optimizer(self):
        if self.triggered:
            return self._asgd
        else:
            return self._sgd

    ### Optimizer methods that need to be redefined ###
    def __getstate__(self):
        return {
            '_triggered': self._triggered,
            '_sgd': self._sgd,
            '_asgd': self._asgd
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        return {
            '_triggered': self._triggered,
            '_sgd': self._sgd.state_dict(),
            '_asgd': self._asgd.state_dict()
        }

    def __repr__(self):
        return f'NTASGDOptimizer(triggered={self._triggered})'

    def load_state_dict(self, state_dict):
        state_dict = deepcopy(state_dict)
        self._triggered = state_dict['_triggered']
        self._sgd.load_state_dict(state_dict['_sgd'])
        self._asgd.load_state_dict(state_dict['_asgd'])

    ### Methods deferred to the active optimizer ###
    def zero_grad(self):
        self.active_optimizer.zero_grad()

    def step(self, closure=None):
        self.active_optimizer.step(closure)

    def add_param_group(self, param_group):
        self.active_optimizer.add_param_group(param_group)

    @property
    def param_groups(self):
        return self.active_optimizer.param_groups

    @property
    def state(self):
        return self.active_optimizer.state



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
        logger.debug('Optimizer: %s', self.optimizer.active_optimizer)
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

