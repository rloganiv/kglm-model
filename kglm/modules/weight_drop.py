import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNBase
from functools import wraps


class BackHook(torch.nn.Module):
    def __init__(self, hook):
        super(BackHook, self).__init__()
        self._hook = hook
        self.register_backward_hook(self._backward)

    def forward(self, *inp):
        return inp

    @staticmethod
    def _backward(self, grad_in, grad_out):
        self._hook()
        return None


class WeightDrop(torch.nn.Module):
    """
    Implements drop-connect, as per Merity et al https://arxiv.org/abs/1708.02182
    """
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        self.hooker = BackHook(lambda: self._backward())

    def _setup(self):
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            if self.training:
                mask = raw_w.new_ones((raw_w.size(0), 1))
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
                setattr(self, name_w + "_mask", mask)
            else:
                w = raw_w
            rnn_w = getattr(self.module, name_w)
            rnn_w.data.copy_(w)

    def _backward(self):
        # transfer gradients from embeddedRNN to raw params
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            rnn_w = getattr(self.module, name_w)
            raw_w.grad = rnn_w.grad * getattr(self, name_w + "_mask")

    def forward(self, *args):
        self._setweights()
        return self.module(*self.hooker(*args))
