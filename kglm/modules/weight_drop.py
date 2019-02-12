import torch
from torch.nn import Parameter
import torch.nn.functional as F


class WeightDrop(torch.nn.Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module, weights, dropout=0):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        for weight in self.weights:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, weight)
            self.register_parameter(f'{weight}_raw', Parameter(w.data))
            self.module._parameters[weight] = F.dropout(w, p=self.dropout, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for weight in self.weights:
            raw_w = getattr(self, f'{weight}_raw')
            self.module._parameters[weight] = F.dropout(raw_w, p=self.dropout, training=self.training)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

    def reset(self):
        for weight in self.weights:
            raw_w = getattr(self, f'{weight}_raw')
            self.module._parameters[weight] = F.dropout(raw_w, p=self.dropout, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()
