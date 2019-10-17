from typing import Dict, List, Optional, Tuple

from overrides import overrides
import torch
from torch.nn import Parameter
import torch.nn.functional as F


LstmState = Tuple[torch.FloatTensor, torch.FloatTensor]
StateDict = Dict[str, LstmState]


class WeightDrop(torch.nn.Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."
    # pylint: disable=protected-access

    def __init__(self, module, weights, dropout=0):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        for weight in self.weights:
            #Makes a copy of the weights of the selected layers.
            raw_w = getattr(self.module, weight)
            self.register_parameter(f'{weight}_raw', Parameter(raw_w.data))
            self.module._parameters[weight] = F.dropout(raw_w, p=self.dropout, training=False)

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
        if hasattr(self.module, 'reset'):
            self.module.reset()


class WeightDroppedLstm(torch.nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_dim: int,
                 hidden_size: int,
                 dropout: float) -> None:
        super().__init__()

        self._num_layers = num_layers
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._state: Optional[StateDict] = None

        # Create an LSTM for each layer and apply weight drop.
        rnns: List[torch.nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                input_size = embedding_dim
            else:
                input_size = hidden_size
            if i == num_layers - 1:
                output_size = embedding_dim
            else:
                output_size = hidden_size
            rnns.append(torch.nn.LSTM(input_size, output_size, batch_first=True))
        rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=dropout) for rnn in rnns]

        self._rnns = torch.nn.ModuleList(rnns)

    @overrides
    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:  # pylint: disable=arguments-differ
        current_input = embeddings
        hidden_list = []
        for layer, rnn in enumerate(self._rnns):
            # Retrieve previous hidden state for layer. Weird syntax in order to appease MyPy.
            prev_hidden: Optional[LstmState] = None
            if self._state is not None:
                prev_hidden = self._state['layer_%i' % layer]
            # Forward-pass.
            output, hidden = rnn(current_input, prev_hidden)
            output = output.contiguous()
            # Update hidden state for layer.
            hidden = tuple(h.detach() for h in hidden)
            hidden_list.append(hidden)
            current_input = output
        self._state = {'layer_%i' % i: h for i, h in enumerate(hidden_list)}
        return current_input

    def reset(self, reset: torch.ByteTensor) -> None:
        """Resets the internal hidden states"""
        # pylint: disable=invalid-name
        if self._state is None:
            return
        for layer in range(self._num_layers):
            h, c = self._state['layer_%i' % layer]
            h[:, reset, :] = torch.zeros_like(h[:, reset, :])
            c[:, reset, :] = torch.zeros_like(c[:, reset, :])
            self._state['layer_%i' % layer] = (h, c)
