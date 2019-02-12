from typing import Dict

from allennlp.data.fields import ArrayField, SequenceField
import numpy as np
from overrides import overrides
import torch


class SequentialArrayField(ArrayField, SequenceField):
    """
    Behaves the same as ``ArrayField``, but also inherits from ``SequenceField`` to indicate that
    data is a sequence.
    """
    def __init__(self,
                 array: np.ndarray,
                 dtype: np.dtype,
                 sequence_dim: int = 0,
                 padding_value: int = 0) -> None:
        ArrayField.__init__(self, array=array, padding_value=padding_value)
        self._dtype = dtype
        self._sequence_dim = sequence_dim

    @overrides
    def sequence_length(self) -> int:
        return self.array.shape[self._sequence_dim]

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        max_shape = [padding_lengths["dimension_{}".format(i)]
                     for i in range(len(padding_lengths))]

        # Convert explicitly to an ndarray just in case it's an scalar (it'd end up not being an ndarray otherwise)
        return_array = np.asarray(np.full(max_shape, self.padding_value), dtype=self._dtype)

        # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
        # form the right shaped list of slices for insertion into the final tensor.
        slicing_shape = list(self.array.shape)
        if len(self.array.shape) < len(max_shape):
            slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(self.array.shape))]
        slices = tuple([slice(0, x) for x in slicing_shape])
        return_array[slices] = self.array
        tensor = torch.from_numpy(return_array)
        return tensor

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        # Pass the padding_value, so that any outer field, e.g., `ListField[ArrayField]` uses the
        # same padding_value in the padded ArrayFields
        return ArrayField(np.array([], dtype=self._dtype), padding_value=self.padding_value)
