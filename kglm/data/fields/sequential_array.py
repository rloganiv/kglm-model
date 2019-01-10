import numpy

from allennlp.data.fields import ArrayField, SequenceField
from overrides import overrides


class SequentialArrayField(ArrayField, SequenceField):
    """
    Behaves exactly the same as ``ArrayField``, but also inherits from ``SequenceField`` to
    indicate that data is a sequence.
    """
    def __init__(self,
                 array: numpy.ndarray,
                 sequence_dim: int,
                 padding_value: int = 0) -> None:
        ArrayField.__init__(self, array=array, padding_value=padding_value)
        self._sequence_dim = sequence_dim

    @overrides
    def sequence_length(self) -> int:
        return self.array.shape[self._sequence_dim]
