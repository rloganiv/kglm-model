# pylint: disable=no-self-use
from typing import Any, Dict, List

from allennlp.data.fields.field import Field
from overrides import overrides


class GlobalObject(Field):
    """
    This "Field" (double-quotes intended) passes a global object created by a ``DatasetReader``
    to a ``Model``. These kinds of objects might store something like a large knowledge-base or
    some other container of information which:
        1. Is static.
        2. Is too large to tensorize each time a ``Batch`` is generated.
        3. Needs to be accessed both during dataset creating and during model training.
    Of course, you could just pass whatever you want...surely this won't have any unintended
    consequences.

        "With great power comes great responsibility"
            - Uncle Ben
    """
    def __init__(self, global_object: Any) -> None:
        self._global_object = global_object

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Any:
        # pylint: disable=unused-argument
        return self._global_object

    @overrides
    def empty_field(self) -> 'GlobalObject':
        return GlobalObject(None)

    @classmethod
    @overrides
    def batch_tensors(cls, tensor_list: List[Any]) -> Any:  # type: ignore
        return tensor_list[0]  # We only need one...

    def __str__(self) -> str:
        return "GlobalObject(%r)" % self._global_object
