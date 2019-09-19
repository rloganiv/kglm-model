from collections import deque
from copy import deepcopy
import logging
import itertools
import random
from typing import Deque, Dict, Iterable, Iterator, List, Tuple, Union

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Batch
from allennlp.data.fields import Field, ListField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import add_epoch_number, DataIterator
import numpy as np
import torch

from kglm.data.fields import SequentialArrayField

logger = logging.getLogger(__name__)

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name


@DataIterator.register('fancy')
class FancyIterator(DataIterator):
    """Fancy cause it's really expensive."""
    def __init__(self,
                 batch_size: int,
                 split_size: int,
                 splitting_keys: List[str],
                 truncate: bool = True,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        super(FancyIterator, self).__init__(
            batch_size=batch_size,
            instances_per_epoch=instances_per_epoch,
            max_instances_in_memory=max_instances_in_memory,
            cache_instances=cache_instances,
            track_epoch=track_epoch,
            maximum_samples_per_batch=maximum_samples_per_batch)
        self._splitting_keys = splitting_keys
        self._split_size = split_size
        self._truncate = truncate

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = False) -> Iterator[TensorDict]:

        key = id(instances)
        starting_epoch = self._epochs[key]

        # In order to ensure that we are (almost) constantly streaming data to the model we
        # need to have all of the instances in memory ($$$)
        instance_list = list(instances)

        if (self._batch_size > len(instance_list)) and self._truncate:
            raise ConfigurationError('FancyIterator will not return any data when the batch size '
                                     'is larger than number of instances and truncation is enabled. '
                                     'To fix this either use a smaller batch size (better for '
                                     'training) or disable truncation (better for validation).')

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:

            if shuffle:
                random.shuffle(instance_list)

            # We create queues for each instance in the batch, and greedily fill them to try and
            # ensure each queue's length is roughly equal in size.
            queues: List[Deque[Instance]] = [deque() for _ in range(self._batch_size)]
            queue_lengths = np.zeros(self._batch_size, dtype=int)
            for instance in instances:

                # Now we split the instance into chunks.
                chunks, length = self._split(instance)

                # Next we identify which queue is the shortest and add the chunks to that queue.
                destination = np.argmin(queue_lengths)
                queues[destination].extend(chunks)
                queue_lengths[destination] += length

            # We need a NULL instance to replace the output of an exhausted queue if we are evaluating
            prototype = deepcopy(chunks[-1])
            new_fields: Dict[str, Field] = {}
            for name, field in prototype.fields.items():
                if isinstance(field, MetadataField):
                    new_fields[name] = field
                else:
                    new_fields[name] = field.empty_field()
            blank_instance = Instance(new_fields)

            for batch in self._generate_batches(queues, blank_instance):
                if self._track_epoch:
                    add_epoch_number(batch, epoch)

                if self.vocab is not None:
                    batch.index_instances(self.vocab)

                padding_lengths = batch.get_padding_lengths()
                yield batch.as_tensor_dict(padding_lengths), 1

            self._epochs[key] = epoch + 1

    def _split(self, instance: Instance) -> Tuple[List[Instance], int]:
        # Determine the size of the sequence inside the instance.
        true_length = len(instance['source'])
        if (true_length % self._split_size) != 0:
            offset = 1
        else:
            offset = 0
        padded_length = self._split_size * (true_length // self._split_size + offset)

        # Determine the split indices.
        split_indices = list(range(0, true_length, self._split_size))
        if true_length > split_indices[-1]:
            split_indices.append(true_length)

        # Determine which fields are not going to be split
        constant_fields = [x for x in instance.fields if x not in self._splitting_keys]

        # Create the list of chunks
        chunks: List[Instance] = []

        for i, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):

            # Copy all of the constant fields from the instance to the chunk.
            chunk_fields = {key: instance[key] for key in constant_fields}

            # Determine whether or not to signal model to reset.
            if i == 0:
                reset = SequentialArrayField(np.array(1), dtype=np.uint8)
            else:
                reset = SequentialArrayField(np.array(0), dtype=np.uint8)
            chunk_fields['reset'] = reset

            # Obtain splits derived from sequence fields.
            for key in self._splitting_keys:
                source_field = instance[key]
                # pylint: disable=protected-access
                if isinstance(source_field, TextField):
                    split_field = TextField(source_field.tokens[start:end],
                                            source_field._token_indexers)
                elif isinstance(source_field, SequentialArrayField):
                    # TODO: Figure out how to use sequence dim here...
                    split_field = SequentialArrayField(source_field.array[start:end],
                                                       dtype=source_field._dtype)
                elif isinstance(source_field, ListField):
                    split_field = ListField(source_field.field_list[start:end])
                else:
                    raise NotImplementedError('FancyIterator currently only supports splitting '
                                              '`TextField`s or `SequentialArrayField`s.')
                chunk_fields[key] = split_field
            chunks.append(Instance(chunk_fields))

        return chunks, padded_length

    def _generate_batches(self,
                          queues: List[Deque[Instance]],
                          blank_instance: Instance) -> Iterator[Batch]:
        num_iter = max(len(q) for q in queues)
        for _ in range(num_iter):
            instances: List[Instance] = []
            for q in queues:
                try:
                    instance = q.popleft()
                except IndexError:  # A queue is depleted
                    # If we're training, we break to avoid densely padded inputs (since this biases
                    # the model to overfit the longer sequences).
                    if self._truncate:
                        return
                    # But if we're evaluating we do want the padding, so that we don't skip anything.
                    else:
                        instance = blank_instance
                instances.append(instance)
            batch = Batch(instances)
            yield batch

    def get_num_batches(self, instances: Iterable[Instance]) -> float:
        return 0
