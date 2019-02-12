import logging
import itertools
import random
from typing import Dict, Iterable, Iterator, List, Tuple, Union

from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
import numpy as np
import torch

logger = logging.getLogger(__name__)

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name


@DataIterator.register('awd')
class AwdIterator(DataIterator):
    def __init__(self,
                 split_size: int,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        super(AwdIterator, self).__init__(
                batch_size=batch_size,
                instances_per_epoch=instances_per_epoch,
                max_instances_in_memory=max_instances_in_memory,
                cache_instances=cache_instances,
                track_epoch=track_epoch,
                maximum_samples_per_batch=maximum_samples_per_batch)
        self._split_size = split_size

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = False) -> Iterator[Tuple[TensorDict, float]]:
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        # Following the original implementation we will simply concatenate all
        # of the tensors together, then split it into batch_size pieces and
        # yield little chunks of the big array. Although the chunk sizes will
        # vary, the array will otherwise always be in the same order.
        for epoch in epochs:
            instance_list = list(instances)
            for instance in instance_list:
                instance.index_fields(self.vocab)

            tensor_dicts = [instance.as_tensor_dict() for instance in instance_list]
            big_ass_sequence = torch.cat([x['tokens']['tokens'] for x in tensor_dicts],
                                         dim=0)
            n_batch = big_ass_sequence.shape[0] // self._batch_size
            big_ass_sequence = big_ass_sequence.narrow(0, 0, n_batch * self._batch_size)
            big_ass_sequence = big_ass_sequence.view(self._batch_size, -1)
            total_length = big_ass_sequence.shape[1]
            i = 0
            while i < total_length - 2:
                if shuffle:
                    bptt = self._split_size if np.random.random() < 0.95 else self._split_size / 2
                    sequence_length = max(5, int(np.random.normal(bptt, 5)))
                    sequence_length = min(sequence_length, total_length - 1 - i)
                else:
                    bptt = self._split_size
                    sequence_length = min(self._split_size, total_length - 1 -i)
                if i == 0:
                    reset = torch.ones(self._batch_size, dtype=torch.uint8)
                else:
                    reset = torch.zeros(self._batch_size, dtype=torch.uint8)

                out_dict: TensorDict = {
                    'source': {'tokens': big_ass_sequence[:, i:i+sequence_length]},
                    'target': {'tokens': big_ass_sequence[:, i+1:i+sequence_length+1]},
                    'reset': reset
                }
                lr_mult = sequence_length / bptt
                i += sequence_length

                yield out_dict, lr_mult

            self._epochs[key] = epoch + 1
