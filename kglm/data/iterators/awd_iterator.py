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
                 shuffle: bool = False) -> Iterator[TensorDict]:
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            # In order to ensure that we are (almost) constantly streaming data to the model we
            # need to have all of the instances in memory ($$$)
            instance_list = list(instances)
            if shuffle:
                random.shuffle(instance_list)
            for instance in instance_list:
                instance.index_fields(self.vocab)
            tensor_dicts = [instance.as_tensor_dict() for instance in instance_list]

            source = [d['source']['tokens'] for d in tensor_dicts]
            omni_source_tensor = torch.cat(source, 0)

            target = [d['target']['tokens'] for d in tensor_dicts]
            omni_target_tensor = torch.cat(target, 0)

            truncate_to = self._batch_size * (omni_source_tensor.shape[0] // self._batch_size)

            omni_source_tensor = omni_source_tensor[:truncate_to]
            omni_source_tensor = omni_source_tensor.view(self._batch_size, -1)

            omni_target_tensor = omni_target_tensor[:truncate_to]
            omni_target_tensor = omni_target_tensor.view(self._batch_size, -1)

            last = 0
            while last < omni_source_tensor.shape[1]:
                if random.random() > 0.5:
                    delta = int(np.random.normal(self._split_size, 5))
                else:
                    delta = int(np.random.normal(self._split_size / 2, 5))
                start = last
                last += delta
                if last >= (omni_source_tensor.shape[1] - 1):
                    break
                dict = {
                    'source': {'tokens': omni_source_tensor[:, start:last]},
                    'target': {'tokens': omni_target_tensor[:, start:last]},
                    'reset': torch.zeros(self._batch_size, dtype=torch.uint8)
                }
                yield dict

            self._epochs[key] = epoch + 1

