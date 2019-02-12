import itertools
import logging
import random
from typing import Iterable, Iterator, List, Tuple, Union

from allennlp.common.registrable import Registrable
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict, add_epoch_number
from allennlp.data.instance import Instance
import numpy as np
from overrides import overrides
import torch

logger = logging.getLogger(__name__)


def get_sequence_length(tensorized_field: Union[torch.Tensor, TensorDict]) -> int:
    if isinstance(tensorized_field, torch.Tensor):
        return tensorized_field.shape[1]
    elif isinstance(tensorized_field, dict):
        # We are making the extreme assumption that all of the tensors in a nested TensorDict have
        # the same sequence length.
        tensorized_subfield = next(iter(tensorized_field.values()))  # Get any value
        return get_sequence_length(tensorized_subfield)
    else:
        raise RuntimeError('Failed to get sequence length of one of the fields.')


class Splitter(Registrable):
    """
    An abstract ``Splitter`` class.

    Parameters
    ----------
    splitting_keys : List[str]
        List of fields that need to be split. All keys shoulds correspond to sequences with
        shared sequence dimension.
    """
    default_implementation = 'fixed'

    def __init__(self, splitting_keys: List[str]) -> None:
        self._splitting_keys = splitting_keys

    def __call__(self, tensor_dict: TensorDict, truncate_at: int) -> Iterable[TensorDict]:

        if not all(key in tensor_dict for key in self._splitting_keys):
            missing_keys = [key for key in self._splitting_keys if key not in tensor_dict]
            raise RuntimeError('Tensor dict is missing splitting keys: %s' % missing_keys)

        sequence_length = truncate_at

        split_indices = self._create_split_indices(sequence_length)
        # If the last split is too small, then merge with the second to last
        # split.
        if (split_indices[-1] - split_indices[-2]) <= 1:
            split_indices[-2] = split_indices[-1]
            del split_indices[-1]
        for i, (start, stop) in enumerate(zip(split_indices[:-1], split_indices[1:])):
            sliced_tensor_dict = self._slice_tensor_dict(tensor_dict, start, stop)
            if i == 0:
                sliced_tensor_dict['reset'] = True
            else:
                sliced_tensor_dict['reset'] = False
            yield sliced_tensor_dict

    def _create_split_indices(self, sequence_length) -> List[int]:
        raise NotImplementedError

    def _get_sequence_length(self, tensor_dict: TensorDict) -> int:
        sequence_lengths = []
        for key in self._splitting_keys:
            # Sometimes tensor dicts can be nested. E.g. for TextFeilds.
            tensorized_field = tensor_dict[key]
            sequence_length = get_sequence_length(tensorized_field)
            sequence_lengths.append(sequence_length)
        if not all(length == sequence_lengths[0] for length in sequence_lengths):
            raise RuntimeError('Cannot split sequences of unequal lengths')
        return sequence_lengths.pop()

    def _slice_tensor_dict(self, tensor_dict: TensorDict, start: int, end: int) -> TensorDict:

        def _recursion(tensor_or_dict):
            if isinstance(tensor_or_dict, torch.Tensor):
                return tensor_or_dict[:, start:end]
            elif isinstance(tensor_or_dict, dict):
                return {key: _recursion(value) for key, value in tensor_or_dict.items()}
            else:
                raise ValueError('Splitter encountered unexpected value in tensor_dict')

        other_keys = [key for key in tensor_dict.keys() if key not in self._splitting_keys]
        out = {key: tensor_dict[key] for key in other_keys}
        for key in self._splitting_keys:
            out[key] = _recursion(tensor_dict[key])

        return out


@Splitter.register('fixed')
class FixedSplitter(Splitter):
    def __init__(self,
                 split_size: int,
                 splitting_keys: List[str]):
        super().__init__(splitting_keys=splitting_keys)
        self._split_size = split_size

    @overrides
    def _create_split_indices(self, sequence_length: int) -> Iterable[int]:
        return list(range(0, sequence_length, self._split_size)) + [sequence_length]


@Splitter.register('random')
class RandomSplitter(Splitter):
    def __init__(self,
                 mean_split_size: int,
                 max_split_size: int,
                 min_split_size: int,
                 splitting_keys: List[str]) -> None:
        super().__init__(splitting_keys=splitting_keys)
        self._mean_split_size = mean_split_size
        self._max_split_size = max_split_size
        self._min_split_size = min_split_size

    @overrides
    def _create_split_indices(self, sequence_length: int) -> List[int]:
        split_indices = []
        last = 0
        while last < sequence_length:
            split_indices.append(last)
            delta = int(np.random.normal(self._mean_split_size, 5))
            delta = max(self._min_split_size, delta)
            delta = min(self._max_split_size, delta)
            last += delta
        split_indices.append(sequence_length)
        return split_indices


@DataIterator.register('split')
class SplitIterator(BucketIterator):
    """
    A modified version of ``BucketIterator`` which uses a ``Splitter`` to split large ``Tensors``
    into smaller chunks.

    Parameters
    ----------
    splitter : Splitter
        Tensors can be split in multiple ways (e.g. into fixed vs. random length chunks). A
        splitter produces a generator that yields chunks of the large tensor when called.
    sorting_keys : List[Tuple[str, str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.
        For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
        "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
        "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
        "num_token_characters" of the "sentence1" field.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.
        Note that if you specify ``max_instances_in_memory``, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        See :class:`BasicIterator`.
    """
    def __init__(self,
                 splitter: Splitter,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        super().__init__(sorting_keys=sorting_keys,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         cache_instances=False,
                         track_epoch=track_epoch,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._splitter = splitter

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.

        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset. IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        num_epochs : ``int``, optional (default=``None``)
            How times should we iterate over this dataset?  If ``None``, we will iterate over it
            forever.
        shuffle : ``bool``, optional (default=``True``)
            If ``True``, we will shuffle the instances in ``dataset`` before constructing batches
            and iterating over the data.
        """
        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            if self._cache_instances and key in self._cache:
                # Serve the results from the cache.
                tensor_dicts = self._cache[key]

                if shuffle:
                    random.shuffle(tensor_dicts)
                for tensor_dict in tensor_dicts:
                    if self._track_epoch:
                        # The tensor_dict already has an "epoch_num" tensor,
                        # so just fill it with the right value.
                        epoch_tensor: torch.Tensor = tensor_dict['epoch_num']
                        epoch_tensor.fill_(epoch)
                    for split_tensor_dict in self._splitter(tensor_dict):
                        yield split_tensor_dict
            else:
                batches = self._create_batches(instances, shuffle)

                # Should we add the instances to the cache this epoch?
                add_to_cache = self._cache_instances and key not in self._cache

                for batch in batches:
                    if self._track_epoch:
                        add_epoch_number(batch, epoch)

                    if self.vocab is not None:
                        batch.index_instances(self.vocab)


                    # In order to make  gradient updates fair in expectation,
                    # we randomly choose a sequence to cutoff at.
                    all_instance_lengths = [instance.get_padding_lengths() for
                                            instance in batch.instances]
                    random_instance = random.choice(all_instance_lengths)
                    truncate_at = random_instance['tokens']['num_tokens']
                    padding_lengths = batch.get_padding_lengths()
                    logger.debug('trunacate at: %s', truncate_at)
                    logger.debug('padding_lengths: %s', padding_lengths)

                    tensor_dict = batch.as_tensor_dict(padding_lengths)

                    if add_to_cache:
                        self._cache[key].append(tensor_dict)

                    for split_tensor_dict in self._splitter(tensor_dict, truncate_at):
                        yield split_tensor_dict

            # Increment epoch tracker
            self._epochs[key] = epoch + 1

    def get_num_batches(self, instances: Iterable[Instance]) -> float:
        return 1
