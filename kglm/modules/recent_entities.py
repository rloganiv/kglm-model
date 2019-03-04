from typing import Dict, List, Tuple

from allennlp.modules.token_embedders import TokenEmbedder
from overrides import overrides
import torch

from kglm.nn.util import nested_enumerate


class RecentEntities:
    """
    Module for tracking a dynamically changing list of entities.

    Parameters
    ----------
    cutoff : ``int``
        Number of time steps that an entity is considered 'recent'.
    """
    def __init__(self,
                 cutoff: int) -> None:
        self._cutoff = cutoff
        self._remaining: List[Dict[int, int]] = []

    def __call__(self,
                 entity_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the set of valid parent entities for a given mention.

        Parameters
        ----------
        entity_ids : ``torch.LongTensor``
            A tensor of shape ``(batch_size, sequence_length)`` whose elements are the ids
            of the corresponding token in the ``target`` sequence.

        Returns
        -------
        A tuple ``(candidate_ids, candidate_mask)`` containings the following elements:
        candidate_ids : ``torch.LongTensor``
            A tensor of shape ``(batch_size, n_candidates)`` of all of the candidates for each
            batch element.
        candidate_mask : ``torch.LongTensor``
            A tensor of shape ``(batch_size, sequence_length, n_candidates)`` defining which
            subset of candidates can be selected at the given point in the sequence.
        """
        batch_size, sequence_length = entity_ids.shape[:2]

        # TODO: See if we can get away without nested loops / cast to CPU.
        candidate_ids = self._get_candidates(entity_ids)
        candidate_lookup = [{parent_id: j for j, parent_id in enumerate(l)} for l in candidate_ids.tolist()]

        # Create mask
        candidate_mask = entity_ids.new_zeros(size=(batch_size, sequence_length, candidate_ids.shape[-1]),
                                              dtype=torch.uint8)

        # Start by accounting for unfinished masks that remain from the last batch
        for i, lookup in enumerate(self._remaining):
            for parent_id, remainder in lookup.items():
                # Find index w.r.t. the **current** set of candidates
                k = candidate_lookup[i][parent_id]
                # Fill in the remaining amount of mask
                candidate_mask[i, :remainder, k] = 1
                # If splits are really short, then we might still have some remaining
                lookup[parent_id] -= sequence_length

        # Cast to list so we can use elements as keys (not possible for tensors)
        parent_id_list = entity_ids.tolist()
        for i, j, *_, parent_id in nested_enumerate(parent_id_list):
            if parent_id == 0:
                continue
            else:
                # Fill in mask
                k = candidate_lookup[i][parent_id]
                candidate_mask[i, j + 1 : j + self._cutoff + 1, k] = 1
                # Track how many sequence elements remain
                remainder = sequence_length - (j + self._cutoff + 1)
                self._remaining[i][parent_id] = (j + self._cutoff + 1) - sequence_length

        # Remove any ids for non-recent parents (e.g. those without remaining mask)
        for i, lookup in enumerate(self._remaining):
            self._remaining[i] = {key: value for key, value in lookup.items() if value > 0}

        return candidate_ids, candidate_mask

    # pylint: disable=redefined-builtin
    def _get_candidates(self,
                        entity_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Combines the unique ids from the current batch with the previous set of ids to form the
        collection of **all** relevant ids.

        Parameters
        ----------
        entity_ids : ``torch.LongTensor``
            A tensor of shape ``(batch_size, sequence_length)`` whose elements are the ids
            of the corresponding token in the ``target`` sequence.

        Returns
        -------
        unique_entity_ids : ``torch.LongTensor``
            A tensor of shape ``(batch_size, max_num_parents)`` containing all of the unique
            candidate ids.
        """
        # Get the tensors of unique ids for each batch element and store them in a list
        all_unique: List[torch.LongTensor] = []
        for i, ids in enumerate(entity_ids):
            if self._remaining[i] is not None:
                previous_ids = list(self._remaining[i].keys())
                previous_ids = entity_ids.new_tensor(previous_ids)
                ids = torch.cat((ids.view(-1), previous_ids), dim=0)
            unique = torch.unique(ids, sorted=True)
            all_unique.append(unique)

        # Convert the list to a tensor by adding adequete padding.
        batch_size = entity_ids.shape[0]
        max_num_parents = max(unique.shape[0] for unique in all_unique)
        unique_entity_ids = entity_ids.new_zeros(size=(batch_size, max_num_parents))
        for i, unique in enumerate(all_unique):
            unique_entity_ids[i, :unique.shape[0]] = unique

        return unique_entity_ids

    def reset(self, reset: torch.ByteTensor) -> None:
        """
        Parameters
        ----------
        reset : ``torch.ByteTensor``
            A tensor of shape ``(batch_size,)`` indicating whether the state (e.g. list of
            previously seen entities) for the corresponding batch element should be reset.
        """
        if (len(reset) != len(self._remaining)) and not reset.all():
            raise RuntimeError('Changing the batch size without resetting all internal states is '
                               'undefined.')

        # If everything is being reset, then we treat as if the Module has just been initialized.
        # This simplifies the case where the batch_size has been
        if reset.all():
            batch_size = reset.shape[0]
            self._remaining = [dict() for _ in range(batch_size)]

        # Otherwise only reset the internal state for the indicated batch elements
        else:
            for i, should_reset in enumerate(reset):
                if should_reset:
                    self._remaining[i] = {}

    # TODO: Check
    def insert(self, values: torch.LongTensor, mask: torch.ByteTensor = None) -> None:
        """
        To deal with case in sampling where tail ids are only known after __call__
        """
        batch_size = values.shape[0]
        for i in range(batch_size):
            if mask is not None:
                if mask[i]:
                    self._remaining[i][values[i].item()] = self._cutoff + 1
            else:
                self._remaining[i][values[i].item()] = self._cutoff + 1

