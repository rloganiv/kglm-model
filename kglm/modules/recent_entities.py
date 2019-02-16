from typing import Any, Dict, List, Optional

from allennlp.modules.token_embedders import TokenEmbedder
import torch


class RecentEntities(torch.nn.Module):
    """
    Module for tracking a dynamically changing list of entities.

    Parameters
    ----------
    entity_embedder : ``TokenEmbedder``
        Lookup the embeddings of recent entities
    cutoff : ``int``
        Number of time steps that an entity is considered 'recent'.
    """
    def __init__(self,
                 entity_embedder: TokenEmbedder,
                 cutoff: int) -> None:
        self._entity_embedder = entity_embedder
        self._cutoff = cutoff
        self._previous_ids: List[torch.LongTensor] = []
        self._last_seen: List[Dict[int, int]] = []

    def __forward__(self,
                    hidden: torch.FloatTensor,
                    parent_ids: torch.LongTensor) -> None:
        """
        Computes the log-probability of selecting the provided of entity ids conditioned on the
        current hidden state, as well as updates the ``RecentEntities`` hidden state.

        Parameters
        ----------
        hidden : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, sequence_length, hidden_dim)`` containing the
            hidden states for the current batch.
        parent_ids : ``torch.LongTensor``
            The target entity ids.
        """

    # pylint: disable=redefined-builtin
    def _get_candidates(self,
                        parent_ids: torch.LongTensor,
                        sorted: bool = False) -> torch.LongTensor:
        """
        Combines the unique ids from the current batch with the previous set of ids to form the
        collection of **all** relevant parent entities.

        Parameters
        ----------
        parent_ids : ``torch.LongTensor``
            A tensor of shape ``(batch_size, seq_length, num_parents)`` containing the ids of all
            possible parents of the corresponding mention.
        sorted : ``bool`` (default=``False``)
            Whether or not to sort the parent ids.

        Returns
        -------
        unique_parent_ids : ``torch.LongTensor``
            A tensor of shape ``(batch_size, max_num_parents)`` containing all of the unique
            candidate parent ids.
        """
        # Get the tensors of unique ids for each batch element and store them in a list
        all_unique: List[torch.LongTensor] = []
        for i, id_sequence in enumerate(parent_ids):
            if self._previous_ids is not None:
                combined = torch.cat((self._previous_ids[i], id_sequence.view(-1)), dim=0)
                unique = torch.unique(combined, sorted=sorted)
            else:
                unique = torch.unique(id_sequence, sorted=sorted)
            all_unique.append(unique)

        # Convert the list to a tensor by adding adequete padding.
        batch_size = parent_ids.shape[0]
        max_num_parents = max(unique.shape[0] for unique in all_unique)
        unique_parent_ids = parent_ids.new_zeros(size=(batch_size, max_num_parents))
        for i, unique in enumerate(all_unique):
            unique_parent_ids[i, :unique.shape[0]] = unique

        return unique_parent_ids

    def _update_state(self,
                      parent_ids: torch.LongTensor):
        # We know for certain that everything before the cutoff cannot be carried over.
        truncated_parent_ids = parent_ids[:, -self._cutoff:]

        sequence_length = truncated_parent_ids.shape[1]
        for batch_index, batch_element in enumerate(truncated_parent_ids):
            for timestep, _parent_ids in enumerate(batch_element):
                for parent_id in _parent_ids:
                    self._last_seen[batch_index][parent_id] = sequence_length - timestep - 1
            # Get rid of anything past the cutoff
            self._last_seen[batch_index] = {key: value for key, value in self._last_seen[batch_index].items()
                                            if abs(value) < self._cutoff}
            # Update previous ids
            previous_ids = list(self._last_seen[batch_index].keys())
            self._previous_ids[batch_index] = self._previous_ids[batch_index].new_tensor(previous_ids)

    def reset(self, reset: torch.ByteTensor) -> None:
        """
        Parameters
        ----------
        reset : ``torch.ByteTensor``
            A tensor of shape ``(batch_size,)`` indicating whether the state (e.g. list of
            previously seen entities) for the corresponding batch element should be reset.
        """
        if (len(reset) != len(self._previous_ids)) and not reset.all():
            raise RuntimeError('Changing the batch size without resetting all internal states is '
                               'undefined.')

        # If everything is being reset, then we treat as if the Module has just been initialized.
        # This simplifies the case where the batch_size has been
        if reset.all():
            batch_size = reset.shape[0]
            self._last_seen = [{}] * batch_size
            self._previous_ids = [reset.new_empty(size=(0,), dtype=torch.int64)] * batch_size

        # Otherwise only reset the internal state for the indicated batch elements
        else:
            for i, should_reset in enumerate(reset):
                if should_reset:
                    self._previous_ids[i] = self._previous_ids[i].new_empty(size=(0,))
                    self._last_seen[i] = {}
