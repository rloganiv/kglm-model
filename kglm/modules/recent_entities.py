from typing import Any, Dict, List, Optional

from allennlp.modules.token_embedders import TokenEmbedder
import torch


class RecentEntities(torch.nn.Module):
    """
    Module for tracking a dynamically changing list of entities.

    Parameters
    ----------
    entity_embedder : ``TokenEmbedder``
        Used to create embeddings for entities.
    batch_size : ``int``
        The batch size.
    maximum_entities : ``int``
        The maximum number of entities that can be considered recent.
    lifespan : ``int``
        The number of time steps that must elapse before an entity is removed from the set of
        recent entities.
    """
    def __init__(self,
                 entity_embedder: TokenEmbedder,
                 batch_size: int,  # TODO: Can we avoid explicitly defining this?
                 maximum_entities: int,  # TODO: Determine if this is needed.
                 lifespan: int) -> None:
        self._entity_embedder = entity_embedder
        self._batch_size = batch_size
        self._maximum_entities = maximum_entities
        self._last_seen: Optional[List[Any]] = None
        self._lifespan = lifespan

    def __forward__(self,
                    hidden: torch.FloatTensor,
                    entity_ids: torch.LongTensor) -> None:
        """
        Computes the log-probability of selecting the provided of entity ids conditioned on the
        current hidden state, as well as updates the ``RecentEntities`` hidden state.

        Parameters
        ----------
        hidden : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, sequence_length, hidden_dim)`` containing the
            hidden states for the current batch.
        entity_ids : ``torch.LongTensor``
            The target entity ids.
        """
        raise NotImplementedError

    def _create_logit_mask(self,
                           entity_ids: torch.LongTensor = None) -> torch.ByteTensor:
        """
        Creates a mask which specifying the valid entities choices at each time step.

        Parameters
        ----------
        entity_ids : ``torch.LongTensor``
            The target entity ids.

        Returns
        -------
        logit_mask : ``torch.ByteTensor``
            A mask indicating which entities are valid at each time step.
        """
        raise NotImplementedError

    # TODO: Settle on name. Figure out what this should return: a tensor of (normalized) logits, with or without the corresponding mask? How will user know which entity a logit corresponds to?
    def get_logits(self,
                   hidden: torch.Tensor):
        """
        Parameters
        ----------
        hidden : ``torch.Tensor``
            A tensor of shape ``(batch_size)`` containing the hidden state for the current time
            step.
        """
        if self.training:
            raise Warning(
                '``RecentEntities.get_logits`` is not intended to be used during training. Are '
                'you sure you don\'t mean to use ``__forward__``?')
        raise NotImplementedError

    # TODO: Settle on name. Should we allow multiple entity ids to be selected during inference?
    def update(self, entity_ids: torch.LongTensor) -> None:
        """
        Parameters
        ----------
        entity_ids : ``torch.LongTensor``
            A tensor of shape ``(batch_size,)``
        """
        if self.training:
            raise Warning(
                '``RecentEntities.update`` is not intended to be used during training. Are '
                'you sure you don\'t mean to use ``__forward__``?')
        raise NotImplementedError

    def reset(self, reset: torch.ByteTensor) -> None:
        """
        Parameters
        ----------
        reset : ``torch.ByteTensor``
            A tensor of shape ``(batch_size,)`` indicating whether the state (e.g. list of
            previously seen entities) for the corresponding batch element should be reset.
        """
        for i in reset:
            self._last_seen[i] = []
