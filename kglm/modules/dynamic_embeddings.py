from copy import deepcopy
import logging
from typing import Dict, Optional

from overrides import overrides
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F

from allennlp.nn.util import masked_log_softmax

logger = logging.getLogger(__name__)


class DynamicEmbedding(Module):
    """Dynamic embedding module.

    PyTorch friendly implementation of the dynamic entity embeddings from:
        Cite entity_nlm
    Designed so that entity lookups and updates can be efficiently done in batch.
    The tricks are:
        1. To pre-allocate a tensor for storing the entity embeddings on each reset.
        2. To use logical indexing to perform the updates.

    Parameters
    ----------
    embedding_dim : ``int``
        Dimension of the entity embeddings.
    max_embeddings : ``int``
        Maximum number of allowed embeddings.
    """
    def __init__(self,
                 embedding_dim: int,
                 max_embeddings: int) -> None:
        super(DynamicEmbedding, self).__init__()

        self._embedding_dim = embedding_dim
        self._max_embeddings = max_embeddings
        self._initial_embedding = Parameter(F.normalize(torch.randn(embedding_dim), dim=0))

        self._distance_scalar = Parameter(torch.tensor(1e-6))  # pylint: disable=E1102
        self._embedding_projection = torch.nn.Linear(in_features=embedding_dim,
                                                     out_features=embedding_dim,
                                                     bias=False)
        self._delta_projection = torch.nn.Linear(in_features=embedding_dim,
                                                 out_features=embedding_dim,
                                                 bias=False)

        self.embeddings: torch.Tensor = None  # Storage for embeddings
        self.num_embeddings: torch.Tensor = None  # Tracks how many embeddings are in use
        self.last_seen: torch.Tensor = None  # Tracks last time embedding was seen

    def reset_states(self, reset: torch.ByteTensor) -> None:
        """
        Resets the DynamicEmbedding module for use on a new batch of sequences.

        Parameters
        ----------
        batch_size : ``int``
            The batch_size of the new sequence.
        """
        batch_size = reset.shape[0]
        if self.embeddings is not None:
            if (batch_size != self.embeddings.shape[0]) and not reset.all():
                raise RuntimeError('Changing the batch size without resetting all internal states is '
                                   'undefined.')

        # If everything is being reset, then we treat as if the Module has just been initialized.
        # This simplifies the case where the batch_size has been
        if reset.all():
            self.embeddings = self._initial_embedding.new_zeros(batch_size, self._max_embeddings,
                                                                self._embedding_dim)
            self.num_embeddings = self._initial_embedding.new_zeros(batch_size, dtype=torch.int64)
            self.last_seen = self._initial_embedding.new_zeros(batch_size, self._max_embeddings,
                                                               dtype=torch.int64)
        else:
            self.embeddings[reset] = 0
            self.num_embeddings[reset] = 0
            self.last_seen[reset] = 0

        self.add_embeddings(0, reset)

    def detach_states(self) -> None:
        """
        Detaches embeddings from the computation graph. This can be neccesary when working with
        long sequences.
        """
        self.embeddings = self.embeddings.detach()

    def add_embeddings(self,
                       timestep: int,
                       mask: Optional[torch.Tensor] = None) -> None:
        """
        Adds new embeddings to the current collection of embeddings.

        Parameters
        ----------
        timestep: ``int``, (optional)
            The current time step.
        mask: ``Optional[torch.Tensor]``
            A tensor of shape ``(batch_size)`` indicating which sequences to add a new dynamic
            embedding to. If no mask is provided then a new embedding is added for each sequence
            in the batch.
        """
        if mask is None:
            batch_size = self.num_embeddings.shape[0]
            mask = self.num_embeddings.new_ones(batch_size, dtype=torch.uint8)
        elif not mask.any():
            return

        # Embeddings are initialized by adding a small amount of random noise to the initial
        # embedding tensor then normalizing.
        initial = self._initial_embedding.repeat((mask.sum(), 1, 1))
        noise = 1e-4 * torch.randn_like(initial)  # 1e-4 is a magic number from the original implementation
        unnormalized = initial + noise
        normalized = F.normalize(unnormalized, dim=-1)

        self.embeddings[mask, self.num_embeddings[mask]] = normalized.squeeze()
        self.last_seen[mask, self.num_embeddings[mask]] = timestep
        self.num_embeddings[mask] += 1

        if self.num_embeddings.max() == (self._max_embeddings - 1):
            logger.warning('Embeddings full')


    def update_embeddings(self,
                          hidden: torch.Tensor,
                          update_indices: torch.Tensor,
                          timestep: int,
                          mask: Optional[torch.Tensor] = None) -> None:
        """
        Updates existing embeddings.

        Parameters
        ----------
        hidden : ``torch.Tensor``
            A tensor of shape ``(batch_size, embedding_dim)`` used to update existing embeddings.
        update_indices : ``torch.Tensor``
            A tensor of shape ``(batch_size)`` whose elements specify which of the existing
            embeddings to update. Only one embedding per sequence can be updated at a time.
        timestep : ``int``
            The current time step.
        mask: ``Optional[torch.Tensor]``
            A tensor of shape ``(batch_size)`` indicating which sequences in the batch to update
            the dynamic embeddings for. If a mask is not provided then all sequences will be
            updated.
        """
        if mask is None:
            batch_size = self.num_embeddings.shape[0]
            mask = self.num_embeddings.new_ones(batch_size, dtype=torch.uint8)
        elif mask.sum() == 0:
            return
        else:
            batch_size = mask.sum()

        embeddings = self.embeddings[mask, update_indices[mask]]
        hidden = hidden.clone()[mask]

        # Equation 8 in the paper.
        projected = self._delta_projection(embeddings)
        score = torch.bmm(hidden.view(batch_size, 1, -1),
                          projected.view(batch_size, -1, 1))
        score = score.view(batch_size, 1)
        delta = torch.sigmoid(score)

        unnormalized = delta * embeddings + (1 - delta) * hidden
        normalized = F.normalize(unnormalized, dim=-1)

        # If the batch size is one, our approach of indexing with masks will drop the batch
        # dimension when accessing self.embeddings. Accordingly, the batch dimension of
        # normalized needs to be dropped in this case in order for assignment to work.
        self.embeddings[mask, update_indices[mask]] = normalized.squeeze(0)
        self.last_seen[mask, update_indices[mask]] = timestep

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                hidden: torch.Tensor,
                timestep: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Computes logits over the existing embeddings given the current hidden state. If target
        ids are provided then a loss is returned as well.

        Parameters
        ----------
        hidden : ``torch.Tensor``
            A tensor with shape ``(batch_size, embedding_dim)`` containing the current hidden
            states.
        target : ``Optional[torch.Tensor]``
            An optional tensor with shape ``(batch_size,)`` containing the target ids.
        mask : ``Optional[torch.Tensor]``
            An optional tensor with shape ``(batch_size,)`` indicating which terms to include in
            the final loss.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``torch.Tensor``
            The entity prediction logits.
        logit_mask : ``torch.Tensor``
            Mask for the logit tensor.
        loss : ``Optional[torch.Tensor]``
            The loss.
        """
        if mask is None:
            batch_size = self.num_embeddings.shape[0]
            mask = self.num_embeddings.new_ones(batch_size, dtype=torch.uint8)
        elif mask.sum() == 0:
            return {'loss': 0.0}
        else:
            batch_size = mask.sum()

        # First half of equation 4.
        embeddings = self.embeddings[mask]
        projected_embeddings = self._embedding_projection(embeddings)
        hidden = hidden[mask].unsqueeze(1)
        bilinear = torch.bmm(hidden, projected_embeddings.transpose(1, 2))
        bilinear = bilinear.view(batch_size, -1)

        # Second half of equation 4.
        distance_score = torch.exp(self._distance_scalar * (self.last_seen[mask].float() - timestep))
        logits = bilinear + distance_score

        # Since we pre-allocate the embedding array, logits includes scores for all of the
        # embeddings which have not yet been initialized. We create a mask to indicate which scores
        # should be used for prediction / loss calculation.
        num_embeddings = self.num_embeddings[mask].unsqueeze(1)
        arange = torch.arange(self._max_embeddings, device=num_embeddings.device).repeat(mask.sum(), 1)
        logit_mask = arange.lt(num_embeddings)
        logits[logit_mask != 1] = 1e-34

        out = {
                'logits': logits,
                'logit_mask': logit_mask
        }

        if target is not None:
            target = target[mask].unsqueeze(-1)
            log_probs = masked_log_softmax(logits, logit_mask)
            loss = log_probs.gather(-1, target)
            out['loss'] = loss

        return out

    def beam_state(self):
        beam_state = {
            'embeddings': self.embeddings.detach(),
            'num_embeddings': self.num_embeddings.detach(),
            'last_seen': self.last_seen.detach()
        }
        return beam_state

    def load_beam_state(self, beam_state):
        self.embeddings = beam_state.get('embeddings', None)
        self.num_embeddings = beam_state.get('num_embeddings', None)
        self.last_seen = beam_state.get('last_seen', None)
