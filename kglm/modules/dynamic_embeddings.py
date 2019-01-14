import logging
from typing import List

from overrides import overrides
import torch

logger = logging.getLogger(__name__)


class DynamicEmbedding(torch.nn.Module):
    def __init__(self,
                 mean_embedding: torch.Tensor,
                 distance_scalar: torch.Tensor,
                 embedding_projection: torch.nn.Linear,
                 delta_projection: torch.nn.Linear) -> None:
        super(DynamicEmbedding, self).__init__()

        self._mean_embedding = mean_embedding
        self._distance_scalar = distance_scalar
        self._embedding_projection = embedding_projection
        self._delta_projection = delta_projection

        self.embeddings: List[torch.Tensor] = []
        self.last_seen: List[int] = []
        self.add_entity(0)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                hidden: torch.Tensor,
                timestep: int) -> torch.Tensor:
        """Computes logits for entity id prediction.

        Parameters
        ----------
        hidden : ``torch.Tensor(shape=embedding_dim)``
            Current hidden state.

        Returns
        -------
        logits : ``torch.Tensor(shape=num_entities)``
            Logits corresponding to which entity is predicted.
        """
        if len(self.embeddings) > 1:
            embedding_tensor = self._embedding_projection(torch.cat(self.embeddings))
            embedding_feature = torch.einsum('j,ij->i', hidden, embedding_tensor)
        else:
            embedding = self.embeddings[0]
            embedding_feature = torch.dot(hidden, self._embedding_projection(embedding).squeeze())

        last_seen = torch.tensor(self.last_seen, dtype=torch.float32, device=hidden.device)  # pylint: disable=not-callable
        distance_feature = torch.exp(self._distance_scalar * (timestep - last_seen))

        logits = embedding_feature + distance_feature

        return logits

    def detach(self) -> None:
        self.embeddings = [e.detach() for e in self.embeddings]

    def add_entity(self, timestep: int) -> None:
        mean = self._mean_embedding.clone()
        noisy = mean + 0.0001 * torch.randn(self._mean_embedding.shape,
                                            device=self._mean_embedding.device)
        normalized = noisy / torch.norm(noisy, p=2)
        self.embeddings.append(normalized)
        self.last_seen.append(timestep)

    def update_entity(self,
                      hidden: torch.Tensor,
                      entity_idx: torch.Tensor,
                      timestep: int) -> None:
        embedding = self.embeddings[entity_idx].clone()
        delta = torch.sigmoid(torch.dot(hidden, self._delta_projection(embedding).squeeze()))
        new_embedding = delta * embedding + (1 - delta) * hidden
        normalized = new_embedding / torch.norm(new_embedding, p=2)
        self.embeddings[entity_idx] = normalized
        self.last_seen[entity_idx] = timestep
