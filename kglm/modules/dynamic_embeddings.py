import logging
from typing import Dict, List, Optional, Tuple

from overrides import overrides
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DynamicEmbedding(torch.nn.Module):
    def __init__(self,
                 mean_embedding: torch.Tensor,
                 embedding_projection: torch.nn.Linear,
                 delta_projection: torch.nn.Linear) -> None:
        super(DynamicEmbedding, self).__init__()

        self._mean_embedding = mean_embedding
        self._embedding_projection = embedding_projection
        self._delta_projection = delta_projection

        self.embeddings: List[torch.Tensor] = []
        self._add_entity()

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                hidden: torch.Tensor,
                entity_idx: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if len(self.embeddings) > 1:
            embedding_tensor = self._embedding_projection(torch.stack(self.embeddings))
            logits = torch.einsum('j,ij->i', hidden, embedding_tensor)
        else:
            embedding = self.embeddings[0]
            logits = torch.dot(hidden, self._embedding_projection(embedding))

        if entity_idx is not None:
            if entity_idx >= len(self.embeddings):
                logger.info('idx: %i - len: %i', entity_idx, len(self.embeddings))
                raise RuntimeError('Entity index is too great')
            loss = F.cross_entropy(logits.view(1, -1), entity_idx.view(1))

            self._update_entity(hidden, entity_idx)
            if entity_idx == (len(self.embeddings) - 1):
                self._add_entity()
        else:
            loss = None

        return logits, loss

    def detach(self) -> None:
        self.embeddings = [e.detach() for e in self.embeddings]

    def _add_entity(self) -> None:
        mean = self._mean_embedding.clone()
        noisy = mean + 0.0001 * torch.randn(self._mean_embedding.shape,
                                         device=self._mean_embedding.device)
        normalized = noisy / torch.norm(noisy, p=2)
        self.embeddings.append(normalized)

    def _update_entity(self, hidden: torch.Tensor, entity_idx: int) -> None:
        embedding = self.embeddings[entity_idx].clone()
        delta = torch.sigmoid(torch.dot(hidden, self._delta_projection(embedding)))
        new_embedding = delta * embedding + (1 - delta) * hidden
        normalized = new_embedding / torch.norm(new_embedding, p=2)
        self.embeddings[entity_idx] = normalized
