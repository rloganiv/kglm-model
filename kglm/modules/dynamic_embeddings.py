from typing import Dict, List

from overrides import overrides
import torch
import torch.nn.functional as F

# pylint: disable=invalid-name


class DynamicEmbedding(torch.nn.Module):
    def __init__(self,
                 r: torch.Tensor,
                 dim: int) -> None:
                 #fe_dim: int):
        super(DynamicEmbedding, self).__init__()

        self.W_ent = torch.nn.Linear(in_features=dim, out_features=dim)
        self.W_delta = torch.nn.Linear(in_features=dim, out_features=dim)
        # self.Wdist = torch.nn.Linear(in_features=fe_dim, out_features=1)
        self.entity_embeddings: List[torch.Tensor] = []
        self._add_entity(r)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                h: torch.Tensor,
                r: torch.Tensor,
                entity_idx: int = None) -> Dict[str, torch.Tensor]:

        logits = []
        for e in self.entity_embeddings:
            score = torch.dot(h, self.W_ent(e))
            logits.append(score)
        logits = torch.cat(logits)
        out = {'logits': logits}

        if entity_idx:
            assert entity_idx < len(self.entity_embeddings)
            loss = F.cross_entropy(logits, entity_idx)
            out['loss'] = loss

            self._update_entity(h, entity_idx)
            if entity_idx == len(self.entity_embeddings):
                self._add_entity(r)

        return out

    def detach(self) -> None:
        # TODO: Check if this needs to be called after training on one batch.
        self.entity_embeddings = [e.detach() for e in self.entity_embeddings]

    def _add_entity(self, r: torch.Tensor) -> None:
        u = r + 0.01 * torch.randn(r.shape, device=r.device)
        e = u / torch.norm(u, p=2)
        self.entity_embeddings.append(e)

    def _update_entity(self, h: torch.Tensor, entity_idx: int) -> None:
        e = self.entity_embeddings[entity_idx]
        delta = torch.sigmoid(torch.dot(h, self.W_delta(e)))
        u = delta * e + (1 - delta) * h
        e_new = u / torch.norm(u, p=2)
        self.entity_embeddings[entity_idx] = e_new
