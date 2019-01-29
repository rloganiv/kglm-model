from typing import Tuple

import torch

def sample_from_logp(logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draws samples from a tensor of log probabilities. API matches ``torch.max()``.

    Parameters
    ----------
    logp : ``torch.Tensor``
        Tensor of shape ``(batch_size, ..., n_categories)`` of log probabilities.

    Returns
    -------
    A tuple consisting of:
    selected_logp : ``torch.Tensor``
        Tensor of shape ``(batch_size, ...)`` containing the selected log probabilities.
    selected_idx : ``torch.Tensor``
        Tensor of shape ``(batch_size, ...)`` containing the selected indices.
    """
    pdf = torch.exp(logp)
    cdf = torch.cumsum(pdf, dim=-1)
    rng = torch.rand(logp.shape[:-1], device=logp.device).unsqueeze(-1)
    selected_idx = cdf.lt(rng).sum(dim=-1)
    hack = torch.ones(logp.shape[:-1], device=logp.device, dtype=torch.uint8)
    selected_logp = logp[hack, selected_idx[hack]]
    return selected_logp, selected_idx
