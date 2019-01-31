from collections import Counter
import gc
import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def log_torch_garbage(verbose=False):
    """Outputs a list / summary of all tensors to the console."""
    logger.debug('Logging PyTorch garbage')
    obj_counts = Counter()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if verbose:
                    logger.debug('type: %s, size: %s, is_cuda: %s',
                                 type(obj), obj.size(), obj.is_cuda)
                obj_counts[(type(obj), obj.is_cuda)] += 1
            elif hasattr(obj, 'data'):
                if torch.is_tensor(obj.data):
                    if verbose:
                        logger.debug('type: %s, size: %s, is_cuda: %s',
                                     type(obj), obj.size(), obj.is_cuda)
                    obj_counts[(type(obj), obj.is_cuda)] += 1
        except (KeyError, OSError, RuntimeError):
            continue
    logger.debug('Summary stats')
    for key, count in obj_counts.most_common():
        obj_type, is_cuda = key
        logger.debug('type: %s, is_cuda: %s, count: %i', obj_type, is_cuda, count)


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
