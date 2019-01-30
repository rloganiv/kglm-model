from collections import Counter
import gc
import logging

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
