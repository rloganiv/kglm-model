"""
Common types needed by model.
"""
from typing import Dict, Optional, Union

import torch


TensorDict = Dict[str, torch.Tensor]
StateDict = Optional[Dict[str, Union[torch.Tensor, TensorDict]]]
