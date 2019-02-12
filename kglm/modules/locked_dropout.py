import torch

class LockedDropout(torch.nn.Module):
    def forward(self,  # pylint: disable=arguments-differ
                x: torch.Tensor,
                dropout: float = 0.5) -> torch.Tensor:
        if not self.training or not dropout:
            return x
        mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
