from flaky import flaky
import torch

import kglm.nn.util as util


@flaky(max_runs=10, min_passes=1)
def test_sample_from_logp():
    # (batch_size, n_classes)
    unnormalized = torch.randn(10, 30)
    normalized = unnormalized / unnormalized.sum(-1, keepdim=True)
    logp = torch.log(normalized)
    logits_a, sample_a = util.sample_from_logp(logp)
    logits_b, sample_b = util.sample_from_logp(logp)
    assert not torch.allclose(logits_a, logits_b)
    assert not torch.allclose(sample_a, sample_b)

