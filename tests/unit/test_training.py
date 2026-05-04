import numpy as np
import torch

from mmvlm4scd.training.losses import (cox_partial_likelihood_loss,
                                       multitask_loss)
from mmvlm4scd.training.metrics import concordance_index_torch


def test_cox_loss_finite_and_decreases_with_correct_ranking():
    rng = np.random.default_rng(0)
    n = 64
    time = torch.as_tensor(rng.uniform(1, 20, size=n), dtype=torch.float32)
    event = torch.as_tensor((rng.uniform(size=n) < 0.6).astype(int), dtype=torch.long)
    risk_random = torch.randn(n)
    risk_correct = -time   # higher risk for shorter time
    l_rand = cox_partial_likelihood_loss(risk_random, time, event)
    l_corr = cox_partial_likelihood_loss(risk_correct, time, event)
    assert torch.isfinite(l_rand) and torch.isfinite(l_corr)
    assert l_corr < l_rand


def test_multitask_loss_returns_components():
    n = 32
    logits = torch.randn(n, 3, requires_grad=True)
    target = torch.randint(0, 3, (n,))
    risk = torch.randn(n, requires_grad=True)
    time = torch.rand(n) * 20 + 0.1
    event = torch.randint(0, 2, (n,))
    total, comps = multitask_loss(logits, target, risk, time, event,
                                  alpha=1.0, beta=0.5)
    total.backward()
    assert torch.isfinite(total)
    assert {"ce", "cox", "total"} <= set(comps.keys())


def test_c_index_perfect_ordering():
    risk = np.array([3.0, 2.0, 1.0])
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 1])
    assert concordance_index_torch(risk, time, event) == 1.0
