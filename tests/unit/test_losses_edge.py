"""Edge-case tests for Cox partial-likelihood loss + multitask loss."""

import torch

from mmvlm4scd.training.losses import (cox_partial_likelihood_loss,
                                       multitask_loss)


def test_cox_loss_zero_when_no_events():
    risk = torch.randn(8, requires_grad=True)
    time = torch.linspace(1, 8, 8)
    event = torch.zeros(8, dtype=torch.long)
    loss = cox_partial_likelihood_loss(risk, time, event)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_cox_loss_handles_single_event():
    risk = torch.tensor([1.0, 2.0, 0.5, -1.0], requires_grad=True)
    time = torch.tensor([3.0, 1.0, 4.0, 2.0])
    event = torch.tensor([0, 1, 0, 0])
    loss = cox_partial_likelihood_loss(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()
    assert risk.grad is not None and torch.isfinite(risk.grad).all()


def test_cox_loss_handles_tied_times():
    """Tied event times must not crash and must give a finite gradient."""
    risk = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    time = torch.tensor([1.0, 1.0, 2.0, 2.0])
    event = torch.tensor([1, 1, 1, 0])
    loss = cox_partial_likelihood_loss(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(risk.grad).all()


def test_cox_loss_invariant_to_risk_shift():
    """Cox is invariant to a constant shift in risk."""
    n = 32
    torch.manual_seed(0)
    risk = torch.randn(n)
    time = torch.rand(n) * 10 + 0.1
    event = torch.randint(0, 2, (n,))
    a = cox_partial_likelihood_loss(risk, time, event)
    b = cox_partial_likelihood_loss(risk + 5.0, time, event)
    assert torch.allclose(a, b, atol=1e-5)


def test_multitask_loss_alpha_beta_weights_apply():
    n = 16
    torch.manual_seed(0)
    logits = torch.randn(n, 3, requires_grad=True)
    target = torch.randint(0, 3, (n,))
    risk = torch.randn(n, requires_grad=True)
    time = torch.rand(n) * 10 + 1
    event = torch.randint(0, 2, (n,))
    total_a, comps_a = multitask_loss(logits, target, risk, time, event,
                                      alpha=1.0, beta=0.5)
    total_b, comps_b = multitask_loss(logits, target, risk, time, event,
                                      alpha=0.0, beta=0.0)
    assert torch.isfinite(total_a) and torch.isfinite(total_b)
    assert float(total_b) == 0.0
