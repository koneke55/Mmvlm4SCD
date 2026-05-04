"""Loss functions for the dual-head multimodal SCD model.

* ``cox_partial_likelihood_loss``: negative log Breslow partial likelihood
  for survival, taking a per-sample risk score and (time, event).
* ``multitask_loss``: weighted CE + Cox.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cox_partial_likelihood_loss(risk: torch.Tensor,
                                time: torch.Tensor,
                                event: torch.Tensor,
                                eps: float = 1e-8) -> torch.Tensor:
    """Negative Breslow partial log-likelihood.

    For each uncensored subject i, the contribution is
        risk_i - log( sum_{j: t_j >= t_i} exp(risk_j) ).
    We sort by descending time so that the cumulative sum gives the
    risk-set normaliser efficiently.
    """
    if event.dtype != torch.bool:
        event = event.bool()
    if event.sum() == 0:
        return risk.new_tensor(0.0)
    order = torch.argsort(time, descending=True)
    risk_s = risk[order]
    event_s = event[order]
    log_cum = torch.logcumsumexp(risk_s, dim=0)
    log_lik = (risk_s - log_cum)[event_s]
    return -log_lik.mean() if log_lik.numel() else risk.new_tensor(0.0)


def multitask_loss(severity_logits: torch.Tensor,
                   severity_target: torch.Tensor,
                   risk: torch.Tensor,
                   time: torch.Tensor,
                   event: torch.Tensor,
                   alpha: float = 1.0,
                   beta: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Combined CE + Cox loss; returns (total_loss, components_dict)."""
    ce = F.cross_entropy(severity_logits, severity_target)
    cox = cox_partial_likelihood_loss(risk, time, event)
    total = alpha * ce + beta * cox
    return total, {"ce": ce.detach(), "cox": cox.detach(), "total": total.detach()}
