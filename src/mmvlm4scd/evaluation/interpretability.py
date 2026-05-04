"""Per-modality importance via gradient * input on the fused embedding.

For each modality, compute the L2 norm of d(severity_logit) / d(modality_input)
over the test set. Larger means the model is more sensitive to that
modality's signal at decision time.
"""

from __future__ import annotations

from typing import Dict

import torch


def gradient_modality_importance(model, loader, device: str = "cpu") -> Dict[str, float]:
    model.eval().to(device)
    sums = {"clinical": 0.0, "genomic": 0.0, "imaging": 0.0, "temporal": 0.0}
    counts = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        for k in sums:
            batch[k] = batch[k].clone().detach().requires_grad_(True)
        out = model(batch)
        # Use predicted-class logit as scalar.
        cls = out["severity_logits"].argmax(dim=1)
        sel = out["severity_logits"].gather(1, cls.unsqueeze(1)).sum()
        sel.backward()
        for k in sums:
            g = batch[k].grad
            if g is not None:
                sums[k] += float(g.detach().pow(2).sum().sqrt())
        counts += 1
        model.zero_grad(set_to_none=True)
    if counts == 0:
        return {k: 0.0 for k in sums}
    return {k: v / counts for k, v in sums.items()}
