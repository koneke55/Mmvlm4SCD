import torch

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort)
from mmvlm4scd.data.synthetic import SCDSyntheticConfig
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel


def _mini_batch(fusion: str = "attention"):
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=16, seed=0))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    x = pre.transform(cohort["clinical"])
    cfg = ModelConfig(
        clinical_input_dim=x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=32,
        fusion=fusion,
    )
    model = MultimodalSCDModel(cfg)
    batch = {
        "clinical": torch.as_tensor(x, dtype=torch.float32),
        "genomic": torch.as_tensor(cohort["genomic"], dtype=torch.float32),
        "imaging": torch.as_tensor(cohort["imaging"], dtype=torch.float32),
        "temporal": torch.as_tensor(cohort["temporal"], dtype=torch.float32),
        "severity": torch.as_tensor(cohort["severity"], dtype=torch.long),
        "survival_time": torch.as_tensor(cohort["survival_time"], dtype=torch.float32),
        "survival_event": torch.as_tensor(cohort["survival_event"], dtype=torch.long),
    }
    return model, batch


def test_attention_fusion_forward_shapes():
    model, batch = _mini_batch("attention")
    out = model(batch)
    assert out["severity_logits"].shape == (16, 3)
    assert out["risk_score"].shape == (16,)


def test_cross_attention_fusion_forward_shapes():
    model, batch = _mini_batch("cross")
    out = model(batch)
    assert out["severity_logits"].shape == (16, 3)
    assert out["risk_score"].shape == (16,)


def test_late_fusion_forward_shapes():
    model, batch = _mini_batch("late")
    out = model(batch)
    assert out["severity_logits"].shape == (16, 3)
    assert out["risk_score"].shape == (16,)
