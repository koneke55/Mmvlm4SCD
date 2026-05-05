"""Round-trip checkpoint serialization should preserve outputs."""

import torch

from mmvlm4scd.data import StandardPreprocessor, generate_synthetic_cohort
from mmvlm4scd.data.synthetic import SCDSyntheticConfig
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel


def _build_pair():
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=8, seed=0))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    x = pre.transform(cohort["clinical"])
    cfg = ModelConfig(clinical_input_dim=x.shape[1],
                      genomic_input_dim=cohort["genomic"].shape[1],
                      imaging_input_dim=cohort["imaging"].shape[1],
                      temporal_input_dim=cohort["temporal"].shape[2],
                      embed_dim=32, fusion="attention", dropout=0.0)
    a = MultimodalSCDModel(cfg).eval()
    b = MultimodalSCDModel(cfg).eval()
    batch = {
        "clinical": torch.as_tensor(x, dtype=torch.float32),
        "genomic": torch.as_tensor(cohort["genomic"], dtype=torch.float32),
        "imaging": torch.as_tensor(cohort["imaging"], dtype=torch.float32),
        "temporal": torch.as_tensor(cohort["temporal"], dtype=torch.float32),
        "severity": torch.as_tensor(cohort["severity"], dtype=torch.long),
        "survival_time": torch.as_tensor(cohort["survival_time"], dtype=torch.float32),
        "survival_event": torch.as_tensor(cohort["survival_event"], dtype=torch.long),
    }
    return a, b, batch


def test_state_dict_roundtrip_preserves_outputs(tmp_path):
    a, b, batch = _build_pair()
    ckpt = tmp_path / "ckpt.pt"
    torch.save(a.state_dict(), ckpt)
    b.load_state_dict(torch.load(ckpt, map_location="cpu"))
    with torch.no_grad():
        out_a = a(batch); out_b = b(batch)
    assert torch.allclose(out_a["severity_logits"], out_b["severity_logits"], atol=1e-6)
    assert torch.allclose(out_a["risk_score"], out_b["risk_score"], atol=1e-6)


def test_model_eval_mode_is_deterministic():
    a, _, batch = _build_pair()
    a.eval()
    with torch.no_grad():
        o1 = a(batch); o2 = a(batch)
    assert torch.allclose(o1["severity_logits"], o2["severity_logits"], atol=1e-6)
    assert torch.allclose(o1["risk_score"], o2["risk_score"], atol=1e-6)
