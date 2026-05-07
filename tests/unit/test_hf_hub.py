"""Hugging Face Hub bundle round-trip (optional ``[hf]`` deps)."""

import json
from pathlib import Path

import pytest
import torch

from mmvlm4scd.data import StandardPreprocessor, generate_synthetic_cohort
from mmvlm4scd.data.synthetic import SCDSyntheticConfig
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel

pytest.importorskip("safetensors")
pytest.importorskip("huggingface_hub")

from mmvlm4scd.utils.hf_hub import (  # noqa: E402
    hub_dict_to_model_config,
    load_mmvlm_local,
    model_config_to_hub_dict,
    save_mmvlm_pretrained,
)


def test_hub_config_json_roundtrip():
    cfg = ModelConfig(
        clinical_input_dim=19,
        genomic_input_dim=32,
        imaging_input_dim=64,
        temporal_input_dim=6,
        embed_dim=32,
        fusion="cross",
        dropout=0.0,
        head_hidden_dim=48,
        extra={"note": "test"},
    )
    blob = json.dumps(model_config_to_hub_dict(cfg))
    back = hub_dict_to_model_config(json.loads(blob))
    assert back == cfg


def test_save_pretrained_load_local_roundtrip(tmp_path):
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=8, seed=0))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    x = pre.transform(cohort["clinical"])
    cfg = ModelConfig(
        clinical_input_dim=x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=32,
        fusion="attention",
        dropout=0.0,
    )
    m = MultimodalSCDModel(cfg).eval()
    batch = {
        "clinical": torch.as_tensor(x, dtype=torch.float32),
        "genomic": torch.as_tensor(cohort["genomic"], dtype=torch.float32),
        "imaging": torch.as_tensor(cohort["imaging"], dtype=torch.float32),
        "temporal": torch.as_tensor(cohort["temporal"], dtype=torch.float32),
        "severity": torch.as_tensor(cohort["severity"], dtype=torch.long),
        "survival_time": torch.as_tensor(cohort["survival_time"], dtype=torch.float32),
        "survival_event": torch.as_tensor(cohort["survival_event"], dtype=torch.long),
    }
    with torch.no_grad():
        ref = m(batch)

    out_dir = tmp_path / "hub_bundle"
    save_mmvlm_pretrained(out_dir, m, cfg)
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "model.safetensors").is_file()

    m2, cfg2 = load_mmvlm_local(out_dir)
    assert cfg2 == cfg
    m2.eval()
    with torch.no_grad():
        got = m2(batch)
    assert torch.allclose(ref["severity_logits"], got["severity_logits"], atol=1e-5)
    assert torch.allclose(ref["risk_score"], got["risk_score"], atol=1e-5)
