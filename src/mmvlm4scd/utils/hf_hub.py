"""Package :class:`MultimodalSCDModel` weights + :class:`ModelConfig` for Hugging Face Hub.

Requires optional dependencies::

    pip install -e .[hf]

Layout written per repo (single model):

* ``config.json`` — architecture hyperparameters (custom schema, ``model_type: mmvlm4scd``).
* ``model.safetensors`` — PyTorch state dict (SafeTensors).

See ``src/scripts/push_to_hf_hub.py`` to upload (e.g. to https://huggingface.co/koneke55).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from mmvlm4scd.models import ModelConfig, MultimodalSCDModel

MODEL_TYPE = "mmvlm4scd"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


def _require_hf_deps() -> tuple[Any, Any]:
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ImportError(
            "Hugging Face Hub / SafeTensors helpers require optional deps: "
            "pip install -e '.[hf]'"
        ) from exc
    return hf_hub_download, (load_file, save_file)


def model_config_to_hub_dict(cfg: ModelConfig) -> dict[str, Any]:
    """JSON-serializable dict for ``config.json`` (Hub custom schema)."""
    return {
        "model_type": MODEL_TYPE,
        "clinical_input_dim": cfg.clinical_input_dim,
        "genomic_input_dim": cfg.genomic_input_dim,
        "imaging_input_dim": cfg.imaging_input_dim,
        "temporal_input_dim": cfg.temporal_input_dim,
        "embed_dim": cfg.embed_dim,
        "num_severity_classes": cfg.num_severity_classes,
        "fusion": cfg.fusion,
        "dropout": cfg.dropout,
        "head_hidden_dim": cfg.head_hidden_dim,
        "extra": dict(cfg.extra),
    }


def hub_dict_to_model_config(data: dict[str, Any]) -> ModelConfig:
    if data.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"config.json model_type must be {MODEL_TYPE!r}, got {data.get('model_type')!r}"
        )
    fusion = data["fusion"]
    if fusion not in ("attention", "cross", "late"):
        raise ValueError(f"Unknown fusion: {fusion!r}")
    fusion_t: Literal["attention", "cross", "late"] = fusion  # type: ignore[assignment]
    return ModelConfig(
        clinical_input_dim=int(data["clinical_input_dim"]),
        genomic_input_dim=int(data["genomic_input_dim"]),
        imaging_input_dim=int(data["imaging_input_dim"]),
        temporal_input_dim=int(data["temporal_input_dim"]),
        embed_dim=int(data["embed_dim"]),
        num_severity_classes=int(data["num_severity_classes"]),
        fusion=fusion_t,
        dropout=float(data["dropout"]),
        head_hidden_dim=int(data["head_hidden_dim"]),
        extra=dict(data.get("extra") or {}),
    )


def save_mmvlm_pretrained(
    save_directory: str | Path,
    model: nn.Module,
    config: ModelConfig,
) -> None:
    """Write ``config.json`` + ``model.safetensors`` under ``save_directory``."""
    _, (_, save_file) = _require_hf_deps()
    root = Path(save_directory)
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / CONFIG_NAME
    cfg_path.write_text(
        json.dumps(model_config_to_hub_dict(config), indent=2),
        encoding="utf-8",
    )
    state = model.state_dict()
    save_file(state, str(root / WEIGHTS_NAME))


def load_mmvlm_local(load_directory: str | Path,
                   *,
                   map_location: str | torch.device | None = "cpu",
                   ) -> tuple[MultimodalSCDModel, ModelConfig]:
    """Load model + config from a local directory (Hub-style layout)."""
    _, (load_file, _) = _require_hf_deps()
    root = Path(load_directory)
    cfg = hub_dict_to_model_config(
        json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    )
    model = MultimodalSCDModel(cfg)
    tensors = load_file(str(root / WEIGHTS_NAME), device=str(map_location))
    model.load_state_dict(tensors)
    return model, cfg


def load_mmvlm_from_hub(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    map_location: str | torch.device | None = "cpu",
) -> tuple[MultimodalSCDModel, ModelConfig]:
    """Download ``config.json`` + ``model.safetensors`` from the Hub and build the model."""
    hf_hub_download, (load_file, _) = _require_hf_deps()
    cfg_path = hf_hub_download(
        repo_id, CONFIG_NAME, revision=revision, cache_dir=cache_dir
    )
    weights_path = hf_hub_download(
        repo_id, WEIGHTS_NAME, revision=revision, cache_dir=cache_dir
    )
    cfg = hub_dict_to_model_config(
        json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    )
    model = MultimodalSCDModel(cfg)
    tensors = load_file(weights_path, device=str(map_location))
    model.load_state_dict(tensors)
    return model, cfg
