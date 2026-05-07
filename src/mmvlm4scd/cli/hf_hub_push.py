"""CLI entry for ``mmvlm4scd-push-hf`` (``pyproject.toml`` console script)."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from mmvlm4scd.models import MultimodalSCDModel
from mmvlm4scd.utils.hf_hub import hub_dict_to_model_config, save_mmvlm_pretrained
from mmvlm4scd.utils.logging_utils import get_logger

DEFAULT_CARD = """---
license: mit
library_name: pytorch
tags:
  - multimodal
  - sickle-cell-disease
  - survival-analysis
  - pytorch
---

# Mmvlm4SCD checkpoint

Multimodal Sickle Cell Disease **severity** (3-class) + **survival risk** PyTorch weights published from [Mmvlm4SCD](https://github.com/koneke55/Mmvlm4SCD).

## Usage

Install this repo (models are **not** ``transformers.AutoModel``; architecture lives in ``mmvlm4scd``)::

    pip install git+https://github.com/koneke55/Mmvlm4SCD.git
    pip install huggingface_hub safetensors

Then::

    from mmvlm4scd.utils.hf_hub import load_mmvlm_from_hub
    model, cfg = load_mmvlm_from_hub("{repo_id}")

Forward pass expects a batch dict with keys ``clinical``, ``genomic``, ``imaging``, ``temporal`` (plus labels for training).

## Limitations

Training data and preprocessor statistics are **not** bundled unless you add separate artefacts—dimension fields in ``config.json`` must match your preprocessing pipeline.
"""


def main(argv: list[str] | None = None) -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="Publish Mmvlm4SCD weights to Hugging Face Hub.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt file containing a ``state_dict``")
    p.add_argument("--model-config", required=True,
                   help="JSON file with mmvlm4scd Hub config (model_type: mmvlm4scd)")
    p.add_argument("--repo-id", required=True,
                   help="Hub repo id, e.g. koneke55/mmvlm4scd-demo")
    p.add_argument("--readme", default=None,
                   help="Optional README.md for model card (default: generated stub)")
    p.add_argument("--private", action="store_true",
                   help="Create/use a private Hub repo")
    p.add_argument("--dry-run", action="store_true",
                   help="Only build the bundle locally; print path and skip upload")
    args = p.parse_args(argv)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install Hugging Face tooling: pip install -e '.[hf]'") from exc

    raw_cfg = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    cfg = hub_dict_to_model_config(raw_cfg)
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
    if not isinstance(ckpt, dict) or not ckpt:
        raise ValueError("--checkpoint must contain a flat state_dict mapping")
    if not all(isinstance(k, str) for k in ckpt.keys()):
        raise ValueError("--checkpoint must contain string keys (state_dict)")

    model = MultimodalSCDModel(cfg)
    model.load_state_dict(ckpt)

    tmp = Path(tempfile.mkdtemp(prefix="mmvlm_hf_"))
    try:
        save_mmvlm_pretrained(tmp, model, cfg)
        readme = DEFAULT_CARD.replace("{repo_id}", args.repo_id)
        if args.readme:
            readme = Path(args.readme).read_text(encoding="utf-8")
        (tmp / "README.md").write_text(readme, encoding="utf-8")

        if args.dry_run:
            log.info("Dry-run bundle at %s", tmp)
            print(tmp)
            return

        api = HfApi()
        api.create_repo(args.repo_id, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(tmp), repo_id=args.repo_id)
        log.info("Uploaded to https://huggingface.co/%s", args.repo_id)
    finally:
        if not args.dry_run:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
