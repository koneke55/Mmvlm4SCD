"""Write ``configs/hf_hub_*.json`` from a checkpoint ``state_dict`` (no Hub deps)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mmvlm4scd.utils.hf_hub import infer_hub_dict_from_state_dict


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Infer Hugging Face Hub config.json fields from a .pt state_dict.",
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSON path (e.g. configs/hf_hub_my_run.json)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Training dropout (not in weights); default 0.1")
    args = p.parse_args(argv)

    sd = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(sd, dict) or not sd:
        raise SystemExit("Checkpoint must be a flat state_dict mapping.")
    if not all(isinstance(k, str) for k in sd.keys()):
        raise SystemExit("state_dict keys must be strings.")

    rel = str(args.checkpoint.as_posix())
    blob = infer_hub_dict_from_state_dict(
        sd,
        checkpoint_path=rel,
        dropout=args.dropout,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
