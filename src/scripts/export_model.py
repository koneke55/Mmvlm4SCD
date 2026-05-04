"""Export a trained model state dict as a ``.pt`` checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.utils import get_logger


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True, help="Path to .pt with state_dict")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    log = get_logger()
    state = torch.load(args.state, map_location="cpu")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    log.info("Saved checkpoint to %s", out)


if __name__ == "__main__":
    main()
