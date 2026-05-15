"""Thin wrapper for ``python src/scripts/infer_hf_hub_config.py``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.cli.infer_hf_config import main  # noqa: E402

if __name__ == "__main__":
    main()
