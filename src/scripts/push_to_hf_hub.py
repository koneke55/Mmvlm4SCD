"""Thin wrapper: run ``python src/scripts/push_to_hf_hub.py`` without editable install."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.cli.hf_hub_push import main  # noqa: E402

if __name__ == "__main__":
    main()
