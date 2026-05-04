"""Tiny YAML/JSON config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r") as f:
        if p.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, default=_default)
    return p


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
