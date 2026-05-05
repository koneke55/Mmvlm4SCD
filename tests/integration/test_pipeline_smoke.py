"""End-to-end smoke test: run the orchestrator in --smoke mode and check
   that the expected artefacts land on disk with sane contents.

Marked slow because it trains a model. Should still finish in <60 s on CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_full_pipeline_smoke(tmp_path: Path):
    """Run run_full_experiment.py --smoke into a temp output dir and
    assert expected artefacts."""
    cfg_text = (ROOT / "configs" / "default.yaml").read_text()
    out_root = tmp_path / "experiments" / "results"
    out_root.mkdir(parents=True)
    cfg_text = cfg_text.replace("experiments/results", str(out_root))
    cfg_path = tmp_path / "default.yaml"
    cfg_path.write_text(cfg_text)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "scripts" / "run_full_experiment.py"),
         "--config", str(cfg_path), "--smoke"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"smoke run failed: stdout={proc.stdout[-2000:]} stderr={proc.stderr[-2000:]}")

    exp_dir = out_root / "mmvlm4scd_default"
    assert exp_dir.exists(), f"missing experiment dir {exp_dir}"
    for name in ("training_history.json", "test_metrics.json", "summary.json",
                 "seed_runs.json"):
        path = exp_dir / name
        assert path.exists(), f"missing artefact: {path}"
        json.loads(path.read_text())

    fig_dir = exp_dir / "figures"
    for fig in ("training_curves.png", "confusion.png", "km_by_risk.png",
                "calibration.png"):
        f = fig_dir / fig
        assert f.exists() and f.stat().st_size > 200, f"missing/empty figure {f}"

    metrics = json.loads((exp_dir / "test_metrics.json").read_text())
    for k in ("accuracy", "f1_macro", "auroc_ovr", "c_index", "confusion_matrix"):
        assert k in metrics
