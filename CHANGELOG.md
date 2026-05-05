# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-05

### Fixed

- **Bootstrap AUROC** — `evaluation.bootstrap._auroc_ovr` could return
  `1 - AUROC` because scores were ranked descending inside a Mann–Whitney U
  step. Bootstrap summaries now use `sklearn.metrics.roc_auc_score`, matching
  `training.metrics.severity_metrics`, with a regression test.
- **Archived results** — Corrected stored bootstrap `auroc_ovr` blocks in
  `experiments/results/subgroups/subgroups.json` and
  `experiments/results/fusion_comparison/per_fusion.json` via the affine
  inverse consistent with per-resample inversion; **Table 2 (bootstrap)**
  AUROC row in `paper/paper.tex` updated accordingly.
- **Paper** — `Table~\ref{tab:sg}` cross-reference in the Fairness gap
  subsection: `\\ref` → `\ref`.

### Changed

- **CI** — `ruff check src tests scripts` runs on every PR (gated correctness
  rules only; see `[tool.ruff]` in `pyproject.toml`).
- **Contributor workflow** — Pre-commit hooks and README / CONTRIBUTING
  updates aligned with pytest layout (`PYTHONPATH=src`) and the current test
  count.

## [0.1.0] - 2026-05-04

Initial public release (`mmvlm4scd` on PyPI-compatible metadata): multimodal
encoders, fusion strategies, multitask trainer, synthetic cohort, evaluation
stack, LaTeX/ReportLab paper builders, and the public data registry scaffold.
