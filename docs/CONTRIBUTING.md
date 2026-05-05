# Contributing

Thanks for your interest in Mmvlm4SCD.

## Getting set up

```bash
python -m venv .venv
source .venv/bin/activate         # POSIX; Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e .[dev]
PYTHONPATH=src pytest -q
```

Packages live under `src/`; **`PYTHONPATH=src`** (or `pip install -e .`) must be set or `pytest`
will raise `ModuleNotFoundError` for `mmvlm4scd`.

## Lint hooks (optional)

CI runs `ruff check src tests scripts` (configuration in `[tool.ruff]` in `pyproject.toml`).
After installing the dev extra you can mirror that locally via [pre-commit](https://pre-commit.com/):

```bash
pre-commit install
```

## Pull-request checklist

- Unit tests added/updated and passing locally
- New modules typed and documented with module-level docstrings
- No commits of patient-level data, credentials or large binary blobs
- Paper artefacts regenerated via `python paper/build_paper.py` or
  `python paper/build_tex.py` plus `pdflatex` if the experimental
  pipeline or LaTeX source changed materially

## Reporting vulnerabilities

Please open a private email to the repository owner rather than a public
issue.
