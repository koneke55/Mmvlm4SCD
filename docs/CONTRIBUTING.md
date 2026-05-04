# Contributing

Thanks for your interest in Mmvlm4SCD.

## Getting set up

```bash
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e .[dev]
pytest -q
```

## Pull-request checklist

- Unit tests added/updated and passing locally
- New modules typed and documented with module-level docstrings
- No commits of patient-level data, credentials or large binary blobs
- Paper artefacts regenerated via `python paper/build_paper.py` if the
  experimental pipeline changed

## Reporting vulnerabilities

Please open a private email to the repository owner rather than a public
issue.
