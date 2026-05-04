# Paper

Author: **Sambou Kone**.

The same quantitative study is rendered two ways. Both pull live numbers
from the JSON artefacts in `experiments/results/`, so they stay in sync
with whatever experiment was last run.

## 1. ReportLab PDF (no TeX required)

`paper/paper.pdf` is built by

```bash
python paper/build_paper.py --out paper/paper.pdf
```

This is the path used in the repository's CI: it works on any machine
with the project's Python dependencies installed and produces a
self-contained two-column PDF.

## 2. LaTeX source

`paper/paper.tex` + `paper/references.bib` are generated from the same
JSON artefacts by

```bash
python paper/build_tex.py
```

To compile the PDF you need a LaTeX distribution (`texlive`, `MikTeX`,
or `tectonic`):

```bash
# Option A: pdflatex + bibtex (TeX Live / MiKTeX)
cd paper
make

# Option B: tectonic (single-binary, fetches packages on demand)
cd paper
make tectonic
```

The Makefile runs `pdflatex` -> `bibtex` -> `pdflatex` -> `pdflatex` to
resolve cross-references and the BibTeX bibliography. Output:
`paper/paper.pdf` (overwrites the ReportLab PDF if you want a single
deliverable).

## End-to-end pipeline

To rebuild every number in the paper from scratch:

```bash
python src/scripts/run_full_experiment.py --config configs/default.yaml --seeds 3 --ablate
python src/scripts/run_baselines.py
python src/scripts/run_fusion_comparison.py --seeds 3 --epochs 20
python src/scripts/run_subgroup_analysis.py --epochs 20
python src/scripts/run_survival_horizons.py --epochs 20

python paper/build_paper.py
python paper/build_tex.py
```

LaTeX dependencies used by `paper.tex`: `geometry`, `microtype`,
`graphicx`, `booktabs`, `array`, `caption`, `xcolor`, `hyperref`,
`authblk`, `abstract`, `amsmath`, `amssymb`, `textcomp`, `enumitem`,
`cite`. All ship with a default TeX Live install.
