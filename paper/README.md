# Paper -- Q1 journal submission package

Author: **Sambou Kone**.

This directory holds two consistent renderings of the same study, plus
the auxiliary files a Q1 medical-AI journal will ask for.

## Files

| File | Purpose |
|---|---|
| `paper.pdf` | Two-column ReportLab PDF (preprint-style; no LaTeX needed). |
| `paper.tex` | Single-column, line-numbered, 1.5x line-spaced **Q1 journal manuscript** with structured abstract, Highlights, Key Points, Statistical Analysis, Discussion split into Principal findings / Comparison / Strengths & limitations / Implications / Future work, full Declarations (Funding, COI, Ethics, Data availability, Code availability, CRediT), and a TRIPOD+AI Table~S1 checklist. |
| `references.bib` | 15 BibTeX entries covering SCD biology, severity scoring, Cox / DeepSurv survival modelling, multimodal medical AI, blood-smear CNNs, Harrell C-index, Graf IPCW Brier, and TRIPOD+AI \[Collins 2024]. |
| `cover_letter.tex` | Journal-style cover letter template ready to compile. |
| `Makefile` | `make` -> `pdflatex` + `bibtex`; `make tectonic` -> single-binary build. |
| `build_paper.py` | Generates `paper.pdf` from JSON artefacts. |
| `build_tex.py` | Generates `paper.tex` + `references.bib` (Q1 layout) from JSON artefacts. |
| `make_graphical_abstract.py` | Generates `figures/graphical_abstract.{png,pdf}`. |
| `figures/graphical_abstract.png` | Single-figure overview used as the journal graphical abstract and as Figure S1. |

## Compile the journal manuscript

LaTeX engine required (`texlive`, `MikTeX`, or `tectonic`):

```bash
cd paper
make             # pdflatex + bibtex + 2 reruns
# or
make tectonic    # single-binary tectonic
# cover letter
pdflatex cover_letter.tex
```

## End-to-end pipeline

```bash
# 1. experiments
python src/scripts/run_full_experiment.py --config configs/default.yaml --seeds 3 --ablate
python src/scripts/run_baselines.py
python src/scripts/run_fusion_comparison.py --seeds 3 --epochs 20
python src/scripts/run_subgroup_analysis.py --epochs 20
python src/scripts/run_survival_horizons.py --epochs 20
python src/scripts/run_clinical_eval.py --config configs/default.yaml

# 2. paper artefacts
python paper/make_graphical_abstract.py
python paper/build_paper.py        # ReportLab PDF
python paper/build_tex.py          # Q1 LaTeX source

# 3. compile the journal version
cd paper && make
```

## Q1 journal compliance checklist

Items below are required by most Q1 medical-AI journals; the
manuscript covers each:

- [x] Structured abstract (Background / Methods / Results / Conclusions)
- [x] Highlights box (4 bullets)
- [x] Key Points box (Question / Findings / Meaning)
- [x] Graphical abstract (`figures/graphical_abstract.png`)
- [x] Line numbers and 1.5x line spacing for review
- [x] Statistical analysis subsection
- [x] Subgroup analysis with figure + table
- [x] Calibration analysis (reliability diagram + ECE/MCE + Brier/IBS)
- [x] **Clinical utility** (decision-curve analysis, sensitivity/specificity at thresholds, PPV/NPV, per-class AUROC)
- [x] **Robustness** to missing modalities (test-time dropout sweep)
- [x] **Fairness gap** (max-min subgroup AUROC and C-index)
- [x] **External-cohort simulation** (out-of-distribution synthetic seeds)
- [x] Limitations explicit (synthetic data, no real external validation, PH assumption untested)
- [x] Funding, COI, Ethics, Data availability, Code availability, CRediT declarations
- [x] **TRIPOD+AI** reporting checklist (Table~S1)
- [x] Acknowledgements
- [x] Numbered references with full BibTeX file
- [x] Cover letter

## LaTeX dependencies

`geometry`, `microtype`, `graphicx`, `booktabs`, `array`, `caption`,
`xcolor`, `hyperref`, `authblk`, `abstract`, `amsmath`, `amssymb`,
`textcomp`, `enumitem`, `cite`, `lineno`, `setspace`, `tcolorbox`,
`longtable`. All ship with a default TeX Live install.
