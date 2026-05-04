# Notebooks

Exploratory analyses for each modality + the multimodal fusion. To keep
diffs small in version control, please use `jupytext --sync` or strip
outputs before committing.

Suggested order:

1. `01-eda-clinical.ipynb` - distributional checks on synthetic clinical
   tables.
2. `02-eda-genomic.ipynb`  - HBB variant frequencies vs published values.
3. `03-eda-imaging.ipynb`  - smoke-test the imaging-embedding signal.
4. `04-multimodal-fusion.ipynb` - end-to-end fusion sanity check.

Run them after `pip install -e .` so the `mmvlm4scd` package is on the
import path.
