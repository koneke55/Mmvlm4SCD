# Notebooks

Exploratory analyses for each modality + the multimodal fusion. To keep
diffs small in version control, please use `jupytext --sync` or strip
outputs before committing.

Suggested order:

1. `01-eda-clinical.ipynb` — distributional checks on synthetic clinical
   tables and `StandardPreprocessor` output shape (feeds `clinical_input_dim`).
2. `02-eda-genomic.ipynb` — synthetic genotype mix; optional West Africa
   rs334 MAF tilt vs baseline.
3. `03-eda-imaging.ipynb` — imaging embedding dimensions vs severity / labs.
4. `04-multimodal-fusion.ipynb` — `MultimodalSCDDataset`, `DataLoader`, and
   `MultimodalSCDModel` for all three fusion modes plus a one-step CE+Cox
   backward pass (same loss stack as training).

**Optional (Google Colab + real DHS microdata):** after the local stack above,
see `05_colab_west_africa_experiment.ipynb`, `05_colab_nigeria_ndhs2018.ipynb`,
and `06_colab_mali_dhs2018.ipynb` for cohort builders that align survey fields
with the package encoders (some modalities may be zero-padded where not
collected).

Run notebooks (1)–(4) after `pip install -e .` so the `mmvlm4scd` package is on the
import path.
