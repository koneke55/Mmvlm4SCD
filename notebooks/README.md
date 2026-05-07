# Notebooks

Exploratory analyses for each modality + the multimodal fusion. To keep
diffs small in version control, please use `jupytext --sync` or strip
outputs before committing.

**All-in-one:** `00-mmvlm4scd-full-pipeline.ipynb` runs Parts 1–6 in one place (single Colab setup, then synthetic EDA, fusion demo, Nigeria NDHS, Mali DHS). The same material also lives in the smaller notebooks below if you prefer shorter sessions.

**Google Colab:** each notebook includes an **Open in Colab** badge where helpful.
For free GPU tiers and Runtime tips, follow [Unsloth's Google Colab guide](https://docs.unsloth.ai/get-started/install/google-colab) ([unsloth.ai](https://unsloth.ai)). This project uses vanilla PyTorch + `pip install -e .`; it does **not** require the `unsloth` package—only the Colab workflow notes.

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
