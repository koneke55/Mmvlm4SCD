# [ Mmvlm4SCD ] MMultimodal AI Model for Sickle Cell Disease severity and survival prediction          
                                  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)    
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)     
        
A multimodal deep learning framework for comprehensive analysis of Sickle Cell Disease using clinical, genomic, imaging, and temporal data. **Release v0.1.1** — see [`CHANGELOG.md`](CHANGELOG.md).

## Features
- **Multimodal Fusion**: Advanced fusion techniques for heterogeneous data types
- **Interpretability**: Built-in model interpretation and visualization tools
- **Scalable**: Support for distributed training and inference
- **Clinical Focus**: Domain-specific preprocessing and evaluation metrics

## Tests and evaluation stack

Run the full suite (200 tests: 199 unit + 1 `@pytest.mark.slow` integration):

```bash
PYTHONPATH=src pytest -q                       # full suite (~30s on CPU)
PYTHONPATH=src pytest -m "not slow" -q          # skip integration smoke
```

### Lint and pre-commit

The project uses [ruff](https://docs.astral.sh/ruff/) for linting; configuration
lives in `pyproject.toml` under `[tool.ruff]`. CI runs `ruff check src tests
scripts` on every push and pull request.

To get the same diagnostics locally before committing, install
[pre-commit](https://pre-commit.com/) once:

```bash
pip install -e .[dev]
pre-commit install
```

After that, every `git commit` runs ruff (with `--fix`), trailing-whitespace
trimming, end-of-file fixing, and the standard YAML/TOML/merge-conflict checks.
Run all hooks against the whole tree at any time with:

```bash
pre-commit run --all-files
```

Coverage spans:

- **Data** (`tests/unit/test_data.py`, `test_dataloaders.py`): registry, synthetic cohort shapes, preprocessor round-trip, train/val/test split partitioning, DataLoader keys/dtypes/drop_last semantics.
- **Models** (`test_encoders.py`, `test_models.py`, `test_serialization.py`): per-encoder shapes & gradients, all three fusion strategies, state-dict round-trip preserves outputs.
- **Training** (`test_trainer.py`, `test_training.py`, `test_losses_edge.py`): trainer reduces loss, deterministic for fixed seed, Cox loss handles tied times / single-event / no-event, Cox is invariant to constant risk shift.
- **Evaluation** (`test_evaluation.py`, `test_clinical_eval.py`, `test_extras.py`, `test_visualization.py`): bootstrap CIs, Brier/IBS, decision-curve analysis, ECE/MCE, per-class AUROC, sensitivity/specificity, modality-dropout sweep, fairness gap, all visualisation entry-points.
- **Integration** (`tests/integration/test_pipeline_smoke.py`): end-to-end run of `run_full_experiment.py --smoke` writes the expected artefacts and figures to disk.

- **Hugging Face Hub** (`test_hf_hub.py`): Hub JSON round-trip, checkpoint→config inference, SafeTensors save/load vs forward pass.

Beyond rank metrics, evaluation now includes:

- **Decision-curve analysis** (`evaluation.clinical.decision_curve`)
- **Calibration error** ECE/MCE (`evaluation.clinical.expected_calibration_error`)
- **Per-class one-vs-rest AUROC** (`evaluation.clinical.per_class_auroc`)
- **Sensitivity / specificity at thresholds** (`evaluation.clinical.sens_spec_at_thresholds`)
- **Test-time modality-dropout robustness sweep** (`evaluation.extras.modality_dropout_sweep`)
- **Subgroup fairness gap** (`evaluation.extras.fairness_gap`)
- **External-cohort distribution-shift simulation** (`evaluation.extras.external_cohort_simulation`)
   
## Quick Start   
```bash   
# Clone repository
git clone https://github.com/koneke55/Mmvlm4SCD.git
cd Mmvlm4SCD

# Install dependencies
pip install -e .

# Run training
python src/scripts/train.py --config configs/default.yaml
```

### Hugging Face Hub ([profile](https://huggingface.co/koneke55))

Mmvlm4SCD is a **custom PyTorch** graph (not `transformers.AutoModel`). To publish checkpoints under your Hub namespace, bundle **`config.json`** + **`model.safetensors`**:

Infer Hub JSON from an existing ``.pt`` ``state_dict`` (dimensions + fusion kind):

```bash
mmvlm4scd-infer-hf-config \
  --checkpoint experiments/checkpoints/best_model.pt \
  --out configs/hf_hub_best_model.pt.json
```

Then push as above with ``--model-config configs/hf_hub_best_model.pt.json``. A checked-in example for the default synthetic checkpoint is ``configs/hf_hub_best_model.pt.json``.

```bash
pip install -e ".[hf]"
huggingface-cli login   # or set HF_TOKEN

# Copy and edit dims to match your checkpoint / preprocessor
cp configs/hf_hub_model_config.example.json my_hub_config.json

mmvlm4scd-push-hf \
  --checkpoint experiments/checkpoints/best_model.pt \
  --model-config my_hub_config.json \
  --repo-id koneke55/mmvlm4scd-<run-name>
```

Load after install:

```python
from mmvlm4scd.utils.hf_hub import load_mmvlm_from_hub

model, cfg = load_mmvlm_from_hub("koneke55/mmvlm4scd-<run-name>")
```

Use **one Hub model repo per checkpoint** (or branches/revisions). Helpers live in `mmvlm4scd.utils.hf_hub`; optional `[hf]` installs `huggingface_hub` and `safetensors`.

```
# Structure  
Mmvlm4SCD/
│
├── .github/
│   ├── workflows/
│   │   ├── ci-cd.yml
│   │   └── model-testing.yml
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   ├── genomic/
│   │   ├── clinical/
│   │   └── temporal/
│   ├── processed/
│   ├── external/
│   └── README.md
│
├── docs/
│   ├── api/
│   ├── tutorials/
│   ├── architecture.md
│   ├── dataset.md
│   └── CONTRIBUTING.md
│
├── notebooks/
│   ├── 00-mmvlm4scd-full-pipeline.ipynb
│   ├── 01-eda-clinical.ipynb
│   ├── 02-eda-genomic.ipynb
│   ├── 03-eda-imaging.ipynb
│   ├── 04-multimodal-fusion.ipynb
│   ├── 05_colab_west_africa_experiment.ipynb
│   ├── 05_colab_nigeria_ndhs2018.ipynb
│   ├── 06_colab_mali_dhs2018.ipynb
│   └── README.md
│
├── src/
│   ├── mmvlm4scd/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── dataloaders.py
│   │   │   ├── preprocessing.py
│   │   │   ├── multimodal_dataset.py
│   │   │   └── augmentations.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── encoders/
│   │   │   │   ├── clinical_encoder.py
│   │   │   │   ├── genomic_encoder.py
│   │   │   │   ├── imaging_encoder.py
│   │   │   │   └── temporal_encoder.py
│   │   │   │
│   │   │   ├── fusion/
│   │   │   │   ├── attention_fusion.py
│   │   │   │   ├── cross_attention.py
│   │   │   │   └── late_fusion.py
│   │   │   │
│   │   │   └── multimodal_model.py
│   │   │
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainers.py
│   │   │   ├── losses.py
│   │   │   └── metrics.py
│   │   │
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── evaluators.py
│   │   │   ├── interpretability.py
│   │   │   └── visualization.py
│   │   │
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── config_utils.py
│   │   │   ├── logging_utils.py
│   │   │   └── device_utils.py
│   │   │
│   │   └── configs/
│   │       ├── default.yaml
│   │       ├── clinical_config.yaml
│   │       ├── genomic_config.yaml
│   │       └── multimodal_config.yaml
│   │
│   └── scripts/
│       ├── train.py
│       ├── evaluate.py
│       ├── preprocess_data.py
│       ├── export_model.py
│       └── inference.py
│
├── tests/
│   ├── unit/
│   │   ├── test_data.py
│   │   ├── test_models.py
│   │   └── test_training.py
│   ├── integration/
│   └── conftest.py
│
├── experiments/
│   ├── logs/
│   ├── checkpoints/
│   └── results/
│
├── environment/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── environment.yml
│   └── setup.py
│
├── .gitignore
├── LICENSE
├── README.md
├── CODE_OF_CONDUCT.md
├── CITATION.cff
├── pyproject.toml
└── mkdocs.yml
```
