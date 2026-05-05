# Mmvlm4SCD   
                        
# MMultimodal AI Model for Sickle Cell Disease             
                                  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)    
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)     
        
A multimodal deep learning framework for comprehensive analysis of Sickle Cell Disease using clinical, genomic, imaging, and temporal data.  
                                    
## Features
- **Multimodal Fusion**: Advanced fusion techniques for heterogeneous data types
- **Interpretability**: Built-in model interpretation and visualization tools
- **Scalable**: Support for distributed training and inference
- **Clinical Focus**: Domain-specific preprocessing and evaluation metrics

## Tests and evaluation stack

Run all tests (52 unit + 1 end-to-end integration smoke):

```bash
PYTHONPATH=src pytest -q                    # full suite (~12s)
PYTHONPATH=src pytest -m "not slow" -q      # skip the integration smoke
```

Coverage spans:

- **Data** (`tests/unit/test_data.py`, `test_dataloaders.py`): registry, synthetic cohort shapes, preprocessor round-trip, train/val/test split partitioning, DataLoader keys/dtypes/drop_last semantics.
- **Models** (`test_encoders.py`, `test_models.py`, `test_serialization.py`): per-encoder shapes & gradients, all three fusion strategies, state-dict round-trip preserves outputs.
- **Training** (`test_trainer.py`, `test_training.py`, `test_losses_edge.py`): trainer reduces loss, deterministic for fixed seed, Cox loss handles tied times / single-event / no-event, Cox is invariant to constant risk shift.
- **Evaluation** (`test_evaluation.py`, `test_clinical_eval.py`, `test_extras.py`, `test_visualization.py`): bootstrap CIs, Brier/IBS, decision-curve analysis, ECE/MCE, per-class AUROC, sensitivity/specificity, modality-dropout sweep, fairness gap, all visualisation entry-points.
- **Integration** (`tests/integration/test_pipeline_smoke.py`): end-to-end run of `run_full_experiment.py --smoke` writes the expected artefacts and figures to disk.

Beyond rank metrics, evaluation now includes:

- **Decision-curve analysis** (`evaluation.clinical.decision_curve`)
- **Calibration error** ECE/MCE (`evaluation.clinical.expected_calibration_error`)
- **Per-class one-vs-rest AUROC** (`evaluation.clinical.per_class_auroc`)
- **Sensitivity / specificity at thresholds** (`evaluation.clinical.sens_spec_at_thresholds`)
- **Test-time modality-dropout robustness sweep** (`evaluation.extras.modality_dropout_sweep`)
- **Subgroup fairness gap** (`evaluation.extras.fairness_gap`)
- **External-cohort distribution-shift simulation** (`evaluation.extras.external_cohort_simulation`)

All seven are wired into `src/scripts/run_clinical_eval.py`, which writes JSON artefacts under `experiments/results/<exp>/clinical/` and is read by both the LaTeX (`paper/build_tex.py`) and ReportLab (`paper/build_paper.py`) paper builders.
   
## Quick Start   
```bash   
# Clone repository
git clone https://github.com/yourusername/Mmvlm4SCD.git
cd Mmvlm4SCD

# Install dependencies
pip install -e .

# Run training
python src/scripts/train.py --config configs/default.yaml

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
│   ├── 01-eda-clinical.ipynb
│   ├── 02-eda-genomic.ipynb
│   ├── 03-eda-imaging.ipynb
│   ├── 04-multimodal-fusion.ipynb
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
