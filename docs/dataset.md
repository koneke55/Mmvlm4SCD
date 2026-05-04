# Datasets

The canonical, machine-readable list of public SCD data sources lives in
`src/mmvlm4scd/data/registry.py` and is reproduced below.

Run

```bash
python -m mmvlm4scd.data.registry
```

to print the table. Programmatic filtering:

```python
from mmvlm4scd.data import list_sources

open_sources       = list_sources(access="open")
imaging_sources    = list_sources(modality="imaging")
clinical_dua_only  = list_sources(modality="clinical", access="dua-required")
```

| Access tier      | Meaning                                                 |
|------------------|---------------------------------------------------------|
| `open`           | Direct download or open API.                            |
| `registered`     | Free account / portal registration required.            |
| `dua-required`   | Requires Data Use Agreement (e.g. dbGaP, UK Biobank).   |

The synthetic cohort shipped with the repository is **not** real patient
data. See the synthetic-data disclaimer in `paper/paper.pdf`.
