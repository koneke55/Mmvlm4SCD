# `data/`

This directory is intentionally **empty** in version control. Real data
must never be committed: dbGaP, MIMIC-IV, UK Biobank and CuRe-SCD all
require Data Use Agreements that forbid republication.

Layout convention:

```
data/
  raw/         <- as-downloaded archives, immutable
    images/
    genomic/
    clinical/
    temporal/
  processed/   <- normalised, ready-for-model parquet/npz
  external/    <- third-party reference panels (e.g. gnomAD)
```

The synthetic cohort is materialised under `processed/` by

```bash
python src/scripts/preprocess_data.py --n 2000 --seed 7
```

and the catalogue of public sources is queried via
`mmvlm4scd.data.registry`.
