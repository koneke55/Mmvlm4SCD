# Data Harmonization Schema

Every real-data loader (`src/mmvlm4scd/data/loaders/<source>.py`) must
emit instances of the unified schema declared in
`src/mmvlm4scd/data/harmonization.py`. This document is the
authoritative spec.

## Goals

1. **Single contract.** The model never sees a source-specific format.
2. **Explicit dtypes and units.** No implicit unit conversions; missing
   data is `None`, never `0`.
3. **Modality-aware partial observation.** A patient may have any
   subset of modalities present; the model handles modality masking
   at fusion time.
4. **PHI-safe by construction.** The schema does not have fields for
   names, addresses, or full dates of birth.

## Patient record

Each patient is represented by a `PatientRecord` dataclass with the
following modality blocks. All fields are optional unless marked
`(required)`.

### Identity (required, deidentified)

| Field | Type | Notes |
|---|---|---|
| `patient_id` | str (required) | Site-scoped, opaque, non-reversible. Hash of `(source, original_id, project_salt)`. |
| `source` | str (required) | One of: `mimiciv`, `dbgap_phs001514`, `dbgap_phs001599`, `ukbb`, `cure_scd`, `topmed`, `synthetic`. |
| `cohort_split` | str (required) | One of: `train`, `val`, `test`, `external`. Locked at pre-registration time. |

### Clinical (one row per patient)

Demographics + baseline labs + comorbidities at index encounter.

| Field | Type | Unit | Notes |
|---|---|---|---|
| `age_years` | float | year | At index encounter. Clip ages > 89 to 90. |
| `sex` | enum | -- | `F`, `M`, `unknown`. |
| `ancestry` | enum | -- | Self-reported, harmonised to NIH-OD categories: `african`, `african_american`, `caribbean`, `hispanic_latino`, `middle_eastern`, `south_asian`, `mixed`, `other`, `unknown`. |
| `genotype` | enum | -- | `HbSS`, `HbSC`, `HbSbeta+`, `HbSbeta0`, `HbSother`, `unknown`. |
| `hb_g_dl` | float | g/dL | Baseline haemoglobin. |
| `hbf_pct` | float | % | Baseline foetal Hb. |
| `wbc_k_uL` | float | x10^3/uL | White-cell count. |
| `plt_k_uL` | float | x10^3/uL | Platelet count. |
| `ldh_u_l` | float | U/L | Lactate dehydrogenase. |
| `bili_total_mg_dl` | float | mg/dL | Total bilirubin. |
| `crp_mg_l` | float | mg/L | C-reactive protein. |
| `bmi_kg_m2` | float | kg/m^2 | At index encounter. |
| `acs_history` | bool | -- | Acute chest syndrome ever recorded. |
| `stroke_history` | bool | -- | Any stroke event ever recorded. |
| `hydroxyurea_ever` | bool | -- | Hydroxyurea exposure flag. |
| `chronic_transfusion` | bool | -- | Chronic transfusion programme flag. |
| `voc_per_year` | float | crises/yr | Annual vaso-occlusive crisis rate at baseline. |

### Genomic (one row per patient)

| Field | Type | Notes |
|---|---|---|
| `hbb_variants` | list[str] | ClinVar/HGVS-formatted variant identifiers (e.g. `NM_000518.5:c.20A>T`). |
| `polygenic_score` | float | Standardised polygenic-like score (TOPMed-derived for Tier-C cohorts; gnomAD AF-weighted for open cohorts). |
| `prs_source` | str | Provenance label for `polygenic_score` so paired vs disjoint sources are not silently mixed. |

### Imaging (one row per patient)

| Field | Type | Notes |
|---|---|---|
| `image_embedding` | np.ndarray (D,) float32 | Output of the Phase-1 ResNet18/ConvNeXt-Tiny encoder pretrained on `erythrocytesIDB + Kaggle Sickle RBC`. D = 512. |
| `image_count` | int | Number of slides aggregated. |
| `imaging_source` | str | `pbs_microscopy`, `peripheral_smear_scanned`, etc. |

### Temporal (irregular time series)

Stored as a list of dictionaries because real data are irregular.

| Field | Type | Notes |
|---|---|---|
| `timeline` | list[TemporalPoint] | Sorted by relative day from index encounter. |

Each `TemporalPoint`:

| Field | Type | Unit | Notes |
|---|---|---|---|
| `delta_days` | int | days from index | Negative for pre-index, zero or positive for post. |
| `hr_bpm` | float | bpm | Heart rate. |
| `spo2_pct` | float | % | Pulse oximetry. |
| `sbp_mmHg` | float | mmHg | Systolic blood pressure. |
| `hb_g_dl` | float | g/dL | Same units as clinical block. |
| `wbc_k_uL` | float | x10^3/uL | -- |
| `pain_vas` | float | 0-10 | Patient-reported pain. |
| `observed_mask` | dict[str, bool] | -- | Which fields were observed at this point; unused fields are not imputed silently. |

### Outcomes (one row per patient)

| Field | Type | Unit | Notes |
|---|---|---|---|
| `severity_label` | enum | -- | `mild`, `moderate`, `severe`, `unknown`. Derived per the locked formula in `docs/severity_label_definition.md`. |
| `severity_score` | float | -- | Continuous version used to derive the label; useful for ordinal regression sensitivity analyses. |
| `time_to_event_years` | float | year | Time from index to event or censor. |
| `event_observed` | bool | -- | True for death; False for censoring. |
| `cause_of_death` | str | -- | Free-text or ICD-10; optional. |

## Validation rules

The `PatientRecord.validate()` method enforces:

* `patient_id` is non-empty and not a date-of-birth-ish pattern.
* `cohort_split` is one of the allowed values.
* If `severity_label != 'unknown'` then `severity_score` is finite.
* If `event_observed` is True then `time_to_event_years > 0`.
* If `image_embedding` is set then its length is exactly the expected
  encoder dimension.
* All temporal points have `delta_days` strictly increasing.

## Versioning

The schema version is exposed as `harmonization.SCHEMA_VERSION` (a
`MAJOR.MINOR.PATCH` string). Any patient artefact written to disk
includes this version. Loaders refuse to read records whose `MAJOR`
mismatches the runtime.
