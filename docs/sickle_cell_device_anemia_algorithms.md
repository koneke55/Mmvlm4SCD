# Structured Algorithms for Sickle Cell Device Anemia

This document provides a structured, implementation-ready algorithm set for a sickle cell **device anemia** workflow (device-assisted anemia monitoring and decision support for SCD patients), plus a tagging system for recent West Africa datasets. The dataset examples are intentionally marked as **EXAMPLE** placeholders so they can be replaced with verified sources.

## 1) End-to-End Algorithm Map

**Goal:** fuse clinical, lab, imaging, and device telemetry signals to detect anemia episodes and support triage.

### 1.1 Pipeline Overview
1. **Data Intake**
   - Clinical: demographics, vitals, labs (Hb, reticulocytes, bilirubin), medications.
   - Imaging: smear microscopy, retinal images, ultrasound (if available).
   - Genomics: hemoglobin variants (HbSS, HbSC, HbSβ-thal).
   - Device telemetry: wearable SpO₂, HR, PPG, temperature, blood pressure cuffs.
2. **Normalization & QC**
   - Validate ranges (e.g., Hb 3–20 g/dL), detect missingness, unit harmonization.
   - De-identify PHI, timestamp normalization to UTC.
3. **Feature Engineering**
   - Rolling trend features: Hb change in 7 days, HRV, SpO₂ desaturation events.
   - Event windows: pain crisis, transfusion, hospitalization windows.
4. **Risk Scoring & Detection**
   - Multi-branch encoders (clinical, lab, imaging, telemetry).
   - Fusion module + calibrated classifier/regressor for anemia severity.
5. **Decision Support**
   - Provide alerts with explainability (top features, trend drivers).
   - Recommended actions (confirmatory lab test, clinician review).
6. **Continuous Learning**
   - Feedback loop from clinician labels and outcomes.

---

## 2) Structured Algorithms (Pseudo-code)

### 2.1 Data Intake & Harmonization
```
Algorithm IntakeAndHarmonize
Input: clinical_table, lab_table, imaging_paths, genomic_table, telemetry_stream
Output: harmonized_dataset

1: validate_schema(clinical_table, lab_table, genomic_table)
2: clinical_table ← normalize_units(clinical_table)
3: lab_table ← normalize_units(lab_table)
4: telemetry_stream ← resample_to_fixed_rate(telemetry_stream, rate=1Hz)
5: imaging_data ← load_images(imaging_paths)
6: merged ← merge_on_patient_id_and_time(clinical_table, lab_table, genomic_table)
7: merged ← join_telemetry(merged, telemetry_stream)
8: return {merged, imaging_data}
```

### 2.2 Feature Engineering
```
Algorithm BuildFeatures
Input: merged, imaging_data
Output: features

1: features_tabular ← rolling_features(merged, window=7 days)
2: features_tabular ← add_event_windows(features_tabular, events=[transfusion, crisis])
3: features_img ← imaging_encoder(imaging_data)
4: features ← concat(features_tabular, features_img)
5: return features
```

### 2.3 Anemia Detection & Severity Scoring
```
Algorithm AnemiaSeverity
Input: features
Output: anemia_score, risk_class

1: z ← fusion_model(features)
2: anemia_score ← calibrated_regressor(z)
3: risk_class ← threshold(anemia_score, bins=[mild, moderate, severe])
4: return anemia_score, risk_class
```

---

## 3) Reference Implementation (Python Skeleton)

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class HarmonizedData:
    tabular: "pd.DataFrame"
    images: np.ndarray


def validate_schema(*tables):
    # TODO: enforce required columns
    return True


def normalize_units(df):
    # TODO: convert Hb, bilirubin units to standardized mg/dL, g/dL
    return df


def intake_and_harmonize(clinical, labs, genomics, telemetry, images) -> HarmonizedData:
    validate_schema(clinical, labs, genomics)
    clinical = normalize_units(clinical)
    labs = normalize_units(labs)
    # TODO: resample telemetry, align timestamps
    merged = clinical.merge(labs, on=["patient_id", "timestamp"], how="left")
    merged = merged.merge(genomics, on=["patient_id"], how="left")
    return HarmonizedData(tabular=merged, images=images)


def build_features(data: HarmonizedData) -> Dict[str, np.ndarray]:
    # TODO: rolling stats, event windows, encoder outputs
    tabular_feats = np.asarray(data.tabular.select_dtypes(include=[np.number]))
    image_feats = np.mean(data.images, axis=(1, 2, 3), keepdims=True)
    return {"tabular": tabular_feats, "image": image_feats}


def fusion_model(features: Dict[str, np.ndarray]) -> np.ndarray:
    # TODO: replace with trained multimodal model
    return np.concatenate([features["tabular"], features["image"]], axis=1)


def anemia_severity(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    z = fusion_model(features)
    anemia_score = z.mean(axis=1)
    risk_class = np.digitize(anemia_score, bins=[8.0, 11.0, 13.0])
    return {"anemia_score": anemia_score, "risk_class": risk_class}
```

---

## 4) Clinical Codes & Data Tags

This section provides **common** coding references used for SCD and anemia workflows. Verify local coding policies and payer requirements before using in production.

### 4.1 ICD-10-CM (Common SCD/Anemia Codes)
- **D57.0**: Hb-SS disease with crisis *(umbrella code)*.
- **D57.00**: Hb-SS disease with crisis, unspecified.
- **D57.01**: Hb-SS disease with acute chest syndrome.
- **D57.02**: Hb-SS disease with splenic sequestration.
- **D57.1**: Hb-SS disease without crisis.
- **D57.2**: Hb-SC disease.
- **D57.4**: Sickle-cell thalassemia.
- **D64.9**: Anemia, unspecified.

### 4.2 LOINC (Common Lab Measurements)
- **718-7**: Hemoglobin [Mass/volume] in Blood.
- **4544-3**: Hematocrit [Volume Fraction] of Blood.
- **6690-2**: WBC [#/volume] in Blood by Automated count.

### 4.3 Device/Telemetry Tags (Suggested)
- `ppg_hr`: photoplethysmography heart rate
- `spo2`: pulse oximetry oxygen saturation
- `skin_temp_c`: skin temperature (Celsius)
- `cuff_bp_sys`, `cuff_bp_dia`: blood pressure
- `actigraphy_steps`: activity steps (daily)

---

## 5) Dataset Tagging: West Africa (Recent)

**Tagging goals:** ensure datasets are searchable by geography, recency, modality, and data access status.

### 5.1 Tag Schema
- `dataset_id`: unique slug
- `title`: dataset name
- `countries`: list (e.g., `"Ghana"`, `"Nigeria"`)
- `region`: `"West Africa"`
- `years_covered`: `[start_year, end_year]`
- `recency_tag`: `"recent"` if `end_year >= (current_year - 5)`
- `modalities`: `clinical`, `lab`, `imaging`, `genomic`, `telemetry`
- `access`: `public`, `restricted`, `application_required`
- `source_url`: canonical reference
- `verification_status`: `unverified`, `verified`
- `last_verified`: `YYYY-MM-DD` or `null`
- `notes`

### 5.2 Example Tagging Entries (PLACEHOLDERS)
> Replace these **EXAMPLE** entries with verified datasets and sources.

```yaml
- dataset_id: example-west-africa-scd-1
  title: "EXAMPLE: West Africa SCD Cohort"
  countries: ["Ghana", "Nigeria"]
  region: "West Africa"
  years_covered: [2020, 2024]
  recency_tag: "recent"
  modalities: ["clinical", "lab", "telemetry"]
  access: "application_required"
  source_url: "https://example.org/dataset"
  verification_status: "unverified"
  last_verified: null
  notes: "Replace with verified dataset source."

- dataset_id: example-west-africa-scd-2
  title: "EXAMPLE: SCD Imaging Registry"
  countries: ["Senegal"]
  region: "West Africa"
  years_covered: [2019, 2023]
  recency_tag: "recent"
  modalities: ["imaging"]
  access: "restricted"
  source_url: "https://example.org/dataset"
  verification_status: "unverified"
  last_verified: null
  notes: "Replace with verified dataset source."
```

---

## 6) Tagging Utility (Python)

```python
from datetime import datetime


def compute_recency_tag(end_year: int, window: int = 5) -> str:
    current_year = datetime.utcnow().year
    return "recent" if end_year >= (current_year - window) else "historical"


def tag_dataset(entry: dict) -> dict:
    entry = entry.copy()
    end_year = entry.get("years_covered", [None, None])[1]
    if end_year:
        entry["recency_tag"] = compute_recency_tag(end_year)
    entry["region"] = "West Africa"
    return entry
```

---

## 7) Integration Guidance
- Add real datasets to `data/datasets/west_africa_recent.yaml` using the schema above.
- Update `docs/dataset.md` (if present) with citations and access instructions.
- Keep notes on ethics approvals, data-sharing agreements, and local IRB constraints.
