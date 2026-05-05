# Mmvlm4SCD -- Pre-Registration Template

> Post a copy of this document to **OSF** (`https://osf.io/`) before
> requesting any DUA. Do not edit it after the access date of the first
> credentialed cohort. If a hypothesis or metric must change, file an
> **amendment** with a date and rationale; do not silently rewrite.
>
> This is a fillable template. Replace each `[TODO]` with the actual
> commitment.

| Field | Value |
|---|---|
| Title | Multimodal Modeling of Sickle Cell Disease: External Validation on Credentialed Cohorts |
| Author | Sambou Kone |
| Pre-registration date | `[TODO YYYY-MM-DD]` |
| Pre-registration DOI / OSF ID | `[TODO]` |
| Anticipated first DUA submission | `[TODO YYYY-MM-DD]` |
| Code version (git tag) | `[TODO v0.x.y]` |
| Funding | Not required for this study (author-funded). |

## 1. Hypotheses

Write each hypothesis as **directional** and **falsifiable**.
Standard convention: H1, H2 are primary; H3+ are secondary.

* **H1 (severity discrimination).** A jointly-trained multimodal model
  achieves macro one-vs-rest AUROC at least 0.05 higher than a
  clinical-only logistic regression baseline on the held-out
  external cohort.
* **H2 (survival discrimination).** The multimodal model achieves a
  Harrell C-index at least 0.03 higher than a clinical-only Cox PH
  model on the held-out external cohort.
* **H3 (calibration).** The multimodal model achieves Integrated Brier
  Score (IBS) over [1, 20] years no worse than the Cox PH baseline
  by more than 0.02 on the held-out external cohort.
* **H4 (clinical utility).** The multimodal model has higher net
  benefit than "treat all" and "treat none" policies for the
  severe-vs-not framing across thresholds p in [0.2, 0.6].
* **H5 (robustness).** Test-time AUROC degradation under per-modality
  dropout at p = 0.25 is less than 0.05 on the held-out external
  cohort.
* **H6 (fairness, exploratory).** Subgroup max-min AUROC gap across
  ancestry strata is less than 0.10 on the held-out external cohort.
  This hypothesis is exploratory and *will not* be used to declare
  the model "fair"; a wider gap is a publishable negative finding.

### Geographic-transfer hypotheses (added because Africa + South Asia carry the global SCD burden)

* **H7 (Africa-trained -> Africa-evaluated, primary).** A multimodal
  model trained on the SickleInAfrica / SPARCO Registry plus the
  Muhimbili Sickle Cohort and evaluated on a held-out Africa-side
  partition (held-out site or held-out time window) achieves macro
  AUROC at least 0.80 and Harrell C-index at least 0.65, both with
  the lower bound of the 95% bootstrap CI strictly above the
  Sebastiani / Quinn clinical-only baselines re-estimated on the same
  cohort.
* **H8 (South-Asia-trained -> South-Asia-evaluated, primary).** A
  multimodal model trained on ICMR-NIRTH Jabalpur (and, where
  available, AIIMS / MGM Indore) and evaluated on the
  Lok-Biradari-Prakalp / Hemalkasa community-clinic cohort achieves
  macro AUROC at least 0.75 and C-index at least 0.62. Lower thresholds
  reflect the smaller, more heterogeneous Indian cohorts.
* **H9 (cross-region transfer, exploratory).** A model trained on
  Africa-side cohorts and evaluated on South-Asia-side cohorts (and
  vice versa) shows macro-AUROC degradation >= 0.05 versus
  within-region evaluation. This hypothesis is **directional toward a
  positive degradation**: we expect transfer to be hard and we want
  the manuscript to honestly report it.
* **H10 (haplotype stratification).** Per-haplotype subgroup
  performance (Benin vs Bantu/CAR vs Senegal vs Cameroon vs
  Arab-Indian) shows max-min AUROC gap >= 0.05; this is reported
  rather than tested for "fairness" because the haplotypes carry
  genuine biological signal.
* **H11 (care-setting confounding).** Stratifying performance by
  `CareSettingTier` (TERTIARY_HIC / TERTIARY_LMIC / SECONDARY_LMIC /
  COMMUNITY) reveals at least 0.05 AUROC variation, indicating that
  any naive global metric conflates "the disease is mild" with
  "treatment access is poor."

## 2. Population

* **Inclusion.** Patients with an SCD diagnosis confirmed by ICD-10
  D57.0 / D57.1 / D57.2 / D57.4 / D57.8, or by genotype where
  available (HbSS, HbSC, HbSbeta+, HbSbeta0).
* **Exclusion.** Patients < 1 year old at first encounter; patients
  with fewer than two encounters spaced >= 90 days apart (avoids
  one-shot encounters that cannot anchor a longitudinal model).
* **Time window.** Index encounter is the first encounter satisfying
  inclusion criteria after `[TODO YYYY-MM-DD]`. Follow-up is the
  minimum of (a) last recorded encounter, (b) date of death, or (c)
  cohort cut-off.

## 3. Outcomes

* **Severity (3-class ordinal).** mild / moderate / severe, derived
  from a pre-registered formula combining annual VOC count,
  hospitalisation days, transfusion utilisation and acute-chest-
  syndrome history. The formula is locked in
  `docs/severity_label_definition.md` and frozen at pre-registration
  time.
* **Survival.** All-cause mortality time and event flag. Censored at
  the cohort cut-off date or last encounter.
* **Secondary outcomes (exploratory).** Time to first VOC after
  index; time to first stroke; transfusion-free survival.

## 3.bis Geographic-priority data plan

Training prioritises Africa- and South-Asia-anchored cohorts;
North-American and European cohorts (MIMIC-IV, dbGaP SCDIC,
Walk-PHaSST, UK Biobank, NHLBI TOPMed) are **reference** cohorts used
for transferability analysis only. Specifically:

* **Primary training cohort.** SickleInAfrica / SPARCO Registry
  (Tanzania + Ghana + Nigeria + Cameroon + Mali) joined with the
  Muhimbili Sickle Cohort (Tanzania) and the ICMR-NIRTH Jabalpur
  tribal cohort (India). Patients are tagged with `Region`,
  `country`, `HbHaplotype`, `CareSettingTier` and
  `HydroxyureaAccess` per the unified harmonization schema.
* **Primary external-validation cohort.** A held-out site of
  SickleInAfrica/SPARCO that was not used in training, joined with
  the Lok Biradari Prakalp (Hemalkasa) community-clinic cohort.
* **Reference external cohorts.** dbGaP SCDIC, Walk-PHaSST and the
  UK Biobank SCD subset, evaluated as transferability checks; their
  results are reported separately and **never** as the headline
  metrics.
* **Pediatric stratum.** CONSA newborn-screening data drives a
  separate pediatric severity model evaluated at 1-, 2- and
  5-year horizons; reported alongside the adult model with explicit
  age-band tagging.

## 4. Predictors

Locked feature lists per modality. New features may not be added
post-hoc; missing features are imputed only via methods declared here.

* **Clinical (tabular).** Age, sex, ancestry self-report, genotype,
  baseline Hb, HbF%, WBC, platelet count, LDH, total bilirubin, CRP,
  history of acute chest syndrome, history of stroke, hydroxyurea
  exposure flag, chronic transfusion flag, BMI.
* **Genomic.** HBB pathogenic-variant indicators (ClinVar curated
  set, Phase 1 deliverable). For Tier-C cohorts: a polygenic risk
  score derived from `[TODO PRS source]`.
* **Imaging.** Per-patient peripheral-blood-smear embedding produced
  by the Phase-1 ResNet18 / ConvNeXt-Tiny encoder pretrained on
  `erythrocytesIDB + Kaggle Sickle RBC`.
* **Temporal.** Monthly aggregates of vitals (HR, SpO2, BP), labs
  (Hb, WBC, LDH), and pain VAS for the 24 months preceding index
  (when available).

## 5. Statistical analysis

* **Splitting.** Train, val, test split is **patient-level** and
  **temporally stratified**: the test cohort is drawn from a
  geographically and/or temporally disjoint partition of the data.
  No patient appears in more than one split.
* **Primary analysis.** Frozen attention-fusion model trained on
  cohort A; evaluated *once* on the locked external cohort B.
* **Sensitivity analyses.** (i) Site-specific recalibration via
  temperature scaling and isotonic regression; (ii) per-modality
  ablation; (iii) fusion-strategy sweep (attention, cross-attention,
  late); (iv) subgroup-aware loss reweighting.
* **Uncertainty quantification.** 3 random initialisations + 1000
  non-parametric bootstrap resamples for 95% percentile CIs on every
  reported metric.
* **Multiple-comparison policy.** No formal multiple-comparison
  correction is applied to subgroup analyses, which are descriptive.
  Only H1-H5 are confirmatory; H6 and all subgroup numbers are
  flagged exploratory.
* **Stopping rule for hyperparameter search.** Hyperparameters are
  frozen on the validation split of cohort A *before* any access to
  cohort B. After cohort B access is granted, no hyperparameter
  re-tuning is performed.

## 6. Sample size

`[TODO -- Hanley-McNeil approximation against the H1 effect size; lock the
exact value before access.]` Provisional target n_test >= 600 patients
to detect AUROC = 0.85 vs 0.80 with alpha = 0.05 and power = 0.80.

## 7. Missing data

* Tabular features missing in < 30% of patients are imputed by the
  median (continuous) or the mode (categorical) computed on the
  *training* split only.
* Patients missing all features in a modality are still included; the
  modality's contribution is masked at fusion time. The model already
  supports this via `evaluation.extras.modality_dropout_sweep`.
* Patients missing the outcome variable are excluded.

## 8. Privacy and ethics

See `docs/data_access_checklist.md` for the full IRB / DUA / HIPAA /
GDPR plan. Highlights:

* All real-data processing happens on a HIPAA-compliant workstation.
* No raw data, derived features at the patient level, or trained
  weights that could memorize identifiers will be committed to the
  public repository.
* Aggregate-level results (metrics, bootstrap CIs, calibration plots
  on >= 50 patients per bin) are public.
* Model checkpoints will be released only after a memorisation /
  membership-inference audit.

## 9. Reporting

The Phase-4 manuscript will follow TRIPOD+AI (Collins 2024). A
checklist mirroring `paper/paper.tex` Table~S1 will be appended.

## 10. Deviation log

| Date | Section | Change | Rationale |
|---|---|---|---|
| `[YYYY-MM-DD]` | `[2.1]` | `[change]` | `[rationale]` |

## 11. Sign-off

* Author: Sambou Kone -- `[signature placeholder]`
* Methods reviewer (independent): `[TODO name + signature]`
* Date: `[TODO]`
