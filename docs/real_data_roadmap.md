# Real-Human-Data Roadmap for Mmvlm4SCD

> **Purpose.** This document is the controlling plan for transitioning
> Mmvlm4SCD from a literature-calibrated synthetic benchmark to a
> validated, regulator-ready clinical AI tool trained and evaluated on
> real Sickle Cell Disease (SCD) cohorts.
>
> Author: **Sambou Kone**.
> Status: **draft v1** -- to be locked at first DUA submission.
> Companion files: `docs/preregistration.md`, `docs/data_access_checklist.md`,
> `docs/data_harmonization.md`, `src/mmvlm4scd/data/registry.py`.

## 0. Why this plan exists

The current paper reports results on a 2,000-patient synthetic cohort.
Two TRIPOD+AI items (12.1 *Development vs Validation* and 14.2
*External Validity*) explicitly call this out as a limitation. Real
data unlock four irreplaceable scientific milestones:

1. **External validity.** Train on one credentialed cohort, test on a
   geographically and temporally disjoint cohort.
2. **Clinical credibility.** Decision-curve analysis and calibration
   metrics on real outcomes are required before a journal will
   classify the work as a *clinical prediction model* rather than a
   methods paper.
3. **Equity assessment.** Real ancestry, socioeconomic and care-access
   distributions are needed before the fairness gap can be
   interpreted.
4. **Regulatory pathway.** FDA SaMD or EU MDR submission requires
   prospective evaluation on real data; this plan is the first step
   toward that pathway, not a substitute for it.

## 1. Guiding principles

1. **Pre-register, then access.** Lock the model spec, splits and
   metrics on a public registry (OSF, AsPredicted) *before* requesting
   any DUA. See `docs/preregistration.md`.
2. **Minimize re-identification risk.** Treat all real data as PHI by
   default; apply HIPAA Safe-Harbor or Expert Determination; never
   commit raw or de-identified data to git.
3. **Reproducibility-by-construction.** Every figure and number in
   downstream papers must be regenerable from versioned code +
   versioned data manifests; no manual spreadsheet steps.
4. **Modality independence first.** When a credentialed cohort
   provides only a subset of modalities, evaluate the corresponding
   *masked* model rather than synthesising the missing modality.
5. **Open by default, gated only where required.** Code, schemas,
   pre-registration, and aggregate-level results stay public; only
   patient-level data are gated.

## 2. Data-source hierarchy

The 14 sources catalogued in `src/mmvlm4scd/data/registry.py` fall
into three access tiers. Phasing follows the path of least friction.

### Tier A -- Open (no application required)

| Source | Modality | Volume | Use |
|---|---|---|---|
| `erythrocytesIDB` | imaging | ~600 RBC images | CNN imaging encoder pretraining + benchmark |
| `Kaggle Sickle RBC` | imaging | ~10k patches | CNN augmentation set |
| `GEO GSE53441` | transcriptomics | ~30 SCD vs control | feature engineering for genomic encoder |
| `GEO GSE35007` | transcriptomics | pediatric SCD | external transcriptomic validation |
| `ClinVar -- HBB` | variants | curated set | ground-truth pathogenic-variant labels |
| `gnomAD v4 -- HBB` | variants | population | allele frequencies for polygenic-like score calibration |
| `WHO SCD profiles` | epidemiology | country-level | sanity-check synthetic prevalences |
| `GBD 2021 SCD` | epidemiology | country-level | re-calibrate Weibull hazard parameters |

### Tier B -- Registered (free credential, online training)

| Source | Modality | Volume | Use |
|---|---|---|---|
| `MIMIC-IV` | clinical, temporal | full ICU/ED, D57.* subset ~1.5k SCD encounters | longitudinal labs/vitals encoder; first real Cox survival fit |
| `NHLBI CuRe-SCD Data Hub` | multimodal | longitudinal multi-site | flagship multimodal cohort |

PhysioNet credentialed-researcher status (CITI training + signed DUA)
is sufficient for MIMIC-IV.

### Tier C -- DUA-required (months-long applications)

| Source | Modality | Volume | Use |
|---|---|---|---|
| `dbGaP phs001514` (SCDIC Registry) | clinical | ~2.5k patients | severity targets, transfusion / hydroxyurea outcomes |
| `dbGaP phs001599` (Walk-PHaSST) | clinical, hemodynamic | ~720 adults | survival validation cohort |
| `NHLBI TOPMed` | WGS | tens of thousands | replace polygenic-like block with real PRS |
| `UK Biobank SCD subset` | multimodal | low N, but rich modalities | external validation |

## 3. Phased timeline

```
Phase 0 -- Pre-registration & ethics       (now -- 4 weeks)
Phase 1 -- Tier A open sources             (months 1-3)
Phase 2 -- Tier B registered access        (months 2-5)
Phase 3 -- Tier C DUA-gated cohorts        (months 4-9)
Phase 4 -- External validation paper       (months 9-12)
Phase 5 -- Prospective / regulatory        (months 12+)
```

### Phase 0 -- Pre-registration & ethics (4 weeks)

Deliverables:

- [ ] `docs/preregistration.md` posted on OSF as a public, time-stamped
      pre-registration.
- [ ] `docs/data_access_checklist.md` reviewed with an institutional
      IRB office (home institution or commercial IRB if independent).
- [ ] HIPAA training (CITI Biomedical Researcher track) completed.
- [ ] Data Management & Sharing Plan drafted per NIH 2023 policy.
- [ ] PHI-handling SOP written (`docs/phi_sop.md`, internal).
- [ ] Encrypted-storage workstation provisioned (full-disk
      encryption, audited access, no cloud sync).
- [ ] Funding statement: *not required* for this study; compute is
      author-funded (matches paper Declarations).

### Phase 1 -- Tier A open sources (months 1-3)

Deliverables:

- [ ] `src/mmvlm4scd/data/loaders/erythrocytes_idb.py`
- [ ] `src/mmvlm4scd/data/loaders/kaggle_sickle_rbc.py`
- [ ] `src/mmvlm4scd/data/loaders/geo_gse53441.py`
- [ ] `src/mmvlm4scd/data/loaders/clinvar_hbb.py`
- [ ] `src/mmvlm4scd/data/loaders/gnomad_hbb.py`
- [ ] CNN imaging encoder upgraded from MLP-on-embedding to
      ResNet18 / ConvNeXt-Tiny, pretrained on `erythrocytesIDB +
      Kaggle Sickle RBC`.
- [ ] HBB pathogenic-variant block built from ClinVar + gnomAD AFs
      (re-uses `data/harmonization.py` Genomic schema).
- [ ] Re-calibrate synthetic Weibull hazard against GBD 2021 country
      profiles to reduce the simulator's distance from real
      epidemiology.
- [ ] All Tier A loaders ship with unit tests using a small
      committed-to-repo fixture (anonymised public).

### Phase 2 -- Tier B registered access (months 2-5)

Deliverables:

- [ ] PhysioNet credentialed-researcher status approved.
- [ ] `src/mmvlm4scd/data/loaders/mimiciv_scd.py` extracts the
      D57.* ICD-10 cohort, joins labs (`labevents`), vitals
      (`chartevents`), and admissions; outputs the unified schema
      (`data/harmonization.py` -- Clinical, Temporal).
- [ ] First real-data multitask fit (severity surrogate via
      transfusion/hydroxyurea utilisation; Cox survival from death
      flag in `patients.dod`).
- [ ] Pre-registered metrics computed (no peeking).
- [ ] CuRe-SCD Data Hub access approved (Phase 2b -- multimodal).

### Phase 3 -- Tier C DUA-gated cohorts (months 4-9)

Deliverables:

- [ ] dbGaP DUA submitted for phs001514 (SCDIC) and phs001599
      (Walk-PHaSST) with an attached Data Use Statement that
      mirrors the pre-registration.
- [ ] NHLBI TOPMed application (PRS upgrade).
- [ ] UK Biobank application (multimodal external validation).
- [ ] `src/mmvlm4scd/data/loaders/dbgap_phs001514.py`,
      `dbgap_phs001599.py`, `topmed_scd.py`, `uk_biobank_scd.py`.
- [ ] Federation-friendly evaluation: each cohort runs a frozen
      pre-registered evaluation script, returns *only* aggregate
      metrics + bootstrap CIs to the central repo.
- [ ] Site-specific calibration recipe (temperature scaling +
      isotonic regression as a sensitivity analysis).

### Phase 4 -- External validation paper (months 9-12)

Deliverables:

- [ ] Manuscript v2: same structure as the synthetic paper, but
      headlined by external validation on SCDIC + UK Biobank,
      decision-curve analysis on real outcomes, fairness gap on real
      ancestry/sex distributions, and a prospective model card.
- [ ] Pre-registered model card (`docs/model_card.md`) specifying
      intended use, contraindications, and known failure modes.
- [ ] CONSORT-AI compliance check (extension of TRIPOD+AI for
      prospective evaluations).

### Phase 5 -- Prospective / regulatory (months 12+)

Optional, contingent on Phase 4 results:

- [ ] Prospective study protocol drafted in collaboration with one or
      two SCD comprehensive-care centres.
- [ ] FDA SaMD pre-submission (Q-sub) if claims warrant.
- [ ] EU MDR Class IIa technical documentation (CE-mark) if EU
      deployment is intended.

## 4. Engineering deltas

The current code expects four pre-aligned tensors per patient. Real
data are messy, asynchronous, and partially observed. Required
upgrades:

1. **Unified schema** (`src/mmvlm4scd/data/harmonization.py`) defines
   `Clinical`, `Genomic`, `Imaging`, `Temporal` Pydantic-style
   dataclasses with explicit dtypes, units, and required vs optional
   fields. Every loader returns instances of these.
2. **Deidentification utilities**
   (`src/mmvlm4scd/data/deidentification.py`): HIPAA Safe-Harbor
   stripping (remove PHI columns), date-shifting per patient,
   k-anonymity check, and a "raw-data-must-not-leak" guard that
   refuses to serialise patient identifiers.
3. **Modality-aware DataLoader.** The current
   `MultimodalSCDDataset` assumes all four modalities are present.
   Real cohorts will not. The next iteration should support
   per-sample modality availability masks; the model already handles
   modality-dropout at test time (`evaluation.extras.
   modality_dropout_sweep`), so the change is in the loader/collator.
4. **CNN imaging encoder.** Replace the MLP-on-precomputed-embedding
   with a `torch.hub` ResNet18 / ConvNeXt-Tiny initialised from
   ImageNet, then fine-tuned on `erythrocytesIDB + Kaggle Sickle RBC`.
   This is the single largest expected uplift for genuinely
   informative imaging.
5. **Time-series alignment.** Real labs are irregular; switch from
   a fixed monthly grid to time-aware GRU input
   `(value, delta_time_since_prev)` or use a discretised binned
   imputation with explicit "observed" mask.
6. **Federated evaluation harness.** Sites that cannot share
   patient-level data still need to participate. Ship a
   `src/scripts/run_federated_eval.py` that consumes a frozen model
   checkpoint + the pre-registered metric spec, runs the eval at the
   site, and emits a JSON of aggregate metrics + bootstrap CIs.
7. **Provenance ledger.** Every artefact in `experiments/results/`
   gets a sidecar JSON recording the source dataset, version,
   commit hash, and seed; this becomes the reproducibility appendix.

## 5. Statistical and reporting plan

Pre-registered metrics (locked before any real-data access):

* **Discrimination.** Macro one-vs-rest AUROC, per-class AUROC,
  Harrell C-index.
* **Calibration.** Reliability diagram, ECE, MCE, time-dependent IPCW
  Brier, Integrated Brier Score (IBS) over [1, 20] years.
* **Clinical utility.** Decision-curve analysis (severe vs not),
  sensitivity/specificity at p in {0.3, 0.5, 0.7}, PPV, NPV.
* **Robustness.** Modality-dropout sweep at p in {0, 0.1, 0.25, 0.5}.
* **Fairness.** Subgroup max-min gap on AUROC and C-index across
  ancestry, sex, age bands; report all subgroups with n >= 50.
* **External validity.** Train on cohort A, test on cohorts B, C with
  cohort identity as the dominant axis of variation.

Sample-size and power calculation (placeholder; lock in
pre-registration):

* For binary "severe vs not" with prevalence ~30%, detecting an
  AUROC change of 0.85 -> 0.80 with alpha=0.05 power=0.80 by
  DeLong's test requires roughly n_test = 600 (Hanley-McNeil
  approximation, exact value re-computed at pre-registration time).

## 6. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | DUA delays > 9 months | Medium | High | Phase 1+2 deliver standalone scientific contributions even if Phase 3 is late. |
| R2 | Privacy breach | Low | Catastrophic | HIPAA SOP, encrypted disks, no cloud sync, k-anonymity check, automated PHI scanner in CI. |
| R3 | Distribution-shift catastrophic drop | Medium | High | Pre-register a sensitivity analysis with site-specific re-calibration; report unrecalibrated and recalibrated metrics side by side. |
| R4 | Fairness gap widens on real ancestry | Medium | High | Pre-register a subgroup-aware loss-weighting sensitivity analysis; refuse to publish "best" subset only -- report all. |
| R5 | Imaging encoder underperforms in real microscopy | Medium | Medium | Phase 1 imaging pretraining is the dedicated mitigation. |
| R6 | Model card / regulatory misuse | Low | High | Publish model card with explicit out-of-scope examples; refuse personalised predictions in any public demo. |
| R7 | Selection bias in MIMIC-IV (ICU only) | High | Medium | Frame MIMIC results as ICU-cohort only; use ambulatory cohorts (SCDIC, CuRe-SCD) for generalisable claims. |
| R8 | Reviewer concerns about synthetic-only paper | High | Low | Already addressed: synthetic paper is framed as the methods paper; this roadmap is the cited bridge to the validation paper. |

## 7. Acceptance criteria for "done"

Phase 4 paper is *submission-ready* when **all** the following are true:

1. Pre-registration is publicly time-stamped and version-locked
   before the access date of the validation cohort.
2. At least one Tier-C cohort has been used as an *external*
   validation set (i.e., not used during training or tuning).
3. Headline metrics include AUROC, C-index, IBS, ECE, decision-curve
   max net benefit, fairness gap, and modality-dropout robustness
   curve, each with bootstrap CIs.
4. A model card (`docs/model_card.md`) is published with explicit
   intended-use and out-of-scope statements.
5. All code that produced the results is at a tagged release; all
   numbers regenerate from `make paper` against the pinned
   environment.
6. A patient/community advisory letter (or equivalent
   public-engagement record) is included in the supplementary
   material; SCD is a community where patient consultation is
   particularly important and the synthetic paper does not yet have
   this.

## 8. Pointers

* Pre-registration template -- `docs/preregistration.md`
* IRB / DUA / HIPAA checklist -- `docs/data_access_checklist.md`
* Schema spec -- `docs/data_harmonization.md`
* Data registry (machine-readable) -- `src/mmvlm4scd/data/registry.py`
* Loader stubs -- `src/mmvlm4scd/data/loaders/`
* Synthetic-paper companion (Q1 LaTeX) -- `paper/paper.tex`
