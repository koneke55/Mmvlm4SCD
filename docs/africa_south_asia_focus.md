# Africa and South Asia: the Geographic Centre of the Mmvlm4SCD Plan

> **Why this document exists.** The default registry of clinical-AI
> datasets is dominated by US and UK cohorts. Sickle Cell Disease
> follows the opposite distribution: Sub-Saharan Africa and South Asia
> together account for the vast majority of SCD births and SCD deaths
> worldwide. Building a useful SCD model therefore means inverting the
> usual "train on rich-country cohorts, hope it transfers" pipeline and
> putting Africa- and South-Asia-led cohorts at the centre of training,
> evaluation, and accountability.
>
> This document is the operational addendum that re-anchors
> `docs/real_data_roadmap.md` around that geographic priority. It also
> sets the equity, governance, and capacity-building requirements that
> any real-data work in these regions must satisfy.

## 1. Why the geographic focus matters

### 1.1 Epidemiology

The disease burden is concentrated in two regions:

* **Sub-Saharan Africa.** Nigeria, the Democratic Republic of the
  Congo, Tanzania, Ghana, Cameroon, Uganda, Kenya, Mali, Senegal and
  Sierra Leone together account for the largest share of annual SCD
  births. Nigeria alone has the largest absolute SCD birth burden of
  any country.
* **South Asia.** India is the second-largest contributor to global
  SCD births, with the disease concentrated in the central and western
  tribal belts (Chhattisgarh, Madhya Pradesh, Maharashtra, Gujarat,
  Odisha) and in pockets of Sri Lanka and Bangladesh.

A model whose evaluation cohorts are 90% North American and European
cannot honestly be called a Sickle Cell Disease model. It is, at
best, a model of how SCD presents in a non-representative slice of
the patient population.

### 1.2 Genetic background

Severity, baseline foetal haemoglobin, and response to hydroxyurea
all depend on the **HbS haplotype background**, which differs by
geography:

| Haplotype | Typical population | Baseline HbF | Severity tendency |
|---|---|---|---|
| Benin | West Africa, African-American diaspora | low-mid | moderate-high |
| Bantu / CAR | Central / East Africa | low | high |
| Senegal | Senegambian | mid | moderate |
| Cameroon | Cameroon | low | moderate |
| **Arab-Indian** | India, Eastern Saudi Arabia | high | mild-moderate |

Treating these as exchangeable -- as a US-trained model implicitly
does -- under-predicts severity for Bantu/CAR patients and
over-predicts severity for Arab-Indian patients. Mmvlm4SCD's
`HbHaplotype` enum and `Region` field exist precisely so that this
modelling choice is explicit, testable, and reportable.

### 1.3 Health-system context

Resource availability differs dramatically across cohorts:

* MIMIC-IV captures care delivered in a US tertiary academic centre
  with full hydroxyurea access and routine chronic transfusion
  programmes.
* Many CONSA / SickleInAfrica sites operate in district hospitals
  where hydroxyurea supply is intermittent and chronic transfusion is
  rare.
* The Lok Biradari Prakalp / Hemalkasa cohort represents
  community-clinic care for tribal populations with limited routine
  treatment access.

A severity score trained without `CareSettingTier` and
`HydroxyureaAccess` fields will conflate "the disease is mild" with
"the patient cannot access treatment." The schema captures both so
the model can be evaluated and re-trained correctly.

## 2. Cohorts at the centre of the plan

The registry now lists 14+ Africa- and South-Asia-anchored sources
(see `src/mmvlm4scd/data/registry.py`). The priority cohorts:

### Sub-Saharan Africa

| Cohort | Country | Why central |
|---|---|---|
| **SickleInAfrica / SPARCO Registry** | Tanzania, Ghana, Nigeria, Cameroon, Mali | Multi-site harmonised African registry; flagship Phase-3 target. |
| **Muhimbili Sickle Cohort (MSC)** | Tanzania | Largest single-site East African cohort with deep longitudinal phenotype + biospecimen archive. |
| **CONSA (Newborn Screening in Africa)** | 9-country newborn screening network | Pediatric and under-5 survival data that no Western cohort can supply at scale. |
| **LUTH and UCH Ibadan** | Nigeria | The single country with the largest SCD birth burden globally. |
| **Komfo Anokye + Korle-Bu** | Ghana | Longest-running African newborn-screening + hydroxyurea programme. |
| **MalariaGEN HBB / globin** | Pan-African | Open population genomics for haplotype-aware features. |
| **H3Africa SCD bioinformatics** | Pan-African | African modifier-gene background (BCL11A, HMIP-2, MYB). |

### South Asia

| Cohort | Country | Why central |
|---|---|---|
| **NSCAEM** | India (17 states) | National elimination mission screening tens of millions; the only population-level SCD screening programme outside Africa. |
| **ICMR-NIRTH Jabalpur tribal cohort** | India (MP, Chhattisgarh) | Reference cohort for the Arab-Indian haplotype + tribal-population SCD natural history. |
| **AIIMS New Delhi SCD Registry** | India | Tertiary-centre Indian phenotype with paired HBB/HBA genotypes. |
| **Lok Biradari Prakalp / Hemalkasa** | India (Gadchiroli) | Community-clinic care in the Madia Gond tribal population; canonical low-resource reference. |
| **MGM Indore SCD programme** | India (MP) | Treatment-effect heterogeneity (hydroxyurea, pregnancy outcomes). |
| **Sickle Cell Society of Sri Lanka** | Sri Lanka | Distinct ancestry and care system; complementary to the Indian registries. |

## 3. Governance, ethics and data sovereignty

The defaults that work for MIMIC-IV do **not** transfer to these
regions. Specific frameworks apply:

### 3.1 Pan-African

* **H3Africa Ethics Working Group** guidance: secondary use of African
  genomic data requires re-consent or an affirmative IRB ruling at
  the host institution.
* **CARE Principles for Indigenous Data Governance**
  (Collective Benefit, Authority to Control, Responsibility, Ethics).
* **African Union Convention on Cyber Security and Personal Data
  Protection** (Malabo Convention) at the continental level, plus
  national data-protection laws (Nigeria NDPR, Kenya DPA 2019, South
  Africa POPIA, Ghana DPA 2012).

### 3.2 South Asia / India

* **ICMR National Ethical Guidelines for Biomedical and Health
  Research Involving Human Participants** (2017, with 2023 SCD-screening updates).
* **Indian DBT / BIRAC Guidelines for International Collaborative
  Biomedical Research** -- material transfer agreements (MTAs) are
  required for any biospecimen movement.
* **Digital Personal Data Protection Act 2023** (DPDPA).
* **Tribal Health Bureau approval** for any work involving
  Scheduled-Tribe populations; CARE Principles apply here as well.

### 3.3 Country-specific IRBs

| Country | Body | Notes |
|---|---|---|
| Nigeria | NHREC + state Ministry of Health committees | site-level approval also required |
| Ghana | Ghana Health Service Ethics Review Committee | per-site addenda for KATH and Korle-Bu |
| Tanzania | NIMR Medical Research Coordinating Committee (MRCC) | applies for foreign collaborators |
| Kenya | KEMRI Scientific & Ethics Review Unit (SERU) | + NACOSTI permit |
| Uganda | Uganda National Council for Science and Technology (UNCST) + local IRB | |
| DRC | Comite National d'Ethique de la Sante | |
| South Africa | HREC at host university + POPIA registration | |
| India | ICMR national bioethics + institutional IEC + Health Ministry's Screening Committee for international collaborations | + Tribal Health Bureau where applicable |
| Sri Lanka | Ethics Review Committee, Ministry of Health | |

### 3.4 Data flow rules

* Default: **patient-level data stays within the country of origin**.
  Aggregate metrics, model weights and trained-model evaluation
  artefacts can be returned to the central repo only after a host-
  country review.
* Federated evaluation harness: each site runs the frozen evaluation
  script locally and returns only `clinical_summary.json`-style
  aggregates (already supported by the existing
  `run_clinical_eval.py`).
* Material Transfer Agreements (MTAs) are required for any movement
  of biospecimens; we plan no biospecimen movement.

## 4. Equity-aware evaluation requirements

Beyond the metrics already implemented:

* **Stratified-by-region performance.** All headline metrics must be
  reported separately for Africa, South Asia, and the
  North-America / Europe reference set.
* **Stratified-by-haplotype performance.** AUROC and C-index
  reported by `HbHaplotype` (Benin, Bantu/CAR, Senegal, Cameroon,
  Arab-Indian). Helper: `cohort_geographic_breakdown`.
* **Stratified-by-care-setting performance.** Helps separate "model
  is wrong" from "treatment access is poor."
* **Fairness gap as a hard publication gate.** Phase 4 manuscript
  refuses to make a generalisability claim if the max-min subgroup
  AUROC gap on `Region` exceeds 0.10 without an explicit
  re-calibration analysis.
* **Pediatric-specific evaluation.** CONSA data unlock under-5
  outcomes; the model must be evaluated separately on the pediatric
  cohort with appropriate horizons (1, 2, 5 years) rather than the
  default adult-survival horizons (1, 2, 5, 10, 15, 20).

## 5. Capacity-building commitments

A model that is "trained on African data and used elsewhere" without
return value is a transfer of resource, not a collaboration. The
project commits to:

1. **Open code, open weights (after audit).** All code and
   appropriately audited model weights are public; no per-cohort
   "academic licence" gating.
2. **Co-authorship by data-contributing sites.** Manuscripts using
   data from a contributing site list the local PI as a co-author
   per ICMJE criteria; if those criteria are not met for individuals,
   the contributing institution is acknowledged in the affiliations
   block of the model card and any submitted manuscript.
3. **Local-language model card.** The Phase-4 model card is
   translated into Swahili, Yoruba, Hausa, Hindi, Marathi and Tamil
   (where contributors exist for translation review). Plain-language
   summaries for patient communities are produced with the relevant
   patient-organisation partners (SCFN, SCDAA, etc.).
4. **Low-resource inference budget.** Final models targeting
   community-clinic deployment must run inference in <500 ms on a
   single CPU thread and use <2 GB of RAM, so they remain useful on
   commodity hardware in low-bandwidth settings.
5. **No paywall on derived statistics.** Any aggregated statistic
   computed on a contributing cohort is published openly (under
   `experiments/results/<exp>/clinical/` and in the manuscript),
   never sold, never gated.

## 6. Updated phasing for the African and South-Asian branches

This supersedes the geographic ordering in
`docs/real_data_roadmap.md` Section 3.

```
Phase 0 -- Pre-registration & ethics      (now -- 4 weeks)
Phase 1 -- Open foundations               (months 1-3)
Phase 2A -- Africa Tier-B (registered)    (months 2-6)
                CONSA, MUHAS / Muhimbili, NSCAEM India
Phase 2B -- South Asia Tier-B             (months 2-6)
                NSCAEM-linked screening + AIIMS letters of support
Phase 3A -- Africa DUA (months 4-9)
                SPARCO, LUTH, UCH Ibadan, KATH/Korle-Bu, H3Africa
Phase 3B -- South Asia DUA (months 4-9)
                ICMR-NIRTH, AIIMS-RDA, MGM Indore,
                Lok Biradari Prakalp (community MoU)
Phase 3C -- North America / Europe reference (months 6-9)
                MIMIC-IV, dbGaP SCDIC + Walk-PHaSST,
                UK Biobank, NHLBI TOPMed -- evaluated as
                *reference* cohorts, not as the primary training set
Phase 4 -- External-validation paper      (months 9-12)
                Headline cohort: SickleInAfrica / SPARCO + ICMR-NIRTH
                Reference cohorts: SCDIC + UK Biobank
Phase 5 -- Prospective / regulatory       (months 12+)
                If indicated, work with national
                programmes (NSCAEM India,
                Ministries of Health for Tanzania, Ghana, Nigeria)
```

The reordering is deliberate: the model is trained predominantly on
African and South-Asian patients and evaluated on Western reference
cohorts only as a sanity check on transferability *out* of the burden
regions, not as the primary endpoint.

## 7. What this means for the next 30 days

* [ ] Identify and reach out to one local PI in each of: Tanzania
      (MUHAS), Nigeria (LUTH or UCH Ibadan), Ghana (KATH or Korle-Bu),
      and India (ICMR-NIRTH or AIIMS).
* [ ] Pre-register the Phase-4 protocol on OSF using the template in
      `docs/preregistration.md`, with explicit
      geographic-transfer hypotheses (now H7-H9 in the
      preregistration).
* [ ] Run the existing `run_clinical_eval.py` against the synthetic
      cohort with the new `Region`, `HbHaplotype` and
      `CareSettingTier` fields populated, to verify the
      stratified-evaluation harness works end-to-end before any real
      data arrive.
* [ ] Submit CITI Biomedical Research and HIPAA training renewals.
* [ ] Prepare a 1-page plain-language project summary in English,
      Swahili, French, Hindi and Marathi for community-organisation
      review.
