# Real-Data Access, Ethics and Privacy Checklist

This document is the operational checklist that gates real-human-data
access for Mmvlm4SCD. It pairs with `docs/real_data_roadmap.md`
(strategic plan) and `docs/preregistration.md` (scientific
commitments). It is intentionally conservative.

## 1. Researcher credentials

- [ ] CITI Biomedical Research course (current within 12 months).
- [ ] CITI Health Privacy and Information Security (HIPAA, current).
- [ ] CITI Conflict of Interest course (current).
- [ ] PhysioNet Credentialed-Researcher status (for MIMIC-IV).
- [ ] dbGaP authorised user with valid eRA Commons login.
- [ ] UK Biobank approved researcher (one-off application).

## 2. Institutional review

- [ ] Determine whether the planned use is **Not Human-Subjects
      Research** (only fully de-identified secondary data with no
      intervention) or **Exempt** under 45 CFR 46.104. File the
      determination request with an IRB.
- [ ] If working independently (no university affiliation), engage a
      commercial IRB (e.g., WCG IRB, Advarra) for documented review
      before submitting any DUA.
- [ ] Document the IRB determination letter in
      `docs/ethics/irb_letter.pdf` (gitignored, internal).

## 3. Data Use Agreements

For each cohort the DUA must be reviewed *before* download and
re-reviewed annually:

- [ ] **dbGaP phs001514 (SCDIC).** Project-specific Data Access
      Request; consent group restrictions noted in the dbGaP
      catalogue; Data Use Statement aligned with
      `docs/preregistration.md`.
- [ ] **dbGaP phs001599 (Walk-PHaSST).** As above.
- [ ] **MIMIC-IV (PhysioNet).** Sign the Data Use Agreement at
      download time; the PhysioNet credentialed account is the access
      gate.
- [ ] **NHLBI TOPMed.** TOPMed-specific application; coordinate with
      the relevant cohort working group.
- [ ] **UK Biobank.** Application to UKB Access Management System;
      annual reporting required.
- [ ] **NHLBI CuRe-SCD Data Hub.** Cure SCD Initiative data access
      committee.

## 4. HIPAA Safe-Harbor de-identification (US-side)

The following HIPAA Safe-Harbor identifiers must not appear in any
patient-level artefact stored on disk or in version control. The
helper `mmvlm4scd.data.deidentification.assert_no_phi_columns`
enforces this on every loader.

- [ ] Names
- [ ] Geographic subdivisions smaller than a state (incl. ZIP codes
      below the first three digits)
- [ ] All elements of dates (except year) directly related to an
      individual; ages > 89 collapsed to "90+"
- [ ] Telephone numbers, fax numbers, email addresses
- [ ] Social Security numbers; medical record numbers; health-plan
      beneficiary numbers; account numbers; certificate / licence
      numbers
- [ ] Vehicle, device identifiers and serial numbers
- [ ] Web URLs, IP addresses
- [ ] Biometric identifiers (fingerprints, voice prints)
- [ ] Full-face photographs and any comparable images
- [ ] Any other unique identifying number, characteristic, or code

If any of the above is required for analysis, switch to **Expert
Determination** (45 CFR 164.514(b)(1)) and engage a qualified
statistician / privacy officer for the certification.

## 5. GDPR (EU-side, e.g. UK Biobank)

- [ ] Lawful basis identified (typically Article 6(1)(e) public
      interest + Article 9(2)(j) scientific research with appropriate
      safeguards).
- [ ] Data Protection Impact Assessment (DPIA) completed.
- [ ] Records of Processing Activities (ROPA) maintained.
- [ ] Subject-rights handling plan documented (access, erasure,
      restriction). Most large biobanks delegate this to the source.

## 5.bis Sub-Saharan Africa frameworks

For any cohort with `region == "africa"`:

- [ ] Country-level IRB approval secured before any data movement.
      See the country table below.
- [ ] **H3Africa Ethics Working Group** guidance reviewed; secondary
      use is constrained by the original consent.
- [ ] **CARE Principles for Indigenous Data Governance** (Collective
      Benefit, Authority to Control, Responsibility, Ethics) explicitly
      applied; documented in a one-page CARE-alignment memo per
      cohort.
- [ ] **African Union Convention on Cyber Security and Personal Data
      Protection (Malabo Convention)** plus the relevant national
      data-protection law:
  - [ ] Nigeria -- NDPR (Nigerian Data Protection Regulation 2019);
        site-level NHREC + state-level approval.
  - [ ] Ghana -- Ghana DPA 2012; GHS-ERC + KATH/Korle-Bu site IRBs.
  - [ ] Tanzania -- NIMR MRCC; Personal Data Protection Act 2022.
  - [ ] Kenya -- DPA 2019; KEMRI SERU + NACOSTI permit.
  - [ ] Uganda -- UNCST + local IRB; Data Protection and Privacy
        Act 2019.
  - [ ] DRC -- Comite National d'Ethique de la Sante.
  - [ ] South Africa -- POPIA 2013 + HREC at host university.
  - [ ] Cameroon -- Comite National d'Ethique de la Recherche pour
        la Sante Humaine.
  - [ ] Mali -- Comite Institutionnel d'Ethique du USTTB.
- [ ] Material Transfer Agreement (MTA) signed if any biospecimen
      moves; we plan no biospecimen movement, so the MTA path is
      avoided.
- [ ] Data Sovereignty contract: patient-level data remain in the
      country of origin; only aggregate metrics return to the central
      repository.
- [ ] Local PI listed as a co-author of any manuscript using the
      cohort, per ICMJE criteria. If criteria are not met for a
      named individual, the contributing institution is acknowledged
      in the affiliations block.
- [ ] Plain-language project summary translated into the
      cohort-relevant language(s) (Swahili, Yoruba, Hausa, Igbo,
      French where applicable) and shared with the local SCD
      patient-association before data extraction.

## 5.ter South Asia frameworks (India, Sri Lanka, Bangladesh)

For any cohort with `region == "south_asia"`:

- [ ] **ICMR National Ethical Guidelines for Biomedical and Health
      Research Involving Human Participants** (2017, with 2023 SCD-
      screening updates) reviewed.
- [ ] **Indian DBT / BIRAC Guidelines for International Collaborative
      Biomedical Research** followed; MTAs in place if biospecimens
      move.
- [ ] **Digital Personal Data Protection Act 2023** (DPDPA) compliance
      documented.
- [ ] Institutional Ethics Committee (IEC) approval at each host
      institution (ICMR-NIRTH Jabalpur, AIIMS New Delhi, MGM Indore,
      etc.).
- [ ] Health Ministry's Screening Committee (HMSC) clearance for
      international collaboration where applicable.
- [ ] **Tribal Health Bureau approval** for any work involving
      Scheduled-Tribe populations (NSCAEM, NIRTH-Jabalpur, Hemalkasa).
- [ ] **CARE Principles for Indigenous Data Governance** explicitly
      applied for tribal-population cohorts.
- [ ] Local-language plain-language project summary (Hindi, Marathi,
      Gujarati, Telugu, Tamil, Sinhala) shared with patient
      organisations.
- [ ] Sri Lanka -- Ethics Review Committee, Ministry of Health.
- [ ] Bangladesh -- BMRC (Bangladesh Medical Research Council) if /
      when Bangladeshi cohorts join the registry.

## 6. Storage and compute

- [ ] Workstation: full-disk encryption (LUKS / FileVault /
      BitLocker), screen-lock at 5 min, password manager only.
- [ ] No automatic cloud sync of any folder containing real data
      (Dropbox, iCloud, Google Drive, OneDrive, GitHub Desktop).
- [ ] Backup: encrypted external drive in a locked location; backups
      contain the same retention window as the DUA.
- [ ] Compute: local GPU or HIPAA-eligible cloud (AWS HIPAA-eligible
      services with BAA, Azure with BAA). No general-purpose Colab /
      Kaggle for real data.
- [ ] Network: SSH only with key auth; no public-facing services on
      the workstation.

## 7. Repository hygiene

- [ ] `.gitignore` blocks `data/raw/**`, `data/processed/**`,
      `data/external/**`, `experiments/checkpoints/*.pt`. Already
      enforced.
- [ ] Pre-commit hook (`pre-commit-config.yaml`) runs the PHI scanner
      `mmvlm4scd.data.deidentification.scan_path_for_phi` on staged
      files.
- [ ] `experiments/results/` is allowed to contain *aggregate*
      metrics only; never raw predictions on identified patients.
- [ ] Trained model weights are released only after passing a
      memorisation audit (see Section 9).

## 8. Logging and audit

- [ ] Every data-access event is logged in
      `experiments/logs/data_access.jsonl` with: timestamp, dataset
      identifier, project ID, user, purpose. Review monthly.
- [ ] DUA renewal reminders set in the project calendar 60 days
      before each expiry.
- [ ] Annual data-destruction certification when a DUA window ends.

## 9. Model release controls

- [ ] Membership-inference probe run before release; report attack
      AUROC on training vs holdout patients, refuse release if > 0.6.
- [ ] No raw training-set patient IDs ever appear in a model card,
      figure, or example.
- [ ] Public model card (`docs/model_card.md`) follows the
      Mitchell et al. (2019) template; specifies intended use,
      out-of-scope use, ethical considerations, and known failure
      modes.
- [ ] Inference demo (if any) has rate limiting and never echoes the
      input back in logs.

## 10. Patient and community engagement

- [ ] Identify a Sickle Cell community advisory body (e.g. Sickle Cell
      Disease Association of America, Sickle Cell Society UK) and
      send a plain-language project summary at Phase 1.
- [ ] Solicit feedback on subgroup definitions, fairness metrics, and
      acceptable false-positive / false-negative trade-offs.
- [ ] Co-author or acknowledge the advisory body in any real-data
      paper, per their preference.

## 11. Final gate before downloading any real data

All boxes in Sections 1-3 and 6-7 must be checked, and the
pre-registration must be public. A short "go / no-go" memo is signed
by the author and the methods reviewer and stored in
`docs/ethics/go_nogo_<dataset>_<date>.pdf` (gitignored).
