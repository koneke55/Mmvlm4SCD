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
