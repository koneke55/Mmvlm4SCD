"""Registry of public Sickle Cell Disease (SCD) data sources.

The aim of this module is to give a single, citable catalogue of public
SCD datasets and the access steps for each. We intentionally do not
auto-download credential-gated archives (dbGaP, UK Biobank, MIMIC); those
require an authorized data-use agreement signed by the researcher.

Each entry exposes:
    - ``name``               human-readable dataset name
    - ``modality``           clinical | genomic | imaging | temporal | multimodal
    - ``url``                landing page / portal
    - ``access``             open | registered | dua-required
    - ``citation``           short bibliographic key
    - ``notes``              one-line summary

The registry is consumed by ``data/loaders`` and by the documentation
generator in ``docs/dataset.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass(frozen=True)
class DataSourceSpec:
    name: str
    modality: str
    url: str
    access: str  # "open" | "registered" | "dua-required"
    citation: str
    notes: str
    tags: tuple = field(default_factory=tuple)


PUBLIC_SCD_DATASETS: List[DataSourceSpec] = [
    DataSourceSpec(
        name="NHLBI CuRe-SCD Data Hub",
        modality="multimodal",
        url="https://curesicklecell.org/",
        access="registered",
        citation="NHLBI-CureSCD-2020",
        notes="Cure Sickle Cell Initiative cohort with longitudinal clinical, "
        "genomic and patient-reported outcomes.",
        tags=("longitudinal", "clinical", "genomic"),
    ),
    DataSourceSpec(
        name="dbGaP phs001514 - Sickle Cell Disease Implementation Consortium",
        modality="clinical",
        url="https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001514",
        access="dua-required",
        citation="dbGaP-phs001514",
        notes="SCDIC Registry: pain, transfusion, hydroxyurea adherence outcomes.",
        tags=("registry", "phenotype"),
    ),
    DataSourceSpec(
        name="dbGaP phs001599 - Walk-PHaSST",
        modality="clinical",
        url="https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001599",
        access="dua-required",
        citation="dbGaP-phs001599",
        notes="Pulmonary hypertension and SCD natural-history cohort.",
        tags=("survival", "hemodynamics"),
    ),
    DataSourceSpec(
        name="GEO GSE53441 - Sickle Cell Whole Blood Transcriptome",
        modality="genomic",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53441",
        access="open",
        citation="Hounkpe2014",
        notes="Whole-blood microarray expression in SCD patients vs controls.",
        tags=("transcriptomics", "microarray"),
    ),
    DataSourceSpec(
        name="GEO GSE35007 - Pediatric SCD Erythrocyte Transcriptome",
        modality="genomic",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35007",
        access="open",
        citation="GSE35007",
        notes="Pediatric SCD reticulocyte expression profiling.",
        tags=("pediatric", "transcriptomics"),
    ),
    DataSourceSpec(
        name="MIMIC-IV (SCD ICD-10 D57.* subset)",
        modality="clinical",
        url="https://physionet.org/content/mimiciv/",
        access="dua-required",
        citation="Johnson2023MIMICIV",
        notes="ICU and ED encounters; query D57.0/D57.1/D57.4 ICD-10 codes "
        "to extract SCD admissions, labs and vitals time series.",
        tags=("icu", "labs", "temporal"),
    ),
    DataSourceSpec(
        name="UK Biobank SCD subset",
        modality="multimodal",
        url="https://www.ukbiobank.ac.uk/",
        access="dua-required",
        citation="UKBiobank2015",
        notes="ICD-10 D57.* coded participants with imaging, biochemistry, "
        "genotype array and longitudinal follow-up.",
        tags=("biobank", "longitudinal"),
    ),
    DataSourceSpec(
        name="erythrocytesIDB - Peripheral blood smear sickle cells",
        modality="imaging",
        url="http://erythrocytesidb.uib.es/",
        access="open",
        citation="GonzalezHidalgo2015",
        notes="Annotated peripheral blood-smear images of normal, sickle and "
        "other deformed erythrocytes; classic SCD imaging benchmark.",
        tags=("microscopy", "benchmark"),
    ),
    DataSourceSpec(
        name="Kaggle Red Blood Cell (RBC) Sickle Dataset",
        modality="imaging",
        url="https://www.kaggle.com/datasets/sandeshb/sickle-cell",
        access="open",
        citation="KaggleSickleRBC",
        notes="Public Kaggle mirror of sickle vs normal RBC microscopy "
        "patches suitable for CNN benchmarks.",
        tags=("microscopy",),
    ),
    DataSourceSpec(
        name="ClinVar - HBB variants",
        modality="genomic",
        url="https://www.ncbi.nlm.nih.gov/clinvar/?term=HBB%5Bgene%5D",
        access="open",
        citation="ClinVar2020",
        notes="Curated HBB pathogenic variants underlying HbS/HbC/Hb beta-thalassemia.",
        tags=("variants", "HBB"),
    ),
    DataSourceSpec(
        name="gnomAD v4 - HBB allele frequencies",
        modality="genomic",
        url="https://gnomad.broadinstitute.org/gene/ENSG00000244734",
        access="open",
        citation="Karczewski2020",
        notes="Population allele frequencies for HBB variants.",
        tags=("population-genetics",),
    ),
    DataSourceSpec(
        name="NHLBI TOPMed - SCD-relevant cohorts",
        modality="genomic",
        url="https://topmed.nhlbi.nih.gov/",
        access="dua-required",
        citation="TOPMed2021",
        notes="Whole-genome sequencing across SCD-relevant cohorts (e.g. "
        "WGHS, JHS, BAGS).",
        tags=("wgs",),
    ),
    DataSourceSpec(
        name="WHO Sickle Cell Disease Country Profiles",
        modality="clinical",
        url="https://www.who.int/health-topics/sickle-cell-disease",
        access="open",
        citation="WHO2023SCD",
        notes="Aggregate prevalence and mortality statistics by country / region.",
        tags=("epidemiology",),
    ),
    DataSourceSpec(
        name="GBD 2021 - Sickle Cell Disorders",
        modality="clinical",
        url="https://vizhub.healthdata.org/gbd-results/",
        access="open",
        citation="GBD2021SCD",
        notes="Global Burden of Disease estimates of SCD incidence, mortality, "
        "DALYs by age and region.",
        tags=("epidemiology", "burden"),
    ),
]


def list_sources(modality: str | None = None,
                 access: str | None = None) -> List[DataSourceSpec]:
    """Filter the registry by modality and/or access tier."""
    out: Iterable[DataSourceSpec] = PUBLIC_SCD_DATASETS
    if modality is not None:
        out = (s for s in out if s.modality == modality)
    if access is not None:
        out = (s for s in out if s.access == access)
    return list(out)


def to_markdown_table() -> str:
    rows = ["| Dataset | Modality | Access | Citation | Notes |",
            "|---|---|---|---|---|"]
    for s in PUBLIC_SCD_DATASETS:
        rows.append(f"| [{s.name}]({s.url}) | {s.modality} | {s.access} | "
                    f"{s.citation} | {s.notes} |")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    print(to_markdown_table())
