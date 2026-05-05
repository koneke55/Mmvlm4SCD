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
    region: str = "global"  # "africa" | "south_asia" | "north_america" | "europe" | "global"
    countries: tuple = field(default_factory=tuple)


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
        region="north_america",
        countries=("USA",),
    ),
    DataSourceSpec(
        name="dbGaP phs001514 - Sickle Cell Disease Implementation Consortium",
        modality="clinical",
        url="https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001514",
        access="dua-required",
        citation="dbGaP-phs001514",
        notes="SCDIC Registry: pain, transfusion, hydroxyurea adherence outcomes.",
        tags=("registry", "phenotype"),
        region="north_america",
        countries=("USA",),
    ),
    DataSourceSpec(
        name="dbGaP phs001599 - Walk-PHaSST",
        modality="clinical",
        url="https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001599",
        access="dua-required",
        citation="dbGaP-phs001599",
        notes="Pulmonary hypertension and SCD natural-history cohort.",
        tags=("survival", "hemodynamics"),
        region="north_america",
        countries=("USA",),
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
        region="north_america",
        countries=("USA",),
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
        region="europe",
        countries=("UK",),
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
        region="global",
    ),

    # ----------------------------------------------------------------- #
    #              Sub-Saharan Africa cohorts and resources             #
    # ----------------------------------------------------------------- #

    DataSourceSpec(
        name="SickleInAfrica / SPARCO Registry",
        modality="multimodal",
        url="https://sickleinafrica.org/",
        access="dua-required",
        citation="Makani2020SickleInAfrica",
        notes="Sickle Pan-African Research Consortium; harmonised electronic "
        "registry across Tanzania, Ghana, Nigeria (with Cameroon, Mali "
        "expanding); flagship African SCD cohort with longitudinal "
        "clinical, genomic and outcome data.",
        tags=("registry", "longitudinal", "h3africa"),
        region="africa",
        countries=("Tanzania", "Ghana", "Nigeria", "Cameroon", "Mali"),
    ),
    DataSourceSpec(
        name="Muhimbili Sickle Cohort (MSC), Tanzania",
        modality="multimodal",
        url="https://www.muhas.ac.tz/",
        access="registered",
        citation="Makani2011MuhimbiliCohort",
        notes="Largest single-site East African SCD cohort (>5,000 patients) "
        "at Muhimbili University of Health and Allied Sciences; rich "
        "longitudinal phenotype and biospecimen archive.",
        tags=("longitudinal", "biorepository", "east_africa"),
        region="africa",
        countries=("Tanzania",),
    ),
    DataSourceSpec(
        name="CONSA - Consortium on Newborn Screening in Africa",
        modality="clinical",
        url="https://consortiumonnewbornscreeninginafrica.org/",
        access="registered",
        citation="CONSA2020",
        notes="Newborn-screening + early-childhood follow-up across Liberia, "
        "Ghana, Tanzania, Uganda, DRC, Kenya, Madagascar, Zambia, Nigeria; "
        "primary source for under-5 SCD survival and intervention data.",
        tags=("newborn_screening", "pediatric", "survival"),
        region="africa",
        countries=("Liberia", "Ghana", "Tanzania", "Uganda", "DRC",
                   "Kenya", "Madagascar", "Zambia", "Nigeria"),
    ),
    DataSourceSpec(
        name="Lagos University Teaching Hospital (LUTH) SCD Clinic",
        modality="clinical",
        url="https://luth.gov.ng/",
        access="dua-required",
        citation="Akinyanju1989LUTH",
        notes="One of the longest-running SCD clinical cohorts in West Africa; "
        "Nigeria carries the largest absolute SCD birth burden globally.",
        tags=("west_africa", "longitudinal"),
        region="africa",
        countries=("Nigeria",),
    ),
    DataSourceSpec(
        name="University College Hospital (UCH) Ibadan SCD Cohort",
        modality="clinical",
        url="https://uch-ibadan.org.ng/",
        access="dua-required",
        citation="UCH-Ibadan-SCD",
        notes="Adult and pediatric SCD outpatient and inpatient records; "
        "supports HbF modifier and treatment-response studies.",
        tags=("west_africa", "adult_pediatric"),
        region="africa",
        countries=("Nigeria",),
    ),
    DataSourceSpec(
        name="Komfo Anokye / Korle-Bu SCD Programmes, Ghana",
        modality="clinical",
        url="https://kathhsp.org/",
        access="dua-required",
        citation="Ohene-Frempong2008Ghana",
        notes="Two flagship Ghanaian programmes integrating newborn "
        "screening, hydroxyurea provision and longitudinal follow-up; "
        "anchors of the SickleInAfrica West-Africa node.",
        tags=("newborn_screening", "hydroxyurea", "west_africa"),
        region="africa",
        countries=("Ghana",),
    ),
    DataSourceSpec(
        name="MalariaGEN HBB / globin variant data",
        modality="genomic",
        url="https://www.malariagen.net/",
        access="open",
        citation="MalariaGEN2014",
        notes="Open population genomics across West, East and Central African "
        "cohorts; provides HbS/HbC/HbE allele frequencies and HBB variant "
        "calls suitable for population-stratified PRS construction.",
        tags=("population_genetics", "hbs_haplotype"),
        region="africa",
        countries=("Nigeria", "Ghana", "Mali", "Tanzania", "Cameroon",
                   "Burkina_Faso", "Kenya"),
    ),
    DataSourceSpec(
        name="H3Africa SCD bioinformatics archive (H3ABioNet)",
        modality="genomic",
        url="https://h3africa.org/",
        access="dua-required",
        citation="H3Africa2014",
        notes="Pan-African genomics consortium; SCD-specific sub-projects "
        "include modifier-gene studies (BCL11A, HMIP-2, MYB) on African "
        "haplotype backgrounds.",
        tags=("modifier_genes", "h3africa"),
        region="africa",
        countries=("Nigeria", "Ghana", "Tanzania", "South_Africa",
                   "Uganda", "Cameroon"),
    ),

    # ----------------------------------------------------------------- #
    #                 South Asia cohorts and resources                  #
    # ----------------------------------------------------------------- #

    DataSourceSpec(
        name="National Sickle Cell Anaemia Elimination Mission (NSCAEM), India",
        modality="clinical",
        url="https://sickle.nhm.gov.in/",
        access="registered",
        citation="NSCAEM2023India",
        notes="Government-of-India 2023 mission targeting screening of ~70 M "
        "people in 17 SCD-endemic states (predominantly tribal populations); "
        "integrated district-level registries and point-of-care HemeChip / "
        "SickleSCAN data.",
        tags=("population_screening", "tribal", "policy"),
        region="south_asia",
        countries=("India",),
    ),
    DataSourceSpec(
        name="ICMR-NIRTH Jabalpur Tribal SCD Cohort",
        modality="clinical",
        url="https://nirth.res.in/",
        access="dua-required",
        citation="NIRTH-Jabalpur-SCD",
        notes="National Institute for Research in Tribal Health: longitudinal "
        "tribal-population SCD cohort across Madhya Pradesh and "
        "Chhattisgarh; central reference for the Indian Arab-Indian "
        "haplotype background.",
        tags=("tribal", "central_india", "arab_indian_haplotype"),
        region="south_asia",
        countries=("India",),
    ),
    DataSourceSpec(
        name="AIIMS New Delhi SCD Registry",
        modality="multimodal",
        url="https://www.aiims.edu/",
        access="dua-required",
        citation="AIIMS-SCD-Registry",
        notes="Tertiary-centre SCD registry with paired clinical, biochemical "
        "and HBB/HBA genotype data; supports adult-onset complication "
        "studies.",
        tags=("tertiary_centre", "hba_genotype"),
        region="south_asia",
        countries=("India",),
    ),
    DataSourceSpec(
        name="Lok Biradari Prakalp (Hemalkasa) SCD Cohort",
        modality="clinical",
        url="https://lokbiradariprakalp.org/",
        access="dua-required",
        citation="LBPHemalkasa-SCD",
        notes="Community-based SCD care in the Madia Gond tribal population "
        "of Gadchiroli (Maharashtra); rare longitudinal record of "
        "low-resource SCD natural history.",
        tags=("tribal", "community_clinic", "low_resource"),
        region="south_asia",
        countries=("India",),
    ),
    DataSourceSpec(
        name="MGM Medical College Indore SCD Programme",
        modality="clinical",
        url="https://www.mgmmcindore.in/",
        access="dua-required",
        citation="MGM-Indore-SCD",
        notes="Western-India regional SCD programme covering screening, "
        "hydroxyurea provision and pregnancy outcomes; useful for "
        "treatment-effect heterogeneity studies.",
        tags=("treatment_response", "pregnancy", "west_india"),
        region="south_asia",
        countries=("India",),
    ),
    DataSourceSpec(
        name="Sickle Cell Society of Sri Lanka",
        modality="clinical",
        url="https://www.sicklecellsl.org/",
        access="registered",
        citation="SCSSL-Cohort",
        notes="Smaller but well-characterised Sri Lankan cohort; complements "
        "Indian registries with a distinct ancestry and care-system "
        "context.",
        tags=("sri_lanka",),
        region="south_asia",
        countries=("Sri_Lanka",),
    ),
]


def list_sources(modality: str | None = None,
                 access: str | None = None,
                 region: str | None = None,
                 country: str | None = None) -> List[DataSourceSpec]:
    """Filter the registry by modality, access tier, region and/or country."""
    out: Iterable[DataSourceSpec] = PUBLIC_SCD_DATASETS
    if modality is not None:
        out = (s for s in out if s.modality == modality)
    if access is not None:
        out = (s for s in out if s.access == access)
    if region is not None:
        out = (s for s in out if s.region == region)
    if country is not None:
        out = (s for s in out if country in s.countries)
    return list(out)


def coverage_by_region() -> dict[str, int]:
    """Count entries per geographic region. Useful for sanity-checking
    that the registry is not US/UK-centric."""
    out: dict[str, int] = {}
    for s in PUBLIC_SCD_DATASETS:
        out[s.region] = out.get(s.region, 0) + 1
    return out


def to_markdown_table() -> str:
    rows = ["| Dataset | Region | Modality | Access | Citation | Notes |",
            "|---|---|---|---|---|---|"]
    for s in PUBLIC_SCD_DATASETS:
        rows.append(f"| [{s.name}]({s.url}) | {s.region} | {s.modality} | "
                    f"{s.access} | {s.citation} | {s.notes} |")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    print(to_markdown_table())
