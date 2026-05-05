"""Unit tests for real-data loader stubs.

Until each cohort's access is configured, every loader must raise a
clear ``DataAccessError`` rather than silently returning empty data or,
worse, attempting an unauthenticated download.
"""

import pytest

from mmvlm4scd.data.loaders import (
    AIIMSDelhiLoader, CONSALoader, ClinVarHBBLoader, CureSCDLoader,
    DataAccessError, DbGaPSCDICLoader, DbGaPWalkPhasstLoader,
    ErythrocytesIDBLoader, GEOGSE53441Loader, GhanaKATHKorleBuLoader,
    GnomADHBBLoader, H3AfricaSCDLoader, ICMRNIRTHJabalpurLoader,
    KaggleSickleRBCLoader, LUTHNigeriaLoader, LokBiradariHemalkasaLoader,
    MGMIndoreLoader, MalariaGENHBBLoader, MimicIVSCDLoader,
    MuhimbiliMSCLoader, NSCAEMIndiaLoader, SCSSriLankaLoader,
    SPARCORegistryLoader, TOPMedSCDLoader, UCHIbadanLoader,
    UKBiobankSCDLoader,
)

OPEN_GLOBAL_LOADERS = [
    ErythrocytesIDBLoader, KaggleSickleRBCLoader, GEOGSE53441Loader,
    ClinVarHBBLoader, GnomADHBBLoader,
]
NORTH_AMERICA_EUROPE_LOADERS = [
    MimicIVSCDLoader, DbGaPSCDICLoader, DbGaPWalkPhasstLoader,
    UKBiobankSCDLoader, TOPMedSCDLoader, CureSCDLoader,
]
AFRICA_LOADERS = [
    SPARCORegistryLoader, MuhimbiliMSCLoader, CONSALoader,
    LUTHNigeriaLoader, UCHIbadanLoader, GhanaKATHKorleBuLoader,
    MalariaGENHBBLoader, H3AfricaSCDLoader,
]
SOUTH_ASIA_LOADERS = [
    NSCAEMIndiaLoader, ICMRNIRTHJabalpurLoader, AIIMSDelhiLoader,
    LokBiradariHemalkasaLoader, MGMIndoreLoader, SCSSriLankaLoader,
]
ALL_LOADERS = (OPEN_GLOBAL_LOADERS + NORTH_AMERICA_EUROPE_LOADERS
               + AFRICA_LOADERS + SOUTH_ASIA_LOADERS)


@pytest.mark.parametrize("cls", ALL_LOADERS)
def test_loader_raises_data_access_error_until_configured(cls, tmp_path):
    loader = cls(root=tmp_path, project_salt="x" * 16)
    with pytest.raises(DataAccessError):
        loader.load()


@pytest.mark.parametrize("cls", ALL_LOADERS)
def test_loader_declares_source_and_access(cls):
    assert cls.source != "synthetic"
    assert cls.access in {"open", "registered", "dua-required"}


@pytest.mark.parametrize("cls", ALL_LOADERS)
def test_loader_todo_is_actionable(cls):
    """Every stub must say what is left to do for whoever picks it up."""
    assert isinstance(cls.todo, str) and len(cls.todo) >= 20


def test_dbgap_loaders_are_dua_required():
    assert DbGaPSCDICLoader.access == "dua-required"
    assert DbGaPWalkPhasstLoader.access == "dua-required"
    assert TOPMedSCDLoader.access == "dua-required"
    assert UKBiobankSCDLoader.access == "dua-required"


def test_open_loaders_are_actually_open():
    for cls in OPEN_GLOBAL_LOADERS + [MalariaGENHBBLoader]:
        assert cls.access == "open"


def test_mimic_and_curescd_are_registered():
    assert MimicIVSCDLoader.access == "registered"
    assert CureSCDLoader.access == "registered"


def test_africa_loader_set_covers_priority_sources():
    """The African branch must include SPARCO, Muhimbili, CONSA and at
    least one Nigerian and one Ghanaian site."""
    sources = {cls.source for cls in AFRICA_LOADERS}
    assert "sparco" in sources
    assert "muhimbili_msc" in sources
    assert "consa" in sources
    nigeria = sources & {"luth_nigeria", "uch_ibadan"}
    ghana = sources & {"ghana_kath_korlebu"}
    assert nigeria, "expected at least one Nigerian-site loader"
    assert ghana, "expected at least one Ghanaian-site loader"


def test_south_asia_loader_set_covers_priority_sources():
    """The South Asia branch must include NSCAEM, an ICMR/NIRTH cohort,
    an AIIMS cohort and at least one community-based Indian cohort."""
    sources = {cls.source for cls in SOUTH_ASIA_LOADERS}
    assert "nscaem_india" in sources
    assert "icmr_nirth_jabalpur" in sources
    assert "aiims_delhi" in sources
    community = sources & {"lokbiradari_hemalkasa"}
    assert community, "expected at least one community-clinic loader"


def test_geographic_loader_balance():
    """Africa + South Asia loaders together must outnumber the
    North-America + Europe reference set."""
    assert (len(AFRICA_LOADERS) + len(SOUTH_ASIA_LOADERS)
            >= len(NORTH_AMERICA_EUROPE_LOADERS))
