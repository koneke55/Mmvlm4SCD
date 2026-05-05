"""Real-data loaders for Mmvlm4SCD.

Each module in this package implements one source from the
``data.registry`` catalogue. Loaders return ``List[PatientRecord]``;
they do not write to disk and they do not silently impute. PHI-handling
is enforced via ``data.deidentification``.

Phase 1 (open) sources are stubbed and ready for implementation; Phase
2 (registered) and Phase 3 (DUA-required) loaders raise a clear
``DataAccessError`` until the corresponding access has been granted
and configured locally.

Usage::

    from mmvlm4scd.data.loaders import (
        ErythrocytesIDBLoader, MimicIVSCDLoader, DataAccessError,
    )

    records = ErythrocytesIDBLoader(root="data/raw/erythrocytes_idb",
                                     project_salt=os.environ["MMVLM_SALT"]
                                     ).load()
"""

from __future__ import annotations

from .base import BaseLoader, DataAccessError, LoaderConfig

# Open / global foundations
from .erythrocytes_idb import ErythrocytesIDBLoader
from .kaggle_sickle_rbc import KaggleSickleRBCLoader
from .geo_gse53441 import GEOGSE53441Loader
from .clinvar_hbb import ClinVarHBBLoader
from .gnomad_hbb import GnomADHBBLoader

# North America / Europe reference cohorts
from .mimiciv_scd import MimicIVSCDLoader
from .dbgap_phs001514 import DbGaPSCDICLoader
from .dbgap_phs001599 import DbGaPWalkPhasstLoader
from .uk_biobank_scd import UKBiobankSCDLoader
from .topmed_scd import TOPMedSCDLoader
from .cure_scd import CureSCDLoader

# Sub-Saharan Africa cohorts and resources (priority)
from .sparco import SPARCORegistryLoader
from .muhimbili_msc import MuhimbiliMSCLoader
from .consa import CONSALoader
from .luth_nigeria import LUTHNigeriaLoader
from .uch_ibadan import UCHIbadanLoader
from .ghana_kath_korlebu import GhanaKATHKorleBuLoader
from .malariagen_hbb import MalariaGENHBBLoader
from .h3africa_scd import H3AfricaSCDLoader

# South Asia cohorts and resources (priority)
from .nscaem_india import NSCAEMIndiaLoader
from .icmr_nirth_jabalpur import ICMRNIRTHJabalpurLoader
from .aiims_delhi import AIIMSDelhiLoader
from .lokbiradari_hemalkasa import LokBiradariHemalkasaLoader
from .mgm_indore import MGMIndoreLoader
from .scs_sri_lanka import SCSSriLankaLoader

__all__ = [
    "BaseLoader",
    "DataAccessError",
    "LoaderConfig",
    # Open / global
    "ErythrocytesIDBLoader",
    "KaggleSickleRBCLoader",
    "GEOGSE53441Loader",
    "ClinVarHBBLoader",
    "GnomADHBBLoader",
    # North America / Europe
    "MimicIVSCDLoader",
    "DbGaPSCDICLoader",
    "DbGaPWalkPhasstLoader",
    "UKBiobankSCDLoader",
    "TOPMedSCDLoader",
    "CureSCDLoader",
    # Sub-Saharan Africa
    "SPARCORegistryLoader",
    "MuhimbiliMSCLoader",
    "CONSALoader",
    "LUTHNigeriaLoader",
    "UCHIbadanLoader",
    "GhanaKATHKorleBuLoader",
    "MalariaGENHBBLoader",
    "H3AfricaSCDLoader",
    # South Asia
    "NSCAEMIndiaLoader",
    "ICMRNIRTHJabalpurLoader",
    "AIIMSDelhiLoader",
    "LokBiradariHemalkasaLoader",
    "MGMIndoreLoader",
    "SCSSriLankaLoader",
]
