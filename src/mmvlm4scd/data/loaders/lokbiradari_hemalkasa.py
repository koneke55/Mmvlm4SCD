"""Lok Biradari Prakalp (Hemalkasa) Madia Gond SCD cohort loader.

Community-based SCD care in Gadchiroli (Maharashtra). Provides a rare
longitudinal record of low-resource SCD natural history without
hydroxyurea routinely available. Treat as the canonical
``CareSettingTier.COMMUNITY`` reference.
"""

from __future__ import annotations

from .base import StubLoader


class LokBiradariHemalkasaLoader(StubLoader):
    source = "lokbiradari_hemalkasa"
    access = "dua-required"
    todo = ("community-led MoU with the Lok Biradari Prakalp leadership; "
            "default CareSettingTier.COMMUNITY and "
            "HydroxyureaAccess.INTERMITTENT or .UNAVAILABLE; honour "
            "CARE Principles before any secondary derivation")
