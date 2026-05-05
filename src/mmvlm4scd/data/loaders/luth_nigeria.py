"""Lagos University Teaching Hospital SCD Clinic loader (DUA, Nigeria)."""

from __future__ import annotations

from .base import StubLoader


class LUTHNigeriaLoader(StubLoader):
    source = "luth_nigeria"
    access = "dua-required"
    todo = ("Lagos University / NHREC ethics; MoU with the LUTH "
            "haematology unit; Nigeria carries the largest absolute SCD "
            "birth burden globally so this cohort anchors the West "
            "African node")
