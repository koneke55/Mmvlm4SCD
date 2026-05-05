"""erythrocytesIDB loader (open).

Phase 1 priority. Once implemented, returns ``PatientRecord`` instances
with an ``Imaging`` block populated by a ResNet18/ConvNeXt-Tiny
embedding of each patient's slide(s).

Source: http://erythrocytesidb.uib.es/
License: CC-BY (verify in the dataset README at download time).
"""

from __future__ import annotations

from .base import StubLoader


class ErythrocytesIDBLoader(StubLoader):
    source = "erythrocytes_idb"
    access = "open"
    todo = ("download the erythrocytesIDB images, run the Phase-1 CNN "
            "encoder, and emit one PatientRecord per image-set with "
            "an `imaging.image_embedding` of shape (512,)")
