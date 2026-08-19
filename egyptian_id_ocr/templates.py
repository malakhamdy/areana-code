"""Versioned layouts in normalized canonical-card coordinates.

Regions are search zones, not final fixed crops. The localizer refines them using ink
and OCR-anchor evidence after perspective correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .schemas import DocumentSide


@dataclass(frozen=True)
class FieldTemplate:
    field: str
    search_region: tuple[float, float, float, float]
    expected_type: str
    padding: tuple[float, float, float, float]  # left, top, right, bottom, relative to card
    anchor_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CardTemplate:
    template_id: str
    side: DocumentSide
    fields: Mapping[str, FieldTemplate]
    notes: str


FRONT_V1 = CardTemplate(
    template_id="egypt_front_legacy_v1",
    side=DocumentSide.FRONT,
    fields={
        "portrait": FieldTemplate(
            "portrait", (0.012, 0.055, 0.270, 0.585), "image", (0, 0, 0, 0)
        ),
        "name": FieldTemplate(
            "name",
            (0.385, 0.205, 0.600, 0.285),
            "arabic_text",
            (0.008, 0.010, 0.012, 0.010),
            ("الاسم",),
        ),
        "address": FieldTemplate(
            "address",
            (0.445, 0.465, 0.545, 0.255),
            "arabic_multiline",
            (0.012, 0.015, 0.010, 0.015),
            ("العنوان",),
        ),
        "national_id": FieldTemplate(
            "national_id",
            (0.390, 0.755, 0.600, 0.205),
            "national_id",
            (0.010, 0.010, 0.008, 0.010),
            ("الرقم القومي",),
        ),
    },
    notes="Legacy/current front with portrait left, identity text right, NID along bottom.",
)

BACK_V1 = CardTemplate(
    template_id="egypt_back_generic_v1",
    side=DocumentSide.BACK,
    fields={
        "barcode": FieldTemplate(
            "barcode", (0.035, 0.665, 0.930, 0.300), "pdf417", (0.005, 0.005, 0.005, 0.005)
        ),
        "back_text": FieldTemplate(
            "back_text", (0.255, 0.130, 0.710, 0.510), "arabic_multiline", (0.01, 0.01, 0.01, 0.01)
        ),
    },
    notes="Generic back; barcode is decoded independently and front-only fields are not forced.",
)

TEMPLATE_REGISTRY = {FRONT_V1.template_id: FRONT_V1, BACK_V1.template_id: BACK_V1}


def template_for_side(side: DocumentSide) -> tuple[CardTemplate, float]:
    if side == DocumentSide.FRONT:
        return FRONT_V1, 0.88
    if side == DocumentSide.BACK:
        return BACK_V1, 0.72
    # Unknown images get no front extraction; generic back is the least assumptive layout.
    return BACK_V1, 0.25
