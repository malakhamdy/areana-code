"""Helpers for safe logging/export of sensitive identity results."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .normalization import mask_national_id


def redacted_result(payload: dict[str, Any]) -> dict[str, Any]:
    safe = deepcopy(payload)
    for section in ("fields", "derived"):
        national = safe.get(section, {}).get("national_id")
        if national:
            for key in ("raw", "normalized"):
                if national.get(key):
                    national[key] = mask_national_id(national[key])
            for candidate in national.get("candidates", []):
                for key in ("raw", "normalized"):
                    if candidate.get(key):
                        candidate[key] = mask_national_id(candidate[key])
    debug = safe.get("debug", {})
    if "ocr_candidates" in debug:
        nid_candidates = debug["ocr_candidates"].get("national_id", [])
        for candidate in nid_candidates:
            for key in ("raw", "normalized"):
                if candidate.get(key):
                    candidate[key] = mask_national_id(candidate[key])
    return safe
