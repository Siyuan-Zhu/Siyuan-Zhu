"""The photo critic's two pass marks, matched per judge model from one private run.

Today's marks (0.70 meaning, 0.68 composite) were tuned for no model in particular. A private run scores the
same photos under Flash and under another judge; that judge's marks are set so the same number of photos
pass as pass under Flash at today's marks. The stored document holds numbers and a public model name only.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

DOCUMENT_NAME = "judge_calibration"
SCHEMA_VERSION = 1
DEFAULT_MARKS: Tuple[float, float] = (0.70, 0.68)
MARK_FLOOR = 0.55
MARK_CEILING = 0.85
MIN_PHOTOS = 9


def _round_down(value: float) -> float:
    # The epsilon keeps a score such as 0.29 (stored as 0.28999...) at 0.29.
    return math.floor(value * 100 + 1e-9) / 100


def _matched_mark(flash: Sequence[float], judge: Sequence[float], flash_mark: float) -> Optional[float]:
    passing = sum(score >= flash_mark for score in flash)
    ranked = sorted(judge, reverse=True)
    if passing and ranked[passing - 1] <= 0.0:
        # The judge scored fewer photos above zero than Flash passes: there is nothing to match.
        return None
    mark = ranked[0] + 0.01 if passing == 0 else ranked[passing - 1]
    return _round_down(min(MARK_CEILING, max(MARK_FLOOR, mark)))


def matched_marks(
    flash_scores: Sequence[Tuple[float, float]],
    judge_scores: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """(meaning, composite) marks that pass as many photos under the judge as pass under Flash today.

    Each tuple is (meaning, composite) for the same photo in the same order. None when fewer than
    MIN_PHOTOS pairs, when the two lists do not pair up, or when on either axis Flash passes k photos but the
    judge's k-th highest score is zero.
    """
    if len(flash_scores) != len(judge_scores) or len(flash_scores) < MIN_PHOTOS:
        return None
    marks = tuple(
        _matched_mark([row[axis] for row in flash_scores], [row[axis] for row in judge_scores], DEFAULT_MARKS[axis])
        for axis in (0, 1)
    )
    return None if None in marks else marks


def build_document(model: str, marks: Tuple[float, float], counts: Mapping[str, int], date_iso: str) -> dict:
    """The private calibration document: numbers and the public model name, no photo ids, queries or text."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model": str(model),
        "meaning_min": float(marks[0]),
        "composite_min": float(marks[1]),
        "photos": int(counts["photos"]),
        "flash_pass_meaning": int(counts["flash_pass_meaning"]),
        "flash_pass_composite": int(counts["flash_pass_composite"]),
        "date": str(date_iso),
    }


def _usable_mark(value: Any) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and MARK_FLOOR <= value <= MARK_CEILING)


def marks_for(model: str, document: Optional[Mapping[str, Any]]) -> Tuple[float, float]:
    """The document's marks when it is well formed, names `model` and both marks are in bounds; else today's.

    Never raises.
    """
    try:
        if not isinstance(document, Mapping) or not isinstance(model, str):
            return DEFAULT_MARKS
        version, photos = document.get("schema_version"), document.get("photos")
        if type(version) is not int or version != SCHEMA_VERSION or document.get("model") != model:
            return DEFAULT_MARKS
        if type(photos) is not int or photos < MIN_PHOTOS:
            return DEFAULT_MARKS
        meaning, composite = document.get("meaning_min"), document.get("composite_min")
        if not (_usable_mark(meaning) and _usable_mark(composite)):
            return DEFAULT_MARKS
        return float(meaning), float(composite)
    except Exception:
        return DEFAULT_MARKS


def load_document(store: Any) -> Optional[Mapping[str, Any]]:
    """The stored judge calibration, or None when it is absent or cannot be read. Never raises."""
    try:
        return store.load_calibration(DOCUMENT_NAME)
    except Exception:
        return None
