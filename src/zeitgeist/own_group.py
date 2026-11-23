"""The own-group check: whether the word and gloss sit closer to their own candidate than to the night's others.

Arithmetic on unit vectors, so a word that fits any candidate shows up as a number. It decides nothing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .methodology import _extract_unit_embedding_f32

SWITCH_ENV = "ZEITGEIST_OWN_GROUP"


def own_group_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH_ENV) == "1"


def _skipped(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def _unit_vectors(items: Sequence[Any]) -> List[np.ndarray]:
    """Each item's unit vector, read as understanding._vectors reads it; items without one are left out."""
    vectors = []
    for item in items:
        try:
            vector = np.asarray(_extract_unit_embedding_f32(item), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if vector.size and np.all(np.isfinite(vector)) and float(np.linalg.norm(vector)) > 1e-12:
            vectors.append(vector)
    return vectors


def own_group_numbers(
    text_vector: Any, own_items: Sequence[Any], other_candidates: Sequence[Sequence[Any]],
) -> Dict[str, Any]:
    """The mean cosine of the text to its own candidate's items and to each other candidate's, and the margin.

    `margin` is `own` minus the highest of `others`, null when no other candidate has a vector. An other
    candidate whose items carry no vector has no entry. Pure: no I/O, no logging.
    """
    own_items = list(own_items or ())
    if not own_items:
        return _skipped("no_items")
    text = np.asarray(text_vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(text)) if text.size else 0.0
    if not text.size or not np.all(np.isfinite(text)) or not norm > 1e-12:
        return _skipped("no_vector")
    text = text / norm
    own = _unit_vectors(own_items)
    others = [_unit_vectors(list(items or ())) for items in other_candidates or ()]
    if any(vector.shape[0] != text.shape[0] for vector in own + [v for group in others for v in group]):
        return _skipped("dimension")
    if not own:
        return _skipped("no_vector")

    def mean_cosine(vectors: List[np.ndarray]) -> float:
        return float(np.mean([float(vector @ text) for vector in vectors]))

    own_mean = mean_cosine(own)
    other_means = [mean_cosine(group) for group in others if group]
    return {
        "status": "ok",
        "own": own_mean,
        "others": other_means,
        "margin": own_mean - max(other_means) if other_means else None,
        "n_own": len(own),
    }
