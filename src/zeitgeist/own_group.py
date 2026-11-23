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


KEY = "own_group"
# What the private run records when the rows carry several embedding models, or none.
_NO_SINGLE_MODEL = frozenset({"mixed", "unknown"})


def _field(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _measure(*, word: Any, gloss: Any, chosen_cluster_id: Any, clusters: Sequence[Any], embedding_model: Any,
             genai_client: Any) -> Dict[str, Any]:
    from . import run_record
    from .collectors import embed_missing_items
    from .spend_guard import SpendBlocked

    groups = list(clusters or ())[:run_record.MAX_GROUPS]
    chosen = None if chosen_cluster_id is None else str(chosen_cluster_id)
    own = next((group for group in groups if str(_field(group, "cluster_id")) == chosen), None)
    if own is None:
        return _skipped("unknown_candidate")
    own_items = list(_field(own, "items") or ())
    others = [list(_field(group, "items") or ()) for group in groups if group is not own]
    # Checked before the paid call: a candidate the arithmetic cannot use is not worth one.
    if not own_items:
        return _skipped("no_items")
    if not _unit_vectors(own_items):
        return _skipped("no_vector")
    model = str(embedding_model or "")
    if not model or model in _NO_SINGLE_MODEL:
        return _skipped("embedding_model")
    if not word or not gloss:
        return _skipped("no_text")
    if genai_client is None:
        return _skipped("no_client")
    try:
        # The collection step's own request: one text, the night's model, 768 dimensions, verified vectors only.
        [embedded] = embed_missing_items(
            [{"url_hash": KEY, "title": "%s. %s" % (word, gloss)}], {},
            genai_client=genai_client, model=model, require_live=True,
        )
    except SpendBlocked:
        return _skipped("refused")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return _skipped("failed")
    return own_group_numbers(_extract_unit_embedding_f32(embedded), own_items, others)


def run_own_group_check(
    store: Any, run_id: str, *, word: Any, gloss: Any, chosen_cluster_id: Any, clusters: Sequence[Any],
    embedding_model: Any, genai_client: Any,
) -> bool:
    """Embed "word. gloss", measure it against the night's first five candidates, store it under tests["own_group"].

    True only when numbers were computed and stored. Anything that stops the measure is stored as
    {"status": "skipped", "reason": ...}. Without a stored record nothing is embedded or written.
    Never raises but KeyboardInterrupt and SystemExit; prints and logs nothing.
    """
    from . import run_record

    try:
        record = store.load_run_record(run_id)
        if not isinstance(record, dict) or not isinstance(record.get("tests"), dict):
            return False
        numbers = _measure(word=word, gloss=gloss, chosen_cluster_id=chosen_cluster_id, clusters=clusters,
                           embedding_model=embedding_model, genai_client=genai_client)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        numbers = _skipped("failed")
    try:
        stored = run_record.add_test_numbers(store, run_id, KEY, numbers)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False
    return bool(stored) and numbers.get("status") == "ok"
