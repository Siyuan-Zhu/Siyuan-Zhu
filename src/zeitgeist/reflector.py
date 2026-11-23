"""The reflector: a group that keeps standing in the window without leading, and whether it has been named.

This part finds the standing group while the night's groups are in memory and stores what the
reflector needs from the run on its record. Arithmetic only: it calls no provider and decides nothing.
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .methodology import _extract_unit_embedding_f32, _get_item_field

SWITCH_ENV = "ZEITGEIST_REFLECTOR"
INPUTS_KEY = "reflector_inputs"
MIN_DAYS = 5


def reflector_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH_ENV) == "1"


def _utc_date(value: Any) -> Optional[date]:
    """The UTC date of a datetime or an ISO string; a time without a zone is read as UTC. None when unreadable."""
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    return None


def _items(cluster: Any) -> List[Any]:
    return list(getattr(cluster, "items", None) or ())


def _distinct_days(cluster: Any) -> int:
    days = {_utc_date(_get_item_field(item, "published_at")) for item in _items(cluster)}
    days.discard(None)
    return len(days)


def _score(cluster: Any) -> float:
    try:
        score = float(getattr(cluster, "commission_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def standing_groups(clusters: Sequence[Any], *, min_days: int = MIN_DAYS) -> list:
    """Every cluster after the first whose items fall on at least `min_days` distinct UTC dates.

    Most distinct days first, then the higher `commission_score`, then the smaller `cluster_id`.
    Items without a readable `published_at` are ignored. Pure.
    """
    counted = [(cluster, _distinct_days(cluster)) for cluster in list(clusters or ())[1:]]
    qualifying = [(cluster, days) for cluster, days in counted if days >= int(min_days)]
    qualifying.sort(key=lambda pair: (-pair[1], -_score(pair[0]), str(getattr(pair[0], "cluster_id", ""))))
    return [cluster for cluster, _ in qualifying]


def centre(cluster: Any) -> Optional[List[float]]:
    """The unit mean of the items' unit vectors, read as understanding._vectors reads them; None without any."""
    vectors = []
    for item in _items(cluster):
        try:
            vector = np.asarray(_extract_unit_embedding_f32(item), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if vector.size and np.all(np.isfinite(vector)) and float(np.linalg.norm(vector)) > 1e-12:
            vectors.append(vector)
    if not vectors:
        return None
    dims = [vector.shape[0] for vector in vectors]
    dim = max(set(dims), key=dims.count)
    mean = np.mean([vector for vector in vectors if vector.shape[0] == dim], axis=0)
    norm = float(np.linalg.norm(mean))
    if not norm > 1e-12:
        return None
    return [float(value) for value in mean / norm]


def reflector_inputs(clusters: Sequence[Any], chosen_cluster_id: Any, embedding_model: Any) -> Dict[str, Any]:
    """The standing group and the chosen candidate as the reflector needs them: block, centre, item hashes.

    `standing` is the first standing group that is not the chosen one; `chosen` is the cluster among the
    first five whose id is `chosen_cluster_id`. Either is None when there is none. Pure.
    """
    from . import run_record
    from .understanding import _render_cluster

    ranked = list(clusters or ())
    chosen_id = None if chosen_cluster_id is None else str(chosen_cluster_id)

    def cluster_id(cluster: Any) -> str:
        return str(getattr(cluster, "cluster_id", ""))

    standing = next((cluster for cluster in standing_groups(ranked) if cluster_id(cluster) != chosen_id), None)
    chosen = next((cluster for cluster in ranked[:run_record.MAX_GROUPS] if cluster_id(cluster) == chosen_id), None)
    standing_entry = None
    if standing is not None:
        standing_entry = {
            "cluster_id": cluster_id(standing),
            "rank": next(rank for rank, cluster in enumerate(ranked, 1) if cluster is standing),
            "days": _distinct_days(standing),
            "item_hashes": [str(value) for value in (_get_item_field(item, "url_hash") for item in _items(standing))
                            if value],
            "block": _render_cluster(standing),
            "centre": centre(standing),
        }
    chosen_entry = None
    if chosen is not None:
        chosen_entry = {"cluster_id": cluster_id(chosen), "block": _render_cluster(chosen), "centre": centre(chosen)}
    return {"standing": standing_entry, "chosen": chosen_entry, "embedding_model": str(embedding_model or "")}


def store_reflector_inputs(
    store: Any, run_id: str, clusters: Sequence[Any], chosen_cluster_id: Any, embedding_model: Any,
) -> bool:
    """Store `reflector_inputs` under tests["reflector_inputs"] of a stored record; True when it was stored.

    Without a stored record nothing is written. Never raises but KeyboardInterrupt and SystemExit;
    prints and logs nothing.
    """
    from . import run_record

    try:
        inputs = reflector_inputs(clusters, chosen_cluster_id, embedding_model)
        return bool(run_record.add_test_numbers(store, run_id, INPUTS_KEY, inputs))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False
