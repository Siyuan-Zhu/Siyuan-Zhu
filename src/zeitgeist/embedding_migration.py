"""Embedding-2 migration: calibration percentiles, the cluster gate, the private calibration document and the run plan.

The calibration document holds only numbers, an ISO date, a status word and the two
public model names. Nothing here prints or stores item text, urls, hashes or sources.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import collectors, spend_guard
from .methodology import (
    ScoredCluster,
    _extract_unit_embedding_f32,
    _get_item_hash,
    is_captionless,
    score_and_cluster_signals,
)

EMBED_001: str = collectors.EMBEDDING_MODEL
EMBED_2: str = spend_guard.EMBEDDING_2_MODEL

# The three cluster cutoffs, then the near-copy cutoff.
LEGACY_CUTOFFS: Tuple[float, float, float, float] = (0.78, 0.72, 0.68, 0.85)

MIN_CALIBRATION_ITEMS = 100
GATE_MIN_MULTI_SOURCE = 2
GATE_MAX_LARGEST_SHARE = 0.35
GATE_MIN_SURVIVORS = 3
CALIBRATION_SCHEMA_VERSION = 1

CAL_NONE = 0
CAL_ACCEPTED = 1
CAL_REJECTED = 2
CAL_INVALID = 3
CAL_UNREADABLE = 4
CAL_UNSUPPORTED = 5

CALIBRATION_DOCUMENT = "embedding_calibration"

_STATUS_CODES = {"accepted": CAL_ACCEPTED, "rejected": CAL_REJECTED}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}", re.ASCII)
_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "date",
        "item_count",
        "from_model",
        "to_model",
        "percentiles",
        "multi_source_clusters",
        "largest_share",
        "top5_survivors",
    }
)


def pairwise_similarities(items: Sequence[Any]) -> np.ndarray:
    """Strict upper triangle (i < j) of the cosine matrix of the items' unit vectors."""
    if len(items) < 2:
        return np.empty(0, dtype=np.float64)
    vectors = np.stack([_extract_unit_embedding_f32(item) for item in items]).astype(np.float64)
    rows, cols = np.triu_indices(len(items), k=1)
    return np.clip((vectors @ vectors.T)[rows, cols], -1.0, 1.0)


def percentile_ranks(
    sims: np.ndarray, cutoffs: Sequence[float] = LEGACY_CUTOFFS
) -> Tuple[float, ...]:
    """For each cutoff, the fraction of similarities strictly below it."""
    return tuple(round(float(np.count_nonzero(sims < c)) / sims.size, 6) for c in cutoffs)


def cutoffs_at(sims: np.ndarray, percentiles: Sequence[float]) -> Tuple[float, ...]:
    """The similarity at each percentile; a percentile of 1.0 maps above every pair."""
    return tuple(
        float(np.quantile(sims, p, method="linear")) if p < 1.0 else 1.0 + 1e-6
        for p in percentiles
    )


def cluster_shape(clusters: Sequence[ScoredCluster]) -> Tuple[int, float]:
    """(clusters with two or more sources, largest cluster's share of all clustered items)."""
    multi_source = sum(1 for c in clusters if c.unique_sources_count >= 2)
    total = sum(len(c.items) for c in clusters)
    if total == 0:
        return multi_source, 0.0
    return multi_source, max(len(c.items) for c in clusters) / total


@dataclass(frozen=True)
class GateReport:
    multi_source: int
    largest_share: float
    top5_survivors: int
    passed: bool


def gate_report(
    clusters_001: Sequence[ScoredCluster], clusters_e2: Sequence[ScoredCluster]
) -> GateReport:
    multi_source, largest_share = cluster_shape(clusters_e2)
    e2_hashes = [{_get_item_hash(item) for item in c.items} for c in clusters_e2]
    survivors = 0
    for cluster in clusters_001[:5]:
        hashes = [_get_item_hash(item) for item in cluster.items]
        if any(2 * sum(h in held for h in hashes) > len(hashes) for held in e2_hashes):
            survivors += 1
    passed = (
        multi_source >= GATE_MIN_MULTI_SOURCE
        and largest_share < GATE_MAX_LARGEST_SHARE
        and survivors >= GATE_MIN_SURVIVORS
    )
    return GateReport(multi_source, largest_share, survivors, passed)


@dataclass(frozen=True)
class Calibration:
    status: str
    date: str
    item_count: int
    percentiles: Tuple[float, float, float, float]
    multi_source: int
    largest_share: float
    top5_survivors: int


def calibration_document(cal: Calibration) -> dict:
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": cal.status,
        "date": cal.date,
        "item_count": int(cal.item_count),
        "from_model": EMBED_001,
        "to_model": EMBED_2,
        "percentiles": [float(p) for p in cal.percentiles],
        "multi_source_clusters": int(cal.multi_source),
        "largest_share": float(cal.largest_share),
        "top5_survivors": int(cal.top5_survivors),
    }


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_unit_float(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value) and 0.0 <= value <= 1.0


def _parse_document(document: Any) -> Tuple[int, Optional[Calibration]]:
    if not isinstance(document, dict) or set(document) != _DOCUMENT_KEYS:
        return CAL_INVALID, None
    schema = document["schema_version"]
    status = document["status"]
    date = document["date"]
    percentiles = document["percentiles"]
    counts = (document["item_count"], document["multi_source_clusters"], document["top5_survivors"])
    if (
        isinstance(schema, bool)
        or not isinstance(schema, int)
        or schema != CALIBRATION_SCHEMA_VERSION
        or not isinstance(status, str)
        or status not in _STATUS_CODES
        or not isinstance(date, str)
        or not _DATE_RE.fullmatch(date)
        or not all(_is_count(c) for c in counts)
        or counts[0] < MIN_CALIBRATION_ITEMS
        or document["from_model"] != EMBED_001
        or document["to_model"] != EMBED_2
        or not _is_unit_float(document["largest_share"])
        or not isinstance(percentiles, list)
        or len(percentiles) != 4
        or not all(_is_unit_float(p) for p in percentiles)
    ):
        return CAL_INVALID, None
    p = tuple(float(v) for v in percentiles)
    if not p[3] >= p[0] >= p[1] >= p[2]:
        return CAL_INVALID, None
    cal = Calibration(
        status=status,
        date=date,
        item_count=counts[0],
        percentiles=p,
        multi_source=counts[1],
        largest_share=float(document["largest_share"]),
        top5_survivors=counts[2],
    )
    return _STATUS_CODES[status], cal


def parse_calibration(document: Any) -> Tuple[int, Optional[Calibration]]:
    """(state code, calibration); never raises."""
    if document is None:
        return CAL_NONE, None
    try:
        return _parse_document(document)
    except Exception:
        return CAL_INVALID, None


@dataclass
class EmbeddingRun:
    mode: str
    calibration_state: int
    percentiles: Optional[Tuple[float, float, float, float]] = None
    used: str = EMBED_001
    fallback: bool = False
    deferred: bool = False
    date_str: str = ""
    items: int = 0
    pending_rows: Optional[Dict[str, Dict[str, Any]]] = None
    pending_items: Optional[List[Any]] = None
    cutoffs: Optional[Tuple[float, ...]] = None
    multi_source: int = -1
    largest_share: Optional[float] = None
    gate: Optional[GateReport] = None
    write_failed: bool = False


def plan_embedding_run(store: Any, requested: str) -> Optional[EmbeddingRun]:
    """The run's embedding mode from the stored calibration; never raises, never writes."""
    if requested != EMBED_2:
        return None
    try:
        load = getattr(store, "load_calibration", None)
    except Exception:
        load = None
    if not callable(load):
        return EmbeddingRun(mode="legacy", calibration_state=CAL_UNSUPPORTED)
    try:
        document = load(CALIBRATION_DOCUMENT)
    except Exception:
        return EmbeddingRun(mode="legacy", calibration_state=CAL_UNREADABLE)
    state, cal = parse_calibration(document)
    if state == CAL_NONE:
        return EmbeddingRun(mode="switch_over", calibration_state=state)
    if state == CAL_ACCEPTED and cal is not None:
        return EmbeddingRun(mode="migrated", calibration_state=state, percentiles=cal.percentiles)
    return EmbeddingRun(mode="legacy", calibration_state=state)


def _joined(values: Optional[Sequence[float]], fmt: str) -> str:
    return ",".join(fmt % v for v in values) if values else "-1"


def _share(value: Optional[float]) -> str:
    return "%.3f" % value if value is not None else "-1"


def embedding_line(run: EmbeddingRun) -> str:
    gate = run.gate
    used = {EMBED_001: 1, EMBED_2: 2}.get(run.used, -1)
    return (
        "[embedding] used=%d switch_over=%d fallback=%d calibration=%d items=%d p=%s cut=%s "
        "multi_source=%d largest_share=%s gate_multi_source=%d gate_largest_share=%s "
        "top5_survivors=%d pass=%d write_failed=%d"
    ) % (
        used,
        1 if run.mode == "switch_over" else 0,
        int(bool(run.fallback)),
        run.calibration_state,
        run.items,
        _joined(run.percentiles, "%.6f"),
        _joined(run.cutoffs, "%.4f"),
        run.multi_source,
        _share(run.largest_share),
        gate.multi_source if gate else -1,
        _share(gate.largest_share if gate else None),
        gate.top5_survivors if gate else -1,
        int(gate.passed) if gate else -1,
        int(bool(run.write_failed)),
    )


def _usable_similarities(items: Sequence[Any], exclude_captionless: bool = False) -> np.ndarray:
    """Pairwise similarities of the items that carry a finite, nonzero vector of the common size."""
    usable: List[Tuple[Any, int]] = []
    for item in items:
        if exclude_captionless and is_captionless(item):
            continue
        try:
            vec = _extract_unit_embedding_f32(item)
        except Exception:
            continue
        if vec.size and bool(np.all(np.isfinite(vec))) and float(np.linalg.norm(vec)) > 0.5:
            usable.append((item, vec.size))
    if len(usable) < 2:
        return np.empty(0, dtype=np.float64)
    dim = Counter(size for _, size in usable).most_common(1)[0][0]
    return pairwise_similarities([item for item, size in usable if size == dim])


def cluster_for_run(
    items: Sequence[Any],
    run: EmbeddingRun,
    *,
    store: Any,
    now_utc: Any,
    interim_grouping: bool = False,
    **score_kwargs: Any,
) -> List[ScoredCluster]:
    """Cluster with the model the run settled on; on a switch-over night, gate and record the result.

    A failure in the cutoff or gate maths never stops the run: it falls back to today's fixed
    cutoffs, writes nothing and clears what it had set. A SpendBlocked still propagates.

    interim_grouping leaves captionless posts out of every grouping and of the percentile maths, and
    runs only the calibrated embedding-2 cutoffs to their loosest tier.
    """
    if interim_grouping:
        score_kwargs = {**score_kwargs, "exclude_captionless": True}
    before = (run.percentiles, run.cutoffs, run.gate)
    try:
        clusters = _clusters_for_run(items, run, store, now_utc, score_kwargs, interim_grouping)
    except spend_guard.SpendBlocked:
        raise
    except Exception:
        run.percentiles, run.cutoffs, run.gate = before
        run.fallback = True
        clusters = score_and_cluster_signals(items, now_utc=now_utc, **score_kwargs)
    run.multi_source, run.largest_share = cluster_shape(clusters)
    return clusters


def _clusters_for_run(
    items: Sequence[Any], run: EmbeddingRun, store: Any, now_utc: Any, score_kwargs: Dict[str, Any],
    interim_grouping: bool = False,
) -> List[ScoredCluster]:
    calibrated_kwargs = {**score_kwargs, "use_loosest_cutoff": True} if interim_grouping else score_kwargs

    def score(rows: Sequence[Any], cut: Optional[Tuple[float, ...]] = None) -> List[ScoredCluster]:
        if cut is None:
            return score_and_cluster_signals(rows, now_utc=now_utc, **score_kwargs)
        return score_and_cluster_signals(
            rows, now_utc=now_utc, cutoffs=cut[:3], near_duplicate_threshold=cut[3], **calibrated_kwargs
        )

    if run.fallback or run.deferred or run.mode not in ("migrated", "switch_over"):
        return score(items)
    sims = _usable_similarities(items, interim_grouping)
    if run.mode == "migrated":
        if sims.size == 0 or run.percentiles is None:
            return score(items)
        run.cutoffs = cutoffs_at(sims, run.percentiles)
        return score(items, run.cutoffs)
    pending_sims = _usable_similarities(run.pending_items or (), interim_grouping)
    if sims.size == 0 or pending_sims.size == 0:
        # Too few vectors to calibrate: no decision tonight, no write.
        run.deferred = True
        return score(items)
    clusters_001 = score(items)
    percentiles = percentile_ranks(sims)
    cut = cutoffs_at(pending_sims, percentiles)
    clusters_e2 = score(run.pending_items, cut)
    gate = gate_report(clusters_001, clusters_e2)
    run.percentiles, run.cutoffs, run.gate = percentiles, cut, gate
    status = "accepted" if gate.passed else "rejected"
    document = calibration_document(
        Calibration(status, run.date_str, run.items, percentiles, gate.multi_source,
                    gate.largest_share, gate.top5_survivors)
    )
    try:
        store.save_calibration(CALIBRATION_DOCUMENT, document)
    except Exception:
        run.write_failed = True
        return clusters_001
    if not gate.passed:
        return clusters_001
    try:
        store.save_ingestion_cache_day(run.date_str, run.pending_rows)
    except Exception:
        run.write_failed = True
    run.used = EMBED_2
    return clusters_e2
