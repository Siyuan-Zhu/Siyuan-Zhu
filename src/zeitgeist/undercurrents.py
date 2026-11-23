"""Pure arithmetic for cross-realm undercurrents: realms, eligibility, centroids.

No model calls, no I/O, no environment reads. Every threshold is a count or
relative to the day's own similarities; ties break deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .methodology import _extract_unit_embedding_f32, _get_item_field

DEFAULT_CATEGORY_REALMS: Dict[str, str] = {
    "builder_creator": "tech",
    "tech_community": "tech",
    "tech_ai": "tech",
    "financial_markets": "markets",
    "economics": "markets",
    "policy": "civic",
    "macro_institution": "civic",
    "global_news": "civic",
    "culture": "culture",
    "cultural_pulse": "culture",
    "design": "design",
    "visual_aesthetic": "design",
    "science": "science",
}

MIN_ELIGIBLE_ITEMS = 3
MIN_ELIGIBLE_SOURCES = 2
_ZERO_NORM = 1e-12


def source_realms_from_feeds(
    feeds: Sequence[Mapping[str, Any]],
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Map each feed id to the realm of its category; overrides win."""
    out: Dict[str, str] = {}
    for feed in feeds:
        realm = DEFAULT_CATEGORY_REALMS.get(feed.get("category"))
        if realm is not None and feed.get("id") is not None:
            out[feed["id"]] = realm
    if overrides:
        out.update(overrides)
    return out


def cluster_sources(cluster: Any) -> List[str]:
    """Sorted distinct source ids of a cluster's items."""
    found = {_get_item_field(item, "source_id") for item in cluster.items}
    return sorted(s for s in found if s)


def realm_profile(cluster: Any, source_realms: Mapping[str, str]) -> Dict[str, int]:
    """Distinct sources per realm; unlabelled sources are not counted."""
    profile: Dict[str, int] = {}
    for source in cluster_sources(cluster):
        realm = source_realms.get(source)
        if realm is not None:
            profile[realm] = profile.get(realm, 0) + 1
    return profile


def dominant_realm(cluster: Any, source_realms: Mapping[str, str]) -> Optional[str]:
    """Realm with the most sources; ties go to the lexically smaller realm."""
    profile = realm_profile(cluster, source_realms)
    if not profile:
        return None
    return min(profile, key=lambda realm: (-profile[realm], realm))


def is_eligible(cluster: Any) -> bool:
    """At least 3 items and at least 2 distinct sources."""
    return (
        len(cluster.items) >= MIN_ELIGIBLE_ITEMS
        and len(cluster_sources(cluster)) >= MIN_ELIGIBLE_SOURCES
    )


def _modal_dimension_only(values: Sequence[Any], dim_of: Any) -> List[Any]:
    """Keep values of the most common dimension; ties go to the larger dimension."""
    counts: Dict[int, int] = {}
    for value in values:
        dim = int(dim_of(value))
        counts[dim] = counts.get(dim, 0) + 1
    if not counts:
        return []
    modal = max(counts, key=lambda dim: (counts[dim], dim))
    return [v for v in values if int(dim_of(v)) == modal]


def cluster_centroid(cluster: Any) -> np.ndarray:
    """Renormalised mean of the items' unit vectors; zeros if the mean is ~0."""
    if not cluster.items:
        return np.zeros(0, dtype=np.float32)
    vectors = _modal_dimension_only(
        [_extract_unit_embedding_f32(item) for item in cluster.items],
        lambda v: v.shape[0],
    )
    mean = np.mean(np.stack(vectors), axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(mean))
    if norm < _ZERO_NORM:
        return np.zeros_like(mean, dtype=np.float32)
    return (mean / norm).astype(np.float32)


def link_clusters(
    clusters: Sequence[Any],
    source_realms: Mapping[str, str],
    *,
    percentile: float = 90.0,
    k: int = 3,
) -> List[Tuple[int, int, float]]:
    """Mutual top-k cross-realm links at or above the day's own percentile.

    The percentile is taken over all pairs of eligible clusters with non-zero
    centroids (including clusters without a dominant realm). Returns sorted
    (i, j, similarity) with i < j, indices into ``clusters``.
    """
    usable: List[int] = []
    centroids: Dict[int, np.ndarray] = {}
    for idx, cluster in enumerate(clusters):
        if not is_eligible(cluster):
            continue
        centroid = cluster_centroid(cluster)
        if not np.any(centroid):
            continue
        usable.append(idx)
        centroids[idx] = centroid
    keep = _modal_dimension_only(usable, lambda idx: centroids[idx].shape[0])
    usable = [idx for idx in usable if idx in set(keep)]
    if len(usable) < 2:
        return []

    def sim(a: int, b: int) -> float:
        lo, hi = (a, b) if a < b else (b, a)
        return float(np.dot(centroids[lo], centroids[hi]))

    pair_sims = [sim(a, b) for pos, a in enumerate(usable) for b in usable[pos + 1 :]]
    threshold = float(np.percentile(pair_sims, percentile))

    realm_of = {idx: dominant_realm(clusters[idx], source_realms) for idx in usable}
    candidates = [idx for idx in usable if realm_of[idx] is not None]

    top: Dict[int, set] = {}
    for i in candidates:
        others = [j for j in candidates if realm_of[j] != realm_of[i]]
        others.sort(key=lambda j: (-sim(i, j), j))
        top[i] = set(others[: max(k, 0)])

    links: List[Tuple[int, int, float]] = []
    for pos, i in enumerate(candidates):
        for j in candidates[pos + 1 :]:
            if j in top[i] and i in top[j]:
                s = sim(i, j)
                if s >= threshold:
                    links.append((i, j, s))
    return links


@dataclass(frozen=True)
class Undercurrent:
    cluster_ids: Tuple[str, ...]
    realms: Tuple[str, ...]
    source_count: int
    item_keys: Tuple[str, ...]
    mean_link_similarity: float
    score: float


def find_undercurrents(
    clusters: Sequence[Any],
    source_realms: Mapping[str, str],
    *,
    percentile: float = 90.0,
    k: int = 3,
    min_clusters: int = 3,
    min_realms: int = 3,
    min_sources: int = 4,
) -> List[Undercurrent]:
    """Connected components of the link graph that meet all three minimums."""
    links = link_clusters(clusters, source_realms, percentile=percentile, k=k)
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in links:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    members: Dict[int, List[int]] = {}
    for node in list(parent):
        members.setdefault(find(node), []).append(node)

    found: List[Undercurrent] = []
    for root, nodes in members.items():
        nodes.sort()
        realms = sorted({dominant_realm(clusters[n], source_realms) for n in nodes})
        sources = {s for n in nodes for s in cluster_sources(clusters[n])}
        if len(nodes) < min_clusters or len(realms) < min_realms or len(sources) < min_sources:
            continue
        member_set = set(nodes)
        link_sims = [s for i, j, s in links if i in member_set and j in member_set]
        keys = {
            str(h)
            for n in nodes
            for item in clusters[n].items
            for h in [_get_item_field(item, "url_hash")]
            if h
        }
        total = sum(float(clusters[n].commission_score) for n in nodes)
        found.append(
            Undercurrent(
                cluster_ids=tuple(sorted(str(clusters[n].cluster_id) for n in nodes)),
                realms=tuple(realms),
                source_count=len(sources),
                item_keys=tuple(sorted(keys)),
                mean_link_similarity=float(sum(link_sims) / len(link_sims)),
                score=total * math.log(1 + len(realms)),
            )
        )
    found.sort(key=lambda u: (-u.score, u.cluster_ids))
    return found


NULL_PASS_RATIO: float = 2.0


def cross_day_null(
    day_a: Sequence[Any],
    day_b: Sequence[Any],
    source_realms: Mapping[str, str],
    *,
    percentile: float = 90.0,
    k: int = 3,
) -> Dict[str, float]:
    """Within-day versus between-day link rates over the pooled two days.

    Real structure links clusters of the same day far more often than clusters
    of unrelated days. Pairs are counted over eligible, realm-labelled,
    non-zero clusters of different realms.
    """
    pooled = list(day_a) + list(day_b)
    split = len(day_a)
    links = link_clusters(pooled, source_realms, percentile=percentile, k=k)
    within_links = sum(1 for i, j, _ in links if (i < split) == (j < split))
    between_links = len(links) - within_links

    countable: List[Tuple[int, str]] = []
    for idx, cluster in enumerate(pooled):
        if not is_eligible(cluster) or not np.any(cluster_centroid(cluster)):
            continue
        realm = dominant_realm(cluster, source_realms)
        if realm is not None:
            countable.append((idx, realm))
    within_pairs = 0
    between_pairs = 0
    for pos, (i, realm_i) in enumerate(countable):
        for j, realm_j in countable[pos + 1 :]:
            if realm_i == realm_j:
                continue
            if (i < split) == (j < split):
                within_pairs += 1
            else:
                between_pairs += 1

    within_rate = within_links / within_pairs if within_pairs else 0.0
    between_rate = between_links / between_pairs if between_pairs else 0.0
    if between_rate > 0:
        ratio = within_rate / between_rate
    else:
        ratio = float("inf") if within_rate > 0 else 0.0
    return {
        "within_links": within_links,
        "between_links": between_links,
        "within_pairs": within_pairs,
        "between_pairs": between_pairs,
        "within_rate": within_rate,
        "between_rate": between_rate,
        "ratio": ratio,
    }


def realm_report(
    clusters: Sequence[Any],
    source_realms: Mapping[str, str],
    undercurrents: Sequence[Undercurrent],
) -> Dict[str, Dict[str, int]]:
    """Per realm (and "unlabelled"): item, eligible-cluster and undercurrent-cluster counts."""
    report: Dict[str, Dict[str, int]] = {}

    def row(realm: str) -> Dict[str, int]:
        return report.setdefault(
            realm, {"items": 0, "eligible_clusters": 0, "undercurrent_clusters": 0}
        )

    in_undercurrent = {cid for u in undercurrents for cid in u.cluster_ids}
    for cluster in clusters:
        for item in cluster.items:
            realm = source_realms.get(_get_item_field(item, "source_id"))
            row(realm if realm is not None else "unlabelled")["items"] += 1
        realm = dominant_realm(cluster, source_realms) or "unlabelled"
        if is_eligible(cluster):
            row(realm)["eligible_clusters"] += 1
        if str(cluster.cluster_id) in in_undercurrent:
            row(realm)["undercurrent_clusters"] += 1
    return report
