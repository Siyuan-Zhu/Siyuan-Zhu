"""Understanding v1 helpers: typical-titles packet, handle blanking, echo check, realms and links.

Pure functions only: no I/O, no logging. The only environment reads are in
`understanding_v1_enabled`, `links_enabled` and `repair_enabled`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .methodology import (
    _extract_unit_embedding_f32,
    _get_item_field,
    _is_personal_orbit_item,
    legacy_top_cluster,
)
from .undercurrents import find_undercurrents, source_realms_from_feeds

SWITCH_ENV = "ZEITGEIST_UNDERSTANDING_V1"
LINKS_ENV = "ZEITGEIST_LINKS"
REPAIR_ENV = "ZEITGEIST_CHOSEN_GROUP_FIELDS"

MAX_CLUSTERS = 5
TITLE_CHARS = 120
ECHO_NGRAM = 5
LONG_HANDLE_LETTERS = 12

_OWNER_HANDLE_RE = re.compile(r"o[1-4]_ig_([a-z0-9._]{2,30})")
_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_PLAIN_LETTERS_RE = re.compile(r"[a-z]+")
_CLUSTER_ID_RE = re.compile(r"\bcluster[ _#-]?\d+\b", re.IGNORECASE)


def understanding_v1_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH_ENV) == "1"


def links_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(LINKS_ENV) == "1"


def repair_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(REPAIR_ENV) == "1"


def item_handles(item: Any) -> List[str]:
    """Lower-cased, de-duplicated handles tied to an item (mentions plus the owner handle)."""
    found: List[str] = []
    try:
        sources: List[Any] = [_get_item_field(item, "mentioned_handles")]
        extra = _get_item_field(item, "extra")
        if isinstance(extra, Mapping):
            sources.append(extra.get("mentioned_handles"))
        for raw in sources:
            if isinstance(raw, (list, tuple, set, frozenset)):
                for h in raw:
                    if isinstance(h, str):
                        h = h.strip().lstrip("@").lower()
                        if h:
                            found.append(h)
        source_id = str(_get_item_field(item, "source_id", "") or "").lower()
        m = _OWNER_HANDLE_RE.fullmatch(source_id)
        if m:
            found.append(m.group(1))
    except Exception:
        pass
    return list(dict.fromkeys(found))


def _handle_pattern(handle: str) -> "re.Pattern[str]":
    # Not inside a longer word. A trailing dot that ends a sentence still counts as a boundary.
    return re.compile(
        r"(?<![A-Za-z0-9._])" + re.escape(handle) + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])",
        re.IGNORECASE,
    )


def blank_handles(text: str, handles: Sequence[str]) -> str:
    out = text or ""
    cleaned = {str(x).strip().lstrip("@") for x in handles}
    for h in sorted((h for h in cleaned if h), key=len, reverse=True):
        out = _handle_pattern(h).sub("someone", out)
    return out


def _vectors(items: Sequence[Any]) -> np.ndarray:
    vecs: List[Optional[np.ndarray]] = []
    for it in items:
        try:
            vecs.append(_extract_unit_embedding_f32(it).reshape(-1))
        except Exception:
            vecs.append(None)
    dims = [v.shape[0] for v in vecs if v is not None and v.shape[0] > 0]
    dim = max(set(dims), key=dims.count) if dims else 1
    rows = []
    for v in vecs:
        if v is None or v.shape[0] != dim or not np.all(np.isfinite(v)):
            rows.append(np.zeros(dim, dtype=np.float32))
        else:
            rows.append(v.astype(np.float32))
    return np.vstack(rows)


def typical_items(items: Sequence[Any], k: int = 4) -> List[Any]:
    """The k most central items (highest mean cosine to the others); all in order when n <= k."""
    items = list(items)
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return items
    try:
        vecs = _vectors(items)
        sim = vecs @ vecs.T
        mean_sim = (sim.sum(axis=1) - np.diag(sim)) / float(n - 1)
        zero = np.linalg.norm(vecs, axis=1) == 0
        order = sorted(
            range(n),
            key=lambda i: (bool(zero[i]), -round(float(mean_sim[i]), 9), i),
        )
        return [items[i] for i in order[:k]]
    except Exception:
        return items[:k]


def _clean(text: Any, handles: Sequence[str]) -> str:
    s = _URL_RE.sub("", str(text or ""))
    s = blank_handles(s, handles)
    return " ".join(s.split())


def _cut(text: str, limit: int) -> str:
    return text[:limit].rstrip()


def _typical_titles(cluster: Any) -> List[str]:
    titles: List[str] = []
    for it in typical_items(list(getattr(cluster, "items", None) or [])):
        title = _cut(_clean(_get_item_field(it, "title", ""), item_handles(it)), TITLE_CHARS)
        if title:
            titles.append(title)
    return titles


def _render_cluster(cluster: Any) -> str:
    keywords = public_keywords(list(getattr(cluster, "cluster_keywords", None) or []), cluster)
    titles = _typical_titles(cluster)
    return (
        "- Cluster {cid}: commission_score={cs:.2f}, velocity={v:.2f}, orbit_resonance={o:.2f}, "
        "keywords={kw}, sample_titles={titles}"
    ).format(
        cid=getattr(cluster, "cluster_id", ""),
        cs=float(getattr(cluster, "commission_score", 0.0) or 0.0),
        v=float(getattr(cluster, "velocity", 0.0) or 0.0),
        o=float(getattr(cluster, "orbit_resonance", 0.0) or 0.0),
        kw=keywords,
        titles=titles,
    )


def render_evidence_packet(
    clusters: Sequence[Any], now_utc: Any = None, *, links: Optional[Sequence[Any]] = None,
    extra_group: Any = None,
) -> str:
    """The five groups, then each link as one line, then the extra group; with neither, today's packet."""
    clusters = list(clusters or [])[:MAX_CLUSTERS]
    if not clusters:
        return "- No clusters provided"
    lines: List[str] = []
    for c in clusters:
        try:
            lines.append(_render_cluster(c))
        except Exception:
            lines.append(f"- Cluster {getattr(c, 'cluster_id', '')}: unavailable")
    for link in links or ():
        try:
            lines.append(_render_link(link))
        except Exception:
            lines.append(f"- Link {getattr(link, 'link_id', '')}: unavailable")
    if extra_group is not None:
        try:
            lines.append(_render_cluster(extra_group))
        except Exception:
            lines.append(f"- Cluster {getattr(extra_group, 'cluster_id', '')}: unavailable")
    return "\n".join(lines)


def _ngrams(tokens: Sequence[str], n: int) -> set:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def echo_violation(fields: Sequence[str], clusters: Sequence[Any]) -> Optional[str]:
    try:
        items = [it for c in (clusters or []) for it in (getattr(c, "items", None) or [])]
        orbit_grams: set = set()
        handles: List[str] = []
        for it in items:
            handles.extend(item_handles(it))
            if _is_personal_orbit_item(it):
                for key in ("title", "summary"):
                    toks = _TOKEN_RE.findall(str(_get_item_field(it, key, "") or "").lower())
                    orbit_grams |= _ngrams(toks, ECHO_NGRAM)
        texts = [str(f or "") for f in (fields or [])]
        for text in texts:
            if _ngrams(_TOKEN_RE.findall(text.lower()), ECHO_NGRAM) & orbit_grams:
                return "Echoes source text"
        risky = _risky_handles(list(dict.fromkeys(handles)))
        part_sets = _handle_part_sets(risky)
        for text in texts:
            for h in risky:
                if _handle_pattern(h).search(text):
                    return "Names a source account"
            toks = _TOKEN_RE.findall(text.lower())
            for parts in part_sets:
                run_len = 0
                for tok in toks:
                    run_len = run_len + 1 if tok in parts else 0
                    if run_len >= 2:
                        return "Names a source account"
        for text in texts:
            if _CLUSTER_ID_RE.search(text):
                return "Names an internal cluster id"
    except Exception:
        return None
    return None


def _risky_handles(handles: Sequence[str]) -> List[str]:
    return [
        h for h in handles
        if h and (_PLAIN_LETTERS_RE.fullmatch(h) is None or len(h) >= LONG_HANDLE_LETTERS)
    ]


def _handle_part_sets(risky: Sequence[str]) -> List[set]:
    return [{p for p in re.split(r"[^a-z]+", h.lower()) if len(p) >= 2} for h in risky]


def _keywords_without_handles(keywords: Sequence[str], handles: Sequence[str]) -> List[str]:
    """Drops any keyword that is, or (for risky handles) contains, one of the handles. May raise."""
    handles = list(dict.fromkeys(h for h in handles if h))
    risky = _risky_handles(handles)
    part_sets = _handle_part_sets(risky)
    kept: List[str] = []
    for kw in keywords:
        low = str(kw).strip().lower()
        if low in handles or any(_handle_pattern(h).search(low) for h in risky):
            continue
        toks = re.findall(r"[a-z0-9']+", low)
        if len(toks) >= 2 and any(set(toks) <= parts for parts in part_sets):
            continue
        kept.append(kw)
    return kept


def public_keywords(keywords: Sequence[str], cluster: Any) -> List[str]:
    """Keywords safe to publish: drops any that is, or (for risky handles) contains, an item handle."""
    try:
        handles: List[str] = []
        for it in getattr(cluster, "items", None) or []:
            handles.extend(item_handles(it))
        return _keywords_without_handles(keywords, handles)
    except Exception:
        return list(keywords)


def ranking_stats(clusters: Sequence[Any]) -> Tuple[int, int]:
    """(top_changed, new_top_legacy_rank): 1-based legacy rank of the first cluster."""
    remaining = list(clusters or [])
    if not remaining:
        return (0, 0)
    first = remaining[0]
    rank = 0
    while remaining:
        top = legacy_top_cluster(remaining)
        rank += 1
        if top is first:
            return (0, 1) if rank == 1 else (1, rank)
        remaining = [c for c in remaining if c is not top]
    return (0, 0)


# ---- realms: the night's realm map ----

CULTURE_AND_COMMUNITY = "culture_and_community"
REALMS = (CULTURE_AND_COMMUNITY, "design", "tech", "markets", "civic", "science")
_LAYER_CULTURE = "culture"


class RealmMap(dict):
    """Realm by `source_id`. Feeds are stored entries; an Instagram source id
    (one containing `_ig_`) is answered by rule from its handle on lookup.

    Its text form never shows an entry, because the private map feeds it.
    """

    def __init__(self, feed_realms: Mapping[str, str], handle_realms: Mapping[str, str]) -> None:
        super().__init__(feed_realms)
        self._handle_realms = dict(handle_realms)

    def _instagram_realm(self, source_id: Any) -> Optional[str]:
        if not isinstance(source_id, str) or "_ig_" not in source_id:
            return None
        m = _OWNER_HANDLE_RE.fullmatch(source_id.lower())
        if m and m.group(1) in self._handle_realms:
            return self._handle_realms[m.group(1)]
        return CULTURE_AND_COMMUNITY

    def __missing__(self, source_id: Any) -> str:
        realm = self._instagram_realm(source_id)
        if realm is None:
            raise KeyError("no realm")
        return realm

    def get(self, source_id: Any, default: Any = None) -> Any:
        if dict.__contains__(self, source_id):
            return dict.__getitem__(self, source_id)
        realm = self._instagram_realm(source_id)
        return default if realm is None else realm

    def __contains__(self, source_id: Any) -> bool:
        return dict.__contains__(self, source_id) or self._instagram_realm(source_id) is not None

    def __repr__(self) -> str:
        return f"RealmMap({len(self)} feeds)"

    __str__ = __repr__


def _bare_handle(value: Any) -> str:
    return str(value).strip().lstrip("@").lower()


def source_realm_map(feeds: Any, private_realms: Any, roster: Any) -> RealmMap:
    """The night's realm for each source: feeds by category (the layer's "culture" is
    culture and community), every Instagram source culture and community, a private
    entry first (by feed id, or by bare handle for Instagram), and a discovered account
    the realm all its roster parents share. Never prints or logs."""
    feed_list = [f for f in (feeds if isinstance(feeds, (list, tuple)) else []) if isinstance(f, Mapping)]
    feed_ids = {f.get("id") for f in feed_list if f.get("id") is not None}
    private: dict = {}
    if isinstance(private_realms, Mapping):
        private = {k: v for k, v in private_realms.items() if isinstance(k, str) and v in REALMS}
    feed_private = {k: v for k, v in private.items() if k in feed_ids}
    handle_private = {_bare_handle(k): v for k, v in private.items() if k not in feed_ids}

    feed_realms = {
        source_id: (CULTURE_AND_COMMUNITY if realm == _LAYER_CULTURE else realm)
        for source_id, realm in source_realms_from_feeds(feed_list).items()
    }
    feed_realms.update(feed_private)

    def parent_realm(parent: Any) -> Optional[str]:
        parent = str(parent)
        if parent in feed_realms:
            return feed_realms[parent]
        if "_ig_" not in parent:
            return None
        m = _OWNER_HANDLE_RE.fullmatch(parent.lower())
        if m and m.group(1) in handle_private:
            return handle_private[m.group(1)]
        return CULTURE_AND_COMMUNITY

    discovered: dict = {}
    records = roster.get("handles") if isinstance(roster, Mapping) else None
    for handle, record in (records.items() if isinstance(records, Mapping) else []):
        if not isinstance(record, Mapping):
            continue
        parents = record.get("parent_sources") or record.get("distinct_parent_sources") or []
        if not isinstance(parents, (list, tuple, set)) or not parents:
            continue
        found = {parent_realm(p) for p in parents}
        realm = found.pop() if len(found) == 1 else None
        discovered[_bare_handle(handle)] = realm if realm is not None else CULTURE_AND_COMMUNITY

    return RealmMap(feed_realms, {**discovered, **handle_private})


def realm_counts(items: Sequence[Any], realm_map: Mapping[str, str]) -> dict:
    """Items per realm; an item with no realm is counted under "none"."""
    counts: dict = {}
    for item in items or []:
        realm = realm_map.get(_get_item_field(item, "source_id"))
        key = realm if realm is not None else "none"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---- links: groups in different realms that sit unusually close on the night ----

MAX_LINKS = 2
LINK_KEYWORDS = 8
LINK_TITLES = 4


@dataclass(frozen=True)
class LinkCandidate:
    """One link offered to the Director: its member groups in rank order and the layer's figures."""

    link_id: str
    member_ids: Tuple[str, ...]
    realms: Tuple[str, ...]
    source_count: int
    mean_link_similarity: float
    members: Tuple[Any, ...]

    @property
    def items(self) -> List[Any]:
        return [it for member in self.members for it in (getattr(member, "items", None) or [])]


def link_candidates(clusters: Sequence[Any], realm_map: Mapping[str, str]) -> List[LinkCandidate]:
    """The layer's first two currents over all of the night's groups, as links `link_1`, `link_2`."""
    clusters = list(clusters or [])
    rank = {}
    for position, cluster in enumerate(clusters):
        rank.setdefault(str(getattr(cluster, "cluster_id", "")), position)
    found: List[LinkCandidate] = []
    for number, current in enumerate(find_undercurrents(clusters, realm_map)[:MAX_LINKS], 1):
        positions = sorted(rank[cid] for cid in current.cluster_ids)
        members = tuple(clusters[p] for p in positions)
        found.append(LinkCandidate(
            link_id=f"link_{number}",
            member_ids=tuple(str(getattr(m, "cluster_id", "")) for m in members),
            realms=tuple(current.realms),
            source_count=int(current.source_count),
            mean_link_similarity=float(current.mean_link_similarity),
            members=members,
        ))
    return found


def link_keywords(link: Any) -> List[str]:
    """Members in rank order, each member's keywords without the handles of every member's items, duplicates
    dropped regardless of case; the first eight.

    Fails closed: a member whose keywords cannot be filtered gives none, and unreadable handles give no keywords.
    """
    try:
        handles = [h for member in link.members for it in (getattr(member, "items", None) or [])
                   for h in item_handles(it)]
    except Exception:
        return []
    seen: set = set()
    kept: List[str] = []
    for member in link.members:
        try:
            member_keywords = _keywords_without_handles(list(getattr(member, "cluster_keywords", None) or []), handles)
        except Exception:
            member_keywords = []
        for kw in member_keywords:
            key = str(kw).strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            kept.append(kw)
    return kept[:LINK_KEYWORDS]


def link_titles(link: Any) -> List[str]:
    """One typical title from each member in rank order, round and round, until there are four."""
    per_member = [_typical_titles(member) for member in link.members]
    titles: List[str] = []
    for depth in range(max((len(t) for t in per_member), default=0)):
        for member_titles in per_member:
            if depth < len(member_titles) and len(titles) < LINK_TITLES:
                titles.append(member_titles[depth])
    return titles


def _render_link(link: Any) -> str:
    return (
        "- Link {lid}: groups={groups}, realms={realms}, sources={sources}, similarity={sim:.2f}, "
        "keywords={kw}, sample_titles={titles}"
    ).format(
        lid=link.link_id,
        groups=list(link.member_ids),
        realms=list(link.realms),
        sources=int(link.source_count),
        sim=float(link.mean_link_similarity),
        kw=link_keywords(link),
        titles=link_titles(link),
    )


def shown_groups(
    clusters: Sequence[Any], links: Optional[Sequence[Any]] = None, extra_group: Any = None,
) -> List[Any]:
    """Every group whose items the Director was shown: the five, each link's members, the extra group."""
    shown: List[Any] = list(clusters or [])[:MAX_CLUSTERS]
    seen = {id(c) for c in shown}
    extra = [m for link in links or () for m in getattr(link, "members", ())]
    if extra_group is not None:
        extra.append(extra_group)
    for group in extra:
        if id(group) not in seen:
            seen.add(id(group))
            shown.append(group)
    return shown
