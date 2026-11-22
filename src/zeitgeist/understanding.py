"""Understanding v1 helpers: evidence packet, handle blanking, echo check.

Pure functions only: no I/O, no logging. The only environment read is in
`understanding_v1_enabled`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .methodology import (
    _extract_unit_embedding_f32,
    _get_item_field,
    _is_personal_orbit_item,
    _parse_utc_datetime,
    legacy_top_cluster,
)

SWITCH_ENV = "ZEITGEIST_UNDERSTANDING_V1"

MAX_CLUSTERS = 5
TITLE_CHARS = 120
SUMMARY_CHARS = 240
ECHO_NGRAM = 5
LONG_HANDLE_LETTERS = 12

_OWNER_HANDLE_RE = re.compile(r"o[1-4]_ig_([a-z0-9._]{2,30})")
_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_PLAIN_LETTERS_RE = re.compile(r"[a-z]+")


def understanding_v1_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH_ENV) == "1"


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


def representative_items(items: Sequence[Any], k: int = 6) -> List[Any]:
    """Medoid first, then greedy max-min diversity; all items in original order when n <= k."""
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
        chosen = [int(np.argmax(mean_sim))]
        while len(chosen) < k:
            best_idx, best_val = -1, None
            for i in range(n):
                if i in chosen:
                    continue
                worst = float(max(sim[i, c] for c in chosen))
                if best_val is None or worst < best_val - 1e-9:
                    best_idx, best_val = i, worst
            chosen.append(best_idx)
        return [items[i] for i in chosen]
    except Exception:
        return items[:k]


def _clean(text: Any, handles: Sequence[str]) -> str:
    s = _URL_RE.sub("", str(text or ""))
    s = blank_handles(s, handles)
    return " ".join(s.split())


def _cut(text: str, limit: int) -> str:
    return text[:limit].rstrip()


def _evidence_line(item: Any) -> str:
    handles = item_handles(item)
    title = _cut(_clean(_get_item_field(item, "title", ""), handles), TITLE_CHARS)
    summary = _cut(_clean(_get_item_field(item, "summary", ""), handles), SUMMARY_CHARS)
    if summary and title and summary.lower().startswith(title.lower()):
        return summary
    if title and summary:
        return f"{title} | {summary}"
    return title or summary


def _render_cluster(cluster: Any, now_utc: Optional[datetime]) -> str:
    items = list(getattr(cluster, "items", None) or [])
    keywords = list(getattr(cluster, "cluster_keywords", None) or [])
    lines = [
        "- Cluster {cid}: commission_score={cs:.2f}, velocity={v:.2f}, orbit_resonance={o:.2f}, keywords={kw}".format(
            cid=getattr(cluster, "cluster_id", ""),
            cs=float(getattr(cluster, "commission_score", 0.0) or 0.0),
            v=float(getattr(cluster, "velocity", 0.0) or 0.0),
            o=float(getattr(cluster, "orbit_resonance", 0.0) or 0.0),
            kw=keywords,
        )
    ]
    times = [_parse_utc_datetime(_get_item_field(i, "published_at")) for i in items]
    ref = now_utc if now_utc is not None else (max(times) if times else None)
    last_24 = last_72 = older = 0
    for t in times:
        age = (ref - t) if ref is not None else timedelta(0)
        if age <= timedelta(hours=24):
            last_24 += 1
        elif age <= timedelta(hours=72):
            last_72 += 1
        else:
            older += 1
    sources = getattr(cluster, "unique_sources_count", None)
    if sources is None:
        sources = len(getattr(cluster, "unique_sources", ()) or ())
    lines.append(
        f"  facts: items={len(items)}, sources={int(sources)}, last_24h={last_24}, last_72h={last_72}, older={older}"
    )
    lines.append("  evidence:")
    n = 0
    for it in representative_items(items):
        text = _evidence_line(it)
        if not text:
            continue
        n += 1
        lines.append(f"    {n}. {text}")
    return "\n".join(lines)


def render_evidence_packet(clusters: Sequence[Any], now_utc: Any = None) -> str:
    clusters = list(clusters or [])[:MAX_CLUSTERS]
    if not clusters:
        return "- No clusters provided"
    now = _parse_utc_datetime(now_utc) if now_utc is not None else None
    blocks: List[str] = []
    for c in clusters:
        try:
            blocks.append(_render_cluster(c, now))
        except Exception:
            blocks.append(f"- Cluster {getattr(c, 'cluster_id', '')}: unavailable")
    return "\n".join(blocks)


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
        risky = [
            h
            for h in dict.fromkeys(handles)
            if h and (_PLAIN_LETTERS_RE.fullmatch(h) is None or len(h) >= LONG_HANDLE_LETTERS)
        ]
        for text in texts:
            for h in risky:
                if _handle_pattern(h).search(text):
                    return "Names a source account"
    except Exception:
        return None
    return None


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
