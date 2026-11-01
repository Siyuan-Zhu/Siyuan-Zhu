"""Orbit & Macro ingestion collectors, 1-hop mention graph, scout decay, and batch embeddings."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
import logging
import os
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import httpx
import numpy as np

from .spend_guard import SpendBlocked

logger = logging.getLogger(__name__)

MAX_ITEMS_PER_SOURCE_PER_DAY: int = 3
MAX_ACTIVE_SCOUTS: int = 15
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIM: int = 768
EMBEDDING_BATCH_SIZE: int = 100
FEED_CACHE_TTL_SECONDS: int = 345600

O3_CATEGORIES: tuple[str, ...] = (
    "tech_ai",
    "economics",
    "science",
    "culture",
    "design",
    "policy",
)

TIER_WEIGHTS: dict[str, float] = {
    "O1": 2.0,
    "O2": 1.6,
    "O3": 1.3,
    "O4_SCOUT": 1.1,
    "M1": 1.0,
    "M2": 1.0,
    "M3": 1.0,
    "M4_SCOUT": 1.0,
    "MACRO": 1.0,
}

FEED_CACHE_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "url_hash",
        "title",
        "url",
        "source_id",
        "source_tier",
        "published_at",
        "fetched_at",
        "summary",
        "orbit_weight",
        "schema_version",
        "expires_at",
    }
)

ORBIT_ROSTER_FIRESTORE_KEYS: frozenset[str] = frozenset(
    {
        "state_version",
        "handles",
        "curated_o3_cursor",
        "updated_at",
    }
)

_TRACKING_PARAM_EXACT = frozenset(
    {
        "fbclid",
        "igshid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "ref_url",
    }
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9._])@([A-Za-z0-9._]{2,30})")
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s\"'<>)]+")


def normalize_url(raw_url: str) -> str:
    """Normalize a URL by stripping tracking query parameters and trailing slashes."""
    cleaned = (raw_url or "").strip()
    if not cleaned:
        return ""
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned.rstrip("/")

    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    elif path == "/":
        path = ""

    filtered_params: list[tuple[str, str]] = []
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        k_lower = k.lower()
        if k_lower.startswith("utm_") or k_lower in _TRACKING_PARAM_EXACT:
            continue
        filtered_params.append((k, v))

    query = urlencode(filtered_params, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def compute_url_hash(url: str, *, hex_chars: int = 32) -> str:
    """Return the SHA-256 hex digest of the normalized canonical URL (default 32 hex chars / 128-bit)."""
    canonical = normalize_url(url)
    width = max(16, min(64, int(hex_chars)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:width]


def compute_url_hash_128(url: str) -> str:
    """Return the 128-bit (32-char hex) SHA-256 digest of the normalized canonical URL (`[C-M4]`)."""
    return compute_url_hash(url, hex_chars=32)


def _clean_text(raw: str | None, max_chars: int = 600) -> str:
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    no_tags = _HTML_TAG_RE.sub(" ", unescaped)
    compact = _WHITESPACE_RE.sub(" ", no_tags).strip()
    return compact[:max_chars]


def _parse_timestamp(raw_ts: str | None, fallback_utc: datetime | None = None) -> datetime:
    """Parse RFC-2822, ISO-8601, or numeric epoch timestamps into a timezone-aware UTC datetime."""
    default_dt = fallback_utc or datetime.now(timezone.utc)
    if not raw_ts:
        return default_dt
    text = str(raw_ts).strip()
    if not text:
        return default_dt

    # Try RFC 2822 (RSS pubDate)
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Try ISO 8601 (Atom published/updated)
    try:
        iso_candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    return default_dt


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1].lower()
    return tag.lower()


def _parse_xml_feed_entries(xml_text: str, fallback_utc: datetime) -> list[dict[str, Any]]:
    """Parse RSS 2.0, RDF/RSS 1.0, Atom, YouTube Atom, GitHub Atom, or ArXiv XML into raw entry dicts."""
    entries: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError:
        return entries

    for elem in root.iter():
        tag_name = _local_tag(elem.tag)
        if tag_name not in ("item", "entry"):
            continue

        title = ""
        link = ""
        summary = ""
        raw_date = ""

        for child in elem:
            ctag = _local_tag(child.tag)
            ctext = (child.text or "").strip()
            if ctag == "title" and not title:
                title = ctext
            elif ctag == "link":
                href = (child.attrib.get("href") or "").strip()
                rel = (child.attrib.get("rel") or "alternate").strip().lower()
                if href:
                    if rel == "alternate" or not link:
                        link = href
                elif ctext and not link:
                    link = ctext
            elif ctag in ("description", "summary", "content", "encoded") and not summary:
                summary = ctext
            elif ctag == "group":
                # media:group in YouTube Atom feeds
                for mchild in child:
                    mtag = _local_tag(mchild.tag)
                    if mtag == "description" and not summary:
                        summary = (mchild.text or "").strip()
            elif ctag in ("pubdate", "published", "updated", "date") and not raw_date:
                raw_date = ctext

        if not link and not title:
            continue

        dt = _parse_timestamp(raw_date, fallback_utc=fallback_utc)
        entries.append(
            {
                "title": _clean_text(title, max_chars=240),
                "url": link,
                "summary": _clean_text(summary, max_chars=600),
                "dt": dt,
            }
        )

    return entries


def _parse_json_feed_entries(payload: Any, fallback_utc: datetime) -> list[dict[str, Any]]:
    """Parse HN Algolia JSON or JSON Feed payloads into raw entry dicts."""
    entries: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return entries

    hits = payload.get("hits") or payload.get("items") or []
    if not isinstance(hits, list):
        return entries

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        title = _clean_text(hit.get("title") or hit.get("story_title") or "", max_chars=240)
        object_id = str(hit.get("objectID") or hit.get("id") or "")
        url = hit.get("url") or hit.get("story_url") or ""
        if not url and object_id:
            url = f"https://news.ycombinator.com/item?id={object_id}"
        summary_parts: list[str] = []
        if hit.get("story_text") or hit.get("comment_text") or hit.get("summary"):
            summary_parts.append(
                _clean_text(
                    hit.get("story_text") or hit.get("comment_text") or hit.get("summary"),
                    max_chars=500,
                )
            )
        points = hit.get("points")
        num_comments = hit.get("num_comments")
        if points is not None:
            summary_parts.append(f"points={points} comments={num_comments or 0}")
        raw_date = hit.get("created_at") or hit.get("date_published") or ""
        dt = _parse_timestamp(raw_date, fallback_utc=fallback_utc)
        if url or title:
            entries.append(
                {
                    "title": title,
                    "url": url,
                    "summary": " | ".join(p for p in summary_parts if p),
                    "dt": dt,
                }
            )
    return entries


def _enforce_daily_source_cap(
    raw_items: list[dict[str, Any]],
    max_per_day: int = MAX_ITEMS_PER_SOURCE_PER_DAY,
) -> list[dict[str, Any]]:
    """Sort items descending by timestamp and keep at most `max_per_day` per `(source_id, published_date)`."""
    ordered = sorted(raw_items, key=lambda x: x.get("published_at", ""), reverse=True)
    counts: dict[tuple[str, str], int] = {}
    seen_hashes: set[str] = set()
    kept: list[dict[str, Any]] = []

    for item in ordered:
        url_hash = item.get("url_hash", "")
        if url_hash and url_hash in seen_hashes:
            continue
        key = (str(item.get("source_id", "")), str(item.get("published_date", "")))
        current = counts.get(key, 0)
        if current >= max_per_day:
            continue
        counts[key] = current + 1
        if url_hash:
            seen_hashes.add(url_hash)
        kept.append(item)

    return kept


def _resolve_source_tier(raw_tier: str | None) -> str:
    """Map internal tier labels to Firestore-valid source_tier values ('O1', 'O2', 'O3', 'O4_SCOUT', 'MACRO')."""
    tier = str(raw_tier or "").strip().upper()
    if tier in {"O1", "O2", "O3", "O4_SCOUT"}:
        return tier
    return "MACRO"


def _resolve_orbit_weight(raw_tier: str | None, explicit_weight: Any = None) -> float:
    """Return canonical tier-specific orbit weight (O1=2.0, O2=1.6, O3=1.3, O4_SCOUT=1.1, M1-M3=1.0)."""
    tier = str(raw_tier or "").strip().upper()
    if tier in TIER_WEIGHTS:
        return float(TIER_WEIGHTS[tier])
    if explicit_weight is not None:
        try:
            return float(explicit_weight)
        except (TypeError, ValueError):
            pass
    return 1.0


def _resolve_origin(raw_tier: str | None, explicit_origin: str | None = None) -> str:
    """Resolve canonical item origin tag from tier and optional explicit origin."""
    if explicit_origin and str(explicit_origin).strip():
        return str(explicit_origin).strip()
    tier = str(raw_tier or "").strip().upper()
    if tier in {"O1", "O2", "O3"}:
        return "curated_O1_O3"
    if tier == "O4_SCOUT":
        return "orbit_scout_1hop"
    if tier == "M4_SCOUT":
        return "macro_scout"
    return "curated_M1_M3"


def sanitize_feed_cache_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return a strict 11-key feed_cache document matching firestore.rules allowlist."""
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_url = str(item.get("url") or "").strip()
    canonical_url = normalize_url(raw_url) or raw_url
    url_hash = str(item.get("url_hash") or "").strip() or compute_url_hash(canonical_url)
    published_at = str(item.get("published_at") or now_iso)
    fetched_at = str(item.get("fetched_at") or published_at or now_iso)
    pub_dt = _parse_timestamp(published_at, fallback_utc=now_utc)
    raw_tier = item.get("source_tier") or item.get("tier") or "MACRO"
    source_tier = _resolve_source_tier(str(raw_tier))
    orbit_weight = float(
        item.get("orbit_weight")
        if item.get("orbit_weight") is not None
        else _resolve_orbit_weight(str(item.get("tier") or source_tier), item.get("weight"))
    )
    expires_at = int(item.get("expires_at") or (int(pub_dt.timestamp()) + FEED_CACHE_TTL_SECONDS))
    schema_version = int(item.get("schema_version") or 1)

    return {
        "url_hash": url_hash,
        "title": str(item.get("title") or ""),
        "url": canonical_url,
        "source_id": str(item.get("source_id") or "unknown_source"),
        "source_tier": source_tier,
        "published_at": published_at,
        "fetched_at": fetched_at,
        "summary": str(item.get("summary") or item.get("summary_clean") or ""),
        "orbit_weight": orbit_weight,
        "schema_version": schema_version,
        "expires_at": expires_at,
    }


def select_o3_sources_for_cursor(
    sources: list[dict[str, Any]],
    cursor: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Select O3 feeds for the 2 active domain categories at `cursor` and return `(filtered_sources, next_cursor)`."""
    num_cats = len(O3_CATEGORIES)
    norm_cursor = int(cursor) % num_cats
    active_categories = {
        O3_CATEGORIES[norm_cursor],
        O3_CATEGORIES[(norm_cursor + 1) % num_cats],
    }
    next_cursor = (norm_cursor + 2) % num_cats

    if not isinstance(sources, list) or not sources:
        return [], next_cursor

    o3_entries = [
        s for s in sources if isinstance(s, dict) and str(s.get("tier") or "").strip().upper() == "O3"
    ]
    has_canonical_categories = any(
        str(s.get("category") or "").strip().lower() in O3_CATEGORIES for s in o3_entries
    )

    selected: list[dict[str, Any]] = []
    o3_idx = 0
    for src in sources:
        if not isinstance(src, dict):
            continue
        tier = str(src.get("tier") or "").strip().upper()
        if tier != "O3":
            selected.append(src)
            continue
        cat = str(src.get("category") or "").strip().lower()
        if has_canonical_categories:
            if cat in active_categories:
                selected.append(src)
        else:
            if (o3_idx % num_cats) in {norm_cursor, (norm_cursor + 1) % num_cats}:
                selected.append(src)
        o3_idx += 1

    return selected, next_cursor


rotate_curated_o3_sources = select_o3_sources_for_cursor


def _extract_feeds_from_config(sources_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract feed configurations from top-level `feeds` list or nested `lenses` blocks."""
    if not isinstance(sources_config, dict):
        return []
    feeds = (
        sources_config.get("feeds")
        or sources_config.get("curated_feeds")
        or sources_config.get("sources")
    )
    if isinstance(feeds, list) and feeds:
        return [f for f in feeds if isinstance(f, dict)]

    lenses = sources_config.get("lenses")
    if isinstance(lenses, dict):
        flattened: list[dict[str, Any]] = []
        for lens_cfg in lenses.values():
            if isinstance(lens_cfg, dict):
                for src in lens_cfg.get("sources") or []:
                    if isinstance(src, dict):
                        flattened.append(src)
        return flattened
    return []


def _normalize_feed_entries(
    raw_entries: list[dict[str, Any]],
    feed_cfg: dict[str, Any],
    now_utc: datetime,
) -> list[dict[str, Any]]:
    """Convert raw parsed feed entries into normalized item dicts with full schema & TTL fields."""
    source_id = str(feed_cfg.get("id") or "unknown_source")
    source_name = str(feed_cfg.get("name") or source_id)
    tier = str(feed_cfg.get("tier") or "O1").strip().upper()
    source_tier = _resolve_source_tier(tier)
    bucket = str(feed_cfg.get("bucket") or ("ORBIT" if tier.startswith("O") else "MACRO"))
    origin = _resolve_origin(tier, feed_cfg.get("origin"))
    weight = _resolve_orbit_weight(tier, feed_cfg.get("weight"))
    fetched_at = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    normalized_items: list[dict[str, Any]] = []
    for entry in raw_entries:
        canonical_url = normalize_url(entry.get("url", ""))
        if not canonical_url:
            continue
        dt: datetime = entry["dt"]
        published_at = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        published_date = dt.strftime("%Y-%m-%d")
        title_clean = entry.get("title", "")
        summary_clean = entry.get("summary", "")
        raw_blob = f"{title_clean} {summary_clean}"
        mentioned_handles = [
            m.rstrip(".").lower()
            for m in _MENTION_RE.findall(raw_blob)
            if len(m.rstrip(".")) >= 2
        ]
        normalized_items.append(
            {
                "url_hash": compute_url_hash(canonical_url),
                "url": canonical_url,
                "title": title_clean,
                "summary": summary_clean,
                "summary_clean": summary_clean,
                "mentioned_handles": mentioned_handles,
                "source_id": source_id,
                "source_name": source_name,
                "tier": tier,
                "source_tier": source_tier,
                "bucket": bucket,
                "origin": origin,
                "weight": weight,
                "orbit_weight": weight,
                "published_at": published_at,
                "published_date": published_date,
                "fetched_at": fetched_at,
                "schema_version": 1,
                "expires_at": int(dt.timestamp()) + FEED_CACHE_TTL_SECONDS,
            }
        )

    return _enforce_daily_source_cap(normalized_items, max_per_day=MAX_ITEMS_PER_SOURCE_PER_DAY)


def _fetch_single_feed(
    client: httpx.Client,
    feed_cfg: dict[str, Any],
    timeout_s: float,
    now_utc: datetime,
) -> list[dict[str, Any]]:
    """Fetch and parse a single feed configuration into normalized item dicts capped per day."""
    url = str(feed_cfg.get("url") or feed_cfg.get("uri") or "").strip()
    if not url:
        return []

    feed_type = str(feed_cfg.get("type") or feed_cfg.get("feed_type") or "rss").lower()
    feed_id = str(feed_cfg.get("id") or url)

    response = None
    last_exc: Exception | None = None
    for _attempt in range(2):
        try:
            response = client.get(url, timeout=timeout_s)
            response.raise_for_status()
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
    if last_exc is not None or response is None:
        logger.warning("Failed to fetch curated feed %s (%s): %s", feed_id, url, last_exc)
        return []

    text = response.text or ""
    if feed_type == "hn_algolia" or text.lstrip().startswith("{"):
        try:
            payload = response.json()
        except Exception:
            payload = json.loads(text) if text.strip() else {}
        raw_entries = _parse_json_feed_entries(payload, fallback_utc=now_utc)
    else:
        raw_entries = _parse_xml_feed_entries(text, fallback_utc=now_utc)

    return _normalize_feed_entries(raw_entries, feed_cfg, now_utc)


async def _fetch_single_feed_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    feed_cfg: dict[str, Any],
    timeout_s: float,
    now_utc: datetime,
) -> list[dict[str, Any]]:
    """Asynchronously fetch and parse a single feed under an asyncio.Semaphore."""
    url = str(feed_cfg.get("url") or feed_cfg.get("uri") or "").strip()
    if not url:
        return []

    feed_type = str(feed_cfg.get("type") or feed_cfg.get("feed_type") or "rss").lower()
    feed_id = str(feed_cfg.get("id") or url)

    response = None
    last_exc: Exception | None = None
    async with semaphore:
        for _attempt in range(2):
            try:
                response = await client.get(url, timeout=timeout_s)
                response.raise_for_status()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
        if last_exc is not None or response is None:
            logger.warning("Failed to fetch async curated feed %s (%s): %s", feed_id, url, last_exc)
            return []

    text = response.text or ""
    if feed_type == "hn_algolia" or text.lstrip().startswith("{"):
        try:
            payload = response.json()
        except Exception:
            payload = json.loads(text) if text.strip() else {}
        raw_entries = _parse_json_feed_entries(payload, fallback_utc=now_utc)
    else:
        raw_entries = _parse_xml_feed_entries(text, fallback_utc=now_utc)

    return _normalize_feed_entries(raw_entries, feed_cfg, now_utc)


async def fetch_all_curated_feeds_async(
    sources_config: dict[str, Any],
    timeout_s: float = 12.0,
    max_concurrency: int = 10,
    *,
    curated_o3_cursor: int | None = None,
) -> list[dict[str, Any]]:
    """Concurrently fetch curated feeds using httpx.AsyncClient and asyncio.Semaphore(10)."""
    feeds = _extract_feeds_from_config(sources_config)
    if not feeds:
        return []

    if curated_o3_cursor is not None:
        feeds, _ = select_o3_sources_for_cursor(feeds, curated_o3_cursor)
        if not feeds:
            return []

    now_utc = datetime.now(timezone.utc)
    headers = {
        "User-Agent": "SiyuanZhu-Portfolio-OrbitCollector/2.0 (+https://github.com/Siyuan-Zhu)"
    }
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    limits = httpx.Limits(max_connections=16)

    collected: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=timeout_s,
        limits=limits,
    ) as client:
        tasks = [
            _fetch_single_feed_async(client, semaphore, feed_cfg, timeout_s, now_utc)
            for feed_cfg in feeds
            if isinstance(feed_cfg, dict)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list) and res:
                collected.extend(res)

    return _enforce_daily_source_cap(collected, max_per_day=MAX_ITEMS_PER_SOURCE_PER_DAY)


def fetch_all_curated_feeds(
    sources_config: dict[str, Any],
    timeout_s: float = 12.0,
    max_workers: int = 10,
    *,
    curated_o3_cursor: int | None = None,
) -> list[dict[str, Any]]:
    """Concurrently fetch curated feeds in `sources_config`, normalize URLs, and cap at 3/source/day."""
    feeds = _extract_feeds_from_config(sources_config)
    if not feeds:
        return []

    if curated_o3_cursor is not None:
        feeds, _ = select_o3_sources_for_cursor(feeds, curated_o3_cursor)
        if not feeds:
            return []

    now_utc = datetime.now(timezone.utc)
    collected: list[dict[str, Any]] = []

    headers = {
        "User-Agent": "SiyuanZhu-Portfolio-OrbitCollector/2.0 (+https://github.com/Siyuan-Zhu)"
    }
    worker_count = max(1, min(int(max_workers), len(feeds)))

    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_s) as client:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(_fetch_single_feed, client, feed_cfg, timeout_s, now_utc)
                for feed_cfg in feeds
                if isinstance(feed_cfg, dict)
            ]
            for fut in as_completed(futures):
                try:
                    items = fut.result()
                    if items:
                        collected.extend(items)
                except Exception as exc:
                    logger.warning("Feed worker future raised exception: %s", exc)
                    continue

    return _enforce_daily_source_cap(collected, max_per_day=MAX_ITEMS_PER_SOURCE_PER_DAY)


def extract_one_hop_mentions(
    items: list[dict[str, Any]],
    curated_handles: set[str],
    blocklist: set[str],
    *,
    instagram_only: bool = False,
    require_curated_origin: bool = False,
) -> dict[str, set[str]]:
    """Extract 1-hop @handle mentions and external domains from O1-O3 items."""
    curated_set = {
        str(h).lstrip("@").strip().lower()
        for h in (curated_handles or set())
        if str(h).strip()
    }
    blocked_set = {
        str(b).lstrip("@").strip().lower()
        for b in (blocklist or set())
        if str(b).strip()
    }

    mentions: dict[str, set[str]] = {}

    for item in items or []:
        if not isinstance(item, dict):
            continue
        tier = str(item.get("tier") or "").strip().upper()
        if require_curated_origin and (item.get("origin") != "curated_O1_O3" or tier not in {"O1", "O2", "O3"}):
            continue
        if tier and tier not in {"O1", "O2", "O3"}:
            continue

        source_id = str(item.get("source_id") or item.get("id") or "unknown_source")
        item_url = str(item.get("url") or "")
        own_host = ""
        if item_url:
            try:
                own_host = urlsplit(item_url).netloc.lower().removeprefix("www.")
            except ValueError:
                own_host = ""

        text_parts = [
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            str(item.get("caption") or ""),
            str(item.get("description") or ""),
        ]
        blob = " ".join(p for p in text_parts if p)
        if not blob:
            continue

        # 1. Extract @handle mentions (including pre-extracted mentioned_handles from sanitized captions)
        pre_extracted = [str(m).lstrip("@") for m in (item.get("mentioned_handles") or []) if str(m).strip()]
        for raw_match in list(_MENTION_RE.findall(blob)) + pre_extracted:
            candidate = raw_match.rstrip(".").lower()
            if len(candidate) < 2:
                continue
            if candidate in curated_set or candidate in blocked_set:
                continue
            mentions.setdefault(candidate, set()).add(source_id)

        if instagram_only:
            continue  # An external domain or another network's profile is not an Instagram mention.

        # 2. Extract external links / social handles from embedded URLs
        for raw_link in _URL_IN_TEXT_RE.findall(blob):
            cleaned_link = raw_link.rstrip(".,;:!?)")
            try:
                parts = urlsplit(cleaned_link)
            except ValueError:
                continue
            host = parts.netloc.lower().removeprefix("www.")
            if not host or "." not in host or host == own_host:
                continue

            base_labels = {lbl for lbl in host.split(".") if lbl}
            if host in {"instagram.com", "x.com", "twitter.com"}:
                path_segs = [seg for seg in parts.path.split("/") if seg]
                if path_segs:
                    handle_cand = path_segs[0].lstrip("@").rstrip(".").lower()
                    if (
                        2 <= len(handle_cand) <= 30
                        and handle_cand not in curated_set
                        and handle_cand not in blocked_set
                    ):
                        mentions.setdefault(handle_cand, set()).add(source_id)
                continue

            if (
                host in curated_set
                or host in blocked_set
                or bool(base_labels & blocked_set)
                or bool(base_labels & curated_set)
            ):
                continue

            mentions.setdefault(host, set()).add(source_id)

    return mentions


_ROSTER_META_KEYS: frozenset[str] = frozenset(
    {
        "handles",
        "scouts",
        "tier_map",
        "consecutive_empty_runs",
        "quarantine_until",
        "curated_o3_cursor",
        "rss_o3_cursor",
        "updated_at",
        "state_version",
        "schema_version",
    }
)


class OrbitRosterState(dict[str, Any]):
    """Hybrid scout roster dict supporting both flat `{handle: rec}` access and structured state fields."""

    def __init__(
        self,
        handles_map: dict[str, dict[str, Any]] | None = None,
        *,
        curated_o3_cursor: int = 0,
        updated_at: str = "",
        state_version: int = 1,
        tier_map: dict[str, str] | None = None,
        consecutive_empty_runs: dict[str, int] | None = None,
        quarantine_until: dict[str, str | None] | None = None,
    ) -> None:
        super().__init__(handles_map or {})
        self.state_version: int = int(state_version)
        self.schema_version: int = int(state_version)
        self.curated_o3_cursor: int = int(curated_o3_cursor) % len(O3_CATEGORIES)
        self.updated_at: str = str(updated_at)
        self.tier_map: dict[str, str] = (
            dict(tier_map)
            if tier_map is not None
            else {
                h: str(rec.get("tier") or "O4_SCOUT")
                for h, rec in self.items()
                if isinstance(rec, dict)
            }
        )
        self.consecutive_empty_runs: dict[str, int] = (
            dict(consecutive_empty_runs)
            if consecutive_empty_runs is not None
            else {
                h: int(rec.get("consecutive_empty_runs", 0))
                for h, rec in self.items()
                if isinstance(rec, dict)
            }
        )
        self.quarantine_until: dict[str, str | None] = (
            dict(quarantine_until)
            if quarantine_until is not None
            else {
                h: rec.get("quarantine_until")
                for h, rec in self.items()
                if isinstance(rec, dict)
            }
        )

    def _meta_dict(self) -> dict[str, Any]:
        handles_copy = {k: v for k, v in super().items()}
        return {
            "handles": handles_copy,
            "scouts": handles_copy,
            "tier_map": dict(self.tier_map),
            "consecutive_empty_runs": dict(self.consecutive_empty_runs),
            "quarantine_until": dict(self.quarantine_until),
            "curated_o3_cursor": int(self.curated_o3_cursor),
            "updated_at": str(self.updated_at),
            "state_version": int(self.state_version),
            "schema_version": int(self.schema_version),
        }

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        meta = self._meta_dict()
        if key in meta:
            return meta[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        meta = self._meta_dict()
        if key in meta:
            return meta[key]
        return default

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        return isinstance(key, str) and key in _ROSTER_META_KEYS

    def to_state_dict(self) -> dict[str, Any]:
        """Return full structured state dictionary including tier_map, quarantine_until, and cursor."""
        return self._meta_dict()

    def to_firestore_doc(self) -> dict[str, Any]:
        """Return the strict 4-key orbit_roster state doc allowed by firestore.rules."""
        return {
            "state_version": int(self.state_version),
            "handles": {k: v for k, v in super().items()},
            "curated_o3_cursor": int(self.curated_o3_cursor) % len(O3_CATEGORIES),
            "updated_at": str(self.updated_at),
        }


def sanitize_orbit_roster_state(roster: dict[str, Any]) -> dict[str, Any]:
    """Keep the legacy client shape; v3 adds a server-only independent RSS cursor."""
    if isinstance(roster, OrbitRosterState):
        return roster.to_firestore_doc()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(roster, dict):
        if "handles" in roster and isinstance(roster.get("handles"), dict):
            handles = dict(roster["handles"])
        elif "scouts" in roster and isinstance(roster.get("scouts"), dict):
            handles = dict(roster["scouts"])
        else:
            handles = {
                k: v
                for k, v in roster.items()
                if k not in _ROSTER_META_KEYS and isinstance(v, dict)
            }
        cursor = max(0, int(roster.get("curated_o3_cursor", 0) or 0))
        if "rss_o3_cursor" not in roster:
            cursor %= len(O3_CATEGORIES)
        state_version = int(roster.get("state_version", 1) or 1)
        updated_at = str(roster.get("updated_at") or now_iso)
    else:
        handles = {}
        cursor = 0
        state_version = 1
        updated_at = now_iso

    result = {
        "state_version": state_version,
        "handles": handles,
        "curated_o3_cursor": cursor,
        "updated_at": updated_at,
    }
    if isinstance(roster, dict) and "rss_o3_cursor" in roster:
        result["rss_o3_cursor"] = int(roster["rss_o3_cursor"] or 0) % len(O3_CATEGORIES)
    return result


def update_orbit_roster(
    roster: dict[str, Any],
    mention_parents: dict[str, set[str]],
    now_utc: datetime,
) -> OrbitRosterState:
    """Update the 1-hop scout roster with 7-day half-life decay, promotion, eviction, 15-node cap, and O3 cursor rotation."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    prev_cursor = 0
    prev_tier_map: dict[str, str] = {}
    prev_empty_map: dict[str, int] = {}
    prev_quarantine_map: dict[str, str | None] = {}

    if isinstance(roster, OrbitRosterState):
        raw_scouts = {k: v for k, v in dict.items(roster)}
        prev_cursor = int(roster.curated_o3_cursor)
        prev_tier_map = dict(roster.tier_map)
        prev_empty_map = dict(roster.consecutive_empty_runs)
        prev_quarantine_map = dict(roster.quarantine_until)
    elif isinstance(roster, dict):
        if "curated_o3_cursor" in roster and roster.get("curated_o3_cursor") is not None:
            try:
                prev_cursor = int(roster["curated_o3_cursor"])
            except (TypeError, ValueError):
                prev_cursor = 0
        if isinstance(roster.get("tier_map"), dict):
            prev_tier_map = {str(k).lower(): str(v) for k, v in roster["tier_map"].items()}
        if isinstance(roster.get("consecutive_empty_runs"), dict):
            prev_empty_map = {
                str(k).lower(): int(v) for k, v in roster["consecutive_empty_runs"].items()
            }
        if isinstance(roster.get("quarantine_until"), dict):
            prev_quarantine_map = {
                str(k).lower(): (str(v) if v else None)
                for k, v in roster["quarantine_until"].items()
            }

        if (
            "handles" in roster
            and isinstance(roster.get("handles"), dict)
            and "decayed_score" not in roster.get("handles", {})
        ):
            raw_scouts = dict(roster["handles"])
        elif (
            "scouts" in roster
            and isinstance(roster.get("scouts"), dict)
            and "decayed_score" not in roster.get("scouts", {})
        ):
            raw_scouts = dict(roster["scouts"])
        else:
            raw_scouts = {
                k: v
                for k, v in roster.items()
                if k not in _ROSTER_META_KEYS and isinstance(v, dict)
            }
    else:
        raw_scouts = {}

    next_cursor = (prev_cursor + 2) % len(O3_CATEGORIES)
    working: dict[str, dict[str, Any]] = {}

    # 1. Apply 7-day exponential decay to all existing scouts
    for raw_handle, rec in raw_scouts.items():
        if not isinstance(rec, dict):
            continue
        handle = str(raw_handle).lstrip("@").strip().lower()
        if not handle:
            continue

        prev_score = float(rec.get("decayed_score", rec.get("vitality_score", 0.0)))
        ref_ts = (
            rec.get("last_updated_utc")
            or rec.get("last_seen_utc")
            or rec.get("last_mentioned_at")
            or now_iso
        )
        ref_dt = _parse_timestamp(str(ref_ts), fallback_utc=now_utc)
        days_elapsed = max(0.0, (now_utc - ref_dt).total_seconds() / 86400.0)
        decayed_score = prev_score * (0.5 ** (days_elapsed / 7.0))

        prev_parents = {
            str(p)
            for p in (
                rec.get("distinct_parent_sources")
                or rec.get("parent_sources")
                or []
            )
            if str(p).strip()
        }
        first_seen = str(rec.get("first_seen_utc") or rec.get("last_seen_utc") or now_iso)
        last_seen = str(rec.get("last_seen_utc") or first_seen)
        mention_count = int(rec.get("mention_count", len(prev_parents)))
        prev_status = str(rec.get("status") or "CANDIDATE")
        prev_tier = str(rec.get("tier") or prev_tier_map.get(handle) or "O4_SCOUT")
        prev_empty = int(rec.get("consecutive_empty_runs", prev_empty_map.get(handle, 0)))
        prev_quarantine = rec.get("quarantine_until", prev_quarantine_map.get(handle))

        working[handle] = {
            "handle": handle,
            "handle_or_url": handle,
            "decayed_score": decayed_score,
            "all_parents": prev_parents,
            "mention_count": mention_count,
            "first_seen_utc": first_seen,
            "last_seen_utc": last_seen,
            "prev_status": prev_status,
            "tier": prev_tier,
            "consecutive_empty_runs": prev_empty,
            "quarantine_until": prev_quarantine,
            "mentioned_today": False,
        }

    # 2. Add +1.0 per distinct parent source mentioned today
    for raw_handle, parents_today_raw in (mention_parents or {}).items():
        handle = str(raw_handle).lstrip("@").strip().lower()
        if not handle:
            continue
        parents_today = {str(p) for p in (parents_today_raw or set()) if str(p).strip()}
        if not parents_today:
            continue

        added_score = 1.0 * float(len(parents_today))
        if handle in working:
            entry = working[handle]
            entry["decayed_score"] = float(entry["decayed_score"]) + added_score
            entry["all_parents"] = set(entry["all_parents"]) | parents_today
            entry["mention_count"] = int(entry["mention_count"]) + len(parents_today)
            entry["last_seen_utc"] = now_iso
            entry["consecutive_empty_runs"] = 0
            entry["quarantine_until"] = None
            entry["mentioned_today"] = True
        else:
            working[handle] = {
                "handle": handle,
                "handle_or_url": handle,
                "decayed_score": added_score,
                "all_parents": set(parents_today),
                "mention_count": len(parents_today),
                "first_seen_utc": now_iso,
                "last_seen_utc": now_iso,
                "prev_status": "CANDIDATE",
                "tier": str(prev_tier_map.get(handle) or "O4_SCOUT"),
                "consecutive_empty_runs": 0,
                "quarantine_until": None,
                "mentioned_today": True,
            }

    # 3. Promote (>= 2.0 score and >= 2 distinct parents) and evict (< 0.25)
    survivors: list[dict[str, Any]] = []
    for handle, entry in working.items():
        score = float(entry["decayed_score"])
        if score < 0.25:
            continue

        if not entry.get("mentioned_today"):
            empty_runs = int(entry.get("consecutive_empty_runs", 0)) + 1
            entry["consecutive_empty_runs"] = empty_runs
            if empty_runs >= 3 and not entry.get("quarantine_until"):
                entry["quarantine_until"] = (now_utc + timedelta(days=7)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )

        all_parents_sorted = sorted(entry["all_parents"])
        prev_status = str(entry["prev_status"])
        if len(all_parents_sorted) >= 2 and (score >= 2.0 or prev_status == "ACTIVE_O4_SCOUT"):
            status = "ACTIVE_O4_SCOUT"
        else:
            status = "CANDIDATE"

        tier_val = str(entry.get("tier") or "O4_SCOUT")
        survivors.append(
            {
                "handle": handle,
                "handle_or_url": handle,
                "decayed_score": score,
                "vitality_score": score,
                "distinct_parent_sources": all_parents_sorted,
                "parent_sources": all_parents_sorted,
                "mention_count": int(entry["mention_count"]),
                "promoted": status == "ACTIVE_O4_SCOUT",
                "first_seen_utc": str(entry["first_seen_utc"]),
                "last_seen_utc": str(entry["last_seen_utc"]),
                "last_mentioned_at": str(entry["last_seen_utc"]),
                "last_updated_utc": now_iso,
                "status": status,
                "tier": tier_val,
                "consecutive_empty_runs": int(entry.get("consecutive_empty_runs", 0)),
                "quarantine_until": entry.get("quarantine_until"),
            }
        )

    # 4. Enforce MAX_ACTIVE_SCOUTS = 15 (keep top 15 by (decayed_score, last_seen_utc))
    survivors.sort(
        key=lambda r: (float(r["decayed_score"]), str(r["last_seen_utc"])),
        reverse=True,
    )
    capped = survivors[:MAX_ACTIVE_SCOUTS]
    result_map = {rec["handle"]: rec for rec in capped}
    tier_map = {rec["handle"]: str(rec["tier"]) for rec in capped}
    empty_runs_map = {rec["handle"]: int(rec["consecutive_empty_runs"]) for rec in capped}
    quarantine_map = {rec["handle"]: rec["quarantine_until"] for rec in capped}

    return OrbitRosterState(
        result_map,
        curated_o3_cursor=next_cursor,
        updated_at=now_iso,
        state_version=1,
        tier_map=tier_map,
        consecutive_empty_runs=empty_runs_map,
        quarantine_until=quarantine_map,
    )


APIFY_INSTAGRAM_ENDPOINT = (
    "https://api.apify.com/v2/acts/apify~instagram-post-scraper/run-sync-get-dataset-items"
)
MAX_APIFY_BATCH_HANDLES: int = 20


def public_ingestion_diagnostics(raw, *, token=""):
    """Copy only bounded typed metadata; never export exception text or payloads."""
    if not isinstance(raw, dict):
        return {}
    result = {}
    if raw.get("provider") in {"apify", "gemini", "unsplash", "bfl", "firestore"}:
        result["provider"] = raw["provider"]
    if type(raw.get("status")) is int and 100 <= raw["status"] <= 599:
        result["status"] = raw["status"]
    for key, pattern in (
        ("error_type", r"[A-Za-z_][A-Za-z0-9_.-]{0,79}"),
        ("code", r"[a-z][a-z0-9_.-]{0,79}"),
        ("build", r"\d{1,10}\.\d{1,10}\.\d{1,10}"),
    ):
        value = raw.get(key)
        if (
            isinstance(value, str)
            and re.fullmatch(pattern, value)
            and not (token and token in value)
            and not re.search(r"(?i)(apify_api_|sk-|AIza|ya29\.|gh[pousr]_)", value)
        ):
            result[key] = value
    return result


def public_collection_health(raw):
    """Only source outcome enums, booleans, and counts may enter public reports."""
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key, values in (
        ("mode", {"cache", "live", "seeded"}),
        ("instagram", {"not_requested", "live", "empty", "unavailable"}),
    ):
        if isinstance(raw.get(key), str) and raw[key] in values:
            result[key] = raw[key]
    for key in (
        "publishable",
        "seeded",
        "apify_batch_succeeded",
        "instagram_fetched_successfully",
        "instagram_degraded",
    ):
        if type(raw.get(key)) is bool:
            result[key] = raw[key]
    for key in (
        "live_item_count",
        "cache_item_count",
        "curated_item_count",
        "orbit_item_count",
        "macro_item_count",
        "instagram_item_count",
        "instagram_requested_handle_count",
        "instagram_dataset_record_count",
        "instagram_unavailable_handle_count",
        "instagram_skipped_record_count",
        "instagram_normalized_post_count",
        "instagram_verified_collaboration_record_count",
        "orbit_source_count",
        "macro_source_count",
        "nonsocial_orbit_source_count",
        "nonsocial_macro_source_count",
        "embedding_eligible_item_count",
        "discarded_before_embedding_count",
    ):
        if type(raw.get(key)) is int and 0 <= raw[key] <= 10**9:
            result[key] = raw[key]
    reasons = raw.get("instagram_unavailable_reasons")
    if isinstance(reasons, dict):
        result["instagram_unavailable_reasons"] = {key: value for key, value in reasons.items()
            if key in {"not_found", "no_items"} and type(value) is int and 0 <= value <= 10**9}
    tiers = raw.get("fresh_items_by_tier")
    if isinstance(tiers, dict):
        result["fresh_items_by_tier"] = {
            key: value
            for key, value in tiers.items()
            if key in {"O1", "O2", "O3", "O4_SCOUT", "M1", "M2", "M3", "M4_SCOUT"}
            and type(value) is int
            and 0 <= value <= 10**9
        }
    return result


class IngestionUnavailable(RuntimeError):
    """Required live collection cannot support a fresh publication."""

    def __init__(self, message, *, public_diagnostics=None, collection_health=None):
        super().__init__(message)
        self.public_diagnostics = public_ingestion_diagnostics(public_diagnostics)
        self.public_collection_health = public_collection_health(collection_health)


def _sanitized_provider_error(exc, token):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    detail = {"provider": "apify", "status": status, "error_type": type(exc).__name__}
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            error = response.json().get("error", {})
            code = public_ingestion_diagnostics(
                {"code": error.get("type") or error.get("code")}, token=token
            ).get("code")
            if code:
                detail["code"] = code
            message = str(error.get("message", ""))[:500].replace(token, "[redacted]")
            message = re.sub(r"https?://[^\s]+", "[provider-url]", message)
            detail["message"] = message
        except Exception:
            pass
    return detail


_MACRO_SCOUT_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "m4_scout_hn_frontpage",
        "name": "Hacker News Front Page Velocity",
        "tier": "M4_SCOUT",
        "bucket": "MACRO",
        "origin": "macro_scout",
        "type": "hn_algolia",
        "url": "https://hn.algolia.com/api/v1/search?tags=front_page&numericFilters=points>100&hitsPerPage=15",
        "weight": 1.0,
    },
    {
        "id": "m4_scout_google_trends_us",
        "name": "Google Trends Daily US",
        "tier": "M4_SCOUT",
        "bucket": "MACRO",
        "origin": "macro_scout",
        "type": "rss",
        "url": "https://trends.google.com/trending/rss?geo=US",
        "weight": 1.0,
    },
)


def fetch_instagram_batch(
    handles: list[str],
    apify_token: str | None,
    max_posts_per_handle: int = 5,
    timeout_s: float = 330.0,
    tier_map: dict[str, str] | None = None,
    *,
    spend_guard: Any = None,
    actor_build: str | None = None,
    health: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch public Instagram posts via Apify in a single batched call for up to 20 handles."""
    token = (apify_token or "").strip()
    if not token or not handles:
        return []

    normalized_tier_map: dict[str, str] = {}
    if isinstance(tier_map, dict):
        for raw_k, raw_v in tier_map.items():
            k = str(raw_k or "").lstrip("@").strip().lower()
            v = str(raw_v or "").strip().upper()
            if k and v:
                normalized_tier_map[k] = v

    cleaned_handles: list[str] = []
    seen_handles: set[str] = set()
    for raw_h in handles:
        h = str(raw_h or "").lstrip("@").strip().lower()
        if h and h not in seen_handles:
            seen_handles.add(h)
            cleaned_handles.append(h)
        if len(cleaned_handles) >= MAX_APIFY_BATCH_HANDLES:
            break

    if not cleaned_handles:
        return []

    per_handle_cap = max(1, min(int(max_posts_per_handle), 5))
    payload = {
        "username": cleaned_handles,
        "resultsLimit": per_handle_cap,
        "dataDetailLevel": "basicData",
        "onlyPostsNewerThan": "14 days",
    }
    build = (actor_build or os.environ.get("APIFY_INSTAGRAM_BUILD", "")).strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", build):
        raise IngestionUnavailable("Verified, pinned APIFY_INSTAGRAM_BUILD required")
    # This is a provider-enforced total charge cap, not a predicted bill. Credits
    # and remaining cash are reserved separately by the shared account ledger.
    maximum_charge = max(0.005, len(cleaned_handles) * per_handle_cap * 0.002)
    from .spend_guard import get_spend_guard
    now_utc = datetime.now(timezone.utc)
    fetched_at = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    response_status = None
    request_timeout = httpx.Timeout(timeout_s, connect=min(float(timeout_s), 15.0))
    batch_health = health if isinstance(health, dict) else {}
    batch_health.update(instagram_requested_handle_count=len(cleaned_handles),
        instagram_unavailable_handle_count=0, instagram_skipped_record_count=0,
        instagram_normalized_post_count=0, instagram_verified_collaboration_record_count=0,
        instagram_degraded=False, instagram_unavailable_reasons={})
    unavailable = set()
    unavailable_reasons = {}
    try:
        guard = spend_guard or get_spend_guard(credential_keys={"apify": token})
        def request():
            nonlocal response_status
            with httpx.Client(timeout=request_timeout, follow_redirects=False) as client:
                response = client.post(
                    APIFY_INSTAGRAM_ENDPOINT,
                    headers={"Authorization": "Bearer " + token},
                    params={"build": build, "maxTotalChargeUsd": maximum_charge,
                            "restartOnError": "false", "timeout": 300},
                    json=payload,
                    timeout=request_timeout,
                )
                response_status = response.status_code
                response.raise_for_status()
                return response.json()
        from .provider_lifecycle import begin_lifecycle, fetch_apify_lifecycle, request_digest
        digest = request_digest({"provider": "apify", "actor": "apify~instagram-post-scraper",
            "build": build, "maxTotalChargeUsd": maximum_charge, "timeout": 300, "input": payload})
        lifecycle = begin_lifecycle(guard, "apify", "instagram-post-scraper", maximum_charge, digest)
        if lifecycle is None:
            dataset = guard.call("apify", "instagram-post-scraper", maximum_charge, request)
        else:
            with httpx.Client(timeout=request_timeout, follow_redirects=False) as client:
                dataset = fetch_apify_lifecycle(client, token, payload, build, maximum_charge,
                                               timeout_s, lifecycle, digest)
            response_status = 200
    except SpendBlocked:
        raise
    except Exception as exc:
        detail = _sanitized_provider_error(exc, token)
        logger.warning("Instagram collection unavailable: %s", json.dumps(detail, sort_keys=True))
        # Never blindly retry: even a lost response may have started a paid run.
        public = public_ingestion_diagnostics(dict(detail, build=build), token=token)
        raise IngestionUnavailable("Apify Instagram request failed; retain last publication", public_diagnostics=public, collection_health=batch_health) from None

    def dataset_failure(message, error_type, code):
        detail = public_ingestion_diagnostics({"provider": "apify", "status": response_status,
            "error_type": error_type, "code": code, "build": build}, token=token)
        return IngestionUnavailable(message, public_diagnostics=detail, collection_health=batch_health)

    if not isinstance(dataset, list):
        raise dataset_failure("Apify returned an invalid dataset envelope", "InvalidDatasetEnvelope", "invalid-dataset-envelope")

    batch_health["instagram_dataset_record_count"] = len(dataset)

    def username(value):
        return value.lstrip("@").strip().lower() if isinstance(value, str) else ""

    def verified_unavailable(record):
        if (not isinstance(record, dict) or not isinstance(record.get('error'), str)
                or record['error'] not in {"not_found", "no_items"}):
            return ""
        try:
            url = urlsplit(record.get("url") or "")
            handle = url.path.rstrip('/').lstrip('/').lower()
            stated_handle = username(record.get('username'))
            if (handle not in seen_handles
                    or (record['error'] == 'not_found' and stated_handle != handle)
                    or (stated_handle and stated_handle != handle)):
                return ""
            if (url.scheme != "https" or url.netloc.lower() not in {"instagram.com", "www.instagram.com"}
                    or url.path.rstrip("/").lower() != "/" + handle or url.query or url.fragment
                    or record.get("latestPosts")):
                return ""
            if record.get('inputUrl'):
                original = urlsplit(record['inputUrl'])
                if (original.scheme != 'https'
                        or original.netloc.lower() not in {'instagram.com', 'www.instagram.com'}
                        or original.path.rstrip('/').lower() != '/' + handle
                        or original.query or original.fragment):
                    return ""
        except (TypeError, ValueError):
            return ""
        return handle

    def requested_identity(post, profile_handle):
        owner = username(post.get("ownerUsername") or post.get("username"))
        if owner in seen_handles:
            return owner, {"owner_username": owner, "requested_username": owner,
                           "association": "owner", "verified_by": "ownerUsername" if post.get("ownerUsername") else "username"}
        producers = post.get("coauthorProducers")
        coauthors = {username(producer.get("username")) for producer in producers if isinstance(producer, dict)} if isinstance(producers, list) else set()
        # Requested order is stable even if the provider reorders coauthors.
        matched = next((handle for handle in cleaned_handles if handle in coauthors), None)
        if owner and matched:
            batch_health["instagram_verified_collaboration_record_count"] += 1
            return matched, {"owner_username": owner, "requested_username": matched,
                             "association": "coauthor", "verified_by": "coauthorProducers.username"}
        if not owner and profile_handle in seen_handles:
            return profile_handle, {"owner_username": "", "requested_username": profile_handle,
                                    "association": "profile_result", "verified_by": "profile.username"}
        raise dataset_failure("Apify dataset contains an unknown profile result", "UnknownProfileResult", "unrecognized-profile")

    collected: list[dict[str, Any]] = []
    for record in dataset:
        missing = verified_unavailable(record)
        if missing:
            unavailable.add(missing)
            unavailable_reasons[missing] = record['error']
            batch_health.update(instagram_unavailable_handle_count=len(unavailable),
                instagram_skipped_record_count=batch_health["instagram_skipped_record_count"] + 1,
                instagram_degraded=True, instagram_unavailable_reasons={
                    reason: list(unavailable_reasons.values()).count(reason)
                    for reason in set(unavailable_reasons.values())})
            continue
        if not isinstance(record, dict) or record.get("error") or record.get("errorDescription"):
            code = record.get("error") if isinstance(record, dict) else None
            if isinstance(code, dict):
                code = code.get("type") or code.get("code")
            raise dataset_failure("Apify dataset contains a failed or invalid profile result", "ProfileResultError", code or "invalid-profile-record")

        latest_posts = record.get("latestPosts")
        if isinstance(latest_posts, list):
            profile_handle = username(record.get("username"))
            if profile_handle not in seen_handles:
                raise dataset_failure("Apify dataset contains an unknown profile result", "UnknownProfileResult", "unrecognized-profile")
            post_records = latest_posts
        else:
            profile_handle = ""
            post_records = [record]
        handle_items: list[dict[str, Any]] = []
        for post in post_records:
            if not isinstance(post, dict) or post.get("error") or post.get("errorDescription"):
                code = post.get("error") if isinstance(post, dict) else None
                if isinstance(code, dict):
                    code = code.get("type") or code.get("code")
                raise dataset_failure("Apify dataset contains a failed or invalid post result", "PostResultError", code or "invalid-post-record")
            handle, provenance = requested_identity(post, profile_handle)
            tier = normalized_tier_map.get(handle, "O3")
            source_tier = _resolve_source_tier(tier)
            bucket = "ORBIT" if tier.startswith("O") else "MACRO"
            origin = _resolve_origin(tier)
            weight = _resolve_orbit_weight(tier)
            source_prefix = tier.lower().split("_")[0] if tier.startswith("O") else "o3"
            source_id = f"{source_prefix}_ig_{handle}"
            source_name = "Instagram Orbit Dispatch"
            short_code = str(post.get("shortCode") or post.get("id") or "").strip()
            raw_url = str(post.get("url") or "").strip()
            if not raw_url:
                if short_code:
                    raw_url = f"https://www.instagram.com/p/{short_code}"
                else:
                    raw_url = f"https://www.instagram.com/{handle}"

            canonical_url = normalize_url(raw_url)
            if not canonical_url:
                continue

            raw_caption = str(post.get("caption") or post.get("text") or "")
            mentioned_handles = [
                m.rstrip(".").lower()
                for m in _MENTION_RE.findall(raw_caption)
                if len(m.rstrip(".")) >= 2
            ]
            sanitized_caption = re.sub(r"@([A-Za-z0-9._]+)", r"\1", raw_caption)
            summary = _clean_text(sanitized_caption, max_chars=600)
            raw_title = str(post.get("title") or "")
            sanitized_title = re.sub(r"@([A-Za-z0-9._]+)", r"\1", raw_title)
            title = _clean_text(
                sanitized_title or summary[:120] or "Instagram orbit dispatch",
                max_chars=240,
            )
            dt = _parse_timestamp(
                str(post.get("timestamp") or post.get("takenAtTimestamp") or ""),
                fallback_utc=now_utc,
            )
            handle_items.append(
                {
                    "url_hash": compute_url_hash(canonical_url),
                    "url": canonical_url,
                    "title": title,
                    "summary": summary,
                    "summary_clean": summary,
                    "caption": sanitized_caption,
                    "mentioned_handles": mentioned_handles,
                    "source_id": source_id,
                    "source_name": source_name,
                    "instagram_provenance": provenance,
                    "tier": tier,
                    "source_tier": source_tier,
                    "bucket": bucket,
                    "origin": origin,
                    "weight": weight,
                    "orbit_weight": weight,
                    "published_at": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "published_date": dt.strftime("%Y-%m-%d"),
                    "fetched_at": fetched_at,
                    "schema_version": 1,
                    "expires_at": int(dt.timestamp()) + FEED_CACHE_TTL_SECONDS,
                }
            )

        collected.extend(handle_items)

    order = {handle: index for index, handle in enumerate(cleaned_handles)}
    collected.sort(key=lambda item: order[item["instagram_provenance"]["requested_username"]])
    # URL hashes deduplicate a post exposed through several requested profiles.
    result = _enforce_daily_source_cap(collected, max_per_day=MAX_ITEMS_PER_SOURCE_PER_DAY)
    batch_health["instagram_normalized_post_count"] = len(result)
    return result


_EXTENDED_MACRO_SCOUT_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "m4_scout_reddit_ml",
        "name": "Reddit r/MachineLearning",
        "tier": "M4_SCOUT",
        "bucket": "MACRO",
        "origin": "macro_scout",
        "type": "rss",
        "url": "https://www.reddit.com/r/MachineLearning/top/.rss?t=day",
        "weight": 1.0,
    },
    {
        "id": "m4_scout_reddit_tech",
        "name": "Reddit r/technology",
        "tier": "M4_SCOUT",
        "bucket": "MACRO",
        "origin": "macro_scout",
        "type": "rss",
        "url": "https://www.reddit.com/r/technology/top/.rss?t=day",
        "weight": 1.0,
    },
    {
        "id": "m4_scout_arxiv_ai",
        "name": "ArXiv cs.AI & cs.LG",
        "tier": "M4_SCOUT",
        "bucket": "MACRO",
        "origin": "macro_scout",
        "type": "rss",
        "url": "https://rss.arxiv.org/rss/cs.AI+cs.LG",
        "weight": 1.0,
    },
)

APIFY_GOOGLE_SEARCH_ENDPOINT = (
    "https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items"
)


def fetch_macro_scout_signals(
    timeout_s: float = 8.0,
    *,
    include_extended_scouts: bool = False,
    apify_token: str | None = None,
) -> list[dict[str, Any]]:
    """Query HN Algolia front-page API and Google Trends Daily US RSS (plus optional Reddit/ArXiv/Apify fallback), tagged M4_SCOUT and capped at 3/source/day."""
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {
        "User-Agent": "SiyuanZhu-Portfolio-OrbitCollector/2.0 (+https://github.com/Siyuan-Zhu)"
    }
    collected: list[dict[str, Any]] = []
    sources = list(_MACRO_SCOUT_SOURCES)
    if include_extended_scouts:
        sources.extend(_EXTENDED_MACRO_SCOUT_SOURCES)

    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_s) as client:
        for scout_cfg in sources:
            try:
                items = _fetch_single_feed(client, dict(scout_cfg), timeout_s, now_utc)
                for item in items:
                    item["tier"] = "M4_SCOUT"
                    item["source_tier"] = "MACRO"
                    item["bucket"] = "MACRO"
                    item["origin"] = "macro_scout"
                    item["weight"] = 1.0
                    item["orbit_weight"] = 1.0
                collected.extend(items)
            except Exception as exc:
                logger.warning(
                    "Macro scout source %s failed: %s",
                    scout_cfg.get("id", "unknown"),
                    exc,
                )
                continue


    return _enforce_daily_source_cap(collected, max_per_day=MAX_ITEMS_PER_SOURCE_PER_DAY)


def pack_f16_base64(vec: Any) -> str:
    """Pack a 768-dim float vector or 1536-byte float16 buffer into a base64-encoded string."""
    if isinstance(vec, (bytes, bytearray)) and len(vec) == EMBEDDING_DIM * 2:
        return base64.b64encode(bytes(vec)).decode("ascii")
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return base64.b64encode(np.asarray(arr, dtype=np.float16).tobytes()).decode("ascii")


def unpack_f16_base64(encoded: str | bytes | bytearray) -> np.ndarray:
    """Unpack a base64-encoded or raw byte float16 buffer into a float32 numpy vector."""
    if isinstance(encoded, (bytes, bytearray)):
        raw = bytes(encoded)
    else:
        raw = base64.b64decode(str(encoded).encode("ascii"))
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32).copy()


def _deterministic_fallback_embedding(text: str, dim: int = EMBEDDING_DIM) -> bytes:
    """Generate a deterministic unit-norm float16 vector (1536 bytes for dim=768) hashed from text."""
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return np.asarray(vec, dtype=np.float16).tobytes()


def _extract_cached_bytes(cached_entry: Any) -> bytes | None:
    if isinstance(cached_entry, (bytes, bytearray)) and len(cached_entry) == EMBEDDING_DIM * 2:
        return bytes(cached_entry)
    if isinstance(cached_entry, dict):
        raw = cached_entry.get("embedding_f16") or cached_entry.get("embedding_f16_b64")
        if isinstance(raw, (bytes, bytearray)) and len(raw) == EMBEDDING_DIM * 2:
            return bytes(raw)
        if isinstance(raw, str) and raw.strip():
            try:
                decoded = base64.b64decode(raw.strip().encode("ascii"))
                if len(decoded) == EMBEDDING_DIM * 2:
                    return decoded
            except Exception:
                return None
    return None


def _pack_embedding_vector(raw_emb: Any) -> bytes | None:
    values = None
    if hasattr(raw_emb, "values"):
        values = getattr(raw_emb, "values")
    elif isinstance(raw_emb, dict) and "values" in raw_emb:
        values = raw_emb["values"]
    elif isinstance(raw_emb, (list, tuple, np.ndarray)):
        values = raw_emb

    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != EMBEDDING_DIM or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8 or not np.isfinite(norm):
        return None
    arr = arr / norm
    return np.asarray(arr, dtype=np.float16).tobytes()


def _live_cached_embedding(entry: Any, model: str) -> bytes | None:
    """Unknown model provenance and offline hash vectors cannot enter live clusters."""
    if not isinstance(entry, dict) or entry.get("embedding_model") != model or entry.get("embedding_source") != "gemini":
        return None
    packed = _extract_cached_bytes(entry)
    if packed is None:
        return None
    values = np.frombuffer(packed, dtype=np.float16).astype(np.float32)
    if not np.all(np.isfinite(values)) or float(np.linalg.norm(values)) <= 1e-8:
        return None
    return packed


def embed_missing_items(
    items: list[dict[str, Any]],
    existing_cache: dict[str, dict[str, Any]] | None = None,
    genai_client: Any | None = None,
    *,
    client: Any | None = None,
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    require_live: bool = False,
) -> list[dict[str, Any]]:
    """Embed deltas; live mode admits only verified vectors from the approved model."""
    if not items:
        return []
    if require_live and model != EMBEDDING_MODEL:
        raise IngestionUnavailable("Live embedding model differs from the approved semantic space")

    active_client = genai_client if genai_client is not None else client
    cache_ref = existing_cache if isinstance(existing_cache, dict) else {}
    enriched: list[dict[str, Any]] = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for idx, raw_item in enumerate(items):
        item = raw_item if isinstance(raw_item, dict) else dict(raw_item)
        url_hash = str(item.get("url_hash") or item.get("url_sha256") or "").strip()
        if not url_hash:
            url_hash = compute_url_hash(str(item.get("url") or item.get("title") or idx))
            item["url_hash"] = url_hash

        cached_entry = cache_ref.get(url_hash)
        cached_bytes = (_live_cached_embedding(item, model) or _live_cached_embedding(cached_entry, model)
                        if require_live else _extract_cached_bytes(item) or _extract_cached_bytes(cached_entry))
        if cached_bytes is not None:
            item["embedding_f16"] = cached_bytes
            item["embedding_f16_b64"] = base64.b64encode(cached_bytes).decode("ascii")
            existing_src = item.get("embedding_source") or (
                cached_entry.get("embedding_source") if isinstance(cached_entry, dict) else None
            )
            item["embedding_source"] = str(existing_src or "gemini")
            if require_live:
                item["embedding_model"] = model
                item["embedding_source"] = "gemini"
        else:
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or item.get("caption") or "").strip()
            text = f"{title}\n{summary}".strip() or url_hash
            uncached_indices.append(idx)
            uncached_texts.append(text)

        enriched.append(item)

    if not uncached_indices:
        return enriched
    if require_live and active_client is None:
        raise IngestionUnavailable("Required live embedding provider unavailable; retain last publication")

    step = max(1, int(batch_size))
    for start in range(0, len(uncached_indices), step):
        batch_idx_slice = uncached_indices[start : start + step]
        batch_text_slice = uncached_texts[start : start + step]
        packed_batch: list[bytes | None] = [None] * len(batch_idx_slice)

        if active_client is not None:
            batch_failed = False
            try:
                response = active_client.models.embed_content(
                    model=model,
                    contents=batch_text_slice,
                    config={"output_dimensionality": EMBEDDING_DIM},
                )
                raw_embeddings = getattr(response, "embeddings", None) or []
                for pos, raw_emb in enumerate(raw_embeddings[: len(batch_idx_slice)]):
                    packed_batch[pos] = _pack_embedding_vector(raw_emb)
                if any(b is None for b in packed_batch):
                    batch_failed = True
            except SpendBlocked:
                raise
            except Exception as exc:
                if require_live:
                    raise IngestionUnavailable("Required live embedding request failed; retain last publication") from None
                logger.warning(
                    "Batch embedding call failed for %d items (%s); retrying items individually.",
                    len(batch_idx_slice),
                    exc,
                )
                packed_batch = [None] * len(batch_idx_slice)
                batch_failed = True

            if require_live and batch_failed:
                raise IngestionUnavailable("Required live embedding response incomplete or invalid; retain last publication")
            if batch_failed and len(batch_idx_slice) > 1:
                for pos, single_text in enumerate(batch_text_slice):
                    if packed_batch[pos] is not None:
                        continue
                    try:
                        single_resp = active_client.models.embed_content(
                            model=model,
                            contents=[single_text],
                            config={"output_dimensionality": EMBEDDING_DIM},
                        )
                        single_embs = getattr(single_resp, "embeddings", None) or []
                        if single_embs:
                            packed_batch[pos] = _pack_embedding_vector(single_embs[0])
                    except SpendBlocked:
                        raise
                    except Exception as single_exc:
                        target_hash = enriched[batch_idx_slice[pos]].get("url_hash", "unknown")
                        logger.warning(
                            "Single-item embedding retry failed for %s: %s; using deterministic fallback.",
                            target_hash,
                            single_exc,
                        )

        for pos, item_idx in enumerate(batch_idx_slice):
            emb_bytes = packed_batch[pos]
            if emb_bytes is not None:
                emb_source = "gemini"
            else:
                emb_bytes = _deterministic_fallback_embedding(batch_text_slice[pos])
                emb_source = "deterministic_fallback"
            target_item = enriched[item_idx]
            target_item["embedding_f16"] = emb_bytes
            target_item["embedding_f16_b64"] = base64.b64encode(emb_bytes).decode("ascii")
            target_item["embedding_source"] = emb_source
            target_item["embedding_model"] = model if emb_source == "gemini" else "deterministic-sha256-v1"
            u_hash = str(target_item.get("url_hash") or target_item.get("url_sha256") or "")
            if u_hash:
                cache_ref[u_hash] = dict(target_item)

    return enriched




