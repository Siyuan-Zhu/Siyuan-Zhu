"""Persistence Seam (`StateStore` Port & Adapters) for the Reactive Profile.

Implements the Ports & Adapters pattern (Category 2/3 in `DEEPENING.md`):
  - `StateStore`: Protocol (Port) defining the minimal persistence interface for
    Community Game state, Weighted Pulse state, 30-day rolling word history, and
    1-vote-per-day HMAC-deduplicated mood tallies.
  - `InMemoryStateStore`: Pure in-memory adapter for fast, isolated unit tests.
  - `JsonFileStateStore`: Local repository JSON adapter reading/writing
    `data/game_state.json` and `data/zeitgeist_state.json`.
  - `FirestoreStateStore`: Cloud Firestore Native/Enterprise adapter using
    `google.cloud.firestore` (lazy-imported so offline runs have zero external
    dependencies, with injectable client support for unit testing).
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Protocol, runtime_checkable
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_MOOD_KEYS: tuple[str, ...] = (
    "spot_on",
    "too_sunny",
    "too_heavy",
)
LEGACY_MOOD_ALIASES: dict[str, str] = {
    "resonant": "spot_on",
    "reflective": "too_sunny",
    "wistful": "too_sunny",
    "electric": "too_sunny",
    "skeptical": "too_heavy",
}
DEFAULT_MOOD_TALLY: dict[str, int] = {
    "spot_on": 20,
    "too_sunny": 3,
    "too_heavy": 4,
}


def _normalize_mood_tally(raw_tally: dict[str, Any] | None) -> dict[str, int]:
    """Normalize a mood vote dictionary to the 3 canonical keys (`spot_on`, `too_sunny`, `too_heavy`)."""
    if not isinstance(raw_tally, dict) or not raw_tally:
        return dict(DEFAULT_MOOD_TALLY)
    has_canonical = any(k in raw_tally for k in VALID_MOOD_KEYS)
    has_legacy = any(str(k).strip().lower() in LEGACY_MOOD_ALIASES for k in raw_tally)
    if not has_canonical and not has_legacy:
        return dict(DEFAULT_MOOD_TALLY)
    canonical: dict[str, int] = {
        k: int(raw_tally.get(k, DEFAULT_MOOD_TALLY[k] if not has_legacy else 0))
        for k in VALID_MOOD_KEYS
    }
    for raw_k, raw_v in raw_tally.items():
        k_norm = str(raw_k).strip().lower()
        if k_norm in LEGACY_MOOD_ALIASES:
            target_k = LEGACY_MOOD_ALIASES[k_norm]
            try:
                canonical[target_k] = int(canonical.get(target_k, 0)) + int(raw_v)
            except (TypeError, ValueError):
                pass
    return canonical


PERSONAL_ORBIT_BUCKETS: frozenset[str] = frozenset(
    {"ORBIT", "O1", "O2", "O3", "ORBIT_SCOUT", "PERSONAL_ORBIT"}
)
M3_RETENTION_HOURS: float = 96.0
CACHE_PAGE_MAX_BYTES = 800 * 1024
CACHE_COMMIT_MAX_BYTES = 8 * 1024 * 1024
CACHE_PAGE_MAX_ITEMS = 500


def _is_personal_orbit_item(item: dict[str, Any]) -> bool:
    """Return True if `item` belongs to a Personal Orbit (`O1`-`O3` or 1-hop Orbit Scout) bucket."""
    bucket = str(item.get("bucket") or "").strip().upper()
    origin = str(item.get("origin") or "").strip()
    return bucket in PERSONAL_ORBIT_BUCKETS or origin in (
        "curated_O1_O3",
        "orbit_scout_1hop",
    )


def _item_max_age_hours(item: dict[str, Any]) -> float:
    """Retain Personal Orbit for 336h and every Macro tier for 96h."""
    bucket = str(item.get("bucket") or "").strip().upper()
    if bucket in PERSONAL_ORBIT_BUCKETS or _is_personal_orbit_item(item):
        return 336.0
    if bucket == "M3":
        return M3_RETENTION_HOURS
    return 96.0


def _encode_json_safe(val: Any) -> Any:
    """Recursively encode bytes/bytearray as base64 dicts for local JSON file storage."""
    if isinstance(val, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(val)).decode("ascii")}
    if isinstance(val, list):
        return [_encode_json_safe(x) for x in val]
    if isinstance(val, dict):
        return {
            str(k): _encode_json_safe(v)
            for k, v in val.items()
            if str(k) != "embedding_f32"
        }
    return val


def _decode_json_safe(val: Any) -> Any:
    """Recursively decode base64 dicts back into raw bytes from local JSON file storage."""
    if isinstance(val, list):
        return [_decode_json_safe(x) for x in val]
    if isinstance(val, dict):
        if set(val.keys()) == {"__bytes_b64__"} and isinstance(val["__bytes_b64__"], str):
            return base64.b64decode(val["__bytes_b64__"])
        return {k: _decode_json_safe(v) for k, v in val.items()}
    return val


def _parse_iso_utc(raw_ts: str | None, fallback_date: str | None = None) -> datetime | None:
    """Parse an ISO-8601 timestamp or YYYY-MM-DD date string into a timezone-aware UTC datetime."""
    candidate = (raw_ts or "").strip()
    if candidate:
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    if fallback_date:
        try:
            dt = datetime.strptime(fallback_date.strip()[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            return dt
        except ValueError:
            pass
    return None


def _cache_shard_date(doc_id: str, payload: dict) -> str:
    return str(payload.get("date") or doc_id)[:10]


def _cache_payload_bytes(payload: dict) -> int:
    return len(json.dumps({"fields": {key: _py_to_firestore_value(value)
        for key, value in payload.items()}}, ensure_ascii=True).encode("utf-8"))


def pack_ingestion_cache_pages(date_str: str, items: dict, *, reserve_embedding: bool = False) -> dict[str, dict]:
    """Losslessly page server-owned caches, bounded below Firestore request limits.

    OAuth/IAM writers bypass Firebase client rules. Existing client rules intentionally
    do not grant access to these private date__NNN pages or the page_count marker.
    reserve_embedding validates source metadata before paying for 768-d float16 vectors.
    """
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date_str)) or _parse_iso_utc(date_str) is None:
        raise ValueError("Cache page requires an ISO publication date")
    groups, current, size = [], {}, 1024
    for key, row in sorted(items.items()):
        cleaned = {k: copy.deepcopy(v) for k, v in row.items()
            if k not in {"embedding_f32", "embedding_f16_b64"}}
        raw = row.get("embedding_f16") or row.get("embedding_f16_b64")
        if reserve_embedding:
            raw = bytes(1536)
            cleaned["embedding_model"] = "gemini-embedding-001"
            cleaned["embedding_source"] = "gemini"
        elif isinstance(raw, str):
            raw = base64.b64decode(raw, validate=True)
        if isinstance(raw, (bytes, bytearray)):
            cleaned["embedding_f16"] = bytes(raw)
        encoded = len(json.dumps({str(key): _py_to_firestore_value(cleaned)}, ensure_ascii=True).encode())
        if encoded + 1024 > CACHE_PAGE_MAX_BYTES:
            raise ValueError("Single ingestion item exceeds the cache page byte ceiling; no context truncated")
        if current and (size + encoded > CACHE_PAGE_MAX_BYTES or len(current) >= CACHE_PAGE_MAX_ITEMS):
            groups.append(current)
            current, size = {}, 1024
        current[str(key)] = cleaned
        size += encoded
    groups.append(current)
    if len(groups) > 500:
        raise ValueError("Ingestion cache exceeds the atomic page count ceiling")
    pages = {(date_str if index == 0 else f"{date_str}__{index:03d}"):
        {"date": date_str, "items": group} for index, group in enumerate(groups)}
    if len(groups) > 1:
        pages[date_str]["page_count"] = len(groups)
    sizes = [_cache_payload_bytes(payload) for payload in pages.values()]
    if any(size > CACHE_PAGE_MAX_BYTES for size in sizes) or sum(sizes) + len(pages) * 2048 > CACHE_COMMIT_MAX_BYTES:
        raise ValueError("Ingestion cache exceeds the atomic commit byte ceiling; no items dropped")
    return pages


def _active_cache_pages(shards: dict[str, dict]) -> dict[str, dict]:
    """Ignore retired pages after an atomic repack, including before physical cleanup."""
    active = {}
    for doc_id, payload in shards.items():
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:__(\d{3,}))?", doc_id)
        if not match:
            continue
        base, index = match.group(1), int(match.group(2) or 0)
        count = (shards.get(base) or {}).get("page_count")
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
            raise ValueError("Invalid ingestion cache page count")
        if count is None or index < count:
            active[doc_id] = payload
    return active


def _prune_shard_dict_in_place(
    shards: dict[str, dict[str, Any]],
    now_utc: datetime,
) -> tuple[int, list[str], list[str]]:
    """Prune expired shards (>14d) and expired Macro items (>96h) from `shards`.

    Returns `(pruned_count, deleted_shard_dates, updated_shard_dates)`.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    cutoff_14d_str = (now_utc - timedelta(days=14)).strftime("%Y-%m-%d")
    pruned_count = 0
    deleted_dates: list[str] = []
    updated_dates: list[str] = []

    for shard_date in list(shards.keys()):
        if _cache_shard_date(shard_date, shards.get(shard_date) or {}) < cutoff_14d_str:
            shard_doc = shards.pop(shard_date, {}) or {}
            items_map = shard_doc.get("items") or {}
            pruned_count += max(1, len(items_map))
            deleted_dates.append(shard_date)
            continue

        shard_doc = shards.get(shard_date) or {}
        items_map = shard_doc.get("items")
        if not isinstance(items_map, dict):
            continue

        changed = False
        for url_hash in list(items_map.keys()):
            raw_item = items_map[url_hash]
            if not isinstance(raw_item, dict):
                continue
            max_age_hours = _item_max_age_hours(raw_item)
            pub_dt = _parse_iso_utc(
                str(raw_item.get("published_at") or ""),
                fallback_date=str(shard_doc.get("date") or shard_date),
            )
            if pub_dt is not None:
                age_hours = (now_utc - pub_dt).total_seconds() / 3600.0
                if age_hours > max_age_hours:
                    items_map.pop(url_hash, None)
                    pruned_count += 1
                    changed = True

        if changed:
            updated_dates.append(shard_date)

    return pruned_count, deleted_dates, updated_dates


def _hydrate_ingestion_cache_from_shards(
    shards: dict[str, dict[str, Any]],
    now_utc: datetime,
) -> dict[str, dict[str, Any]]:
    """Filter shard items by asymmetric retention (14d Orbit / 96h Macro) and decode float16 vectors."""
    import numpy as np  # Lazy import so stdlib-only callers never load numpy

    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    cutoff_date_str = (now_utc - timedelta(days=14)).strftime("%Y-%m-%d")
    merged: dict[str, dict[str, Any]] = {}

    for shard_date in sorted(_active_cache_pages(shards)):
        if _cache_shard_date(shard_date, shards[shard_date] or {}) < cutoff_date_str:
            continue
        shard_doc = shards[shard_date] or {}
        items_map = shard_doc.get("items")
        if not isinstance(items_map, dict):
            continue
        for url_hash, raw_item in items_map.items():
            if not isinstance(raw_item, dict):
                continue
            item = copy.deepcopy(raw_item)
            max_age_hours = _item_max_age_hours(item)

            pub_dt = _parse_iso_utc(
                str(item.get("published_at") or ""),
                fallback_date=str(shard_doc.get("date") or shard_date),
            )
            if pub_dt is not None:
                age_hours = (now_utc - pub_dt).total_seconds() / 3600.0
                if age_hours > max_age_hours:
                    continue

            raw_emb = item.get("embedding_f16")
            if isinstance(raw_emb, str):
                raw_emb = base64.b64decode(raw_emb)
            elif isinstance(raw_emb, bytearray):
                raw_emb = bytes(raw_emb)
            if isinstance(raw_emb, bytes) and len(raw_emb) > 0:
                item["embedding_f16"] = raw_emb
                item["embedding_f32"] = np.frombuffer(raw_emb, dtype=np.float16).astype(
                    np.float32
                )
            merged[str(url_hash)] = item

    return merged


def compute_voter_hmac(
    client_ip: str,
    user_agent: str,
    date_str: str,
    salt: str | None = None,
) -> str:
    """Compute a daily salted HMAC-SHA256 digest without storing raw IP or User-Agent."""
    secret = (
        salt
        if salt is not None
        else os.environ.get("PULSE_VOTE_SALT", "siyuan-zhu-daily-pulse-salt")
    )
    material = f"{date_str.strip()}|{client_ip.strip()}|{user_agent.strip()}"
    digest = hmac.new(
        secret.encode("utf-8"),
        material.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest[:32]


def _extract_recent_words(state: dict[str, Any], days: int = 30) -> list[str]:
    """Extract up to `days` unique recent uppercase words from a pulse state dictionary."""
    limit = max(1, int(days))
    words: list[str] = []

    for entry in state.get("recent_words", []):
        if isinstance(entry, str) and entry.strip():
            w = entry.strip().upper()
            if w not in words:
                words.append(w)
        elif isinstance(entry, dict):
            w = str(entry.get("word") or entry.get("word_of_the_day") or "").strip().upper()
            if w and w not in words:
                words.append(w)

    for entry in state.get("word_history", []):
        if isinstance(entry, dict):
            w = str(entry.get("word") or entry.get("word_of_the_day") or "").strip().upper()
            if w and w not in words:
                words.append(w)
        elif isinstance(entry, str) and entry.strip():
            w = entry.strip().upper()
            if w not in words:
                words.append(w)

    current_word = str(
        (state.get("analysis") or {}).get("word_of_the_day")
        or state.get("word")
        or ""
    ).strip().upper()
    if current_word and current_word not in words:
        words.insert(0, current_word)

    return words[:limit]


def _enrich_pulse_state_with_history(
    state: dict[str, Any],
    existing: dict[str, Any] | None = None,
    max_days: int = 30,
) -> dict[str, Any]:
    """Maintain a rolling 30-day list of recent words and preserve tally/rate-limits."""
    merged = copy.deepcopy(state)
    prior = existing or {}

    if "visitor_mood_tally" not in merged:
        merged["visitor_mood_tally"] = _normalize_mood_tally(
            prior.get("visitor_mood_tally", DEFAULT_MOOD_TALLY)
        )
    else:
        merged["visitor_mood_tally"] = _normalize_mood_tally(
            merged.get("visitor_mood_tally")
        )
    if isinstance(merged.get("mood_votes"), dict):
        merged["mood_votes"] = _normalize_mood_tally(merged["mood_votes"])
    if "vote_rate_limits" not in merged and "vote_rate_limits" in prior:
        merged["vote_rate_limits"] = copy.deepcopy(prior["vote_rate_limits"])

    combined_words: list[str] = []
    analysis = merged.get("analysis") or {}
    current_word = str(
        analysis.get("word_of_the_day") or merged.get("word") or ""
    ).strip().upper()
    if current_word:
        combined_words.append(current_word)

    for src in (merged, prior):
        for w in _extract_recent_words(src, days=max_days):
            if w not in combined_words:
                combined_words.append(w)

    merged["recent_words"] = combined_words[:max_days]
    return merged


def _normalize_mood_vote_inputs(
    date_str: str,
    mood_key: str,
    voter_hmac: str,
) -> tuple[bool, str, str, str, str]:
    """Normalize and validate mood vote parameters, returning `(is_valid, norm_date, norm_mood, norm_hmac, dedup_key)`."""
    raw_mood = (mood_key or "").strip().lower()
    norm_mood = LEGACY_MOOD_ALIASES.get(raw_mood, raw_mood)
    norm_date = (date_str or "default").strip()
    norm_hmac = (voter_hmac or "").strip()
    is_valid = norm_mood in VALID_MOOD_KEYS and bool(norm_hmac)
    return is_valid, norm_date, norm_mood, norm_hmac, f"{norm_date}_{norm_hmac}"


def _apply_local_mood_vote(
    state: dict[str, Any],
    date_str: str,
    mood_key: str,
    voter_hmac: str,
) -> tuple[bool, dict[str, int]]:
    """Mutate `state` with a deduplicated mood vote and return `(accepted, updated_tally)`."""
    tally = _normalize_mood_tally(state.get("visitor_mood_tally"))
    state["visitor_mood_tally"] = tally
    is_valid, _norm_date, norm_mood, _norm_hmac, dedup_key = _normalize_mood_vote_inputs(
        date_str, mood_key, voter_hmac
    )
    if not is_valid:
        return False, dict(tally)

    rate_limits: dict[str, str] = state.setdefault("vote_rate_limits", {})
    if dedup_key in rate_limits:
        return False, dict(tally)

    rate_limits[dedup_key] = norm_mood
    tally[norm_mood] = int(tally.get(norm_mood, 0)) + 1
    if isinstance(state.get("mood_votes"), dict):
        state["mood_votes"] = dict(tally)
    return True, dict(tally)


def _strip_transient_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of `state` with transient runtime flags and inline base64 data URIs removed."""
    cleaned = copy.deepcopy(state)
    cleaned.pop("_needs_git_reconcile", None)
    if isinstance(cleaned.get("board_state"), dict):
        cleaned["board_state"].pop("_needs_git_reconcile", None)
    if str(cleaned.get("image_data_uri") or "").startswith("data:"):
        cleaned["image_data_uri"] = ""
    if isinstance(cleaned.get("visitor_mood_tally"), dict):
        cleaned["visitor_mood_tally"] = _normalize_mood_tally(
            cleaned["visitor_mood_tally"]
        )
    if isinstance(cleaned.get("mood_votes"), dict):
        cleaned["mood_votes"] = _normalize_mood_tally(cleaned["mood_votes"])
    if isinstance(cleaned.get("published_pulse"), dict):
        if str(cleaned["published_pulse"].get("image_data_uri") or "").startswith("data:"):
            cleaned["published_pulse"]["image_data_uri"] = ""
        if isinstance(cleaned["published_pulse"].get("mood_votes"), dict):
            cleaned["published_pulse"]["mood_votes"] = _normalize_mood_tally(
                cleaned["published_pulse"]["mood_votes"]
            )
    return cleaned


def _apply_game_rate_limit(
    rate_state: dict[str, Any],
    actor_handle: str,
    now_epoch: int,
    mode: str = "VS_BOT",
    game_completed: bool = False,
    is_new_round: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate and increment Connect 4 rate limits: 15s VS_BOT (45s shared/PVP_DUEL), 24/hr, 3 games/day."""
    updated = copy.deepcopy(rate_state or {})
    users: dict[str, dict[str, Any]] = updated.setdefault("users", {})
    handle = (actor_handle or "").strip().lower() or "anonymous"
    now_ts = int(now_epoch)
    today_str = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    user_rec = users.setdefault(
        handle,
        {
            "last_move_ts": 0,
            "hourly_timestamps": [],
            "daily_games_date": today_str,
            "daily_games_count": 0,
        },
    )
    if user_rec.get("daily_games_date") != today_str:
        user_rec["daily_games_date"] = today_str
        user_rec["daily_games_count"] = 0

    if (is_new_round or game_completed) and int(user_rec.get("daily_games_count", 0)) >= 3:
        return False, "daily_game_limit_exceeded", updated

    other_human_recent = any(
        other_u != handle
        and int(other_rec.get("last_move_ts", 0)) > 0
        and 0 <= (now_ts - int(other_rec.get("last_move_ts", 0))) <= 120
        for other_u, other_rec in users.items()
    )
    cooldown_sec = (
        45 if (str(mode).strip().upper() == "PVP_DUEL" or other_human_recent) else 15
    )
    last_ts = int(user_rec.get("last_move_ts", 0))
    if last_ts > 0 and (now_ts - last_ts) < cooldown_sec:
        return False, f"cooldown_active_{cooldown_sec}s", updated

    hourly = [
        int(ts)
        for ts in user_rec.get("hourly_timestamps", [])
        if 0 <= (now_ts - int(ts)) < 3600
    ]
    if len(hourly) >= 24:
        return False, "hourly_move_limit_exceeded", updated

    hourly.append(now_ts)
    user_rec["hourly_timestamps"] = hourly
    user_rec["last_move_ts"] = now_ts
    if game_completed or is_new_round:
        user_rec["daily_games_count"] = int(user_rec.get("daily_games_count", 0)) + 1

    # Prune inactive user entries older than 24h or when map exceeds 50 entries
    stale_users = [
        u
        for u, rec in users.items()
        if u != handle
        and rec.get("daily_games_date") != today_str
        and rec.get("last_reset_date") != today_str
        and (
            len(users) > 50
            or (
                int(rec.get("last_move_ts", 0)) > 0
                and (now_ts - int(rec.get("last_move_ts", 0))) > 86400
            )
        )
    ]
    for u in stale_users:
        users.pop(u, None)

    updated["state_version"] = int(updated.get("state_version", 0)) + 1
    return True, "ok", updated


def _increment_completed_game_count(
    rate_state: dict[str, Any],
    actor_handle: str,
    now_epoch: int,
) -> dict[str, Any]:
    """Increment daily completed game count for `actor_handle` on the current UTC date."""
    updated = copy.deepcopy(rate_state or {})
    users: dict[str, dict[str, Any]] = updated.setdefault("users", {})
    handle = (actor_handle or "").strip().lower() or "anonymous"
    now_ts = int(now_epoch)
    today_str = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    user_rec = users.setdefault(
        handle,
        {
            "last_move_ts": now_ts,
            "hourly_timestamps": [],
            "daily_games_date": today_str,
            "daily_games_count": 0,
        },
    )
    if user_rec.get("daily_games_date") != today_str:
        user_rec["daily_games_date"] = today_str
        user_rec["daily_games_count"] = 0
    user_rec["daily_games_count"] = int(user_rec.get("daily_games_count", 0)) + 1
    updated["state_version"] = int(updated.get("state_version", 0)) + 1
    return updated


@runtime_checkable
class StateStore(Protocol):
    """Port interface for persisting Community Game and Weighted Pulse state."""

    needs_git_reconcile: bool

    def load_game_state(self) -> dict[str, Any]:
        """Load the current Community Game state dictionary."""

    def save_game_state(self, state: dict[str, Any]) -> None:
        """Persist the updated Community Game state dictionary."""

    def load_pulse_state(self) -> dict[str, Any]:
        """Load the current Weighted Pulse state dictionary."""

    def save_pulse_state(self, state: dict[str, Any]) -> None:
        """Persist the updated Weighted Pulse state dictionary."""

    def record_mood_vote(
        self,
        date_str: str,
        mood_key: str,
        voter_hmac: str,
    ) -> tuple[bool, dict[str, int]]:
        """Record a 1-click mood vote with 1-vote-per-voter_hmac-per-day deduplication."""

    def get_recent_pulse_words(self, days: int = 30) -> list[str]:
        """Return up to `days` recently used pulse words for 30-day deduplication."""

    def load_ingestion_cache(self, now_utc: datetime) -> dict[str, dict[str, Any]]:
        """Load date-sharded ingestion cache items within retention bounds (14d Orbit / 96h Macro)."""

    def save_ingestion_cache_day(
        self,
        date_str: str,
        items_by_hash: dict[str, dict[str, Any]],
    ) -> None:
        """Persist a single daily ingestion cache shard document `ingestion_cache/{date_str}`."""

    def prune_expired_cache_items(self, now_utc: datetime) -> int:
        """Remove Macro items >96h from shards 5..14 and delete shards >14d."""

    def set_needs_git_reconcile(self, flag: bool) -> None:
        """Flag whether local Git state mirrors need reconciliation from Firestore."""

    def reconcile_git_from_firestore_if_needed(self) -> bool:
        """Sync authoritative higher-version Firestore state to local JSON mirrors if needed."""

    def reconcile_to_git(self) -> bool:
        """Alias for `reconcile_git_from_firestore_if_needed()`."""

    def record_completed_game(self, actor_handle: str, now_epoch: int) -> None:
        """Increment `daily_games_count` for `actor_handle` when a game concludes."""

    def load_game_rate_limits(self) -> dict[str, Any]:
        """Load `profile_state/game_rate_limits` dictionary."""

    def save_game_rate_limits(self, state: dict[str, Any]) -> None:
        """Persist `profile_state/game_rate_limits` dictionary."""

    def check_and_increment_game_rate_limit(
        self,
        actor_handle: str,
        now_epoch: int,
        mode: str = "VS_BOT",
        game_completed: bool = False,
        is_new_round: bool = False,
    ) -> tuple[bool, str]:
        """Enforce 15s VS_BOT (45s shared/PVP_DUEL), 24/hr, and 3 games/day rate limits."""

    def load_orbit_roster(self) -> dict[str, Any]:
        """Load 1-hop `OrbitScout` roster from `profile_state/orbit_roster`."""

    def save_orbit_roster(self, roster: dict[str, Any]) -> None:
        """Persist 1-hop `OrbitScout` roster to `profile_state/orbit_roster`."""

    def save_pulse_feed_entry(self, date_str: str, entry: dict[str, Any]) -> None:
        """Persist daily commissioned entry to `pulse_history/{date_str}` and `data/pulse_feed.json` (365-day window)."""

    def load_bot_memory(self) -> dict[str, Any]:
        """Load Tabular TD(lambda) state memory from `profile_state/bot_memory` / `data/bot_brain.json`."""

    def save_bot_memory(self, memory: dict[str, Any]) -> None:
        """Persist Tabular TD(lambda) state memory to `profile_state/bot_memory` and `data/bot_brain.json`."""

    def load_elo_ledger(self) -> dict[str, Any]:
        """Load Arpad ELO rating ledger from `profile_state/elo_ledger` / `data/hall_of_records.json`."""

    def save_elo_ledger(self, ledger: dict[str, Any]) -> None:
        """Persist Arpad ELO rating ledger to `profile_state/elo_ledger` and `data/hall_of_records.json`."""


class InMemoryStateStore:
    """Pure in-memory StateStore adapter for fast, isolated unit tests."""

    def __init__(
        self,
        initial_game_state: dict[str, Any] | None = None,
        initial_pulse_state: dict[str, Any] | None = None,
    ) -> None:
        self.needs_git_reconcile: bool = False
        self._game_state: dict[str, Any] | None = (
            copy.deepcopy(initial_game_state) if initial_game_state is not None else None
        )
        self._pulse_state: dict[str, Any] = _strip_transient_keys(
            initial_pulse_state
            or {
                "visitor_mood_tally": dict(DEFAULT_MOOD_TALLY),
                "recent_words": [],
                "vote_rate_limits": {},
            }
        )
        self._pulse_state["visitor_mood_tally"] = _normalize_mood_tally(
            self._pulse_state.get("visitor_mood_tally")
        )
        self._ingestion_cache_shards: dict[str, dict[str, Any]] = {}
        self._game_rate_limits: dict[str, Any] = {"users": {}, "state_version": 0}
        self._orbit_roster: dict[str, Any] = {
            "handles": {},
            "curated_o3_cursor": {},
            "state_version": 0,
        }
        self._bot_memory: dict[str, Any] = {
            "state_version": 0,
            "games_learned": 0,
            "states": {},
        }
        self._elo_ledger: dict[str, Any] = {
            "state_version": 0,
            "bot_elo": 1200,
            "players": {},
            "recent_rounds": [],
        }
        self._pulse_feed_entries: list[dict[str, Any]] = []

    def load_game_state(self) -> dict[str, Any]:
        if self._game_state is None:
            from src.game_engine import build_midgame_showcase_state

            self._game_state = build_midgame_showcase_state("connect4", round_id=1)
        return copy.deepcopy(self._game_state)

    def save_game_state(self, state: dict[str, Any]) -> None:
        self._game_state = _strip_transient_keys(state)

    def load_pulse_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._pulse_state)

    def save_pulse_state(self, state: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(state)
        self._pulse_state = _enrich_pulse_state_with_history(
            cleaned, existing=self._pulse_state
        )

    def set_needs_git_reconcile(self, flag: bool) -> None:
        self.needs_git_reconcile = bool(flag)

    def reconcile_git_from_firestore_if_needed(self) -> bool:
        if self.needs_git_reconcile:
            self.needs_git_reconcile = False
            return True
        return False

    def reconcile_to_git(self) -> bool:
        return self.reconcile_git_from_firestore_if_needed()

    def record_completed_game(self, actor_handle: str, now_epoch: int) -> None:
        self.save_game_rate_limits(
            _increment_completed_game_count(
                self.load_game_rate_limits(), actor_handle, now_epoch
            )
        )

    def record_mood_vote(
        self,
        date_str: str,
        mood_key: str,
        voter_hmac: str,
    ) -> tuple[bool, dict[str, int]]:
        return _apply_local_mood_vote(
            self._pulse_state, date_str, mood_key, voter_hmac
        )

    def get_recent_pulse_words(self, days: int = 30) -> list[str]:
        return _extract_recent_words(self._pulse_state, days=days)

    def save_ingestion_cache_day(
        self,
        date_str: str,
        items_by_hash: dict[str, dict[str, Any]],
    ) -> None:
        clean_items: dict[str, dict[str, Any]] = {}
        for k, v in items_by_hash.items():
            item_copy = {ik: iv for ik, iv in v.items() if ik != "embedding_f32"}
            clean_items[str(k)] = copy.deepcopy(item_copy)
        self._ingestion_cache_shards[str(date_str)] = {
            "date": str(date_str),
            "items": clean_items,
        }

    def load_ingestion_cache(self, now_utc: datetime) -> dict[str, dict[str, Any]]:
        return _hydrate_ingestion_cache_from_shards(self._ingestion_cache_shards, now_utc)

    def prune_expired_cache_items(self, now_utc: datetime) -> int:
        pruned, _deleted, _updated = _prune_shard_dict_in_place(
            self._ingestion_cache_shards, now_utc
        )
        return pruned

    def load_game_rate_limits(self) -> dict[str, Any]:
        return copy.deepcopy(self._game_rate_limits)

    def save_game_rate_limits(self, state: dict[str, Any]) -> None:
        self._game_rate_limits = copy.deepcopy(state)

    def check_and_increment_game_rate_limit(
        self,
        actor_handle: str,
        now_epoch: int,
        mode: str = "VS_BOT",
        game_completed: bool = False,
        is_new_round: bool = False,
    ) -> tuple[bool, str]:
        ok, reason, updated = _apply_game_rate_limit(
            self.load_game_rate_limits(),
            actor_handle=actor_handle,
            now_epoch=now_epoch,
            mode=mode,
            game_completed=game_completed,
            is_new_round=is_new_round,
        )
        if ok:
            self.save_game_rate_limits(updated)
        return ok, reason

    def load_orbit_roster(self) -> dict[str, Any]:
        return copy.deepcopy(self._orbit_roster)

    def save_orbit_roster(self, roster: dict[str, Any]) -> None:
        self._orbit_roster = _strip_transient_keys(roster)

    def save_pulse_feed_entry(self, date_str: str, entry: dict[str, Any]) -> None:
        item = _strip_transient_keys(entry)
        item["date"] = str(date_str)[:10]
        item["schema_version"] = "2.0"
        if str(item.get("image_data_uri") or "").startswith("data:"):
            item["image_data_uri"] = ""
        filtered = [e for e in self._pulse_feed_entries if str(e.get("date")) != str(date_str)]
        filtered.append(item)
        filtered.sort(key=lambda x: str(x.get("date", "")), reverse=True)
        self._pulse_feed_entries = filtered[:365]

    def load_bot_memory(self) -> dict[str, Any]:
        return copy.deepcopy(self._bot_memory)

    def save_bot_memory(self, memory: dict[str, Any]) -> None:
        self._bot_memory = _strip_transient_keys(memory)

    def load_elo_ledger(self) -> dict[str, Any]:
        return copy.deepcopy(self._elo_ledger)

    def save_elo_ledger(self, ledger: dict[str, Any]) -> None:
        self._elo_ledger = _strip_transient_keys(ledger)


class JsonFileStateStore:
    """Local JSON file adapter backed by `data/game_state.json` and `data/zeitgeist_state.json`."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.needs_git_reconcile: bool = False
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.game_state_path = self.repo_root / "data" / "game_state.json"
        self.pulse_state_path = self.repo_root / "data" / "zeitgeist_state.json"
        self.ingestion_cache_dir = self.repo_root / "data" / "ingestion_cache"
        self.game_rate_limits_path = self.repo_root / "data" / "game_rate_limits.json"
        self.orbit_roster_path = self.repo_root / "data" / "orbit_roster.json"
        self.pulse_feed_path = self.repo_root / "data" / "pulse_feed.json"
        self.bot_memory_path = self.repo_root / "data" / "bot_brain.json"
        self.elo_ledger_path = self.repo_root / "data" / "hall_of_records.json"

    def _peek_local_game_state(self) -> dict[str, Any] | None:
        if self.game_state_path.exists():
            return json.loads(self.game_state_path.read_text(encoding="utf-8"))
        return None

    def _peek_local_pulse_state(self) -> dict[str, Any] | None:
        if self.pulse_state_path.exists():
            return json.loads(self.pulse_state_path.read_text(encoding="utf-8"))
        return None

    def load_game_state(self) -> dict[str, Any]:
        from src.game_engine import build_midgame_showcase_state, normalize_game_state

        raw = self._peek_local_game_state()
        if raw is not None:
            return _strip_transient_keys(normalize_game_state(raw))

        default_state = build_midgame_showcase_state("connect4", round_id=1)
        self.save_game_state(default_state)
        return default_state

    def save_game_state(self, state: dict[str, Any]) -> None:
        from src.game_engine import normalize_game_state

        cleaned = _strip_transient_keys(state)
        normalized = _strip_transient_keys(normalize_game_state(cleaned))
        self.game_state_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(normalized, indent=2) + "\n"
        self.game_state_path.write_text(serialized, encoding="utf-8")

    def load_pulse_state(self) -> dict[str, Any]:
        if self.pulse_state_path.exists():
            state = _strip_transient_keys(
                json.loads(self.pulse_state_path.read_text(encoding="utf-8"))
            )
            state.setdefault("visitor_mood_tally", dict(DEFAULT_MOOD_TALLY))
            return state
        return {
            "visitor_mood_tally": dict(DEFAULT_MOOD_TALLY),
            "recent_words": [],
            "vote_rate_limits": {},
        }

    def save_pulse_state(self, state: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(state)
        existing = self.load_pulse_state() if self.pulse_state_path.exists() else {}
        enriched = _strip_transient_keys(
            _enrich_pulse_state_with_history(cleaned, existing=existing)
        )
        self.pulse_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.pulse_state_path.write_text(
            json.dumps(enriched, indent=2) + "\n",
            encoding="utf-8",
        )

    def set_needs_git_reconcile(self, flag: bool) -> None:
        self.needs_git_reconcile = bool(flag)

    def reconcile_git_from_firestore_if_needed(self) -> bool:
        if self.needs_git_reconcile:
            self.needs_git_reconcile = False
            return True
        return False

    def reconcile_to_git(self) -> bool:
        return self.reconcile_git_from_firestore_if_needed()

    def record_completed_game(self, actor_handle: str, now_epoch: int) -> None:
        self.save_game_rate_limits(
            _increment_completed_game_count(
                self.load_game_rate_limits(), actor_handle, now_epoch
            )
        )

    def record_mood_vote(
        self,
        date_str: str,
        mood_key: str,
        voter_hmac: str,
    ) -> tuple[bool, dict[str, int]]:
        state = self.load_pulse_state()
        accepted, updated_tally = _apply_local_mood_vote(
            state, date_str, mood_key, voter_hmac
        )
        if accepted:
            self.save_pulse_state(state)
        return accepted, updated_tally

    def get_recent_pulse_words(self, days: int = 30) -> list[str]:
        state = self.load_pulse_state()
        return _extract_recent_words(state, days=days)

    def save_ingestion_cache_day(
        self,
        date_str: str,
        items_by_hash: dict[str, dict[str, Any]],
    ) -> None:
        self.ingestion_cache_dir.mkdir(parents=True, exist_ok=True)
        shard_path = self.ingestion_cache_dir / f"{date_str}.json"
        payload = _encode_json_safe({"date": str(date_str), "items": items_by_hash})
        shard_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def load_ingestion_cache(self, now_utc: datetime) -> dict[str, dict[str, Any]]:
        shards: dict[str, dict[str, Any]] = {}
        if self.ingestion_cache_dir.exists():
            for shard_file in sorted(self.ingestion_cache_dir.glob("*.json")):
                raw = json.loads(shard_file.read_text(encoding="utf-8"))
                decoded = _decode_json_safe(raw)
                date_key = str(decoded.get("date") or shard_file.stem)
                shards[date_key] = decoded
        return _hydrate_ingestion_cache_from_shards(shards, now_utc)

    def prune_expired_cache_items(self, now_utc: datetime) -> int:
        shards: dict[str, dict[str, Any]] = {}
        if self.ingestion_cache_dir.exists():
            for shard_file in sorted(self.ingestion_cache_dir.glob("*.json")):
                raw = json.loads(shard_file.read_text(encoding="utf-8"))
                decoded = _decode_json_safe(raw)
                date_key = str(decoded.get("date") or shard_file.stem)
                shards[date_key] = decoded

        pruned, deleted_dates, updated_dates = _prune_shard_dict_in_place(shards, now_utc)
        for d_str in deleted_dates:
            p = self.ingestion_cache_dir / f"{d_str}.json"
            if p.exists():
                p.unlink()
        for d_str in updated_dates:
            doc = shards.get(d_str) or {}
            self.save_ingestion_cache_day(d_str, doc.get("items") or {})
        return pruned

    def load_game_rate_limits(self) -> dict[str, Any]:
        if self.game_rate_limits_path.exists():
            return json.loads(self.game_rate_limits_path.read_text(encoding="utf-8"))
        return {"users": {}, "state_version": 0}

    def save_game_rate_limits(self, state: dict[str, Any]) -> None:
        self.game_rate_limits_path.parent.mkdir(parents=True, exist_ok=True)
        self.game_rate_limits_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )

    def check_and_increment_game_rate_limit(
        self,
        actor_handle: str,
        now_epoch: int,
        mode: str = "VS_BOT",
        game_completed: bool = False,
        is_new_round: bool = False,
    ) -> tuple[bool, str]:
        ok, reason, updated = _apply_game_rate_limit(
            self.load_game_rate_limits(),
            actor_handle=actor_handle,
            now_epoch=now_epoch,
            mode=mode,
            game_completed=game_completed,
            is_new_round=is_new_round,
        )
        if ok:
            self.save_game_rate_limits(updated)
        return ok, reason

    def load_orbit_roster(self) -> dict[str, Any]:
        if self.orbit_roster_path.exists():
            return _strip_transient_keys(
                json.loads(self.orbit_roster_path.read_text(encoding="utf-8"))
            )
        return {"handles": {}, "curated_o3_cursor": {}, "state_version": 0}

    def save_orbit_roster(self, roster: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(roster)
        cleaned.setdefault("handles", {})
        cleaned.setdefault("curated_o3_cursor", {})
        self.orbit_roster_path.parent.mkdir(parents=True, exist_ok=True)
        self.orbit_roster_path.write_text(
            json.dumps(cleaned, indent=2) + "\n", encoding="utf-8"
        )

    def save_pulse_feed_entry(self, date_str: str, entry: dict[str, Any]) -> None:
        item = _strip_transient_keys(entry)
        item["date"] = str(date_str)[:10]
        item["schema_version"] = "2.0"
        if str(item.get("image_data_uri") or "").startswith("data:"):
            item["image_data_uri"] = ""
        existing_entries: list[dict[str, Any]] = []
        if self.pulse_feed_path.exists():
            raw = json.loads(self.pulse_feed_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                existing_entries = [e for e in raw if isinstance(e, dict)]
            elif isinstance(raw, dict):
                existing_entries = [
                    e for e in raw.get("entries", []) if isinstance(e, dict)
                ]
        filtered = [
            e
            for e in existing_entries
            if str(e.get("date") or e.get("date_iso") or "") != str(date_str)[:10]
        ]
        filtered.append(item)
        filtered.sort(
            key=lambda x: str(x.get("date") or x.get("date_iso") or ""),
            reverse=True,
        )
        capped = filtered[:365]
        self.pulse_feed_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0",
            "schema_version": "2.0",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entries": capped,
        }
        self.pulse_feed_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _peek_local_bot_memory(self) -> dict[str, Any] | None:
        if self.bot_memory_path.exists():
            return json.loads(self.bot_memory_path.read_text(encoding="utf-8"))
        return None

    def load_bot_memory(self) -> dict[str, Any]:
        raw = self._peek_local_bot_memory()
        if raw is not None:
            data = _strip_transient_keys(raw)
            data.setdefault("state_version", 0)
            data.setdefault("games_learned", 0)
            data.setdefault("states", {})
            return data
        return {"state_version": 0, "games_learned": 0, "states": {}}

    def save_bot_memory(self, memory: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(memory)
        cleaned.setdefault("state_version", 0)
        cleaned.setdefault("games_learned", 0)
        cleaned.setdefault("states", {})
        self.bot_memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.bot_memory_path.write_text(
            json.dumps(cleaned, indent=2) + "\n", encoding="utf-8"
        )

    def _peek_local_elo_ledger(self) -> dict[str, Any] | None:
        if self.elo_ledger_path.exists():
            return json.loads(self.elo_ledger_path.read_text(encoding="utf-8"))
        return None

    def load_elo_ledger(self) -> dict[str, Any]:
        raw = self._peek_local_elo_ledger()
        if raw is not None:
            data = _strip_transient_keys(raw)
            data.setdefault("state_version", 0)
            data.setdefault("bot_elo", 1200)
            data.setdefault("players", {})
            data.setdefault("recent_rounds", [])
            return data
        return {
            "state_version": 0,
            "bot_elo": 1200,
            "players": {},
            "recent_rounds": [],
        }

    def save_elo_ledger(self, ledger: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(ledger)
        cleaned.setdefault("state_version", 0)
        cleaned.setdefault("bot_elo", 1200)
        cleaned.setdefault("players", {})
        cleaned.setdefault("recent_rounds", [])
        self.elo_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.elo_ledger_path.write_text(
            json.dumps(cleaned, indent=2) + "\n", encoding="utf-8"
        )


def _py_to_firestore_value(val: Any, in_array: bool = False) -> dict[str, Any]:
    """Encode a Python value into the Cloud Firestore REST v1 Value format."""
    if val is None:
        return {"nullValue": None}
    if isinstance(val, bool):
        return {"booleanValue": val}
    if isinstance(val, int):
        return {"integerValue": str(val)}
    if isinstance(val, float):
        return {"doubleValue": val}
    if isinstance(val, (bytes, bytearray)):
        return {"bytesValue": base64.b64encode(bytes(val)).decode("ascii")}
    if isinstance(val, str):
        return {"stringValue": val}
    if isinstance(val, list):
        if in_array:
            # Firestore does not permit directly nested arrays; wrap inner list in a map
            return {
                "mapValue": {
                    "fields": {
                        "_nested_list": {
                            "arrayValue": {
                                "values": [_py_to_firestore_value(x, in_array=True) for x in val]
                            }
                        }
                    }
                }
            }
        return {
            "arrayValue": {
                "values": [_py_to_firestore_value(x, in_array=True) for x in val]
            }
        }
    if isinstance(val, dict):
        return {
            "mapValue": {
                "fields": {
                    str(k): _py_to_firestore_value(v, in_array=False)
                    for k, v in val.items()
                }
            }
        }
    return {"stringValue": str(val)}


def _firestore_value_to_py(raw: dict[str, Any]) -> Any:
    """Decode a Cloud Firestore REST v1 Value object back into a native Python value."""
    if "nullValue" in raw:
        return None
    if "booleanValue" in raw:
        return bool(raw["booleanValue"])
    if "integerValue" in raw:
        return int(raw["integerValue"])
    if "doubleValue" in raw:
        return float(raw["doubleValue"])
    if "bytesValue" in raw:
        return base64.b64decode(raw["bytesValue"])
    if "stringValue" in raw:
        return str(raw["stringValue"])
    if "arrayValue" in raw:
        vals = (raw["arrayValue"] or {}).get("values") or []
        return [_firestore_value_to_py(v) for v in vals]
    if "mapValue" in raw:
        fields = (raw["mapValue"] or {}).get("fields") or {}
        if set(fields.keys()) == {"_nested_list"}:
            return _firestore_value_to_py(fields["_nested_list"])
        return {k: _firestore_value_to_py(v) for k, v in fields.items()}
    return None


class _FirestoreRestDocSnapshot:
    """Snapshot returned by `_FirestoreRestDocRef.get()`."""

    def __init__(self, doc_id: str, exists: bool, data: dict[str, Any] | None = None, update_time: str | None = None) -> None:
        self.id = doc_id
        self.update_time = update_time
        self.exists = exists
        self._data = copy.deepcopy(data) if data is not None else {}

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


class _FirestoreRestDocRef:
    """Document reference for `_FirestoreRestClient`."""

    def __init__(self, client: _FirestoreRestClient, collection_id: str, doc_id: str) -> None:
        self._client = client
        self._collection_id = collection_id
        self._doc_id = doc_id

    def get(self) -> _FirestoreRestDocSnapshot:
        return self._client.get_document(self._collection_id, self._doc_id)

    def set(self, data: dict[str, Any]) -> None:
        self._client.set_document(self._collection_id, self._doc_id, data)

    def delete(self) -> None:
        self._client.delete_document(self._collection_id, self._doc_id)


class _FirestoreRestCollectionRef:
    """Collection reference for `_FirestoreRestClient`."""

    def __init__(self, client: _FirestoreRestClient, collection_id: str) -> None:
        self._client = client
        self._collection_id = collection_id

    def document(self, doc_id: str) -> _FirestoreRestDocRef:
        return _FirestoreRestDocRef(self._client, self._collection_id, doc_id)

    def stream(self) -> list[_FirestoreRestDocSnapshot]:
        return self._client.list_documents(self._collection_id)


class _FirestoreRestClient:
    """Zero-dependency Cloud Firestore v1 REST client using `urllib.request` and OAuth2 Bearer tokens."""

    def __init__(
        self,
        project_id: str,
        database_id: str = "(default)",
        access_token: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.database_id = database_id or "(default)"
        self._access_token = access_token

    def _mint_sa_access_token(self, sa_json_str: str) -> str:
        sa = json.loads(sa_json_str)
        now = int(time.time())
        header = {"alg": "RS256", "typ": "JWT"}
        claims = {
            "iss": sa["client_email"],
            "scope": "https://www.googleapis.com/auth/datastore https://www.googleapis.com/auth/cloud-platform",
            "aud": sa.get("token_uri", "https://oauth2.googleapis.com/token"),
            "iat": now,
            "exp": now + 3600,
        }
        b64_hdr = base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).decode("ascii").rstrip("=")
        b64_claims = base64.urlsafe_b64encode(json.dumps(claims).encode("utf-8")).decode("ascii").rstrip("=")
        signing_input = f"{b64_hdr}.{b64_claims}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=True) as key_file:
            key_file.write(sa["private_key"])
            key_file.flush()
            sig_proc = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", key_file.name],
                input=signing_input.encode("ascii"),
                check=True,
                capture_output=True,
                timeout=5,
            )
        b64_sig = base64.urlsafe_b64encode(sig_proc.stdout).decode("ascii").rstrip("=")
        jwt_assertion = f"{signing_input}.{b64_sig}"

        form_body = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt_assertion,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            sa.get("token_uri", "https://oauth2.googleapis.com/token"),
            data=form_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            tok_data = json.loads(resp.read().decode("utf-8"))
        return str(tok_data["access_token"])

    def _exchange_github_oidc_token(self) -> str:
        req_url = (os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL") or "").strip()
        req_tok = (os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN") or "").strip()
        if not req_url or not req_tok:
            raise RuntimeError("GitHub Actions OIDC environment variables not set")
        wif_provider = (
            os.environ.get("GCP_WIF_PROVIDER")
            or "//iam.googleapis.com/projects/277213321965/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
        ).strip()
        sep = "&" if "?" in req_url else "?"
        oidc_url = f"{req_url}{sep}audience={urllib.parse.quote(wif_provider, safe='')}"
        oidc_req = urllib.request.Request(
            oidc_url,
            headers={
                "Authorization": f"Bearer {req_tok}",
                "Accept": "application/json; api-version=2.0",
            },
            method="GET",
        )
        with urllib.request.urlopen(oidc_req, timeout=10) as oidc_resp:
            gh_jwt = str(json.loads(oidc_resp.read().decode("utf-8"))["value"])

        sts_payload = json.dumps(
            {
                "audience": wif_provider,
                "grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
                "requestedTokenType": "urn:ietf:params:oauth:token-type:access_token",
                "scope": "https://www.googleapis.com/auth/datastore https://www.googleapis.com/auth/cloud-platform",
                "subjectTokenType": "urn:ietf:params:oauth:token-type:jwt",
                "subjectToken": gh_jwt,
            }
        ).encode("utf-8")
        sts_req = urllib.request.Request(
            "https://sts.googleapis.com/v1/token",
            data=sts_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(sts_req, timeout=10) as sts_resp:
            sts_data = json.loads(sts_resp.read().decode("utf-8"))
        return str(sts_data["access_token"])

    def _resolve_token(self) -> str:
        if self._access_token:
            return self._access_token
        env_tok = (
            os.environ.get("FIRESTORE_ACCESS_TOKEN")
            or os.environ.get("GCP_ACCESS_TOKEN")
            or ""
        ).strip()
        if env_tok:
            self._access_token = env_tok
            return env_tok
        if os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL") and os.environ.get(
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN"
        ):
            self._access_token = self._exchange_github_oidc_token()
            return self._access_token
        sa_raw = (
            os.environ.get("GCP_SA_KEY")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            or os.environ.get("FIREBASE_SERVICE_ACCOUNT")
            or ""
        ).strip()
        if not sa_raw:
            cred_path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
            if cred_path and Path(cred_path).is_file():
                sa_raw = Path(cred_path).read_text(encoding="utf-8")
        if sa_raw:
            self._access_token = self._mint_sa_access_token(sa_raw)
            return self._access_token
        res = subprocess.run(
            ["gcloud", "auth", "print-access-token", "--quiet"],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        tok = res.stdout.strip()
        if not tok:
            raise RuntimeError("Empty access token from gcloud")
        self._access_token = tok
        return tok

    def _base_url(self) -> str:
        db_enc = urllib.parse.quote(self.database_id, safe="()")
        return f"https://firestore.googleapis.com/v1/projects/{self.project_id}/databases/{db_enc}/documents"

    def collection(self, collection_id: str) -> _FirestoreRestCollectionRef:
        return _FirestoreRestCollectionRef(self, collection_id)

    def get_document(self, collection_id: str, doc_id: str) -> _FirestoreRestDocSnapshot:
        tok = self._resolve_token()
        url = (
            f"{self._base_url()}/"
            f"{urllib.parse.quote(collection_id, safe='')}/"
            f"{urllib.parse.quote(doc_id, safe='')}"
        )
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"},
            method="GET",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                fields = payload.get("fields") or {}
                decoded = {k: _firestore_value_to_py(v) for k, v in fields.items()}
                return _FirestoreRestDocSnapshot(doc_id=doc_id, exists=True, data=decoded, update_time=payload.get("updateTime"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return _FirestoreRestDocSnapshot(doc_id=doc_id, exists=False, data={})
                raise  # Authentication/API failures keep their exact classification.
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                # GET is read-only. Never apply this retry to commits or other
                # state-changing requests whose acknowledgement may be unknown.
                if attempt == 2:
                    raise
                time.sleep(0.2 * (attempt + 1))

    def set_document(self, collection_id: str, doc_id: str, data: dict[str, Any]) -> None:
        tok = self._resolve_token()
        url = (
            f"{self._base_url()}/"
            f"{urllib.parse.quote(collection_id, safe='')}/"
            f"{urllib.parse.quote(doc_id, safe='')}"
        )
        body_dict = {
            "fields": {
                str(k): _py_to_firestore_value(v, in_array=False)
                for k, v in data.items()
            }
        }
        raw_bytes = json.dumps(body_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=raw_bytes,
            headers={
                "Authorization": f"Bearer {tok}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()

    def delete_document(self, collection_id: str, doc_id: str) -> None:
        tok = self._resolve_token()
        url = (
            f"{self._base_url()}/"
            f"{urllib.parse.quote(collection_id, safe='')}/"
            f"{urllib.parse.quote(doc_id, safe='')}"
        )
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"},
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            raise

    def list_documents(
        self,
        collection_id: str,
        page_size: int = 100,
    ) -> list[_FirestoreRestDocSnapshot]:
        tok = self._resolve_token()
        base_col_url = f"{self._base_url()}/{urllib.parse.quote(collection_id, safe='')}"
        docs: list[_FirestoreRestDocSnapshot] = []
        next_page_token = ""

        while True:
            query: dict[str, str] = {"pageSize": str(max(1, int(page_size)))}
            if next_page_token:
                query["pageToken"] = next_page_token
            url = f"{base_col_url}?{urllib.parse.urlencode(query)}"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return docs
                raise

            for item in payload.get("documents") or []:
                name = str(item.get("name", ""))
                doc_id = name.rsplit("/", 1)[-1] if "/" in name else name
                fields = item.get("fields") or {}
                decoded = {k: _firestore_value_to_py(v) for k, v in fields.items()}
                docs.append(_FirestoreRestDocSnapshot(doc_id=doc_id, exists=True, data=decoded, update_time=item.get("updateTime")))

            next_page_token = str(payload.get("nextPageToken") or "").strip()
            if not next_page_token:
                break

        return docs


class PublicationOutcomeUnknown(RuntimeError):
    """The transport failed after dispatch; reconcile before any new publication."""


class PublicationConflict(RuntimeError):
    """A concurrent writer changed a document; no conditional bundle was applied."""


def _firestore_error_status(payload: Any) -> str | None:
    """Return only Google RPC's fixed status enum, never provider messages."""
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    status = payload["error"].get("status")
    allowed = {
        "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED",
        "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED",
        "INTERNAL", "UNAVAILABLE", "DATA_LOSS", "UNAUTHENTICATED",
    }
    return status if isinstance(status, str) and status in allowed else None


def _commit_documents(client: Any, writes: list[tuple[str, str, dict, Any]]) -> None:
    """Atomically update supplied top-level fields against exact read revisions.

    No sequential .set fallback: a client without conditional atomic commits is
    unsuitable for canonical publications or votes.
    """
    for _collection, _doc_id, _data, snapshot in writes:
        if snapshot.exists and not getattr(snapshot, "update_time", None):
            raise RuntimeError("Conditional commit requires snapshot.update_time")
    if hasattr(client, "commit_documents"):
        try:
            client.commit_documents(writes)
        except (PublicationConflict, PublicationOutcomeUnknown):
            raise
        except BaseException:
            raise PublicationOutcomeUnknown(
                "Canonical commit outcome unknown; reconcile first"
            ) from None
        return
    if isinstance(client, _FirestoreRestClient):
        body_writes = []
        document_prefix = client._base_url().split("/v1/", 1)[1]
        for collection, doc_id, data, snapshot in writes:
            write = {
                "update": {
                    "name": f"{document_prefix}/{collection}/{doc_id}",
                    "fields": {
                        str(k): _py_to_firestore_value(v, in_array=False)
                        for k, v in data.items()
                    },
                },
                "currentDocument": (
                    {"updateTime": str(snapshot.update_time)}
                    if snapshot.exists
                    else {"exists": False}
                ),
            }
            if snapshot.exists:
                write["updateMask"] = {
                    "fieldPaths": [
                        (
                            k
                            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k)
                            else "`" + k.replace("\\", "\\\\").replace("`", "\\`") + "`"
                        )
                        for k in data
                    ]
                }
            body_writes.append(write)
        request = urllib.request.Request(
            client._base_url() + ":commit",
            data=json.dumps({"writes": body_writes}).encode(),
            headers={
                "Authorization": f"Bearer {client._resolve_token()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=15)
        except urllib.error.HTTPError as exc:
            conflict = exc.code in (409, 412)
            if exc.code == 400:
                try:
                    # Firestore stale updateTime is HTTP400 FAILED_PRECONDITION.
                    # Read only for classification; a broken error body stays a
                    # definitive 4xx rejection, never a guessed conflict.
                    conflict = _firestore_error_status(json.loads(exc.read(65537))) in {
                        "FAILED_PRECONDITION", "ABORTED"
                    }
                except Exception:
                    pass
            if conflict:
                raise PublicationConflict(
                    "Canonical documents changed before commit"
                ) from None
            if 400 <= exc.code < 500:
                raise RuntimeError(
                    f"Canonical atomic commit rejected (HTTP {exc.code})"
                ) from None
            raise PublicationOutcomeUnknown(
                f"Canonical commit outcome unknown (HTTP {exc.code}); reconcile first"
            ) from None
        except BaseException:
            # Dispatch may have reached Firestore even if no response object was
            # returned. Cancellation and unexpected transport errors are unknown.
            raise PublicationOutcomeUnknown(
                "Canonical commit outcome unknown; reconcile first"
            ) from None
        try:
            with response:
                acknowledgement = json.loads(response.read().decode("utf-8"))
            results = acknowledgement.get("writeResults")
            if (
                not isinstance(acknowledgement.get("commitTime"), str)
                or not acknowledgement["commitTime"]
                or not isinstance(results, list)
                or len(results) != len(writes)
                or any(
                    not isinstance(result, dict)
                    or not isinstance(result.get("updateTime"), str)
                    or not result["updateTime"]
                    for result in results
                )
            ):
                raise ValueError("Incomplete canonical commit acknowledgement")
        except BaseException:
            # A successful HTTP status is insufficient if read/decode/close or
            # acknowledgement validation failed after the server may have committed.
            raise PublicationOutcomeUnknown(
                "Canonical commit acknowledgement unavailable; reconcile first"
            ) from None
        return
    if not hasattr(client, "batch"):
        raise RuntimeError(
            "Firestore client does not support atomic conditional commits"
        )
    from google.cloud.firestore_v1 import LastUpdateOption

    batch = client.batch()
    for collection, doc_id, data, snapshot in writes:
        reference = client.collection(collection).document(doc_id)
        if snapshot.exists:
            batch.update(reference, data, option=LastUpdateOption(snapshot.update_time))
        else:
            batch.create(reference, data)
    try:
        results = batch.commit(retry=None, timeout=15)
        if (
            not isinstance(results, list)
            or len(results) != len(writes)
            or any(getattr(result, "update_time", None) is None for result in results)
        ):
            raise ValueError("Incomplete SDK commit acknowledgement")
    except BaseException as exc:
        if isinstance(exc, (PublicationConflict, PublicationOutcomeUnknown)):
            raise
        if type(exc).__name__ in {
            "Conflict",
            "FailedPrecondition",
            "AlreadyExists",
            "NotFound",
            "Aborted",
        }:
            raise PublicationConflict(
                "Canonical documents changed before commit"
            ) from None
        if type(exc).__name__ in {
            "InvalidArgument",
            "BadRequest",
            "PermissionDenied",
            "Forbidden",
            "Unauthenticated",
            "Unauthorized",
            "ResourceExhausted",
            "TooManyRequests",
        }:
            raise RuntimeError("Canonical atomic commit rejected by provider") from None
        # Any other error raised by commit/response processing is ambiguous,
        # including SDK decoder errors and interruption after dispatch.
        raise PublicationOutcomeUnknown(
            "Canonical commit outcome unknown; reconcile first"
        ) from None


COLLECTION_PREFIX_RE = re.compile(r"[A-Za-z0-9_-]*")

JsonStateStore = JsonFileStateStore


class FirestoreStateStore:
    """Cloud Firestore adapter storing documents in `profile_state`, `pulse_history`, `pulse_votes`, and `vote_rate_limits`.

    Lazy-imports `google.cloud.firestore` or falls back to `_FirestoreRestClient` so offline
    and stdlib-only environments have zero fragile external dependency.
    The runtime factory selects `canonical_only=True`: authentication/storage failures
    propagate, required game/pulse documents must already exist, and checkout JSON is
    only a write-through projection. Explicit constructor fallback remains available
    for legacy/offline callers; it is never selected by the runtime factory.
    """

    def __init__(
        self,
        client: Any = None,
        project_id: str | None = None,
        database_id: str = "(default)",
        fallback_store: StateStore | None = None,
        collection_prefix: str | None = None,
        repo_root: Path | None = None,
        canonical_only: bool = False,
        mirror_store: JsonFileStateStore | None = None,
    ) -> None:
        raw_prefix = (
            collection_prefix
            if collection_prefix is not None
            else os.environ.get("FIRESTORE_COLLECTION_PREFIX", "")
        )
        cleaned_prefix = str(raw_prefix).strip()
        if not COLLECTION_PREFIX_RE.fullmatch(cleaned_prefix):
            raise ValueError(
                f"Invalid collection_prefix / FIRESTORE_COLLECTION_PREFIX {raw_prefix!r}: must match [A-Za-z0-9_-]*"
            )
        self.collection_prefix: str = cleaned_prefix
        self.needs_git_reconcile: bool = False
        self._client = client
        self._project_id = (
            project_id
            or os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("FIRESTORE_PROJECT_ID")
            or "git-readme-features"
        )
        if not database_id or database_id == "(default)":
            env_db = (os.environ.get("FIRESTORE_DATABASE_ID") or "").strip()
            if env_db and env_db != "git-readme-features":
                database_id = env_db
            else:
                database_id = "(default)"
        elif database_id == "git-readme-features":
            database_id = "(default)"
        self._database_id = database_id
        self.canonical_only = bool(canonical_only)
        if self.canonical_only and fallback_store is not None:
            raise ValueError("Canonical-only mode cannot configure an authority fallback")
        if mirror_store is not None and not isinstance(mirror_store, JsonFileStateStore):
            raise ValueError("Canonical mirror must be a JsonFileStateStore projection")
        if self.canonical_only:
            self._fallback_store = None
            self._mirror_store = mirror_store or (JsonFileStateStore(repo_root=Path(repo_root)) if repo_root is not None else None)
        else:
            if fallback_store is None and repo_root is not None:
                fallback_store = JsonFileStateStore(repo_root=Path(repo_root))
            self._fallback_store = fallback_store
            self._mirror_store = None

    def _col(self, name: str) -> str:
        return f"{self.collection_prefix}{name}"

    def _mirror_projection(self, filename: str, payload: dict[str, Any]) -> None:
        """Write confirmed canonical data only; never merge or read checkout JSON."""
        if self._mirror_store is None:
            return
        path = self._mirror_store.repo_root / "data" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(_encode_json_safe(_strip_transient_keys(payload)), handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _canonical_tally(payload: dict[str, Any]) -> dict[str, int]:
        tally = payload.get("visitor_mood_tally", payload.get("mood_votes"))
        if not isinstance(tally, dict) or any(type(tally.get(key)) is not int or tally[key] < 0 for key in VALID_MOOD_KEYS):
            raise RuntimeError("Canonical Firestore pulse tally is missing or invalid; explicit recovery required")
        return {key: tally[key] for key in VALID_MOOD_KEYS}

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        prefer_rest = bool(
            (
                os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
                and os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
                and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            )
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            or os.environ.get("FIREBASE_SERVICE_ACCOUNT")
            or os.environ.get("GCP_SA_KEY")
            or os.environ.get("FIRESTORE_ACCESS_TOKEN")
        )
        if self.canonical_only:
            # A rejected explicit runtime credential must not trigger another
            # credentials route or a local authority fallback.
            if prefer_rest:
                client = _FirestoreRestClient(self._project_id, self._database_id)
                client._resolve_token()
            else:
                try:
                    from google.cloud import firestore
                except ImportError:
                    client = _FirestoreRestClient(self._project_id, self._database_id)
                    client._resolve_token()
                else:
                    kwargs = {"project": self._project_id}
                    if self._database_id != "(default)":
                        kwargs["database"] = self._database_id
                    client = firestore.Client(**kwargs)
            self._client = client
            return client
        if prefer_rest and self._project_id:
            try:
                rest_client = _FirestoreRestClient(
                    project_id=self._project_id,
                    database_id=self._database_id,
                )
                rest_client._resolve_token()
                self._client = rest_client
                return self._client
            except Exception:
                pass
        try:
            from google.cloud import firestore  # type: ignore[import-untyped]

            kwargs: dict[str, Any] = {}
            if self._project_id:
                kwargs["project"] = self._project_id
            if self._database_id and self._database_id != "(default)":
                kwargs["database"] = self._database_id
            self._client = firestore.Client(**kwargs)
            return self._client
        except Exception:
            try:
                if self._project_id:
                    rest_client = _FirestoreRestClient(
                        project_id=self._project_id,
                        database_id=self._database_id,
                    )
                    rest_client._resolve_token()
                    self._client = rest_client
                    return self._client
            except Exception:
                pass
            if self._fallback_store is not None:
                return None
            raise

    def load_game_state(self) -> dict[str, Any]:
        from src.game_engine import build_midgame_showcase_state, normalize_game_state

        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.load_game_state()

        doc = client.collection(self._col("profile_state")).document("active_game").get()
        if self.canonical_only:
            if not getattr(doc, "exists", False):
                raise RuntimeError("Canonical Firestore active_game is missing; explicit bootstrap required")
            payload = _strip_transient_keys(doc.to_dict() or {})
            board = payload.get("board")
            if not isinstance(board, list) or len(board) != 6 or any(not isinstance(row, list) or len(row) != 7 for row in board) or type(payload.get("turn")) is not int:
                raise RuntimeError("Canonical Firestore active_game is invalid; explicit recovery required")
            return _strip_transient_keys(normalize_game_state(payload))
        if getattr(doc, "exists", False):
            fs_data = _strip_transient_keys(doc.to_dict() or {})
            fs_ver = int(fs_data.get("state_version", 0))
            local_data: dict[str, Any] | None = None
            if self._fallback_store is not None and hasattr(
                self._fallback_store, "_peek_local_game_state"
            ):
                local_data = self._fallback_store._peek_local_game_state()
            local_ver = int((local_data or {}).get("state_version", 0))

            if local_data is not None and local_ver > fs_ver:
                return _strip_transient_keys(normalize_game_state(local_data))

            normalized = _strip_transient_keys(normalize_game_state(fs_data))
            if fs_ver > local_ver:
                self.needs_git_reconcile = True
                normalized["_needs_git_reconcile"] = True
            return normalized

        if self._fallback_store is not None and hasattr(
            self._fallback_store, "_peek_local_game_state"
        ):
            local_existing = self._fallback_store._peek_local_game_state()
            if local_existing is not None:
                return _strip_transient_keys(normalize_game_state(local_existing))

        default_state = build_midgame_showcase_state("connect4", round_id=1)
        self.save_game_state(default_state)
        return default_state

    @staticmethod
    def game_state_document(state: dict[str, Any]) -> dict[str, Any]:
        """Serialize the normalized game domain for scalar or atomic writes."""
        from src.game_engine import encode_bitboard, normalize_game_state

        cleaned = _strip_transient_keys(state)
        normalized = _strip_transient_keys(normalize_game_state(cleaned))
        board = normalized.get("board") or [["." for _ in range(7)] for _ in range(6)]
        b1, b2 = encode_bitboard(board)
        mode_val = str(normalized.get("mode") or "VS_BOT")
        if mode_val not in ("VS_BOT", "PVP_DUEL"):
            mode_val = "VS_BOT"
        status_val = str(normalized.get("status") or "ACTIVE")
        if status_val not in ("ACTIVE", "COMPLETED", "DRAW", "RED_WIN", "YELLOW_WIN"):
            status_val = "ACTIVE"
        move_history = list(
            normalized.get("move_history")
            if isinstance(normalized.get("move_history"), list)
            else normalized.get("move_log") or []
        )[:100]
        raw_last_move = (
            normalized.get("last_move")
            if "last_move" in normalized
            else (
                normalized.get("last_bot_move")
                or normalized.get("last_challenger_move")
            )
        )
        if isinstance(raw_last_move, dict):
            last_move_map: dict[str, Any] | None = dict(raw_last_move)
        elif isinstance(raw_last_move, str) and raw_last_move.strip():
            last_move_map = {"move_id": raw_last_move.strip()}
        elif move_history and isinstance(move_history[-1], dict):
            last_move_map = dict(move_history[-1])
        else:
            last_move_map = None

        raw_participants = normalized.get("round_participants")
        red_contrib = dict(
            normalized.get("red_contributors")
            or (raw_participants.get("red") if isinstance(raw_participants, dict) else None)
            or {}
        )
        yellow_contrib = dict(
            normalized.get("yellow_contributors")
            or (raw_participants.get("yellow") if isinstance(raw_participants, dict) else None)
            or {}
        )

        fs_doc: dict[str, Any] = {
            "schema_version": str(normalized.get("schema_version") or "2.0"),
            "state_version": int(normalized.get("state_version", 1)),
            "round_id": max(1, int(normalized.get("round_id", 1))),
            "turn": max(0, int(normalized.get("turn", 1))),
            "active_game": str(normalized.get("active_game") or "connect4"),
            "mode": mode_val,
            "status": status_val,
            "current_turn_color": str(
                normalized.get("current_turn_color")
                or ("RED" if int(normalized.get("turn", 1)) % 2 == 1 else "YELLOW")
            ),
            "board": board,
            "bitboard_p1": b1,
            "bitboard_p2": b2,
            "winner": normalized.get("winner"),
            "winning_cells": list(
                normalized.get("winning_cells")
                if isinstance(normalized.get("winning_cells"), list)
                else normalized.get("winning_line") or []
            )[:16],
            "last_move": last_move_map,
            "last_mover": (
                normalized.get("last_mover")
                if "last_mover" in normalized
                else normalized.get("last_voter")
            ),
            "last_move_summary": str(normalized.get("last_move_summary") or ""),
            "move_history": move_history,
            "move_log": list(normalized.get("move_log") or move_history[-12:])[:12],
            "red_contributors": red_contrib,
            "yellow_contributors": yellow_contrib,
            "updated_at": str(
                normalized.get("updated_at")
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ),
        }
        # The normalizer owns the game domain schema. Persist every field it
        # initializes (trace, round completion/history, clocks and rate state),
        # while retaining the legacy canonical aliases above. board_state is a
        # derived duplicate rebuilt on read; unrelated runtime baggage is omitted.
        domain_fields = set(normalize_game_state({})) - {"board_state"}
        for key in domain_fields:
            if key not in fs_doc:
                fs_doc[key] = copy.deepcopy(normalized[key])
        return fs_doc

    def save_game_state(self, state: dict[str, Any]) -> None:
        from src.game_engine import normalize_game_state

        fs_doc = self.game_state_document(state)
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            self._fallback_store.save_game_state(normalize_game_state(state))
            return
        client.collection(self._col("profile_state")).document("active_game").set(fs_doc)
        self._mirror_projection("game_state.json", normalize_game_state(fs_doc))
        if self._fallback_store is not None:
            self._fallback_store.save_game_state(normalize_game_state(state))

    def load_pulse_state(self) -> dict[str, Any]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.load_pulse_state()

        doc = client.collection(self._col("profile_state")).document("daily_pulse").get()
        if self.canonical_only:
            if not getattr(doc, "exists", False):
                raise RuntimeError("Canonical Firestore daily_pulse is missing; explicit bootstrap required")
            payload = _strip_transient_keys(doc.to_dict() or {})
            payload["visitor_mood_tally"] = self._canonical_tally(payload)
            return payload
        if getattr(doc, "exists", False):
            fs_data = _strip_transient_keys(doc.to_dict() or {})
            fs_data.setdefault("visitor_mood_tally", dict(DEFAULT_MOOD_TALLY))
            fs_ver = int(fs_data.get("state_version", 0))
            local_data: dict[str, Any] | None = None
            if self._fallback_store is not None and hasattr(
                self._fallback_store, "_peek_local_pulse_state"
            ):
                local_data = self._fallback_store._peek_local_pulse_state()
            local_ver = int((local_data or {}).get("state_version", 0))

            if local_data is not None and local_ver > fs_ver:
                out_local = _strip_transient_keys(local_data)
                out_local.setdefault("visitor_mood_tally", dict(DEFAULT_MOOD_TALLY))
                return out_local

            if fs_ver > local_ver:
                self.needs_git_reconcile = True
                fs_data["_needs_git_reconcile"] = True
            return fs_data

        if self._fallback_store is not None and hasattr(
            self._fallback_store, "_peek_local_pulse_state"
        ):
            local_existing = self._fallback_store._peek_local_pulse_state()
            if local_existing is not None:
                out_local = _strip_transient_keys(local_existing)
                out_local.setdefault("visitor_mood_tally", dict(DEFAULT_MOOD_TALLY))
                return out_local

        return {
            "visitor_mood_tally": dict(DEFAULT_MOOD_TALLY),
            "recent_words": [],
        }

    def save_pulse_state(self, state: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(state)
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            self._fallback_store.save_pulse_state(cleaned)
            return

        existing_doc = client.collection(self._col("profile_state")).document("daily_pulse").get()
        existing = (
            _strip_transient_keys(existing_doc.to_dict() or {})
            if getattr(existing_doc, "exists", False)
            else {}
        )
        if self.canonical_only:
            if "visitor_mood_tally" in cleaned or "mood_votes" in cleaned:
                cleaned["visitor_mood_tally"] = self._canonical_tally(cleaned)
            else:
                cleaned["visitor_mood_tally"] = self._canonical_tally(existing) if existing else {key: 0 for key in VALID_MOOD_KEYS}
        enriched = _strip_transient_keys(
            _enrich_pulse_state_with_history(cleaned, existing=existing)
        )
        daily_doc = copy.deepcopy(enriched)
        daily_doc.pop("image_data_uri", None)
        daily_doc.pop("vote_rate_limits", None)
        pub_map = daily_doc.get("published_pulse")
        if isinstance(pub_map, dict):
            daily_doc.setdefault("schema_version", "2.0")
            for k in (
                "date",
                "word",
                "gloss",
                "commissioned_thesis",
                "visual_medium",
                "image_url",
                "fallback_webp_path",
                "mood_votes",
                "published_at",
                "updated_at",
            ):
                if k not in daily_doc and k in pub_map:
                    daily_doc[k] = copy.deepcopy(pub_map[k])
        client.collection(self._col("profile_state")).document("daily_pulse").set(daily_doc)

        analysis = enriched.get("analysis") or {}
        date_iso = str(
            analysis.get("date_iso")
            or enriched.get("date_iso")
            or enriched.get("date")
            or ""
        ).strip()
        word = str(
            analysis.get("word_of_the_day") or enriched.get("word") or ""
        ).strip().upper()
        if date_iso and word and not isinstance(enriched.get("published_pulse"), dict):
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            visual_dec = enriched.get("visual_decision") or {}
            tally = enriched.get("visitor_mood_tally") or enriched.get("mood_votes") or {}
            mood_votes_3 = {
                "spot_on": int(tally.get("spot_on", tally.get("resonant", 18))),
                "too_sunny": int(tally.get("too_sunny", tally.get("electric", 3))),
                "too_heavy": int(tally.get("too_heavy", tally.get("reflective", 4))),
            }
            history_doc = {
                "schema_version": "2.0",
                "date": date_iso[:10],
                "word": word,
                "gloss": str(analysis.get("explanation_phrase", enriched.get("gloss", "Daily synthesis"))),
                "commissioned_thesis": str(enriched.get("summary") or analysis.get("explanation_phrase") or enriched.get("gloss") or "96h stratified synthesis across Personal Orbit and Macro Velocity."),
                "visual_medium": str(enriched.get("visual_medium") or visual_dec.get("selected_mode") or "SVG_FALLBACK"),
                "image_url": str(enriched.get("image_url") or enriched.get("artwork_url") or visual_dec.get("source_page_url") or ""),
                "fallback_webp_path": str(enriched.get("fallback_webp_path") or f"data/pulse_archive/{date_iso[:10]}.webp"),
                "unsplash_photographer": enriched.get("unsplash_photographer"),
                "unsplash_profile_url": enriched.get("unsplash_profile_url"),
                "flux_model": enriched.get("flux_model"),
                "cluster_keywords": list(enriched.get("cluster_keywords") or enriched.get("top_orbit_themes") or [word.lower()]),
                "orbit_contribution_pct": int(round(float(enriched.get("orbit_weight_ratio", 0.5)) * 100)),
                "macro_contribution_pct": 100 - int(round(float(enriched.get("orbit_weight_ratio", 0.5)) * 100)),
                "orbit_sources_matched": list(enriched.get("orbit_sources_matched") or []),
                "runner_up_clusters": list(enriched.get("runner_up_clusters") or []),
                "mood_votes": mood_votes_3,
                "critic_scores": dict(enriched.get("critic_scores") or {}),
                "published_at": str(enriched.get("published_at") or now_iso),
                "updated_at": str(enriched.get("updated_at") or enriched.get("published_at") or now_iso),
                "headline": str(enriched.get("headline") or f"{word} — {analysis.get('explanation_phrase', enriched.get('gloss', 'Daily synthesis'))}"),
                "summary": str(enriched.get("summary") or analysis.get("explanation_phrase") or enriched.get("gloss") or "96h stratified synthesis across Personal Orbit and Macro Velocity."),
                "24h_shift": str(enriched.get("24h_shift") or enriched.get("shift_24h") or "Steady cross-domain convergence."),
                "dominant_category": str(enriched.get("dominant_category") or "tech_ai"),
                "dominant_signal_type": str(enriched.get("dominant_signal_type") or "HYBRID"),
                "orbit_weight_ratio": float(enriched.get("orbit_weight_ratio", 0.5)),
                "top_orbit_themes": list(enriched.get("top_orbit_themes") or [word.lower()]),
                "top_macro_themes": list(enriched.get("top_macro_themes") or ["systems"]),
                "visual_tier_used": str(enriched.get("visual_tier_used") or "TIER_3_SVG"),
                "artwork_title": str(enriched.get("artwork_title") or visual_dec.get("title") or word),
                "artwork_artist": str(enriched.get("artwork_artist") or visual_dec.get("artist") or "Generative SVG"),
                "artwork_source": str(enriched.get("artwork_source") or visual_dec.get("selected_mode") or "TIER_3_SVG"),
                "artwork_url": str(enriched.get("artwork_url") or visual_dec.get("source_page_url") or "https://github.com/Siyuan-Zhu/Siyuan-Zhu"),
                "image_data_uri": str(enriched.get("image_data_uri") or ""),
                "quote": str(enriched.get("quote") or analysis.get("explanation_phrase") or enriched.get("gloss") or word),
                "quote_attribution": str(enriched.get("quote_attribution") or "Daily Zeitgeist Director"),
                "window_hours": 96,
                "date_iso": date_iso[:10],
                "word_of_the_day": word,
                "explanation_phrase": analysis.get(
                    "explanation_phrase", enriched.get("gloss", "")
                ),
                "composite_score": analysis.get("composite_score", 0.0),
            }
            client.collection(self._col("pulse_history")).document(date_iso[:10]).set(history_doc)

        self._mirror_projection("zeitgeist_state.json", daily_doc)
        if self._fallback_store is not None:
            self._fallback_store.save_pulse_state(enriched)

    def set_needs_git_reconcile(self, flag: bool) -> None:
        self.needs_git_reconcile = bool(flag)

    def canonical_projections(self, *, write_mirrors: bool = False) -> dict[str, dict[str, Any]]:
        """Read required authority before optionally projecting exact JSON mirrors.

        No remote mutation, local history merge, bootstrap or provider inference.
        Read-only README checks leave checkout files untouched.
        """
        if not self.canonical_only:
            raise RuntimeError("Canonical projections require explicit canonical-only mode")
        projections = {
            "game_state.json": self.load_game_state(),
            "zeitgeist_state.json": self.load_pulse_state(),
            "hall_of_records.json": self.load_elo_ledger(),
            "bot_brain.json": self.load_bot_memory(),
        }
        if write_mirrors:
            for filename, payload in projections.items():
                self._mirror_projection(filename, payload)
            self.needs_git_reconcile = False
        return projections

    def reconcile_git_from_firestore_if_needed(self) -> bool:
        if self.canonical_only and self._mirror_store is not None:
            self.canonical_projections(write_mirrors=True)
            return True
        if self._fallback_store is None:
            if self.needs_git_reconcile:
                self.needs_git_reconcile = False
                return True
            return False

        client = self._get_client()
        if client is None:
            return False

        reconciled = bool(self.needs_git_reconcile)

        pulse_doc = client.collection(self._col("profile_state")).document("daily_pulse").get()
        if getattr(pulse_doc, "exists", False):
            fs_pulse = _strip_transient_keys(pulse_doc.to_dict() or {})
            fs_ver = int(fs_pulse.get("state_version", 0))
            local_pulse = (
                self._fallback_store._peek_local_pulse_state()
                if hasattr(self._fallback_store, "_peek_local_pulse_state")
                else None
            )
            local_ver = int((local_pulse or {}).get("state_version", 0))
            if self.needs_git_reconcile or fs_ver > local_ver:
                self._fallback_store.save_pulse_state(fs_pulse)
                reconciled = True

        game_doc = client.collection(self._col("profile_state")).document("active_game").get()
        if getattr(game_doc, "exists", False):
            fs_game = _strip_transient_keys(game_doc.to_dict() or {})
            fs_ver = int(fs_game.get("state_version", 0))
            local_game = (
                self._fallback_store._peek_local_game_state()
                if hasattr(self._fallback_store, "_peek_local_game_state")
                else None
            )
            local_ver = int((local_game or {}).get("state_version", 0))
            if self.needs_git_reconcile or fs_ver > local_ver:
                self._fallback_store.save_game_state(fs_game)
                reconciled = True

        bot_doc = client.collection(self._col("profile_state")).document("bot_memory").get()
        if getattr(bot_doc, "exists", False) and hasattr(
            self._fallback_store, "save_bot_memory"
        ):
            fs_bot = _strip_transient_keys(bot_doc.to_dict() or {})
            fs_ver = int(fs_bot.get("state_version", fs_bot.get("games_played", 0)))
            local_bot = (
                self._fallback_store._peek_local_bot_memory()
                if hasattr(self._fallback_store, "_peek_local_bot_memory")
                else None
            )
            local_ver = int(
                (local_bot or {}).get(
                    "state_version", (local_bot or {}).get("games_played", 0)
                )
            )
            if self.needs_git_reconcile or fs_ver > local_ver:
                self._fallback_store.save_bot_memory(fs_bot)
                reconciled = True

        elo_doc = client.collection(self._col("profile_state")).document("elo_ledger").get()
        if getattr(elo_doc, "exists", False) and hasattr(
            self._fallback_store, "save_elo_ledger"
        ):
            fs_elo = _strip_transient_keys(elo_doc.to_dict() or {})
            fs_ver = int(
                fs_elo.get(
                    "state_version",
                    len(fs_elo.get("recent_matches", fs_elo.get("recent_rounds", []))),
                )
            )
            local_elo = (
                self._fallback_store._peek_local_elo_ledger()
                if hasattr(self._fallback_store, "_peek_local_elo_ledger")
                else None
            )
            local_ver = int(
                (local_elo or {}).get(
                    "state_version",
                    len(
                        (local_elo or {}).get(
                            "recent_matches", (local_elo or {}).get("recent_rounds", [])
                        )
                    ),
                )
            )
            if self.needs_git_reconcile or fs_ver > local_ver:
                self._fallback_store.save_elo_ledger(fs_elo)
                reconciled = True

        self.needs_git_reconcile = False
        return reconciled

    def reconcile_to_git(self) -> bool:
        return self.reconcile_git_from_firestore_if_needed()

    def record_completed_game(self, actor_handle: str, now_epoch: int) -> None:
        self.save_game_rate_limits(
            _increment_completed_game_count(
                self.load_game_rate_limits(), actor_handle, now_epoch
            )
        )

    def record_mood_vote(
        self,
        date_str: str,
        mood_key: str,
        voter_hmac: str,
    ) -> tuple[bool, dict[str, int]]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.record_mood_vote(date_str, mood_key, voter_hmac)

        is_valid, norm_date, norm_mood, norm_hmac, dedup_key = _normalize_mood_vote_inputs(
            date_str, mood_key, voter_hmac
        )

        votes_col, rates_col, pulse_col = (
            self._col("pulse_votes"), self._col("vote_rate_limits"), self._col("profile_state")
        )
        for attempt in range(5):
            # Every read participates in the same conditional commit. A new vote
            # or editorial publication invalidates this snapshot and is reread.
            votes_doc = client.collection(votes_col).document(norm_date).get()
            rate_doc = client.collection(rates_col).document(dedup_key).get()
            pulse_doc = client.collection(pulse_col).document("daily_pulse").get()
            pulse = _strip_transient_keys(pulse_doc.to_dict() or {}) if pulse_doc.exists else {}
            if self.canonical_only:
                if not pulse_doc.exists:
                    raise RuntimeError("Canonical Firestore daily_pulse is missing; explicit bootstrap required")
                self._canonical_tally(pulse)
                if votes_doc.exists:
                    self._canonical_tally({"visitor_mood_tally": votes_doc.to_dict()})
            tally = _normalize_mood_tally(
                votes_doc.to_dict() if votes_doc.exists
                else pulse.get("visitor_mood_tally", pulse.get("mood_votes", DEFAULT_MOOD_TALLY))
            )
            current_date = str(pulse.get("date") or pulse.get("date_iso")
                or (pulse.get("analysis") or {}).get("date_iso") or "")[:10]
            if not is_valid or rate_doc.exists or (current_date and current_date != norm_date):
                return False, dict(tally)

            tally[norm_mood] += 1
            rate = {
                "date": norm_date, "vote_type": norm_mood,
                "voted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date_str": norm_date, "voter_hmac": norm_hmac, "mood_key": norm_mood,
            }
            pulse_patch = {"visitor_mood_tally": dict(tally), "mood_votes": dict(tally)}
            try:
                _commit_documents(client, [
                    (rates_col, dedup_key, rate, rate_doc),
                    (votes_col, norm_date, dict(tally), votes_doc),
                    (pulse_col, "daily_pulse", pulse_patch, pulse_doc),
                ])
            except PublicationConflict:
                if attempt == 4:
                    raise
                continue

            # Editorial fields are never sent in a vote write. Mirror the exact
            # accepted canonical tally locally rather than incrementing stale data.
            self.needs_git_reconcile = True
            self._mirror_projection("zeitgeist_state.json", dict(pulse, **pulse_patch))
            if self._fallback_store is not None:
                self._fallback_store.record_mood_vote(norm_date, norm_mood, norm_hmac)
                self._fallback_store.save_pulse_state(dict(pulse, **pulse_patch))
            return True, dict(tally)
        raise RuntimeError("Vote contention exhausted without a canonical commit")

    def get_recent_pulse_words(self, days: int = 30) -> list[str]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.get_recent_pulse_words(days=days)

        limit = max(1, int(days))
        words: list[str] = []

        history_col = client.collection(self._col("pulse_history"))
        docs = list(history_col.stream()) if hasattr(history_col, "stream") else []
        sorted_docs = sorted(
            docs,
            key=lambda d: str((d.to_dict() or {}).get("date_iso", "")),
            reverse=True,
        )
        for doc in sorted_docs:
            payload = doc.to_dict() or {}
            w = str(
                payload.get("word_of_the_day") or payload.get("word") or ""
            ).strip().upper()
            if w and w not in words:
                words.append(w)
            if len(words) >= limit:
                return words[:limit]

        pulse_state = self.load_pulse_state()
        for w in _extract_recent_words(pulse_state, days=limit):
            if w not in words:
                words.append(w)
            if len(words) >= limit:
                break

        return words[:limit]

    def save_ingestion_cache_day(
        self,
        date_str: str,
        items_by_hash: dict[str, dict[str, Any]],
    ) -> None:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            self._fallback_store.save_ingestion_cache_day(date_str, items_by_hash)
            return

        cache_col = client.collection(self._col("ingestion_cache"))
        snapshots = {str(doc.id): doc for doc in cache_col.stream()}
        same_day = {doc_id: doc.to_dict() or {} for doc_id, doc in snapshots.items()
            if re.fullmatch(re.escape(str(date_str)) + r"(?:__\d{3,})?", doc_id)}
        merged = {}
        for payload in _active_cache_pages(same_day).values():
            merged.update(payload.get("items") or {})
        merged.update(items_by_hash)
        pages = pack_ingestion_cache_pages(str(date_str), merged)
        if "page_count" in (same_day.get(str(date_str)) or {}):
            pages[str(date_str)]["page_count"] = len(pages)
        writes = []
        for doc_id, payload in pages.items():
            snapshot = snapshots.get(doc_id) or cache_col.document(doc_id).get()
            writes.append((self._col("ingestion_cache"), doc_id, payload, snapshot))
        _commit_documents(client, writes)
        # Retired pages remain invisible via page_count until their date expires.
        # Never delete them here: a concurrent repack may have reused their IDs.
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "save_ingestion_cache_day"
        ):
            self._fallback_store.save_ingestion_cache_day(date_str, merged)

    def load_ingestion_cache(self, now_utc: datetime) -> dict[str, dict[str, Any]]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.load_ingestion_cache(now_utc)

        shards: dict[str, dict[str, Any]] = {}
        cache_col = client.collection(self._col("ingestion_cache"))
        docs = list(cache_col.stream()) if hasattr(cache_col, "stream") else []
        for doc in docs:
            payload = doc.to_dict() or {}
            doc_id = str(getattr(doc, "id", "") or payload.get("date") or "")
            if doc_id:
                shards[doc_id] = payload
        return _hydrate_ingestion_cache_from_shards(shards, now_utc)

    def prune_expired_cache_items(self, now_utc: datetime) -> int:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.prune_expired_cache_items(now_utc)

        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        else:
            now_utc = now_utc.astimezone(timezone.utc)

        shards: dict[str, dict[str, Any]] = {}
        cache_col = client.collection(self._col("ingestion_cache"))
        docs = list(cache_col.stream()) if hasattr(cache_col, "stream") else []
        for doc in docs:
            payload = doc.to_dict() or {}
            doc_id = str(getattr(doc, "id", "") or payload.get("date") or "")
            if doc_id:
                shards[doc_id] = payload

        pruned, deleted_dates, updated_dates = _prune_shard_dict_in_place(shards, now_utc)
        for d_str in deleted_dates:
            doc_ref = cache_col.document(d_str)
            if hasattr(doc_ref, "delete"):
                doc_ref.delete()
        for d_str in updated_dates:
            doc = shards.get(d_str) or {}
            cache_col.document(d_str).set(dict(doc, date=_cache_shard_date(d_str, doc), items=doc.get("items") or {}))

        # Also prune pulse_history (> 365d) and vote_rate_limits (> 30d)
        cutoff_365d = (now_utc - timedelta(days=365)).strftime("%Y-%m-%d")
        history_col = client.collection(self._col("pulse_history"))
        for doc in list(history_col.stream()) if hasattr(history_col, "stream") else []:
            payload = doc.to_dict() or {}
            doc_id = str(getattr(doc, "id", "") or payload.get("date") or payload.get("date_iso") or "")
            if doc_id and doc_id[:10] < cutoff_365d:
                ref = history_col.document(doc_id)
                if hasattr(ref, "delete"):
                    ref.delete()
                    pruned += 1

        cutoff_30d = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")
        limits_col = client.collection(self._col("vote_rate_limits"))
        for doc in list(limits_col.stream()) if hasattr(limits_col, "stream") else []:
            payload = doc.to_dict() or {}
            doc_id = str(getattr(doc, "id", "") or "")
            date_part = str(payload.get("date") or payload.get("date_str") or doc_id[:10])
            if date_part and date_part[:10] < cutoff_30d and doc_id:
                ref = limits_col.document(doc_id)
                if hasattr(ref, "delete"):
                    ref.delete()
                    pruned += 1

        if self._fallback_store is not None and hasattr(
            self._fallback_store, "prune_expired_cache_items"
        ):
            self._fallback_store.prune_expired_cache_items(now_utc)
        return pruned

    def load_game_rate_limits(self) -> dict[str, Any]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.load_game_rate_limits()

        doc = client.collection(self._col("profile_state")).document("game_rate_limits").get()
        if getattr(doc, "exists", False):
            data = doc.to_dict() or {}
            data.setdefault("users", {})
            data.setdefault("state_version", 0)
            return data
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "load_game_rate_limits"
        ):
            return self._fallback_store.load_game_rate_limits()
        return {"users": {}, "state_version": 0}

    def save_game_rate_limits(self, state: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(state)
        cleaned.setdefault("users", {})
        cleaned.setdefault("state_version", 0)
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            self._fallback_store.save_game_rate_limits(cleaned)
            return

        client.collection(self._col("profile_state")).document("game_rate_limits").set(cleaned)
        self._mirror_projection("game_rate_limits.json", cleaned)
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "save_game_rate_limits"
        ):
            self._fallback_store.save_game_rate_limits(cleaned)

    def check_and_increment_game_rate_limit(
        self,
        actor_handle: str,
        now_epoch: int,
        mode: str = "VS_BOT",
        game_completed: bool = False,
        is_new_round: bool = False,
    ) -> tuple[bool, str]:
        ok, reason, updated = _apply_game_rate_limit(
            self.load_game_rate_limits(),
            actor_handle=actor_handle,
            now_epoch=now_epoch,
            mode=mode,
            game_completed=game_completed,
            is_new_round=is_new_round,
        )
        if ok:
            self.save_game_rate_limits(updated)
        return ok, reason

    def load_orbit_roster(self) -> dict[str, Any]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.load_orbit_roster()

        doc = client.collection(self._col("profile_state")).document("orbit_roster").get()
        if getattr(doc, "exists", False):
            data = _strip_transient_keys(doc.to_dict() or {})
            data.setdefault("handles", {})
            data.setdefault("curated_o3_cursor", {})
            return data
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "load_orbit_roster"
        ):
            return self._fallback_store.load_orbit_roster()
        return {"handles": {}, "curated_o3_cursor": {}, "state_version": 0}

    def save_orbit_roster(self, roster: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(roster)
        cleaned.setdefault("handles", {})
        cleaned.setdefault("curated_o3_cursor", {})
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            self._fallback_store.save_orbit_roster(cleaned)
            return

        client.collection(self._col("profile_state")).document("orbit_roster").set(cleaned)
        self._mirror_projection("orbit_roster.json", cleaned)
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "save_orbit_roster"
        ):
            self._fallback_store.save_orbit_roster(cleaned)

    def load_pulse_feed_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Read authoritative dated history, never a potentially stale Git mirror."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("Canonical Firestore history unavailable")
        documents = client.collection(self._col("pulse_history")).stream()
        entries = [_strip_transient_keys(doc.to_dict() or {}) for doc in documents
                   if getattr(doc, "exists", True)]
        entries.sort(key=lambda item: str(item.get("date") or item.get("date_iso") or ""), reverse=True)
        return entries[:max(1, int(days))]

    def save_pulse_feed_entry(self, date_str: str, entry: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(entry)
        cleaned["date"] = str(date_str)[:10]
        cleaned["schema_version"] = "2.0"
        if str(cleaned.get("image_data_uri") or "").startswith("data:"):
            cleaned["image_data_uri"] = ""
        client = self._get_client()
        if client is not None:
            client.collection(self._col("pulse_history")).document(str(date_str)[:10]).set(cleaned)
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "save_pulse_feed_entry"
        ):
            self._fallback_store.save_pulse_feed_entry(date_str, cleaned)

    def load_bot_memory(self) -> dict[str, Any]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.load_bot_memory()

        doc = client.collection(self._col("profile_state")).document("bot_memory").get()
        local_data = (
            self._fallback_store._peek_local_bot_memory()
            if self._fallback_store is not None
            and hasattr(self._fallback_store, "_peek_local_bot_memory")
            else None
        )
        local_ver = int((local_data or {}).get("state_version", 0))
        if getattr(doc, "exists", False):
            fs_data = _strip_transient_keys(doc.to_dict() or {})
            fs_ver = int(fs_data.get("state_version", 0))
            if local_data is not None and local_ver > fs_ver:
                out = _strip_transient_keys(local_data)
                out.setdefault("state_version", local_ver)
                out.setdefault("games_learned", 0)
                out.setdefault("states", {})
                return out
            if fs_ver > local_ver:
                self.needs_git_reconcile = True
            fs_data.setdefault("state_version", fs_ver)
            fs_data.setdefault("games_learned", 0)
            fs_data.setdefault("states", {})
            return fs_data

        if local_data is not None:
            out = _strip_transient_keys(local_data)
            out.setdefault("state_version", local_ver)
            out.setdefault("games_learned", 0)
            out.setdefault("states", {})
            return out
        return {"state_version": 0, "games_learned": 0, "states": {}}

    def save_bot_memory(self, memory: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(memory)
        cleaned.setdefault("state_version", 0)
        cleaned.setdefault("games_learned", 0)
        cleaned.setdefault("states", {})
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            self._fallback_store.save_bot_memory(cleaned)
            return

        client.collection(self._col("profile_state")).document("bot_memory").set(cleaned)
        self._mirror_projection("bot_brain.json", cleaned)
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "save_bot_memory"
        ):
            self._fallback_store.save_bot_memory(cleaned)

    def load_elo_ledger(self) -> dict[str, Any]:
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            return self._fallback_store.load_elo_ledger()

        doc = client.collection(self._col("profile_state")).document("elo_ledger").get()
        if self.canonical_only and not getattr(doc,"exists",False):
            return {"state_version":0,"bot_elo":None,"players":{},"recent_rounds":[]}
        local_data = (
            self._fallback_store._peek_local_elo_ledger()
            if self._fallback_store is not None
            and hasattr(self._fallback_store, "_peek_local_elo_ledger")
            else None
        )
        local_ver = int((local_data or {}).get("state_version", 0))
        if getattr(doc, "exists", False):
            fs_data = _strip_transient_keys(doc.to_dict() or {})
            fs_ver = int(fs_data.get("state_version", 0))
            if local_data is not None and local_ver > fs_ver:
                out = _strip_transient_keys(local_data)
                out.setdefault("state_version", local_ver)
                out.setdefault("bot_elo", 1200)
                out.setdefault("players", {})
                out.setdefault("recent_rounds", [])
                return out
            if fs_ver > local_ver:
                self.needs_git_reconcile = True
            fs_data.setdefault("state_version", fs_ver)
            fs_data.setdefault("bot_elo", None if self.canonical_only else 1200)
            fs_data.setdefault("players", {})
            fs_data.setdefault("recent_rounds", [])
            return fs_data

        if local_data is not None:
            out = _strip_transient_keys(local_data)
            out.setdefault("state_version", local_ver)
            out.setdefault("bot_elo", 1200)
            out.setdefault("players", {})
            out.setdefault("recent_rounds", [])
            return out
        return {
            "state_version": 0,
            "bot_elo": 1200,
            "players": {},
            "recent_rounds": [],
        }

    def save_elo_ledger(self, ledger: dict[str, Any]) -> None:
        cleaned = _strip_transient_keys(ledger)
        cleaned.setdefault("state_version", 0)
        cleaned.setdefault("bot_elo", 1200)
        cleaned.setdefault("players", {})
        cleaned.setdefault("recent_rounds", [])
        client = self._get_client()
        if client is None and self._fallback_store is not None:
            self._fallback_store.save_elo_ledger(cleaned)
            return

        client.collection(self._col("profile_state")).document("elo_ledger").set(cleaned)
        self._mirror_projection("hall_of_records.json", cleaned)
        if self._fallback_store is not None and hasattr(
            self._fallback_store, "save_elo_ledger"
        ):
            self._fallback_store.save_elo_ledger(cleaned)


def get_state_store(
    repo_root: Path | None = None,
    backend: str | None = None,
    firestore_client: Any = None,
    collection_prefix: str | None = None,
) -> StateStore:
    """Select an adapter; Firestore runtime authority is canonical-only with JSON mirrors."""
    resolved_backend = (
        backend or os.environ.get("STATE_STORE_BACKEND", "json")
    ).strip().lower()
    root = Path(repo_root) if repo_root is not None else REPO_ROOT

    if resolved_backend in ("memory", "in_memory", "inmemory"):
        return InMemoryStateStore()

    if resolved_backend == "firestore":
        return FirestoreStateStore(
            client=firestore_client,
            canonical_only=True,
            mirror_store=JsonFileStateStore(repo_root=root),
            collection_prefix=collection_prefix,
        )

    return JsonFileStateStore(repo_root=root)



def _public_publication_paths(root: Path) -> list[str]:
    """Closed public projection paths; operational state never enters Git by glob."""
    from scripts.build_readme import VARIANT_FILES
    from src.editorial_typography import ASSETS as TYPOGRAPHY_ASSETS
    paths = [*TYPOGRAPHY_ASSETS, "docs/resonance.md", "README.md", "assets/game_board.svg", "assets/daily_pulse.svg",
             "assets/daily_pulse_light.webp", "assets/daily_pulse_dark.webp", "assets/pulse_manifest.json",
             "config/orbit_sources.json"]
    paths += ["variants/README-" + key + ".md" for key in ("ai-engineering", "fpa-analytics", "bi-data")]
    paths += ["variants/" + name for name in VARIANT_FILES.values()]
    paths += ["data/" + name for name in ("game_state.json", "zeitgeist_state.json", "hall_of_records.json",
                                          "bot_brain.json", "pulse_feed.json", "orbit_roster.json")]
    feed = root / "data/pulse_feed.json"
    if feed.is_file():
        payload = json.loads(feed.read_text())
        for entry in payload.get("entries", []):
            if not isinstance(entry, dict):
                continue
            for key in ("fallback_webp_path", "webp_asset_path"):
                path = entry.get(key, "")
                if isinstance(path, str) and re.fullmatch(r"assets/pulse/\d{4}-\d{2}-\d{2}\.webp", path):
                    paths.append(path)
    manifest = root / "assets/pulse_manifest.json"
    if manifest.is_file():
        date = json.loads(manifest.read_text()).get("date", "")
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError("Invalid public archive date")
        paths.append("assets/pulse/" + date + ".webp")
    return sorted(set(paths))


def _canonical_publication_snapshot(store: FirestoreStateStore) -> dict:
    documents = store.canonical_projections(write_mirrors=False)
    documents["orbit_roster.json"] = store.load_orbit_roster()
    feed = store._get_client().collection(store._col("pulse_feed")).document("v1").get()
    if feed.exists:
        envelope = feed.to_dict() or {}
        if not isinstance(envelope.get("entries"), list):
            raise RuntimeError("Canonical pulse feed is invalid")
        documents["pulse_feed.json"] = envelope
    else:
        documents["pulse_feed.json"] = {"version": "2.0", "schema_version": "2.0",
                                       "entries": store.load_pulse_feed_history(days=365)}
    return documents


def _rebuild_publication(root: Path, source: Path, store: FirestoreStateStore) -> None:
    """Render a frozen canonical projection; reuse validated art, never infer/seed."""
    import shutil
    from scripts.build_readme import build_all_readmes, check_all_readmes
    from .zeitgeist.publication import verify_publication_bundle, verify_current_pulse
    from .zeitgeist.visual_router import update_svg_mood_vote_bar
    documents = _canonical_publication_snapshot(store)
    pulse = documents["zeitgeist_state.json"]
    publication_id = pulse.get("publication_id")
    if not isinstance(publication_id, str) or not re.fullmatch(r"[a-f0-9]{20}", publication_id):
        raise ValueError("Canonical publication identity unavailable")
    selected = None
    for candidate in (root, source, source / ".publication-recovery" / publication_id):
        try:
            manifest = verify_publication_bundle(candidate, require_current_pulse=False)
            if manifest["publication_id"] != publication_id:
                continue
            verify_current_pulse(pulse, manifest)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        selected = candidate, manifest
        break
    if selected is None:
        raise ValueError("Accepted canonical artwork unavailable; retain recovery evidence")
    candidate, manifest = selected
    for relative in ["assets/pulse_manifest.json", *manifest["files"]]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (candidate / relative).resolve() != destination.resolve():
            shutil.copy2(candidate / relative, destination)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.items():
        (data / name).write_text(json.dumps(_encode_json_safe(payload), indent=2) + "\n")
    # Source history/feed and current pulse must identify the same accepted art.
    verify_publication_bundle(root, require_current_pulse=True)
    update_svg_mood_vote_bar(root / "assets/daily_pulse.svg", pulse["visitor_mood_tally"])
    snapshot_store = JsonFileStateStore(root)
    build_all_readmes(root, state_store=snapshot_store)
    if not check_all_readmes(root, state_store=snapshot_store):
        raise ValueError("Canonical README projection did not validate")
    verify_publication_bundle(root, require_current_pulse=True)
    if _canonical_publication_snapshot(store) != documents:
        raise PublicationConflict("Canonical state advanced while rendering; rebuild")


def _publish_canonical_git(root: Path, store: FirestoreStateStore, message: str,
                           attempts: int, sleep_fn: Any) -> bool:
    """Build atop fetched remote HEAD, never rebase conflicting generated snapshots."""
    branch = (os.environ.get("GITHUB_REF_NAME") or "main").strip() or "main"
    def run(args, *, cwd=root, **kwargs):
        return subprocess.run(["git", *args], cwd=str(cwd), check=False, **kwargs)
    # An intentional owner configuration edit is the only source edit overlaid.
    patch = run(["diff", "--binary", "HEAD", "--", "config/orbit_sources.json"], capture_output=True, text=True)
    if patch.returncode != 0:
        return False
    for attempt in range(max(1, int(attempts))):
        if run(["fetch", "origin", branch]).returncode != 0:
            return False
        head = run(["rev-parse", "FETCH_HEAD"], capture_output=True, text=True)
        if head.returncode != 0:
            return False
        # This process imported the event checkout's renderer. Never render with
        # stale code over a newer runtime; a fresh worker can safely read receipts.
        runtime = run(["diff", "--quiet", "HEAD", head.stdout.strip(), "--", "src/", "scripts/",
                       "requirements.txt", "requirements-ci.lock", ".github/workflows/"])
        if runtime.returncode != 0:
            print("Remote runtime changed; rebuild with a fresh worker before publication.")
            return False
        with tempfile.TemporaryDirectory(prefix="profile-git-publication-") as directory:
            checkout = Path(directory) / "checkout"
            if run(["worktree", "add", "--detach", str(checkout), head.stdout.strip()]).returncode != 0:
                return False
            try:
                if patch.stdout and run(["apply", "--3way", "--index"], cwd=checkout,
                                         input=patch.stdout, text=True, capture_output=True).returncode != 0:
                    return False  # Never overwrite a concurrent owner/source edit.
                try:
                    _rebuild_publication(checkout, root, store)
                except PublicationConflict:
                    continue
                except Exception as exc:
                    print(f"Canonical Git publication blocked ({type(exc).__name__}); retained source unchanged.")
                    return False
                paths = [p for p in _public_publication_paths(checkout) if (checkout / p).is_file()]
                if run(["add", "--", *paths], cwd=checkout).returncode != 0:
                    return False
                staged = run(["diff", "--cached", "--name-only", "-z"], cwd=checkout, capture_output=True, text=True)
                if staged.returncode or not set(staged.stdout.split("\0")) - {""} <= set(paths):
                    return False
                diff = run(["diff", "--cached", "--quiet"], cwd=checkout)
                if diff.returncode == 0:
                    store.set_needs_git_reconcile(False)
                    return True
                if diff.returncode != 1 or run(["commit", "-m", message], cwd=checkout).returncode != 0:
                    return False
                if run(["push", "origin", "HEAD:" + branch], cwd=checkout).returncode == 0:
                    store.set_needs_git_reconcile(False)
                    return True
            finally:
                run(["worktree", "remove", "--force", str(checkout)], capture_output=True)
        if attempt + 1 < attempts:
            sleep_fn((2.0, 5.0, 10.0)[min(attempt, 2)])
    return False


def git_commit_and_push_with_retry(
    commit_message: str,
    paths: list[str] | None = None,
    repo_root: Path | None = None,
    max_attempts: int = 3,
    store: StateStore | None = None,
    sleep_fn: Any = time.sleep,
) -> bool:
    """Publish explicit public paths; canonical writers rebuild on current remote HEAD."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    def publication_failed() -> bool:
        resolved_store = store if store is not None else get_state_store(repo_root=root)
        if hasattr(resolved_store, "set_needs_git_reconcile"):
            resolved_store.set_needs_git_reconcile(True)
        return False

    resolved_store = store if store is not None else get_state_store(repo_root=root)
    stage_targets = paths if paths is not None else _public_publication_paths(root)
    subprocess.run(
        ["git", "config", "user.name", "github-actions[bot]"],
        cwd=str(root),
        check=False,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ],
        cwd=str(root),
        check=False,
    )
    if paths is None and isinstance(resolved_store, FirestoreStateStore) and resolved_store.canonical_only:
        if _publish_canonical_git(root, resolved_store, commit_message, max_attempts, sleep_fn):
            return True
        return publication_failed()
    existing_targets = [
        p
        for p in stage_targets
        if paths is not None or (root / p.rstrip("/")).exists()
    ]
    if existing_targets:
        add_proc = subprocess.run(
            ["git", "add", "--", *existing_targets],
            cwd=str(root),
            check=False,
        )
        if add_proc.returncode != 0:
            return publication_failed()
    if paths is None:
        staged = subprocess.run(["git", "diff", "--cached", "--name-only", "-z"], cwd=str(root),
                                check=False, capture_output=True, text=True)
        if staged.returncode or not set(staged.stdout.split("\0")) - {""} <= set(stage_targets):
            return publication_failed()
    diff_proc = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(root),
        check=False,
    )
    if diff_proc.returncode == 0:
        return True
    if diff_proc.returncode != 1:
        return publication_failed()

    commit_proc = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=str(root),
        check=False,
    )
    if commit_proc.returncode != 0:
        return publication_failed()

    target_branch = (os.environ.get("GITHUB_REF_NAME") or "main").strip() or "main"
    backoffs = (2.0, 5.0, 10.0)
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        rebase_proc = subprocess.run(
            ["git", "pull", "--rebase", "origin", target_branch],
            cwd=str(root),
            check=False,
        )
        if rebase_proc.returncode != 0:
            subprocess.run(
                ["git", "rebase", "--abort"],
                cwd=str(root),
                check=False,
            )
            if attempt < max_attempts:
                sleep_fn(backoffs[min(attempt - 1, len(backoffs) - 1)])
            continue
        push_proc = subprocess.run(
            ["git", "push", "origin", f"HEAD:{target_branch}"],
            cwd=str(root),
            check=False,
        )
        if push_proc.returncode == 0:
            return True
        if attempt < max_attempts:
            delay = backoffs[min(attempt - 1, len(backoffs) - 1)]
            sleep_fn(delay)

    return publication_failed()
