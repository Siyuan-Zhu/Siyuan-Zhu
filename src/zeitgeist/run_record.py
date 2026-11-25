"""The private run record: what ran, what ranked, what the Director was shown and chose, what was published.

Pure functions plus one guarded store write. Nothing here prints, logs, or calls a provider
or the network; the store write is the only outside effect, and it never raises.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

RECORD_SCHEMA = 1
MAX_RECORD_BYTES = 900_000
SWITCH_ENV = "ZEITGEIST_RUN_RECORD"
MAX_GROUPS = 5

# Recorded as read from the environment: a string, or None when unset. Names come only from this list.
SWITCH_NAMES = (
    "ZEITGEIST_LIVE_LLM",
    "ZEITGEIST_LIVE_COLLECT",
    "ZEITGEIST_LIVE_UNSPLASH",
    "ZEITGEIST_OFFLINE_DEMO",
    "ZEITGEIST_UNDERSTANDING_V1",
    "ZEITGEIST_RUN_RECORD",
    "ZEITGEIST_DIRECTOR_MODEL",
    "ZEITGEIST_JUDGE_MODEL",
    "ZEITGEIST_EMBEDDING_MODEL",
    "ZEITGEIST_LINKS",
    "ZEITGEIST_CHOSEN_GROUP_FIELDS",
    "ZEITGEIST_OWN_NIGHT",
    "ZEITGEIST_OWN_GROUP",
    "ZEITGEIST_REFLECTOR",
    "ZEITGEIST_REFLECTOR_MODEL",
    "ZEITGEIST_REFLECTOR_CANDIDATE",
    "ZEITGEIST_UNDATED_AS_BACKGROUND",
    "ZEITGEIST_COPY_CAP",
    "ZEITGEIST_INTERIM_GROUPING",
    "ZEITGEIST_EVIDENCE_ARCHIVE",
    "ZEITGEIST_DROP_FALSE_FIELDS",
)


def run_record_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH_ENV) == "1"


def make_run_id(
    kind: str,
    date_iso: str,
    environ: Optional[Mapping[str, str]] = None,
    arm: Optional[str] = None,
    repeat: Optional[int] = None,
) -> str:
    env = os.environ if environ is None else environ
    parts = [str(kind), str(date_iso)[:10], env.get("GITHUB_RUN_ID") or "local", env.get("GITHUB_RUN_ATTEMPT") or "1"]
    if arm is not None:
        parts.append(str(arm))
    if repeat is not None:
        parts.append(str(repeat))
    return ":".join(parts)


def _field(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def is_instagram_item(item: Any) -> bool:
    # Mirrors the pipeline's own Instagram test in require_live_coverage (adk_pipeline.py:327 on main,
    # :329 once this branch's imports land): the collectors give every
    # Instagram item a source_id of the form "<tier>_ig_<handle>".
    return "_ig_" in str(_field(item, "source_id") or "")


def _plain(value: Any) -> Any:
    """Plain JSON values only: numpy scalars to int/float, non-finite floats to None, dates to ISO, sets sorted."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError("value is not plain JSON")


def _number(value: Any) -> Optional[float]:
    return _plain(float(value)) if value is not None else None


def _group(rank: int, cluster: Any) -> Dict[str, Any]:
    items = list(_field(cluster, "items") or ())
    hashes = [(str(_field(item, "url_hash") or ""), is_instagram_item(item)) for item in items]
    return {
        "cluster_id": str(_field(cluster, "cluster_id")),
        "rank": rank,
        "commission_score": _number(_field(cluster, "commission_score")),
        "velocity": _number(_field(cluster, "velocity")),
        "orbit_resonance": _number(_field(cluster, "orbit_resonance")),
        "s_intra": _number(_field(cluster, "s_intra")),
        "unique_sources_count": int(_field(cluster, "unique_sources_count") or 0),
        "item_count": len(items),
        "item_hashes": [h for h, instagram in hashes if not instagram],
        "instagram_item_hashes": [h for h, instagram in hashes if instagram],
    }


def _link(link: Any) -> Dict[str, Any]:
    return {
        "link_id": str(_field(link, "link_id")),
        "member_ids": [str(member) for member in _field(link, "member_ids") or ()],
        "realms": [str(realm) for realm in _field(link, "realms") or ()],
        "source_count": int(_field(link, "source_count") or 0),
        "mean_link_similarity": _number(_field(link, "mean_link_similarity")),
    }


def _commission_sha256(commission: Any) -> Optional[str]:
    if commission is None:
        return None
    payload = json.dumps(commission.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_iso(now_utc: datetime) -> str:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc).isoformat()


def build_run_record(
    *,
    kind: str,
    date_iso: str,
    now_utc: datetime,
    environ: Mapping[str, str],
    run_models: Any,
    embedding_model: Optional[str],
    clusters: Sequence[Any],
    evidence_text: Optional[str],
    commission: Any,
    candidate_words: Optional[Sequence[Any]],
    chosen_cluster_id: Optional[str],
    intake_hash: Optional[str] = None,
    arm: Optional[str] = None,
    repeat: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
    main_commission: Any = None,
    prompt_sha256: Optional[str] = None,
    cutoffs_source: Optional[str] = None,
    links: Optional[Sequence[Any]] = None,
    realm_counts: Optional[Mapping[str, Any]] = None,
    chosen_mapped: Optional[bool] = None,
    reflector_candidate: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The record as plain values.

    `commission` is the published one (word, gloss, thesis, digest). `main_commission` is the
    Director's main pass when photo recovery revised it; the prompt path and attempts come from it.
    `links` and `realm_counts` are null unless the links switch gave them. `chosen_mapped` is null unless the
    links or the repair switch checked the Director's choice. `reflector_candidate` is null unless the
    reflector's candidate switch looked for a pointed group before the Director call.
    """
    main = commission if main_commission is None else main_commission
    groups = [_group(rank, cluster) for rank, cluster in enumerate(list(clusters or ())[:MAX_GROUPS], 1)]
    chosen = None if chosen_cluster_id is None else str(chosen_cluster_id)
    chosen_rank = next((group["rank"] for group in groups if group["cluster_id"] == chosen), None)
    models = None
    if run_models is not None:
        models = {
            "director": run_models.current("director"),
            "judge": run_models.current("judge"),
            "embedding": embedding_model or run_models.current("embedding"),
            "fallback_fired": bool(run_models.fallback_fired),
            "unrecognized": bool(run_models.unrecognized),
        }
    record: Dict[str, Any] = {
        "record_schema": RECORD_SCHEMA,
        "kind": str(kind),
        "date": str(date_iso)[:10],
        "written_at": _utc_iso(now_utc),
        "commit": environ.get("GITHUB_SHA") or None,
        "switches": {name: environ.get(name) for name in SWITCH_NAMES},
        "models": models,
        "prompt_path": None if main is None else ("packet" if getattr(main, "packet_used", False) else "legacy"),
        "attempts_used": None if main is None else int(getattr(main, "attempts_used", 0)),
        "groups": groups,
        "evidence_text": evidence_text,
        "chosen_cluster_id": chosen,
        "chosen_rank": chosen_rank,
        "candidate_words": list(candidate_words or ()),
        "word": getattr(commission, "word", None),
        "gloss": getattr(commission, "gloss", None),
        "thesis": getattr(commission, "commissioned_thesis", None),
        "commission_sha256": _commission_sha256(commission),
        "prompt_sha256": prompt_sha256,
        "revised_by_photo_recovery": main != commission,
        "main_pass": {"word": getattr(main, "word", None), "gloss": getattr(main, "gloss", None)},
        "intake_hash": intake_hash,
        "cutoffs_source": cutoffs_source,
        "arm": arm,
        "repeat": repeat,
        "tests": dict(extra or {}),
        "links": None if links is None else [_link(link) for link in links],
        "realm_counts": None if realm_counts is None else dict(realm_counts),
        "chosen_mapped": None if chosen_mapped is None else bool(chosen_mapped),
        "reflector_candidate": None if reflector_candidate is None else dict(reflector_candidate),
    }
    return _plain(record)


_SURROGATE_RE = re.compile("[" + chr(0xD800) + "-" + chr(0xDFFF) + "]")
_REPLACEMENT = chr(0xFFFD)


def _without_surrogates(value: Any) -> Any:
    """Every string, keys included, with each surrogate code point replaced by U+FFFD.

    A Python string holds a surrogate only when it is unpaired as text; UTF-8 cannot encode one,
    and Firestore refuses the whole document.
    """
    if isinstance(value, str):
        return _SURROGATE_RE.sub(_REPLACEMENT, value)
    if isinstance(value, dict):
        return {_without_surrogates(key): _without_surrogates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_surrogates(item) for item in value]
    return value


def write_run_record(store: Any, run_id: str, record: Dict[str, Any]) -> bool:
    """Write the record once; False when it is too large or anything fails. Never raises, never prints."""
    try:
        document = _without_surrogates(_plain(record))
        # The record carries its own id, so a later step that loads it can find it again.
        document["run_id"] = run_id
        size = len(json.dumps(document, allow_nan=False).encode("utf-8"))
        if size > MAX_RECORD_BYTES:
            return False
        store.save_run_record(run_id, document)
        return True
    except Exception:
        return False


def add_test_numbers(store: Any, run_id: str, key: str, numbers: Dict[str, Any]) -> bool:
    """Set `tests[key]` on the stored record and save it; no other key changes.

    False when the record is absent or anything fails. Never raises, never prints.
    """
    try:
        record = store.load_run_record(run_id)
        if not isinstance(record, dict) or not isinstance(record.get("tests"), dict):
            return False
        record["tests"][str(key)] = _plain(numbers)
        return write_run_record(store, run_id, record)
    except Exception:
        return False


PRUNE_LOAD_LIMIT = 100_000


def _written_at(record: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(record.get("written_at")))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def prune_run_records(store: Any, *, older_than_days: int = 180, now_utc: Optional[datetime] = None) -> int:
    """Delete the records written more than `older_than_days` ago; the count deleted. Never raises, never prints.

    A record without its id or a readable `written_at` is kept; a time without a zone is read as UTC.
    """
    try:
        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cutoff = now - timedelta(days=int(older_than_days))
        records = store.load_run_records(limit=PRUNE_LOAD_LIMIT)
    except Exception:
        return 0
    deleted = 0
    for record in records or ():
        try:
            run_id = record.get("run_id")
            moment = _written_at(record)
            if isinstance(run_id, str) and moment is not None and moment < cutoff:
                store.delete_run_record(run_id)
                deleted += 1
        except Exception:
            continue
    return deleted
