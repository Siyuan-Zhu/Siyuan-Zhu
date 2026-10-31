"""Successful-batch Instagram rotation and the approved v3 scout lifecycle.

The legacy collector roster API remains readable for old callers. Live runs use
this module and retain historical record fields while adding v3 accounting dates.
"""
from __future__ import annotations

import copy
from datetime import date, datetime, timezone
import math
import re
from typing import Any, Sequence

HANDLE = re.compile(r"[a-z0-9._]{2,30}\Z")
MAX_V3_SCOUTS = 20


def _counter(value: Any) -> int:
    return max(0, int(value)) if isinstance(value, (int, str)) and str(value).isdigit() else 0


def _day(value: Any) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        return ""


def _vitality(record: dict) -> float:
    try:
        value = float(record.get("vitality_score", record.get("decayed_score", .5)))
        return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _ranked_scouts(handles: dict, excluded=()) -> list[tuple[str, dict]]:
    return sorted(
        ((handle, record) for handle, record in handles.items()
         if HANDLE.fullmatch(handle) and handle not in excluded and isinstance(record, dict)
         and _vitality(record) >= .2),
        key=lambda pair: (-_vitality(pair[1]), pair[0]),
    )


def select_instagram_batch(roster: dict, curated_handles: Sequence[str], *, blocklist=()) -> list[str]:
    seeds = list(dict.fromkeys(str(h).lower().lstrip("@") for h in curated_handles))
    blocked = {str(h).lower().lstrip("@") for h in blocklist}
    cursor = _counter(roster.get("curated_o3_cursor"))
    selected = [seeds[(cursor + i) % len(seeds)] for i in range(min(8, len(seeds)))] if seeds else []
    scouts = _ranked_scouts(roster.get("handles") or {}, set(seeds) | blocked)
    selected.extend(handle for handle, _record in scouts[:8])
    return [h for h in selected if HANDLE.fullmatch(h) and h not in blocked]


def advance_roster_after_batch(
    roster: dict, mention_parents: dict[str, set[str]], now_utc: datetime, *,
    apify_batch_succeeded: bool, curated_handles: Sequence[str] = (),
) -> dict:
    result = copy.deepcopy(roster)
    if apify_batch_succeeded is not True:
        return result
    result["curated_o3_cursor"] = (_counter(roster.get("curated_o3_cursor")) + 8) % max(1, len(curated_handles) or 12)
    result["rss_o3_cursor"] = (_counter(roster.get("rss_o3_cursor", roster.get("curated_o3_cursor"))) + 2) % 6
    result["updated_at"] = now_utc.astimezone(timezone.utc).isoformat()
    result["state_version"] = _counter(roster.get("state_version")) + 1
    today = now_utc.astimezone(timezone.utc).date().isoformat()
    stamp = now_utc.astimezone(timezone.utc).isoformat()
    seeds = {str(handle).lower().lstrip("@") for handle in curated_handles}
    handles = result.setdefault("handles", {})
    for handle, record in list(handles.items()):
        if not isinstance(record, dict) or handle in seeds:
            continue
        score = _vitality(record)
        decay_date = _day(record.get("last_vitality_decay_date") or record.get("last_updated_utc") or roster.get("updated_at"))
        if decay_date < today:
            score = max(0.0, score - .12)
            record["last_vitality_decay_date"] = today
        record["vitality_score"] = round(score, 10)
    for raw_handle, parents in mention_parents.items():
        handle = str(raw_handle).lower().lstrip("@")
        if not HANDLE.fullmatch(handle) or handle in seeds or not parents:
            continue
        discovered = handle not in handles
        record = handles.setdefault(handle, {
            "handle": handle, "vitality_score": .50, "first_seen_utc": stamp,
            "last_vitality_decay_date": today,
        })
        old_parents = record.get("parent_sources") or record.get("distinct_parent_sources") or []
        old_parents = old_parents if isinstance(old_parents, (list, set, tuple)) else []
        record["parent_sources"] = sorted(set(old_parents) | set(parents))
        record["distinct_parent_sources"] = list(record["parent_sources"])
        last_bonus = _day(record.get("last_remention_bonus_date") or record.get("last_mentioned_at"))
        if discovered or last_bonus < today:
            record["mention_count"] = _counter(record.get("mention_count")) + len(parents)
            if not discovered:
                record["vitality_score"] = round(min(1.0, _vitality(record) + .25), 10)
            record.update(last_remention_bonus_date=today, last_mentioned_at=stamp, last_seen_utc=stamp)
    retained = {handle: record for handle, record in handles.items() if handle in seeds}
    for handle, record in _ranked_scouts(handles, seeds)[:MAX_V3_SCOUTS]:
        record.update(promoted=True, status="ACTIVE_O4_SCOUT", tier="O4_SCOUT",
                      last_updated_utc=stamp)
        record.setdefault("decayed_score", _vitality(record))
        retained[handle] = record
    result["handles"] = retained
    return result


def apply_winning_cluster_bonus(
    roster: dict, winning_items: Sequence[Any], now_utc: datetime, *, apify_batch_succeeded: bool,
) -> dict:
    """Credit only recorded scouts whose actual Instagram items won today."""
    result = copy.deepcopy(roster)
    if apify_batch_succeeded is not True:
        return result
    today = now_utc.astimezone(timezone.utc).date().isoformat()
    changed = False
    for item in winning_items:
        source_id = item.get("source_id", "") if isinstance(item, dict) else getattr(item, "source_id", "")
        match = re.fullmatch(r"o[1-4]_ig_([a-z0-9._]{2,30})", str(source_id).lower())
        if not match:
            continue
        record = result.get("handles", {}).get(match[1])
        if not isinstance(record, dict) or record.get("tier", "O4_SCOUT") != "O4_SCOUT":
            continue
        if _day(record.get("last_winning_bonus_date")) >= today:
            continue
        record["vitality_score"] = round(min(1.0, _vitality(record) + .35), 10)
        record["last_winning_bonus_date"] = today
        changed = True
    if changed:
        result["state_version"] = _counter(result.get("state_version")) + 1
        result["updated_at"] = now_utc.astimezone(timezone.utc).isoformat()
    return result
