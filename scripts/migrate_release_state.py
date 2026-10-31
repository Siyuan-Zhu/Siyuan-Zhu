"""Idempotent State Backup, Migration, and Rollback CLI for v3.0 Cutover (R-02).

Provides:
  - `create_release_backup`: Snapshots tracked repo state/assets + optional Firestore
    documents into `<backup_dir>/` with SHA-256 `manifest.json`.
  - `migrate_release_state`: Normalizes local JSON offline or migrates a verified
    canonical Firestore snapshot through an explicit saved plan and recovery journal.
    Production cannot be seeded from sandbox JSON.
  - `rollback_release_state`: Verifies `manifest.json` SHA-256 digests pre-flight
    and restores files + optional Firestore documents.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game_engine import (  # noqa: E402
    _c4_norm_cell,
    encode_bitboard,
    normalize_game_state,
)
from src.state_store import (  # noqa: E402
    DEFAULT_MOOD_TALLY,
    FirestoreStateStore,
    JsonStateStore,
    _enrich_pulse_state_with_history,
    _normalize_mood_tally,
    _strip_transient_keys,
    _py_to_firestore_value,
    _firestore_value_to_py,
)


BACKUP_GLOBS: tuple[str, ...] = (
    "README.md",
    "PROJECTS.md",
    "variants/*.md",
    "data/*.json",
    "state/*.json",
    "assets/*.svg",
    "assets/*.webp",
    "assets/pulse/*.webp",
    "assets/pulse_manifest.json",
    "assets/fonts/*",
    ".github/workflows/*.yml",
    "src/**/*.py",
    "scripts/*.py",
    "firestore.rules",
    "firestore.indexes.json",
    "firebase.json",
    "requirements.txt",
)

PROFILE_STATE_DOC_IDS: tuple[str, ...] = (
    "daily_pulse",
    "active_game",
    "elo_ledger",
    "bot_memory",
    "orbit_roster",
    "game_rate_limits",
)

VALID_VISUAL_MEDIUMS: frozenset[str] = frozenset(
    {
        "UNSPLASH_PHOTO",
        "FLUX_FAL",
        "FLUX_3",
        "GEMINI_IMAGE",
        "BESPOKE_SVG",
        "SVG_FALLBACK",
        "PHOTOGRAPHY",
        "FINE_ART",
        "ILLUSTRATION",
        "GENERATIVE_SVG",
        "IMAGEN_3",
        "FLUX_PRO",
    }
)

ALLOWED_DAILY_PULSE_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "state_version",
        "date",
        "word",
        "sub_ words",
        "sub_words",
        "pronunciation",
        "part_of_speech",
        "gloss",
        "etymology",
        "thesis",
        "commissioned_thesis",
        "headline",
        "summary",
        "24h_shift",
        "dominant_category",
        "dominant_signal_type",
        "orbit_weight_ratio",
        "top_orbit_themes",
        "top_macro_themes",
        "visual_tier_used",
        "artwork_title",
        "artwork_artist",
        "artwork_source",
        "artwork_url",
        "quote",
        "quote_attribution",
        "window_hours",
        "visual_medium",
        "curatorial_archetype",
        "text_quadrant",
        "quietest_quadrant",
        "primary_image_url",
        "fallback_image_url",
        "image_url",
        "fallback_webp_path",
        "exif_summary",
        "unsplash_photographer",
        "unsplash_profile_url",
        "flux_model",
        "published_pulse",
        "commission",
        "cluster_keywords",
        "silhouette_score",
        "velocity",
        "orbit_resonance",
        "orbit_contribution_pct",
        "macro_contribution_pct",
        "orbit_sources_matched",
        "runner_up_clusters",
        "published_at",
        "mood_votes",
        "history_words_30d",
        "medium_history_30d",
        "archetype_history_30d",
        "palette_hex",
        "attribution",
        "signal_metrics",
        "critic_scores",
        "definition",
        "insight",
        "category",
        "votes",
        "history_words",
        "updated_at",
        "visitor_mood_tally",
        "recent_words",
        "analysis",
        "visual_decision",
        "telemetry",
        "date_iso",
        "word_history",
    }
)

ALLOWED_PULSE_HISTORY_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "date",
        "word",
        "gloss",
        "commissioned_thesis",
        "visual_medium",
        "image_url",
        "fallback_webp_path",
        "unsplash_photographer",
        "unsplash_profile_url",
        "flux_model",
        "cluster_keywords",
        "orbit_contribution_pct",
        "macro_contribution_pct",
        "orbit_sources_matched",
        "runner_up_clusters",
        "mood_votes",
        "critic_scores",
        "published_at",
        "updated_at",
        "headline",
        "summary",
        "24h_shift",
        "dominant_category",
        "dominant_signal_type",
        "orbit_weight_ratio",
        "top_orbit_themes",
        "top_macro_themes",
        "visual_tier_used",
        "artwork_title",
        "artwork_artist",
        "artwork_source",
        "artwork_url",
        "image_data_uri",
        "quote",
        "quote_attribution",
        "window_hours",
        "date_iso",
        "word_of_the_day",
        "explanation_phrase",
        "composite_score",
    }
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_backup_files(repo_root: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in BACKUP_GLOBS:
        for match in sorted(repo_root.glob(pattern)):
            if match.is_file():
                rel = match.relative_to(repo_root).as_posix()
                found[rel] = match
    return [found[k] for k in sorted(found)]


def _export_firestore_state(firestore_store: FirestoreStateStore) -> dict[str, Any]:
    client = firestore_store._get_client()
    if client is None:
        raise RuntimeError("Firestore client is unavailable for backup export.")

    export_payload: dict[str, Any] = {
        "collection_prefix": firestore_store.collection_prefix,
        "collections": {
            "profile_state": {},
            "pulse_feed": {},
            "pulse_history": {},
            "pulse_votes": {},
            "vote_rate_limits": {},
            "processed_events": {},
        },
    }

    profile_col = client.collection(firestore_store._col("profile_state"))
    for doc_id in PROFILE_STATE_DOC_IDS:
        doc = profile_col.document(doc_id).get()
        if getattr(doc, "exists", False):
            export_payload["collections"]["profile_state"][doc_id] = doc.to_dict() or {}

    feed_doc = client.collection(firestore_store._col("pulse_feed")).document("v1").get()
    if getattr(feed_doc, "exists", False):
        export_payload["collections"]["pulse_feed"]["v1"] = feed_doc.to_dict() or {}

    for logical_col in ("pulse_history", "pulse_votes", "vote_rate_limits", "processed_events"):
        col_ref = client.collection(firestore_store._col(logical_col))
        docs = list(col_ref.stream()) if hasattr(col_ref, "stream") else []
        for doc in docs:
            doc_id = str(getattr(doc, "id", "") or "")
            if doc_id:
                export_payload["collections"][logical_col][doc_id] = doc.to_dict() or {}

    return export_payload


def create_release_backup(
    repo_root: Path,
    backup_dir: Path,
    *,
    firestore_store: FirestoreStateStore | None = None,
) -> dict[str, Any]:
    """Snapshot tracked repo state/assets and optional Firestore docs with SHA-256 manifest."""
    root = Path(repo_root).resolve()
    dest_dir = Path(backup_dir).resolve()
    if (dest_dir / "manifest.json").exists():
        raise ValueError("Refusing to overwrite an existing backup manifest")
    files_dir = dest_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    file_checksums: dict[str, str] = {}
    for src_path in _collect_backup_files(root):
        rel_posix = src_path.relative_to(root).as_posix()
        dst_path = files_dir / rel_posix
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        file_checksums[rel_posix] = _sha256_file(dst_path)

    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "target": _target_identity(firestore_store) if firestore_store is not None else None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(root),
        "collection_prefix": (
            firestore_store.collection_prefix if firestore_store is not None else ""
        ),
        "files": file_checksums,
        "absent_managed_files": [
            p for p in ("assets/pulse_manifest.json",) if not (root / p).exists()
        ],
        "firestore_export": None,
        "firestore_export_sha256": None,
    }

    if firestore_store is not None:
        export_payload = _export_firestore_state(firestore_store)
        export_path = dest_dir / "firestore_export.json"
        export_path.write_text(
            json.dumps(export_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        doc_hashes: dict[str, str] = {}
        for logical_col, col_docs in sorted((export_payload.get("collections") or {}).items()):
            if isinstance(col_docs, dict):
                for doc_id, doc_payload in sorted(col_docs.items()):
                    canonical_bytes = json.dumps(doc_payload, sort_keys=True).encode("utf-8")
                    doc_hashes[f"{logical_col}/{doc_id}"] = hashlib.sha256(canonical_bytes).hexdigest()
        manifest["firestore_export"] = "firestore_export.json"
        manifest["firestore_export_sha256"] = _sha256_file(export_path)
        manifest["firestore_document_sha256"] = doc_hashes

    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _target_identity(store: FirestoreStateStore) -> dict[str, str]:
    return {"project": store._project_id, "database": store._database_id,
            "collection_prefix": store.collection_prefix}


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_child(base: Path, relative: str) -> Path:
    child = (base / relative).resolve()
    if not child.is_relative_to(base.resolve()):
        raise ValueError("Backup path escapes its directory")
    return child


def _verified_backup(backup_dir: Path, store: FirestoreStateStore | None = None) -> tuple[dict, dict | None]:
    backup_dir = Path(backup_dir).resolve()
    manifest = json.loads((backup_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid backup manifest schema")
    for rel, expected in manifest["files"].items():
        file = _safe_child(backup_dir / "files", rel)
        if not file.is_file() or _sha256_file(file) != expected:
            raise ValueError(f"Backup checksum mismatch for {rel}")
    export = None
    if store is not None:
        if manifest.get("target") != _target_identity(store):
            raise ValueError("Backup target does not match project/database/collection prefix")
        rel = manifest.get("firestore_export")
        expected = manifest.get("firestore_export_sha256")
        if not rel or not expected:
            raise ValueError("Verified Firestore backup export required")
        file = _safe_child(backup_dir, rel)
        if not file.is_file() or _sha256_file(file) != expected:
            raise ValueError("Firestore backup checksum mismatch")
        export = json.loads(file.read_text())
        if export.get("collection_prefix") != store.collection_prefix:
            raise ValueError("Firestore backup target prefix mismatch")
    return manifest, export


def _flatten_collections(collections: dict) -> dict[str, dict]:
    return {f"{col}/{doc_id}": payload for col, docs in collections.items()
            for doc_id, payload in docs.items()}


def _normalize_canonical_export(export: dict) -> dict:
    """Normalize only documents present in the canonical export; never seed from checkout."""
    desired = copy.deepcopy(export["collections"])
    normalizers = {"daily_pulse": _normalize_daily_pulse_v3,
                   "active_game": _normalize_active_game_v3,
                   "elo_ledger": _normalize_elo_ledger_v3,
                   "bot_memory": _normalize_bot_memory_v3,
                   "orbit_roster": _normalize_orbit_roster_v3,
                   "game_rate_limits": _normalize_game_rate_limits_v3}
    for doc_id, payload in list(desired["profile_state"].items()):
        desired["profile_state"][doc_id] = normalizers[doc_id](payload)
    pulse = desired["profile_state"].get("daily_pulse", {})
    for doc_id, entry in list(desired["pulse_history"].items()):
        desired["pulse_history"][doc_id] = _normalize_pulse_feed_entry_v3(entry, pulse)
    for doc_id, votes in list(desired["pulse_votes"].items()):
        desired["pulse_votes"][doc_id] = _normalize_mood_tally(votes)
    if "v1" in desired["pulse_feed"]:
        feed = desired["pulse_feed"]["v1"]
        entries = feed if isinstance(feed, list) else feed.get("entries", [])
        normalized = [_normalize_pulse_feed_entry_v3(e, pulse) for e in entries]
        desired["pulse_feed"]["v1"] = {
            "schema_version": 3, "version": "2.0", "entries": normalized,
            "updated_at": (feed.get("updated_at") if isinstance(feed, dict) else None)
                          or pulse.get("updated_at") or (normalized[0]["updated_at"] if normalized else ""),
        }
    return desired


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def _conditional_document_write(client: Any, collection: str, doc_id: str,
                                expected: dict | None, desired: dict | None) -> None:
    """Replace/delete with a server precondition so a concurrent write is never lost."""
    if hasattr(client, "compare_and_set_document"):
        client.compare_and_set_document(collection, doc_id, expected, desired)
        return
    if hasattr(client, "_base_url") and hasattr(client, "_resolve_token"):
        headers = {"Authorization": f"Bearer {client._resolve_token()}",
                   "Content-Type": "application/json", "Accept": "application/json"}
        url = f"{client._base_url()}/{urllib.parse.quote(collection, safe='')}/{urllib.parse.quote(doc_id, safe='')}"
        raw = None
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as response:
                raw = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        value = ({k: _firestore_value_to_py(v) for k, v in raw.get("fields", {}).items()}
                 if raw is not None else None)
        if value != expected:
            raise ValueError(f"Live document changed before conditional write: {collection}/{doc_id}")
        if raw is not None and not raw.get("updateTime"):
            raise ValueError("Firestore response lacks required updateTime precondition")
        name = f"projects/{client.project_id}/databases/{client.database_id}/documents/{collection}/{doc_id}"
        condition = {"updateTime": raw["updateTime"]} if raw is not None else {"exists": False}
        write = ({"delete": name} if desired is None else
                 {"update": {"name": name, "fields": {str(k): _py_to_firestore_value(v) for k, v in desired.items()}}})
        write["currentDocument"] = condition
        request = urllib.request.Request(client._base_url() + ":commit",
                    data=json.dumps({"writes": [write]}).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (409, 412):
                raise ValueError(f"Live document changed at conditional write: {collection}/{doc_id}") from exc
            raise
        return
    if hasattr(client, "transaction"):
        from google.cloud.firestore_v1.transaction import transactional
        ref = client.collection(collection).document(doc_id)
        @transactional
        def write_in_transaction(transaction):
            snapshot = ref.get(transaction=transaction)
            current = snapshot.to_dict() if snapshot.exists else None
            if current != expected:
                raise ValueError(f"Live document changed during transaction: {collection}/{doc_id}")
            if desired is None:
                transaction.delete(ref)
            else:
                transaction.set(ref, copy.deepcopy(desired))
        write_in_transaction(client.transaction(max_attempts=1))
        return
    raise RuntimeError("Firestore client does not support conditional recovery writes")


def _migrate_firestore(root: Path, store: FirestoreStateStore, *, backup_dir: Path | None,
                       plan_file: Path | None, seed_from_json: bool, apply: bool,
                       confirm_production: bool) -> dict:
    if apply and not store.collection_prefix and not confirm_production:
        raise PermissionError("Refusing unprefixed migration without --confirm-production")
    if seed_from_json and not store.collection_prefix:
        raise PermissionError("Refusing to seed production from local JSON")
    if backup_dir is None:
        raise ValueError("Firestore migration requires a verified --backup-dir")
    manifest, export = _verified_backup(backup_dir, store)
    assert export is not None
    before = export["collections"]
    if seed_from_json:
        if any(before.values()):
            raise ValueError("Staging JSON seed requires an empty managed target")
        local = migrate_release_state(root, target_backend="json")
        desired = {**copy.deepcopy(before), **local["documents"]}
    else:
        desired = _normalize_canonical_export(export)
    flat_before, flat_desired = _flatten_collections(before), _flatten_collections(desired)
    changed = [k for k in sorted(flat_desired) if flat_before.get(k) != flat_desired[k]]
    plan = {"schema_version": "1.0", "target": _target_identity(store),
            "backup_manifest_sha256": _sha256_file(Path(backup_dir) / "manifest.json"),
            "before": before, "desired": desired, "changed_documents": changed,
            "source": "staging_json_seed" if seed_from_json else "canonical_backup"}
    plan_sha = _payload_hash(plan)
    current = _export_firestore_state(store)["collections"]
    if current != before:
        # An already applied identical plan is a safe idempotent no-op.
        journal_path = Path(backup_dir) / "migration_journal.json"
        if not (current == desired and journal_path.exists()
                and json.loads(journal_path.read_text()).get("plan_sha256") == plan_sha):
            raise ValueError("Live managed state changed since the backup; create a new backup/plan")
        changed = []
    if apply:
        if plan_file is None or not Path(plan_file).is_file():
            raise ValueError("Apply requires a saved --plan-file from dry-run")
        approved = json.loads(Path(plan_file).read_text())
        if approved != {"plan_sha256": plan_sha, "plan": plan}:
            raise ValueError("Saved migration plan differs from the verified backup/desired state")
        journal_path = Path(backup_dir) / "migration_journal.json"
        journal = {"plan_sha256": plan_sha, "plan": plan, "completed": [], "status": "prepared"}
        _write_json(journal_path, journal)
        client = store._get_client()
        try:
            for path in changed:
                col, doc_id = path.split("/", 1)
                _conditional_document_write(client, store._col(col), doc_id,
                                            flat_before.get(path), flat_desired[path])
                journal["completed"].append(path)
                _write_json(journal_path, journal)
            if _export_firestore_state(store)["collections"] != desired:
                raise ValueError("Migration readback differs from desired managed state")
            journal["status"] = "complete"
            _write_json(journal_path, journal)
        except Exception:
            journal["status"] = "partial_or_unknown"
            _write_json(journal_path, journal)
            raise
    elif plan_file is not None:
        _write_json(Path(plan_file), {"plan_sha256": plan_sha, "plan": plan})
    return {"applied": apply, "target_backend": "firestore", "source": plan["source"],
            "plan_sha256": plan_sha, "target": plan["target"],
            "changed_documents": [store._col(p.split("/", 1)[0]) + "/" + p.split("/", 1)[1] for p in changed],
            "documents": desired}


def rollback_release_state(repo_root: Path, backup_dir: Path, *,
                           firestore_store: FirestoreStateStore | None = None,
                           confirm_production: bool = False, dry_run: bool = False) -> dict:
    """Preflight every target before restoring; refuse unknown post-cutover state."""
    root, src = Path(repo_root).resolve(), Path(backup_dir).resolve()
    if firestore_store is not None and not firestore_store.collection_prefix and not confirm_production and not dry_run:
        raise PermissionError("Refusing unprefixed restore without --confirm-production")
    manifest, export = _verified_backup(src, firestore_store)
    to_restore, to_delete = {}, []
    if firestore_store is not None:
        if not firestore_store.collection_prefix and not confirm_production and not dry_run:
            raise PermissionError("Refusing unprefixed restore without --confirm-production")
        before = export["collections"]
        current = _export_firestore_state(firestore_store)["collections"]
        if current != before:
            journal_path = src / "migration_journal.json"
            if not journal_path.exists():
                raise ValueError("Refusing unexpected state without a migration journal; preserve/reconcile new events")
            journal = json.loads(journal_path.read_text())
            plan = journal.get("plan", {})
            if (_payload_hash(plan) != journal.get("plan_sha256")
                    or plan.get("target") != _target_identity(firestore_store)
                    or plan.get("backup_manifest_sha256") != _sha256_file(src / "manifest.json")
                    or plan.get("before") != before):
                raise ValueError("Invalid migration journal/backup target")
            flat_before = _flatten_collections(before)
            flat_current = _flatten_collections(current)
            flat_after = _flatten_collections(plan["desired"])
            for key in set(flat_before) | set(flat_current) | set(flat_after):
                value = flat_current.get(key)
                if value != flat_before.get(key) and value != flat_after.get(key):
                    raise ValueError(f"Refusing unexpected post-cutover document: {key}; reconcile new events first")
                if key not in flat_before and key in flat_current:
                    to_delete.append(key)
                elif key in flat_before and value != flat_before[key]:
                    to_restore[key] = flat_before[key]
    # Check destination paths before the first database or disk mutation.
    for relative in manifest["files"]:
        _safe_child(root, relative)
    absent_managed = manifest.get("absent_managed_files", [])
    if not isinstance(absent_managed, list) or any(
        p != "assets/pulse_manifest.json" or p in manifest["files"] for p in absent_managed
    ):
        raise ValueError("Invalid absent managed publication files in backup")
    if not dry_run:
        if firestore_store is not None:
            client = firestore_store._get_client()
            for key, payload in to_restore.items():
                col, doc_id = key.split("/", 1)
                _conditional_document_write(client, firestore_store._col(col), doc_id,
                                            _flatten_collections(current).get(key), payload)
            for key in to_delete:
                col, doc_id = key.split("/", 1)
                _conditional_document_write(client, firestore_store._col(col), doc_id,
                                            _flatten_collections(current).get(key), None)
            if _export_firestore_state(firestore_store)["collections"] != export["collections"]:
                raise ValueError("Restore readback does not match backup; leave writers disabled")
        for relative in manifest["files"]:
            dst = _safe_child(root, relative)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_safe_child(src / "files", relative), dst)
        for relative in absent_managed:
            _safe_child(root, relative).unlink(missing_ok=True)
    return {"dry_run": dry_run, "restored_file_count": len(manifest["files"]),
            "restored_files": sorted(manifest["files"]),
            "restored_firestore_docs": len(to_restore), "deleted_firestore_docs": len(to_delete)}


def _normalize_daily_pulse_v3(raw_pulse: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_transient_keys(raw_pulse)
    enriched = _strip_transient_keys(
        _enrich_pulse_state_with_history(cleaned, existing=cleaned)
    )
    enriched.pop("image_data_uri", None)
    enriched.pop("vote_rate_limits", None)

    pub_map = enriched.get("published_pulse")
    if isinstance(pub_map, dict):
        pub_clean = copy.deepcopy(pub_map)
        pub_clean.pop("image_data_uri", None)
        if "mood_votes" in pub_clean:
            pub_clean["mood_votes"] = _normalize_mood_tally(pub_clean["mood_votes"])
        enriched["published_pulse"] = pub_clean
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
            if k not in enriched and k in pub_clean:
                enriched[k] = copy.deepcopy(pub_clean[k])

    analysis = enriched.get("analysis") if isinstance(enriched.get("analysis"), dict) else {}
    visual_dec = (
        enriched.get("visual_decision")
        if isinstance(enriched.get("visual_decision"), dict)
        else {}
    )
    date_str = str(
        enriched.get("date")
        or enriched.get("date_iso")
        or analysis.get("date_iso")
        or "2026-03-01"
    ).strip()[:10]
    word_str = str(
        enriched.get("word")
        or analysis.get("word_of_the_day")
        or "RESONANCE"
    ).strip().upper()
    gloss_str = str(
        enriched.get("gloss")
        or analysis.get("explanation_phrase")
        or "Daily synthesis"
    ).strip()
    thesis_str = str(
        enriched.get("commissioned_thesis")
        or enriched.get("summary")
        or gloss_str
    ).strip()
    medium_raw = str(
        enriched.get("visual_medium")
        or visual_dec.get("selected_mode")
        or "SVG_FALLBACK"
    ).strip().upper()
    if medium_raw not in VALID_VISUAL_MEDIUMS:
        medium_raw = "SVG_FALLBACK"

    raw_tally = (
        enriched.get("mood_votes")
        if isinstance(enriched.get("mood_votes"), dict)
        else enriched.get("visitor_mood_tally")
        if isinstance(enriched.get("visitor_mood_tally"), dict)
        else DEFAULT_MOOD_TALLY
    )
    canonical_tally = _normalize_mood_tally(raw_tally)

    published_at = str(
        enriched.get("published_at")
        or enriched.get("updated_at")
        or f"{date_str}T06:00:00Z"
    )
    updated_at = str(enriched.get("updated_at") or published_at)
    fallback_webp = str(
        enriched.get("fallback_webp_path") or f"assets/pulse/{date_str}.webp"
    )
    image_url = str(
        enriched.get("image_url")
        or enriched.get("artwork_url")
        or visual_dec.get("source_page_url")
        or ""
    )
    if image_url.startswith("data:"):
        image_url = ""

    enriched["schema_version"] = 3
    enriched["state_version"] = int(enriched.get("state_version", 1))
    enriched["date"] = date_str
    enriched["date_iso"] = date_str
    enriched["word"] = word_str
    enriched["gloss"] = gloss_str
    enriched["commissioned_thesis"] = thesis_str
    enriched["visual_medium"] = medium_raw
    enriched["image_url"] = image_url
    enriched["fallback_webp_path"] = fallback_webp
    enriched["mood_votes"] = dict(canonical_tally)
    enriched["visitor_mood_tally"] = dict(canonical_tally)
    enriched["published_at"] = published_at
    enriched["updated_at"] = updated_at

    return {
        k: v
        for k, v in enriched.items()
        if k in ALLOWED_DAILY_PULSE_FIELDS and not (isinstance(v, str) and v.startswith("data:image/"))
    }


def _decode_board_state_42(board_str: str) -> list[list[str]]:
    cells = [_c4_norm_cell(ch) for ch in board_str[:42].ljust(42, ".")]
    return [cells[r * 7 : (r + 1) * 7] for r in range(6)]


def _encode_board_state_42(board: list[list[Any]]) -> str:
    chars: list[str] = []
    for r in range(6):
        row = board[r] if r < len(board) and isinstance(board[r], list) else ["."] * 7
        for c in range(7):
            cell = row[c] if c < len(row) else "."
            chars.append(_c4_norm_cell(cell))
    return "".join(chars)


def _normalize_active_game_v3(raw_game: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_transient_keys(raw_game)
    if (
        isinstance(cleaned.get("board_state"), str)
        and len(cleaned["board_state"]) == 42
        and "board" not in cleaned
    ):
        cleaned["board"] = _decode_board_state_42(cleaned["board_state"])

    normalized = _strip_transient_keys(normalize_game_state(cleaned))
    board = normalized.get("board") or [["." for _ in range(7)] for _ in range(6)]
    board_state_42 = _encode_board_state_42(board)
    b1, b2 = encode_bitboard(board)

    mode_val = str(normalized.get("mode") or "VS_BOT").upper()
    if mode_val not in ("VS_BOT", "PVP_DUEL", "COOP_FORUM"):
        mode_val = "VS_BOT"

    status_val = str(normalized.get("status") or "ACTIVE").upper()
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

    turn_val = max(0, int(normalized.get("turn", 1)))
    turn_color = str(
        normalized.get("current_turn_color")
        or ("RED" if turn_val % 2 == 1 else "YELLOW")
    ).upper()
    if turn_color not in {"RED", "YELLOW", "GOLD", "BLUE", "TERRACOTTA", "CHARCOAL"}:
        turn_color = "RED" if turn_val % 2 == 1 else "YELLOW"

    updated_at = str(
        cleaned.get("updated_at")
        or normalized.get("updated_at")
        or normalized.get("last_move_timestamp")
        or "2026-03-01T00:00:00Z"
    )

    return {
        "schema_version": 3,
        "state_version": int(normalized.get("state_version", 1)),
        "round_id": max(1, int(normalized.get("round_id", 1))),
        "turn": turn_val,
        "active_game": str(normalized.get("active_game") or "connect4"),
        "game_key": str(normalized.get("game_key") or "connect4"),
        "mode": mode_val,
        "status": status_val,
        "current_turn_color": turn_color,
        "board": board,
        "board_state": board_state_42,
        "bitboard_p1": int(normalized.get("bitboard_p1", b1)),
        "bitboard_p2": int(normalized.get("bitboard_p2", b2)),
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
        "updated_at": updated_at,
    }


def _normalize_elo_ledger_v3(raw_elo: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_transient_keys(raw_elo)
    allowed = {
        "schema_version",
        "state_version",
        "bot_elo",
        "players",
        "ratings",
        "recent_rounds",
        "round_id",
        "active_game_key",
        "active_stakes",
        "current_round_log",
        "past_rounds",
        "updated_at",
    }
    out = {k: copy.deepcopy(v) for k, v in cleaned.items() if k in allowed}
    out["schema_version"] = 3
    out["state_version"] = int(out.get("state_version", 0))
    out["bot_elo"] = int(out.get("bot_elo", 1200))
    if not isinstance(out.get("players"), dict):
        out["players"] = (
            copy.deepcopy(out["ratings"])
            if isinstance(out.get("ratings"), dict)
            else {}
        )
    if not isinstance(out.get("recent_rounds"), list):
        out["recent_rounds"] = []
    return out


def _normalize_bot_memory_v3(raw_bot: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_transient_keys(raw_bot)
    allowed = {
        "schema_version",
        "state_version",
        "games_learned",
        "games_played",
        "states",
        "state_values",
        "values",
        "updated_at",
    }
    out = {k: copy.deepcopy(v) for k, v in cleaned.items() if k in allowed}
    out["schema_version"] = 3
    out["state_version"] = int(out.get("state_version", 0))
    out["games_learned"] = int(
        out.get("games_learned", out.get("games_played", 0))
    )
    if not isinstance(out.get("states"), dict):
        out["states"] = (
            copy.deepcopy(out["state_values"])
            if isinstance(out.get("state_values"), dict)
            else copy.deepcopy(out["values"])
            if isinstance(out.get("values"), dict)
            else {}
        )
    return out


def _normalize_orbit_roster_v3(raw_roster: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_transient_keys(raw_roster)
    allowed = {
        "schema_version",
        "state_version",
        "handles",
        "curated_o3_cursor",
        "updated_at",
    }
    out = {k: copy.deepcopy(v) for k, v in cleaned.items() if k in allowed}
    out["schema_version"] = 3
    out["state_version"] = int(out.get("state_version", 0))
    if not isinstance(out.get("handles"), dict):
        out["handles"] = {}
    if not isinstance(out.get("curated_o3_cursor"), (int, dict)):
        out["curated_o3_cursor"] = {}
    return out


def _normalize_game_rate_limits_v3(raw_limits: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_transient_keys(raw_limits)
    allowed = {
        "schema_version",
        "state_version",
        "users",
        "cooldowns",
        "updated_at",
    }
    out = {k: copy.deepcopy(v) for k, v in cleaned.items() if k in allowed}
    out["schema_version"] = 3
    out["state_version"] = int(out.get("state_version", 0))
    if not isinstance(out.get("users"), dict):
        out["users"] = {}
    return out


def _normalize_pulse_feed_entry_v3(
    raw_entry: dict[str, Any],
    fallback_pulse: dict[str, Any],
) -> dict[str, Any]:
    cleaned = _strip_transient_keys(raw_entry)
    date_str = str(
        cleaned.get("date")
        or cleaned.get("date_iso")
        or fallback_pulse.get("date")
        or "2026-03-01"
    ).strip()[:10]
    word_str = str(
        cleaned.get("word")
        or cleaned.get("word_of_the_day")
        or fallback_pulse.get("word")
        or "RESONANCE"
    ).strip().upper()
    gloss_str = str(
        cleaned.get("gloss")
        or cleaned.get("explanation_phrase")
        or fallback_pulse.get("gloss")
        or "Daily synthesis"
    ).strip()
    thesis_str = str(
        cleaned.get("commissioned_thesis")
        or cleaned.get("summary")
        or gloss_str
    ).strip()
    medium_raw = str(
        cleaned.get("visual_medium")
        or fallback_pulse.get("visual_medium")
        or "SVG_FALLBACK"
    ).strip().upper()
    if medium_raw not in VALID_VISUAL_MEDIUMS:
        medium_raw = "SVG_FALLBACK"

    raw_votes = (
        cleaned.get("mood_votes")
        if isinstance(cleaned.get("mood_votes"), dict)
        else fallback_pulse.get("mood_votes")
        if isinstance(fallback_pulse.get("mood_votes"), dict)
        else DEFAULT_MOOD_TALLY
    )
    canonical_votes = _normalize_mood_tally(raw_votes)
    published_at = str(
        cleaned.get("published_at")
        or cleaned.get("updated_at")
        or f"{date_str}T06:00:00Z"
    )
    updated_at = str(cleaned.get("updated_at") or published_at)
    image_url = str(cleaned.get("image_url") or cleaned.get("artwork_url") or "")
    if image_url.startswith("data:"):
        image_url = ""

    out = {k: copy.deepcopy(v) for k, v in cleaned.items() if k in ALLOWED_PULSE_HISTORY_FIELDS}
    out["schema_version"] = 3
    out["date"] = date_str
    out["date_iso"] = date_str
    out["word"] = word_str
    out["word_of_the_day"] = word_str
    out["gloss"] = gloss_str
    out["explanation_phrase"] = gloss_str
    out["commissioned_thesis"] = thesis_str
    out["summary"] = str(out.get("summary") or thesis_str)
    out["headline"] = str(out.get("headline") or f"{word_str} — {gloss_str}")
    out["visual_medium"] = medium_raw
    out["image_url"] = image_url
    out["image_data_uri"] = ""
    out["fallback_webp_path"] = str(
        out.get("fallback_webp_path") or f"assets/pulse/{date_str}.webp"
    )
    out["mood_votes"] = dict(canonical_votes)
    out["published_at"] = published_at
    out["updated_at"] = updated_at
    out.setdefault("cluster_keywords", [word_str.lower()])
    out.setdefault("orbit_contribution_pct", 50)
    out.setdefault("macro_contribution_pct", 50)
    out.setdefault("orbit_sources_matched", [])
    out.setdefault("runner_up_clusters", [])
    out.setdefault("critic_scores", {})
    return out


def _normalize_pulse_feed_v3(
    repo_root: Path,
    normalized_daily_pulse: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    candidate_paths = [
        repo_root / "state" / "pulse_feed.json",
        repo_root / "data" / "pulse_feed.json",
    ]
    existing_paths = [p for p in candidate_paths if p.exists()]
    target_paths = existing_paths if existing_paths else [repo_root / "data" / "pulse_feed.json"]

    raw_feed: Any = None
    for p in existing_paths:
        raw_feed = json.loads(p.read_text(encoding="utf-8"))
        break

    raw_entries: list[dict[str, Any]] = []
    existing_updated_at: str | None = None
    if isinstance(raw_feed, list):
        raw_entries = [e for e in raw_feed if isinstance(e, dict)]
    elif isinstance(raw_feed, dict):
        raw_entries = [e for e in (raw_feed.get("entries") or []) if isinstance(e, dict)]
        if isinstance(raw_feed.get("updated_at"), str) and raw_feed["updated_at"].strip():
            existing_updated_at = raw_feed["updated_at"].strip()

    if not raw_entries:
        raw_entries = [copy.deepcopy(normalized_daily_pulse)]

    by_date: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        norm_entry = _normalize_pulse_feed_entry_v3(entry, normalized_daily_pulse)
        by_date[norm_entry["date"]] = norm_entry

    sorted_entries = [
        by_date[d] for d in sorted(by_date.keys(), reverse=True)
    ][:365]

    feed_updated_at = (
        existing_updated_at
        or (sorted_entries[0]["updated_at"] if sorted_entries else None)
        or normalized_daily_pulse.get("updated_at")
        or "2026-03-01T06:00:00Z"
    )
    feed_doc = {
        "schema_version": 3,
        "version": "2.0",
        "updated_at": str(feed_updated_at),
        "entries": sorted_entries,
    }
    return feed_doc, target_paths


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        val = json.loads(path.read_text(encoding="utf-8"))
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def migrate_release_state(
    repo_root: Path,
    *,
    target_backend: str = "json",
    firestore_store: FirestoreStateStore | None = None,
    apply: bool = False,
    confirm_production: bool = False,
    backup_dir: Path | None = None,
    plan_file: Path | None = None,
    seed_from_json: bool = False,
) -> dict[str, Any]:
    """Normalize JSON, or migrate a verified canonical Firestore backup/plan."""
    root = Path(repo_root).resolve()
    backend = str(target_backend or "json").strip().lower()
    if backend not in ("json", "firestore"):
        raise ValueError(f"Unsupported target_backend {target_backend!r}: must be 'json' or 'firestore'")

    if backend == "firestore":
        store = firestore_store or FirestoreStateStore(repo_root=root)
        return _migrate_firestore(root, store, backup_dir=backup_dir, plan_file=plan_file,
                                 seed_from_json=seed_from_json, apply=apply,
                                 confirm_production=confirm_production)

    json_store = JsonStateStore(repo_root=root)

    raw_pulse = (
        _read_json_if_exists(json_store.pulse_state_path)
        or json_store.load_pulse_state()
    )
    raw_game = (
        _read_json_if_exists(json_store.game_state_path)
        or json_store.load_game_state()
    )
    raw_elo = (
        _read_json_if_exists(json_store.elo_ledger_path)
        or json_store.load_elo_ledger()
    )
    raw_bot = (
        _read_json_if_exists(json_store.bot_memory_path)
        or json_store.load_bot_memory()
    )
    raw_roster = (
        _read_json_if_exists(json_store.orbit_roster_path)
        or json_store.load_orbit_roster()
    )
    raw_limits = (
        _read_json_if_exists(json_store.game_rate_limits_path)
        or json_store.load_game_rate_limits()
    )

    norm_pulse = _normalize_daily_pulse_v3(raw_pulse)
    norm_game = _normalize_active_game_v3(raw_game)
    norm_elo = _normalize_elo_ledger_v3(raw_elo)
    norm_bot = _normalize_bot_memory_v3(raw_bot)
    norm_roster = _normalize_orbit_roster_v3(raw_roster)
    norm_limits = _normalize_game_rate_limits_v3(raw_limits)
    norm_feed, feed_paths = _normalize_pulse_feed_v3(root, norm_pulse)

    profile_docs: dict[str, dict[str, Any]] = {
        "daily_pulse": norm_pulse,
        "active_game": norm_game,
        "elo_ledger": norm_elo,
        "bot_memory": norm_bot,
        "orbit_roster": norm_roster,
        "game_rate_limits": norm_limits,
    }
    pulse_history_docs: dict[str, dict[str, Any]] = {}
    pulse_votes_docs: dict[str, dict[str, Any]] = {}
    for entry in norm_feed["entries"]:
        d_str = str(entry["date"])[:10]
        pulse_history_docs[d_str] = copy.deepcopy(entry)
        pulse_votes_docs[d_str] = dict(entry["mood_votes"])

    changed_documents: list[str] = []

    if backend == "json":
        json_targets: list[tuple[str, Path, dict[str, Any]]] = [
            ("profile_state/daily_pulse", json_store.pulse_state_path, norm_pulse),
            ("profile_state/active_game", json_store.game_state_path, norm_game),
            ("profile_state/elo_ledger", json_store.elo_ledger_path, norm_elo),
            ("profile_state/bot_memory", json_store.bot_memory_path, norm_bot),
            ("profile_state/orbit_roster", json_store.orbit_roster_path, norm_roster),
            ("profile_state/game_rate_limits", json_store.game_rate_limits_path, norm_limits),
        ]
        for fpath in feed_paths:
            json_targets.append((f"pulse_feed/{fpath.relative_to(root).as_posix()}", fpath, norm_feed))

        for doc_label, path, desired_payload in json_targets:
            desired_text = json.dumps(desired_payload, indent=2) + "\n"
            current_text = path.read_text(encoding="utf-8") if path.exists() else None
            if current_text != desired_text:
                changed_documents.append(doc_label)
                if apply:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(desired_text, encoding="utf-8")

    return {
        "applied": bool(apply),
        "target_backend": backend,
        "changed_documents": changed_documents,
        "documents": {
            "profile_state": profile_docs,
            "pulse_feed": {"v1": norm_feed},
            "pulse_history": pulse_history_docs,
            "pulse_votes": pulse_votes_docs,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate_release_state",
        description="Idempotent v3 release state backup, migration, and rollback utility.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    backup_p = subparsers.add_parser("backup", help="Create SHA-256 verified release backup.")
    backup_p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory.",
    )
    backup_p.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(".release-backups/pre-v3-cutover"),
        help="Destination directory for backup files and manifest.json.",
    )
    backup_p.add_argument(
        "--include-firestore",
        action="store_true",
        help="Export Firestore documents into firestore_export.json.",
    )
    backup_p.add_argument(
        "--collection-prefix",
        type=str,
        default=None,
        help="Optional Firestore collection prefix override.",
    )

    migrate_p = subparsers.add_parser("migrate", help="Validate and migrate release state to v3.")
    migrate_p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory.",
    )
    migrate_p.add_argument(
        "--target-backend",
        choices=("json", "firestore"),
        default="json",
        help="Target persistence backend.",
    )
    migrate_p.add_argument(
        "--collection-prefix",
        type=str,
        default=None,
        help="Optional Firestore collection prefix override.",
    )
    migrate_p.add_argument(
        "--include-firestore",
        action="store_true",
        default=False,
        help="Select Firestore target; writes still require explicit --apply and a verified backup/plan.",
    )
    migrate_p.add_argument("--backup-dir", type=Path, help="Verified canonical Firestore backup.")
    migrate_p.add_argument("--plan-file", type=Path, help="Dry-run writes this plan; apply requires an exact match.")
    migrate_p.add_argument("--seed-from-json", action="store_true", help="Explicitly initialize an empty prefixed staging target only.")
    mode_group = migrate_p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate and report changes without mutating state (default).",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write normalized v3 documents to the target backend.",
    )
    migrate_p.add_argument(
        "--confirm-production",
        action="store_true",
        default=False,
        help="Required safety confirmation when mutating unprefixed production Firestore collections.",
    )

    rollback_p = subparsers.add_parser("rollback", help="Verify manifest and restore backup.")
    rollback_p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory.",
    )
    rollback_p.add_argument(
        "--backup-dir",
        type=Path,
        required=True,
        help="Backup directory containing manifest.json.",
    )
    rollback_p.add_argument("--dry-run", action="store_true", help="Verify all recovery preconditions without writes.")
    rollback_p.add_argument(
        "--restore-firestore",
        action="store_true",
        help="Restore Firestore documents from firestore_export.json.",
    )
    rollback_p.add_argument(
        "--collection-prefix",
        type=str,
        default=None,
        help="Optional Firestore collection prefix override.",
    )
    rollback_p.add_argument(
        "--confirm-production",
        action="store_true",
        default=False,
        help="Required safety confirmation when restoring unprefixed production Firestore collections.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.subcommand == "backup":
            fs_store = (
                FirestoreStateStore(
                    collection_prefix=args.collection_prefix,
                    repo_root=args.repo_root,
                )
                if args.include_firestore
                else None
            )
            manifest = create_release_backup(
                repo_root=args.repo_root,
                backup_dir=args.backup_dir,
                firestore_store=fs_store,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "subcommand": "backup",
                        "file_count": len(manifest["files"]),
                        "backup_dir": str(Path(args.backup_dir).resolve()),
                    }
                )
            )
            return 0

        if args.subcommand == "migrate":
            target_backend = (
                "firestore" if getattr(args, "include_firestore", False) else args.target_backend
            )
            should_apply = bool(args.apply)
            fs_store = None
            if target_backend == "firestore":
                resolved_prefix = (
                    args.collection_prefix
                    if args.collection_prefix is not None
                    else os.environ.get("FIRESTORE_COLLECTION_PREFIX", "").strip()
                )
                if should_apply and resolved_prefix == "" and not args.confirm_production:
                    raise PermissionError(
                        "Refusing to migrate unprefixed production Firestore collections without --confirm-production"
                    )
                fs_store = FirestoreStateStore(
                    collection_prefix=resolved_prefix,
                    repo_root=args.repo_root,
                )

            report = migrate_release_state(
                repo_root=args.repo_root,
                target_backend=target_backend,
                firestore_store=fs_store,
                apply=should_apply,
                confirm_production=bool(args.confirm_production),
                backup_dir=args.backup_dir,
                plan_file=args.plan_file,
                seed_from_json=args.seed_from_json,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "subcommand": "migrate",
                        "applied": report["applied"],
                        "target_backend": report["target_backend"],
                        "changed_documents": report["changed_documents"],
                        "documents": report["documents"],
                        **{key: report[key] for key in ("source", "target", "plan_sha256") if key in report},
                    }
                )
            )
            return 0

        if args.subcommand == "rollback":
            fs_store = (
                FirestoreStateStore(
                    collection_prefix=args.collection_prefix,
                    repo_root=args.repo_root,
                )
                if args.restore_firestore
                else None
            )
            result = rollback_release_state(
                repo_root=args.repo_root,
                backup_dir=args.backup_dir,
                firestore_store=fs_store,
                confirm_production=bool(args.confirm_production),
                dry_run=args.dry_run,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "subcommand": "rollback",
                        **result,
                    }
                )
            )
            return 0

    except PermissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
