"""Explicitly rehearse migration and recovery in an empty per-run staging namespace.

Importing this module never contacts a service. Invoking the CLI performs guarded
Firestore writes and restores them; no production confirmation flag is supported.
Private backups and step reports must remain outside the public checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state_store import FirestoreStateStore

PROJECT = "git-readme-features"
DATABASE = "(default)"
MANAGED_COLLECTIONS = (
    "profile_state", "pulse_feed", "pulse_history", "pulse_votes", "vote_rate_limits", "processed_events",
)
STAGING_PREFIX = re.compile(r"staging_[1-9][0-9]*_[1-9][0-9]*_\Z")


def _documents(store) -> dict:
    client = store._get_client()
    if client is None:
        raise RuntimeError("Canonical Firestore client unavailable; JSON fallback is forbidden")
    result = {}
    for collection in MANAGED_COLLECTIONS:
        for snapshot in client.collection(store._col(collection)).stream():
            result[f"{collection}/{snapshot.id}"] = snapshot.to_dict()
    return result


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rehearse(
    repo_root: Path, backup_dir: Path, collection_prefix: str, *,
    runner: Callable = subprocess.run, store_factory: Callable = FirestoreStateStore,
) -> dict:
    """Run the public migration CLI and verify its real target, without fallback reads."""
    root = Path(repo_root).resolve()
    backup = Path(backup_dir).resolve()
    if not STAGING_PREFIX.fullmatch(collection_prefix):
        raise ValueError("Expected unique staging_<numeric run>_<numeric attempt>_ namespace")
    if backup.is_relative_to(root):
        raise ValueError("Private recovery backup must be outside the repository")
    if backup.exists() and any(backup.iterdir()):
        raise ValueError("Recovery backup directory must be new or empty")
    migration_script = root / "scripts" / "migrate_release_state.py"
    if not migration_script.is_file():
        raise ValueError("Explicit repo-root does not contain the migration CLI")
    local_pulse = json.loads((root / "data" / "zeitgeist_state.json").read_text())
    local_game = json.loads((root / "data" / "game_state.json").read_text())
    expected_word = str(local_pulse.get("word") or local_pulse.get("analysis", {}).get("word_of_the_day") or "").upper()
    if not expected_word or not isinstance(local_game.get("turn"), int):
        raise ValueError("Staging seed requires explicit pulse word and integer game turn")
    store = store_factory(project_id=PROJECT, database_id=DATABASE, collection_prefix=collection_prefix)
    if (store._project_id, store._database_id, store.collection_prefix) != (PROJECT, DATABASE, collection_prefix):
        raise ValueError("Resolved Firestore target differs from the fixed rehearsal project/database/namespace")
    if _documents(store):
        raise ValueError("Refusing occupied staging namespace; choose a new run/attempt")
    env = dict(os.environ, GCP_PROJECT_ID=PROJECT, GOOGLE_CLOUD_PROJECT=PROJECT,
               FIRESTORE_PROJECT_ID=PROJECT, FIRESTORE_DATABASE_ID=DATABASE,
               FIRESTORE_COLLECTION_PREFIX=collection_prefix, PYTHONDONTWRITEBYTECODE="1")
    plan = backup / "staging_seed_plan.json"
    reports = {}

    def cli(phase, arguments):
        result = runner([sys.executable, str(migration_script), *arguments],
                        cwd=str(root), env=env, text=True, capture_output=True, check=False)
        if result.returncode:
            # Preserve sensitive diagnostic output privately; never print tokens/state.
            backup.mkdir(parents=True, exist_ok=True)
            (backup / f"{phase}.stderr.txt").write_text(result.stderr or "")
            raise RuntimeError(f"Recovery rehearsal {phase} failed; inspect the private backup logs")
        try:
            report = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise RuntimeError(f"Recovery rehearsal {phase} returned invalid JSON") from None
        if report.get("status") != "ok":
            raise RuntimeError(f"Recovery rehearsal {phase} did not report success")
        (backup / f"{phase}.report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports[phase] = report
        return report

    base = ["--repo-root", str(root), "--collection-prefix", collection_prefix,
            "--backup-dir", str(backup)]
    cli("backup", ["backup", *base, "--include-firestore"])
    seed = ["migrate", *base, "--target-backend", "firestore", "--plan-file", str(plan), "--seed-from-json"]
    cli("plan", [*seed, "--dry-run"])
    if _documents(store):
        raise RuntimeError("Dry-run unexpectedly modified the empty namespace")
    applied = False
    try:
        # Mark before execution: a lost response may mean that writes committed.
        applied = True
        cli("apply", [*seed, "--apply"])
        actual = _documents(store)
        pulse = actual.get("profile_state/daily_pulse", {})
        game = actual.get("profile_state/active_game", {})
        if pulse.get("schema_version") != 3 or game.get("schema_version") != 3:
            raise RuntimeError("Firestore readback did not contain v3 pulse/game documents")
        if pulse.get("word") != expected_word or game.get("turn") != local_game["turn"]:
            raise RuntimeError("Firestore content readback differs from the explicit local staging seed")
        planned = reports["plan"]["documents"]
        for collection, documents in planned.items():
            for doc_id, payload in documents.items():
                if actual.get(f"{collection}/{doc_id}") != payload:
                    raise RuntimeError("Firestore readback differs from the saved staging migration plan")
        seeded_count = len(actual)
    finally:
        if applied:
            # Public recovery preflight refuses any unexpected visitor/state change.
            cli("rollback_plan", ["rollback", *base, "--restore-firestore", "--dry-run"])
            cli("rollback", ["rollback", *base, "--restore-firestore"])
    if _documents(store):
        raise RuntimeError("Recovery readback found staging namespace residue; keep writers disabled")
    manifest = json.loads((backup / "manifest.json").read_text())
    if any(_hash_file(root / relative) != expected for relative, expected in manifest["files"].items()):
        raise RuntimeError("Recovery did not restore the backed-up local files")
    return {"status": "ok", "project": PROJECT, "database": DATABASE,
            "collection_prefix": collection_prefix, "seeded_document_count": seeded_count,
            "remaining_document_count": 0, "plan_sha256": reports["plan"]["plan_sha256"],
            "backup_dir": str(backup), "phases": list(reports)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--collection-prefix", required=True)
    args = parser.parse_args(argv)
    try:
        report = rehearse(args.repo_root, args.backup_dir, args.collection_prefix)
    except (ValueError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "message": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
