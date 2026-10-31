"""Prepare first; conditionally commit one canonical publication, then mirror it.

Firestore is the atomic authority. Git/filesystem mirroring is a separate step;
no compensating remote writes are made when that later step fails.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from src.state_store import (
    FirestoreStateStore,
    InMemoryStateStore,
    JsonFileStateStore,
    PublicationConflict,
    _commit_documents,
    _enrich_pulse_state_with_history,
    _normalize_mood_tally,
    _strip_transient_keys,
)


class PublicationMirrorError(RuntimeError):
    """Canonical commit succeeded; local/Git reconciliation is required."""


def editorial_revision(state: dict) -> str:
    """Votes do not invalidate an editorial revision; another publication does."""
    return hashlib.sha256(
        json.dumps(
            {
                "publication_id": state.get("publication_id"),
                "date": state.get("date") or state.get("date_iso"),
                "analysis": state.get("analysis"),
                "word": state.get("word"),
                "medium": state.get("visual_medium"),
                "state_version": state.get("state_version", 0),
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _bundle(
    payload: dict, entry: dict, existing: dict, feed: dict, tally: dict
) -> tuple[dict, dict, dict]:
    state = _enrich_pulse_state_with_history(_strip_transient_keys(payload), existing)
    item = _strip_transient_keys(entry)
    if payload.get("publication_id"):
        item["publication_id"] = payload["publication_id"]
    item["schema_version"] = "2.0"
    item["mood_votes"] = copy.deepcopy(tally)
    state.update(
        schema_version=3,
        visitor_mood_tally=copy.deepcopy(tally),
        mood_votes=copy.deepcopy(tally),
        published_pulse=copy.deepcopy(item),
        state_version=int(existing.get("state_version", 0)) + 1,
    )
    for key in ("date", "word", "gloss", "visual_medium", "published_at", "updated_at"):
        if key in item:
            state[key] = copy.deepcopy(item[key])
    entries = [
        copy.deepcopy(e)
        for e in feed.get("entries", [])
        if isinstance(e, dict) and str(e.get("date", ""))[:10] != item["date"][:10]
    ]
    entries.append(item)
    entries.sort(key=lambda e: str(e.get("date", "")), reverse=True)
    envelope = {
        "version": "2.0",
        "schema_version": "2.0",
        "entries": entries[:365],
        "updated_at": item.get("updated_at") or item.get("published_at"),
    }
    return state, item, envelope


def publish_bundle(
    store: Any,
    payload: dict,
    entry: dict,
    *,
    expected: dict,
    local_feed: dict,
    prepare: Callable[[dict, dict, dict], None],
) -> dict:
    """Validate a merged snapshot, then CAS profile/history/feed/votes together.

    `prepare` may be called again on vote conflicts. It must only write staging
    files, and must raise before commit for any README/asset validation failure.
    An ambiguous transport failure is never automatically rerun.
    """
    if isinstance(store, FirestoreStateStore):
        client = store._get_client()
        if client is None:
            raise RuntimeError(
                "Canonical Firestore unavailable; publication not committed"
            )
        date = entry["date"][:10]
        paths = [
            (store._col("profile_state"), "daily_pulse"),
            (store._col("pulse_history"), date),
            (store._col("pulse_feed"), "v1"),
            (store._col("pulse_votes"), date),
        ]
        for _attempt in range(5):
            snaps = [client.collection(col).document(doc).get() for col, doc in paths]
            existing = snaps[0].to_dict() if snaps[0].exists else {}
            if editorial_revision(existing) != editorial_revision(expected):
                raise PublicationConflict(
                    "A newer editorial publication exists; regenerate from it"
                )
            # A missing canonical feed may only be bootstrapped from canonical
            # history; the workspace JSON may predate accepted remote entries.
            feed = (snaps[2].to_dict() if snaps[2].exists else
                    {"entries": store.load_pulse_feed_history(days=365)})
            tally = _normalize_mood_tally(
                snaps[3].to_dict()
                if snaps[3].exists
                else existing.get("visitor_mood_tally")
            )
            state, item, envelope = _bundle(payload, entry, existing, feed, tally)
            prepare(state, item, envelope)
            writes = [
                (col, doc, data, snapshot)
                for (col, doc), data, snapshot in zip(
                    paths, (state, item, envelope, tally), snaps
                )
            ]
            try:
                _commit_documents(client, writes)
            except PublicationConflict:
                continue
            store.set_needs_git_reconcile(True)
            return state
        raise PublicationConflict(
            "Concurrent votes prevented a stable publication; retry later"
        )
    existing = store.load_pulse_state()
    if editorial_revision(existing) != editorial_revision(expected):
        raise PublicationConflict(
            "A newer local publication exists; regenerate from it"
        )
    state, item, envelope = _bundle(
        payload,
        entry,
        existing,
        local_feed,
        _normalize_mood_tally(existing.get("visitor_mood_tally")),
    )
    prepare(state, item, envelope)
    if isinstance(store, InMemoryStateStore):
        # Assign only after all validation; no callbacks/IO between assignments.
        store._pulse_state, store._pulse_feed_entries = copy.deepcopy(
            state
        ), copy.deepcopy(envelope["entries"])
    elif not isinstance(store, JsonFileStateStore):
        raise RuntimeError(
            "State store does not implement conditional publication bundles"
        )
    # JSON's authority is the validated workspace installed by the caller.
    return state


class PublicationWorkspace:
    """Isolate rendering and validation from the currently published files."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._temporary = tempfile.TemporaryDirectory(prefix="egm-publication-")
        self.path = Path(self._temporary.name)
        for name in ("assets", "data", "state", "config", "variants"):
            if (self.root / name).is_dir():
                shutil.copytree(self.root / name, self.path / name)
        for name in ("README.md", "PROJECTS.md"):
            if (self.root / name).is_file():
                shutil.copy2(self.root / name, self.path / name)

    def close(self):
        self._temporary.cleanup()

    def stage_canonical_game(self, store: Any) -> None:
        """Snapshot game/ELO for this queued GitHub publication, without writes.

        Bypass adapter mirror/version preferences: Firestore is authoritative even
        when a checkout mirror has the same or a larger state_version. The shared
        GitHub writer queue protects the snapshot through rendering/publication.
        """
        if not isinstance(store, FirestoreStateStore):
            return  # Explicit JSON/in-memory demonstrations keep their local behavior.
        from src.game_engine import normalize_game_state

        client = store._get_client()
        if client is None:
            raise RuntimeError("Canonical game unavailable; retain last publication")
        collection = client.collection(store._col("profile_state"))
        game = collection.document("active_game").get()
        if not game.exists:
            raise RuntimeError(
                "Canonical game missing; refusing checkout fixture publication"
            )
        raw_game = _strip_transient_keys(game.to_dict() or {})
        projected_game = normalize_game_state(raw_game)
        board = projected_game.get("board")
        source = (
            raw_game.get("board_state")
            if isinstance(raw_game.get("board_state"), dict) and "turn" not in raw_game
            else raw_game
        )
        if (
            type(source.get("turn")) is not int
            or source["turn"] < 0
            or not isinstance(source.get("board"), list)
            or not isinstance(board, list)
            or len(board) != 6
            or any(not isinstance(row, list) or len(row) != 7 for row in board)
        ):
            raise RuntimeError(
                "Canonical game invalid; refusing synthesized board publication"
            )
        ledger = collection.document("elo_ledger").get()
        # Canonical absence means no recorded ratings; never promote Git fixtures.
        projected_elo = (
            _strip_transient_keys(ledger.to_dict() or {})
            if ledger.exists
            else {"state_version": 0, "players": {}, "recent_rounds": []}
        )
        # Preserve unavailable canonical ratings through the frozen JSON loader;
        # its legacy baseline must not manufacture a rating for missing records.
        projected_elo.setdefault("bot_elo", None)
        data = self.path / "data"
        data.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            ("game_state.json", projected_game),
            ("hall_of_records.json", projected_elo),
        ):
            (data / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        legacy_mirror = data / "connect4_state.json"
        if legacy_mirror.exists():
            legacy_mirror.write_text(
                json.dumps(projected_game, indent=2) + "\n", encoding="utf-8"
            )

    def prepare(self, state: dict, entry: dict, feed: dict):
        from scripts.build_readme import build_all_readmes, check_all_readmes
        from .publication import verify_publication_bundle
        from .visual_router import update_svg_mood_vote_bar

        data = self.path / "data"
        data.mkdir(parents=True, exist_ok=True)
        for filename, payload in (
            ("zeitgeist_state.json", state),
            ("pulse_feed.json", feed),
        ):
            (data / filename).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        update_svg_mood_vote_bar(
            self.path / "assets/daily_pulse.svg", state["visitor_mood_tally"]
        )
        local_store = JsonFileStateStore(self.path)
        build_all_readmes(self.path, state_store=local_store)
        verify_publication_bundle(self.path, require_current_pulse=True)
        if not check_all_readmes(self.path, state_store=local_store):
            raise ValueError("Staged README validation failed")

    def publication_files(self, state: dict):
        paths = [
            Path("README.md"),
            Path("data/zeitgeist_state.json"),
            Path("data/pulse_feed.json"),
            Path("assets/pulse_manifest.json"),
            Path("assets/daily_pulse.svg"),
            Path("assets/daily_pulse_light.webp"),
            Path("assets/daily_pulse_dark.webp"),
            Path(state["published_pulse"]["fallback_webp_path"]),
        ]
        paths.extend(
            p.relative_to(self.path) for p in (self.path / "variants").glob("*.md")
        )
        # Game controls, image and local snapshots describe the same projection.
        # Only these local mirrors move; the publication does not write game/ELO.
        for relative in (
            "assets/game_board.svg",
            "data/game_state.json",
            "data/hall_of_records.json",
            "data/connect4_state.json",
        ):
            if (self.path / relative).is_file():
                paths.append(Path(relative))
        return paths

    def preserve(self, state: dict) -> str:
        """Keep only public bundle files, never copied credentials/config/cache."""
        target = self.root / ".publication-recovery" / state["publication_id"]
        for relative in self.publication_files(state):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path / relative, destination)
        return str(target)

    def install(self, state: dict):
        """Mirror a committed bundle; failures require reconciliation, never undo."""
        for relative in self.publication_files(state):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                handle.write((self.path / relative).read_bytes())
                temporary = Path(handle.name)
            temporary.replace(target)
