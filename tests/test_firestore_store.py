"""Tests for FirestoreStateStore, JsonFileStateStore, PublishedPulse.to_feed_entry v2 projection, and firestore.rules alignment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.state_store import (
    FirestoreStateStore,
    InMemoryStateStore,
    JsonFileStateStore,
    VALID_MOOD_KEYS,
    compute_voter_hmac,
)
from src.zeitgeist.visual_router import PublishedPulse


SECTION_7_1_FEED_KEYS = {
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
}


def _make_sample_published_pulse() -> PublishedPulse:
    return PublishedPulse(
        date="2026-10-28",
        word="Cantilever",
        gloss="Structural poise across asymmetric loads.",
        thesis="When personal craft anchors high-velocity macro systems, tension becomes structural.",
        orbit_resonance_score=0.78,
        velocity_score=0.64,
        visual_medium="PHOTOGRAPHY",
        webp_asset_path="assets/pulse/2026-10-28.webp",
        svg_asset_path="assets/daily_pulse.svg",
        attribution={
            "photographer_name": "Elena Rostova",
            "photographer_url": "https://unsplash.com/@erostova",
            "unsplash_url": "https://unsplash.com/photos/arch-1",
        },
        palette_hex=["#1A1D24", "#2E3540", "#4A5568", "#A0AEC0", "#EDF2F7"],
        mood_votes={"spot_on": 20, "too_sunny": 3, "too_heavy": 4},
        composite_critic_score=0.812,
        semantic_alignment_score=0.84,
        selected_candidate_index=0,
        cluster_keywords=["cantilever", "architecture", "verification"],
        cluster_silhouette=0.42,
        sources=[{"source_id": "o1_ness_labs", "title": "Mindful Systems"}],
        image_data_uri="data:image/webp;base64,UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/vuUAAA=",
        published_at="2026-10-28T06:00:00Z",
        curatorial_archetype="POETIC_CINEMA",
        text_quadrant="upper_left",
    )


def test_published_pulse_to_feed_entry_v2_schema_and_no_base64() -> None:
    """PublishedPulse.to_feed_entry() emits all Section 7.1 v2 keys and strips base64 data URIs."""
    pulse = _make_sample_published_pulse()
    entry = pulse.to_feed_entry()

    assert SECTION_7_1_FEED_KEYS.issubset(set(entry.keys()))
    assert entry["schema_version"] == "2.0"
    assert entry["date"] == "2026-10-28"
    assert entry["word"] == "Cantilever"
    assert entry["gloss"] == "Structural poise across asymmetric loads."
    assert entry["commissioned_thesis"].startswith("When personal craft anchors")
    assert entry["visual_medium"] == "PHOTOGRAPHY"
    assert entry["fallback_webp_path"] == "assets/pulse/2026-10-28.webp"
    assert entry["unsplash_photographer"] == "Elena Rostova"
    assert entry["unsplash_profile_url"] == "https://unsplash.com/@erostova"
    assert entry["flux_model"] is None
    assert entry["cluster_keywords"] == ["cantilever", "architecture", "verification"]
    assert entry["orbit_contribution_pct"] == 62.0
    assert entry["macro_contribution_pct"] == 38.0
    assert entry["orbit_sources_matched"] == ["o1_ness_labs"]
    assert entry["runner_up_clusters"] == []
    assert entry["mood_votes"] == {"spot_on": 20, "too_sunny": 3, "too_heavy": 4}
    assert entry["critic_scores"] == {"semantic_alignment": 0.84, "composite": 0.812}
    assert entry["image_data_uri"] == ""
    assert not str(entry.get("image_data_uri", "")).startswith("data:")


def test_json_file_state_store_save_pulse_feed_entry_preserves_v2_envelope_and_strips_data_uri(
    tmp_path: Path,
) -> None:
    """JsonFileStateStore.save_pulse_feed_entry preserves version='2.0', schema_version='2.0', and strips data: URIs."""
    store = JsonFileStateStore(repo_root=tmp_path)
    pulse = _make_sample_published_pulse()
    raw_entry = pulse.to_feed_entry()
    raw_entry["image_data_uri"] = "data:image/webp;base64,AAAA"

    store.save_pulse_feed_entry("2026-10-28", raw_entry)

    feed_path = tmp_path / "data" / "pulse_feed.json"
    assert feed_path.exists()
    payload = json.loads(feed_path.read_text(encoding="utf-8"))
    assert payload["version"] == "2.0"
    assert payload["schema_version"] == "2.0"
    assert "updated_at" in payload
    assert len(payload["entries"]) == 1
    saved_entry = payload["entries"][0]
    assert saved_entry["date"] == "2026-10-28"
    assert saved_entry["word"] == "Cantilever"
    assert saved_entry["schema_version"] == "2.0"
    assert saved_entry.get("image_data_uri", "") == ""


class _RecordingDocRef:
    def __init__(self, store: dict[str, dict[str, Any]], path: str) -> None:
        self._store = store
        self._path = path

    def get(self) -> Any:
        exists = self._path in self._store
        data = copy.deepcopy(self._store.get(self._path, {}))
        return type("Snap", (), {"id": self._path.rsplit("/", 1)[-1], "exists": exists,
            "update_time": json.dumps(data, sort_keys=True, default=str) if exists else None,
            "to_dict": lambda self=None: copy.deepcopy(data)})()

    def set(self, payload: dict[str, Any]) -> None:
        self._store[self._path] = copy.deepcopy(payload)

    def delete(self) -> None:
        self._store.pop(self._path, None)


class _RecordingColRef:
    def __init__(self, store: dict[str, dict[str, Any]], col: str) -> None:
        self._store = store
        self._col = col

    def document(self, doc_id: str) -> _RecordingDocRef:
        return _RecordingDocRef(self._store, f"{self._col}/{doc_id}")

    def stream(self) -> list[Any]:
        prefix = f"{self._col}/"
        snaps = []
        for path, data in list(self._store.items()):
            if path.startswith(prefix):
                doc_id = path[len(prefix) :]
                d_copy = copy.deepcopy(data)
                snaps.append(
                    type("Snap", (), {"id": doc_id, "exists": True,
                        "update_time": json.dumps(d_copy, sort_keys=True, default=str),
                        "to_dict": lambda self=None, d=d_copy: copy.deepcopy(d)})()
                )
        return snaps


class _RecordingFirestoreClient:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def commit_documents(self, writes) -> None:
        from src.state_store import PublicationConflict
        for col, doc, _data, snapshot in writes:
            path = f"{col}/{doc}"
            current = self.docs.get(path)
            if (current is not None) != snapshot.exists or (snapshot.exists and current != snapshot.to_dict()):
                raise PublicationConflict(f"Concurrent fake write: {path}")
        # Validate every condition before applying any write.
        for col, doc, data, _snapshot in writes:
            path = f"{col}/{doc}"
            self.docs[path] = dict(self.docs.get(path, {}), **copy.deepcopy(data))

    def compare_and_set_document(self, collection: str, doc_id: str, expected: dict | None, desired: dict | None) -> None:
        path = f"{collection}/{doc_id}"
        if self.docs.get(path) != expected:
            raise ValueError(f"Live document changed: {path}")
        if desired is None:
            self.docs.pop(path, None)
        else:
            self.docs[path] = copy.deepcopy(desired)

    def collection(self, col_id: str) -> _RecordingColRef:
        return _RecordingColRef(self.docs, col_id)


def test_firestore_state_store_save_pulse_state_and_feed_entry_strip_data_uri(
    tmp_path: Path,
) -> None:
    """FirestoreStateStore.save_pulse_state and save_pulse_feed_entry write v2 fields and strip data: URIs."""
    rec_client = _RecordingFirestoreClient()
    store = FirestoreStateStore(
        client=rec_client,
        project_id="test-proj",
        fallback_store=JsonFileStateStore(repo_root=tmp_path),
    )

    pulse = _make_sample_published_pulse()
    entry_with_bloat = pulse.to_feed_entry()
    entry_with_bloat["image_data_uri"] = "data:image/webp;base64,BBBB"

    store.save_pulse_state(
        {
            "date": "2026-10-28",
            "word": "Cantilever",
            "gloss": "Structural poise across asymmetric loads.",
            "commissioned_thesis": "When personal craft anchors high-velocity macro systems.",
            "visual_medium": "PHOTOGRAPHY",
            "mood_votes": {"spot_on": 20, "too_sunny": 3, "too_heavy": 4},
            "published_pulse": entry_with_bloat,
        }
    )
    store.save_pulse_feed_entry("2026-10-28", entry_with_bloat)

    assert "profile_state/daily_pulse" in rec_client.docs
    daily_doc = rec_client.docs["profile_state/daily_pulse"]
    assert daily_doc["word"] == "Cantilever"
    assert daily_doc["published_pulse"].get("image_data_uri", "") == ""

    assert "pulse_history/2026-10-28" in rec_client.docs
    hist_doc = rec_client.docs["pulse_history/2026-10-28"]
    assert hist_doc["word"] == "Cantilever"
    assert hist_doc["schema_version"] == "2.0"
    assert hist_doc.get("image_data_uri", "") == ""
    assert SECTION_7_1_FEED_KEYS.issubset(set(hist_doc.keys()))


def test_valid_mood_keys_and_legacy_alias_normalization(tmp_path: Path) -> None:
    """G-07: VALID_MOOD_KEYS is restricted to the 3 canonical keys and legacy mood aliases normalize cleanly."""
    assert VALID_MOOD_KEYS == ("spot_on", "too_sunny", "too_heavy")

    mem = InMemoryStateStore(
        initial_pulse_state={
            "visitor_mood_tally": {
                "spot_on": 10,
                "too_sunny": 2,
                "too_heavy": 1,
                "resonant": 5,
                "reflective": 3,
                "wistful": 2,
                "electric": 1,
                "skeptical": 4,
            }
        }
    )
    loaded = mem.load_pulse_state()
    assert set(loaded["visitor_mood_tally"].keys()) == {"spot_on", "too_sunny", "too_heavy"}
    assert loaded["visitor_mood_tally"]["spot_on"] == 15
    assert loaded["visitor_mood_tally"]["too_sunny"] == 8
    assert loaded["visitor_mood_tally"]["too_heavy"] == 5

    # Legacy vote input aliases map to canonical keys
    h1 = compute_voter_hmac("198.51.100.1", "Agent/1.0", "2026-10-28")
    h2 = compute_voter_hmac("198.51.100.2", "Agent/1.0", "2026-10-28")
    h3 = compute_voter_hmac("198.51.100.3", "Agent/1.0", "2026-10-28")

    ok1, t1 = mem.record_mood_vote("2026-10-28", "resonant", h1)
    assert ok1 is True
    assert set(t1.keys()) == {"spot_on", "too_sunny", "too_heavy"}
    assert t1["spot_on"] == 16

    ok2, t2 = mem.record_mood_vote("2026-10-28", "wistful", h2)
    assert ok2 is True
    assert set(t2.keys()) == {"spot_on", "too_sunny", "too_heavy"}
    assert t2["too_sunny"] == 9

    ok3, t3 = mem.record_mood_vote("2026-10-28", "skeptical", h3)
    assert ok3 is True
    assert set(t3.keys()) == {"spot_on", "too_sunny", "too_heavy"}
    assert t3["too_heavy"] == 6

    # FirestoreStateStore also normalizes legacy aliases and writes only canonical keys
    rec_client = _RecordingFirestoreClient()
    fs_store = FirestoreStateStore(
        client=rec_client,
        project_id="test-proj",
        fallback_store=JsonFileStateStore(repo_root=tmp_path),
    )
    h4 = compute_voter_hmac("198.51.100.4", "Agent/1.0", "2026-10-28")
    ok4, t4 = fs_store.record_mood_vote("2026-10-28", "reflective", h4)
    assert ok4 is True
    assert set(t4.keys()) == {"spot_on", "too_sunny", "too_heavy"}
    assert set(rec_client.docs["pulse_votes/2026-10-28"].keys()) == {
        "spot_on",
        "too_sunny",
        "too_heavy",
    }
    rate_doc = rec_client.docs[f"vote_rate_limits/2026-10-28_{h4}"]
    assert rate_doc["mood_key"] == "too_sunny"
    assert rate_doc["vote_type"] == "too_sunny"


def test_item_max_age_hours_all_macro_96h_retention() -> None:
    """Approved v3: every Macro tier retains 96h and Personal Orbit retains 336h."""
    from datetime import datetime, timezone
    from src.state_store import M3_RETENTION_HOURS, _item_max_age_hours

    assert M3_RETENTION_HOURS == 96.0
    assert _item_max_age_hours({"bucket": "O1"}) == 336.0
    assert _item_max_age_hours({"bucket": "O2"}) == 336.0
    assert _item_max_age_hours({"bucket": "O3"}) == 336.0
    assert _item_max_age_hours({"origin": "orbit_scout_1hop"}) == 336.0
    assert _item_max_age_hours({"bucket": "M3"}) == 96.0
    assert _item_max_age_hours({"bucket": "m3"}) == 96.0
    assert _item_max_age_hours({"bucket": "M1"}) == 96.0
    assert _item_max_age_hours({"bucket": "M2"}) == 96.0

    # At 120h or 180h age, all Macro tiers expire; Orbit remains eligible.
    now_utc = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    mem = InMemoryStateStore()
    mem.save_ingestion_cache_day(
        "2026-10-23",
        {
            "m1_120h": {
                "bucket": "M1",
                "published_at": "2026-10-23T12:00:00Z",
                "title": "Macro 1 at 120h",
            },
            "m3_120h": {
                "bucket": "M3",
                "published_at": "2026-10-23T12:00:00Z",
                "title": "Macro 3 cultural at 120h",
            },
            "o1_120h": {
                "bucket": "O1",
                "published_at": "2026-10-23T12:00:00Z",
                "title": "Orbit 1 at 120h",
            },
        },
    )
    mem.save_ingestion_cache_day(
        "2026-10-21",
        {
            "m3_180h": {
                "bucket": "M3",
                "published_at": "2026-10-21T00:00:00Z",
                "title": "Macro 3 cultural at 180h",
            },
            "o2_180h": {
                "bucket": "O2",
                "published_at": "2026-10-21T00:00:00Z",
                "title": "Orbit 2 at 180h",
            },
        },
    )

    hydrated = mem.load_ingestion_cache(now_utc)
    assert "m1_120h" not in hydrated
    assert "m3_120h" not in hydrated
    assert "o1_120h" in hydrated
    assert "m3_180h" not in hydrated
    assert "o2_180h" in hydrated

    pruned_count = mem.prune_expired_cache_items(now_utc)
    assert pruned_count == 3  # All three expired Macro items are pruned.

    from src.zeitgeist.methodology import load_asymmetric_window

    window_items = load_asymmetric_window(
        [
            {"url_hash": "m1_120h", "bucket": "M1", "source_id": "m1_hn", "published_at": "2026-10-23T12:00:00Z"},
            {"url_hash": "m3_120h", "bucket": "M3", "source_id": "m3_dezeen", "published_at": "2026-10-23T12:00:00Z"},
            {"url_hash": "m3_180h", "bucket": "M3", "source_id": "m3_dezeen", "published_at": "2026-10-21T00:00:00Z"},
            {"url_hash": "o1_120h", "bucket": "O1", "source_id": "o1_ness", "published_at": "2026-10-23T12:00:00Z"},
        ],
        now_utc=now_utc,
    )
    retained_hashes = {it["url_hash"] for it in window_items}
    assert "m3_120h" not in retained_hashes
    assert "o1_120h" in retained_hashes
    assert "m1_120h" not in retained_hashes
    assert "m3_180h" not in retained_hashes


def _assert_valid_pulse_votes(data: dict[str, Any]) -> None:
    allowed = {"spot_on", "too_sunny", "too_heavy"}
    assert set(data.keys()) == allowed
    for k in allowed:
        assert isinstance(data[k], int) and data[k] >= 0


def _assert_valid_daily_pulse_state(data: dict[str, Any], require_v2: bool = False) -> None:
    allowed = {
        "schema_version",
        "state_version",
        "date",
        "date_utc",
        "word",
        "gloss",
        "commissioned_thesis",
        "visual_anchor",
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
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in daily_pulse: {extra}"
    is_v1 = "visitor_mood_tally" in data and set(data["visitor_mood_tally"].keys()) == {
        "spot_on",
        "too_sunny",
        "too_heavy",
    }
    v2_required = {
        "date",
        "word",
        "gloss",
        "commissioned_thesis",
        "visual_medium",
        "fallback_webp_path",
        "mood_votes",
        "published_at",
    }
    is_v2 = v2_required.issubset(data.keys())
    if is_v2:
        assert isinstance(data["date"], str)
        assert isinstance(data["word"], str)
        assert isinstance(data["gloss"], str)
        assert isinstance(data["commissioned_thesis"], str)
        assert data["visual_medium"] in {
            "UNSPLASH_PHOTO",
            "FLUX_FAL",
            "SVG_FALLBACK",
            "PHOTOGRAPHY",
            "FINE_ART",
            "ILLUSTRATION",
            "GENERATIVE_SVG",
        }
        assert isinstance(data["fallback_webp_path"], str)
        _assert_valid_pulse_votes(data["mood_votes"])
    if "visitor_mood_tally" in data:
        _assert_valid_pulse_votes(data["visitor_mood_tally"])
    if require_v2:
        assert is_v2, f"Expected v2 daily_pulse keys {v2_required}, missing {v2_required - set(data.keys())}"
    else:
        assert is_v1 or is_v2


def _assert_valid_active_game_state(data: dict[str, Any]) -> None:
    allowed = {
        "schema_version",
        "state_version",
        "round_id",
        "turn",
        "mode",
        "status",
        "current_turn_color",
        "board",
        "bitboard_p1",
        "bitboard_p2",
        "winning_cells",
        "last_move",
        "move_history",
        "gold_moves",
        "blue_moves",
        "red_team",
        "yellow_team",
        "red_contributors",
        "yellow_contributors",
        "moves_this_round",
        "last_finished_board",
        "finisher_handle",
        "last_elo_delta",
        "episode_trace",
        "last_move_epoch",
        "game_type",
        "updated_at",
        "active_game",
        "game_key",
        "round_over",
        "winner",
        "last_challenger_move",
        "last_bot_move",
        "last_mover",
        "last_move_timestamp",
        "last_move_summary",
        "move_log",
        "past_rounds",
        "rate_limits",
        "swapped_at_epoch",
        "board_state",
        "target_word",
        "max_attempts",
        "guesses",
        "candidate_words",
        "target_id",
        "target_concept_id",
        "target_name",
        "target_category",
        "max_questions",
        "questions_asked",
        "qa_log",
        "asked_questions",
        "remaining_candidates",
    }
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in active_game: {extra}"
    for req in ("round_id", "turn", "mode", "status", "current_turn_color"):
        assert req in data, f"Missing {req} in active_game"
    assert isinstance(data["round_id"], int) and data["round_id"] >= 1
    assert isinstance(data["turn"], int) and data["turn"] >= 0
    assert data["mode"] in {"VS_BOT", "PVP_DUEL", "COOP_FORUM"}
    assert data["current_turn_color"] in {
        "RED",
        "YELLOW",
        "GOLD",
        "BLUE",
        "TERRACOTTA",
        "CHARCOAL",
    }
    if "last_move" in data and data["last_move"] is not None:
        assert isinstance(data["last_move"], dict), f"last_move must be dict or None, got {type(data['last_move'])}"


def _assert_valid_elo_ledger_state(data: dict[str, Any]) -> None:
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
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in elo_ledger: {extra}"
    assert "bot_elo" in data and isinstance(data["bot_elo"], (int, float))
    assert ("players" in data and isinstance(data["players"], dict)) or (
        "ratings" in data and isinstance(data["ratings"], dict)
    )


def _assert_valid_bot_memory_state(data: dict[str, Any]) -> None:
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
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in bot_memory: {extra}"
    assert ("games_learned" in data and isinstance(data["games_learned"], int)) or (
        "games_played" in data and isinstance(data["games_played"], int)
    )
    assert (
        ("states" in data and isinstance(data["states"], dict))
        or ("state_values" in data and isinstance(data["state_values"], dict))
        or ("values" in data and isinstance(data["values"], dict))
    )


def _assert_valid_game_rate_limits_state(data: dict[str, Any]) -> None:
    allowed = {
        "schema_version",
        "state_version",
        "users",
        "cooldowns",
        "updated_at",
    }
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in game_rate_limits: {extra}"
    assert ("users" in data and isinstance(data["users"], dict)) or (
        "cooldowns" in data and isinstance(data["cooldowns"], dict)
    )


def _assert_valid_orbit_roster_state(data: dict[str, Any]) -> None:
    allowed = {
        "schema_version",
        "state_version",
        "handles",
        "curated_o3_cursor",
        "updated_at",
    }
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in orbit_roster: {extra}"
    assert "handles" in data and isinstance(data["handles"], dict)
    assert "curated_o3_cursor" in data and isinstance(data["curated_o3_cursor"], dict)


def _assert_valid_ingestion_cache_shard(data: dict[str, Any], date_id: str) -> None:
    allowed = {"date", "items", "updated_at"}
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in ingestion_cache shard: {extra}"
    assert data.get("date") == date_id
    assert isinstance(data.get("items"), dict)


def _assert_valid_vote_rate_limit(data: dict[str, Any]) -> None:
    allowed = {"date", "vote_type", "voted_at", "date_str", "voter_hmac", "mood_key"}
    extra = set(data.keys()) - allowed
    assert not extra, f"Unexpected keys in vote_rate_limits: {extra}"
    assert data.get("vote_type") in {"spot_on", "too_sunny", "too_heavy"}
    assert data.get("mood_key") in {"spot_on", "too_sunny", "too_heavy"}


from src.zeitgeist.public_schema import LEGACY_EDITORIAL_FIELDS

FIRESTORE_PULSE_HISTORY_ALLOWED_FIELDS = LEGACY_EDITORIAL_FIELDS

def test_firestore_state_store_all_write_methods_conform_to_firestore_rules(
    tmp_path: Path,
) -> None:
    """Slice 4: Every FirestoreStateStore write method emits documents matching firestore.rules."""
    from src.game_engine import build_midgame_showcase_state

    rec_client = _RecordingFirestoreClient()
    fs_store = FirestoreStateStore(
        client=rec_client,
        project_id="test-proj",
        fallback_store=JsonFileStateStore(repo_root=tmp_path),
    )

    pulse = _make_sample_published_pulse()
    feed_entry = pulse.to_feed_entry()

    # 1. save_pulse_state
    fs_store.save_pulse_state(
        {
            "schema_version": "2.0",
            "state_version": 1,
            "date": "2026-10-28",
            "word": "BULKHEAD",
            "gloss": "Structural partitions that isolate failure domains.",
            "commissioned_thesis": feed_entry["commissioned_thesis"],
            "visual_medium": "PHOTOGRAPHY",
            "image_url": feed_entry["image_url"],
            "fallback_webp_path": feed_entry["fallback_webp_path"],
            "mood_votes": {"spot_on": 20, "too_sunny": 3, "too_heavy": 4},
            "visitor_mood_tally": {"spot_on": 20, "too_sunny": 3, "too_heavy": 4},
            "published_at": "2026-10-28T06:00:00Z",
            "updated_at": "2026-10-28T06:00:00Z",
            "analysis": {
                "date_iso": "2026-10-28",
                "word_of_the_day": "BULKHEAD",
                "explanation_phrase": "Structural partitions that isolate failure domains.",
                "composite_score": 8.42,
            },
            "published_pulse": feed_entry,
            "image_data_uri": "data:image/webp;base64,UklGRkAAAABXRUJQ",
        }
    )
    _assert_valid_daily_pulse_state(
        rec_client.docs["profile_state/daily_pulse"], require_v2=True
    )

    # 2. save_pulse_feed_entry
    fs_store.save_pulse_feed_entry("2026-10-28", feed_entry)
    hist_doc = rec_client.docs["pulse_history/2026-10-28"]
    assert set(hist_doc.keys()).issubset(FIRESTORE_PULSE_HISTORY_ALLOWED_FIELDS)
    assert hist_doc["date"] == "2026-10-28"
    assert not str(hist_doc.get("image_data_uri") or "").startswith("data:")

    # 3. save_game_state
    fs_store.save_game_state(build_midgame_showcase_state("connect4", round_id=1))
    _assert_valid_active_game_state(rec_client.docs["profile_state/active_game"])

    # 4. save_elo_ledger
    fs_store.save_elo_ledger(
        {
            "schema_version": "2.0",
            "state_version": 1,
            "bot_elo": 1200,
            "players": {"Siyuan-Zhu": {"elo": 1125}},
            "recent_rounds": [],
        }
    )
    _assert_valid_elo_ledger_state(rec_client.docs["profile_state/elo_ledger"])

    # 5. save_bot_memory
    fs_store.save_bot_memory(
        {"schema_version": "2.0", "state_version": 1, "games_learned": 3, "states": {}}
    )
    _assert_valid_bot_memory_state(rec_client.docs["profile_state/bot_memory"])

    # 6. save_game_rate_limits
    fs_store.save_game_rate_limits(
        {"schema_version": "2.0", "state_version": 1, "users": {}}
    )
    _assert_valid_game_rate_limits_state(
        rec_client.docs["profile_state/game_rate_limits"]
    )

    # 7. save_orbit_roster
    fs_store.save_orbit_roster(
        {
            "schema_version": "2.0",
            "state_version": 1,
            "handles": {},
            "curated_o3_cursor": {},
        }
    )
    _assert_valid_orbit_roster_state(rec_client.docs["profile_state/orbit_roster"])

    # 8. save_ingestion_cache_day
    fs_store.save_ingestion_cache_day(
        "2026-10-28",
        {"hash1": {"bucket": "O1", "published_at": "2026-10-28T05:00:00Z"}},
    )
    _assert_valid_ingestion_cache_shard(
        rec_client.docs["ingestion_cache/2026-10-28"], "2026-10-28"
    )

    # 9. record_mood_vote
    voter_hmac = compute_voter_hmac("198.51.100.99", "Agent/2.0", "2026-10-28")
    accepted, tally = fs_store.record_mood_vote("2026-10-28", "spot_on", voter_hmac)
    assert accepted is True
    _assert_valid_pulse_votes(rec_client.docs["pulse_votes/2026-10-28"])
    _assert_valid_vote_rate_limit(
        rec_client.docs[f"vote_rate_limits/2026-10-28_{voter_hmac}"]
    )
    _assert_valid_daily_pulse_state(
        rec_client.docs["profile_state/daily_pulse"], require_v2=True
    )


def test_public_json_files_match_runtime_schema_and_no_base64_bloat() -> None:
    """Validate the real public projection, including a fresh v3 publication."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    seed_files = [
        "game_state.json",
        "hall_of_records.json",
        "zeitgeist_state.json",
        "pulse_feed.json",
    ]
    for fname in seed_files:
        fpath = data_dir / fname
        assert fpath.is_file(), f"Missing seed file {fpath}"
        raw_text = fpath.read_text(encoding="utf-8")
        assert "data:image/" not in raw_text, f"Found base64 data:image/ bloat in {fname}"
        assert "\x24" not in raw_text, f"Found forbidden literal currency char in {fname}"

    # Legacy v2 fixtures retain their strict wire checks; fresh domain snapshots
    # are validated against the current normalizer rather than fabricated tags.
    game_state = json.loads((data_dir / "game_state.json").read_text(encoding="utf-8"))
    if game_state.get("schema_version") == "2.0":
        _assert_valid_active_game_state(game_state)
    else:
        from src.game_engine import normalize_game_state
        board = game_state["board"]
        assert len(board) == 6 and all(len(row) == 7 for row in board)
        assert all(cell in {0, 1, 2, ".", "C", "B", "R", "Y"} for row in board for cell in row)
        assert type(game_state["turn"]) is int and game_state["turn"] >= 1
        assert type(game_state["round_id"]) is int and game_state["round_id"] >= 1
        normalized = normalize_game_state(game_state)
        assert normalized["board"] == board
        assert normalized["turn"] == game_state["turn"]
        assert normalized["board_state"]["board"] == board

    elo_ledger = json.loads((data_dir / "hall_of_records.json").read_text(encoding="utf-8"))
    if elo_ledger.get("schema_version") == "2.0":
        _assert_valid_elo_ledger_state(elo_ledger)
    else:
        _assert_valid_elo_ledger_state(JsonFileStateStore(repo_root=repo_root).load_elo_ledger())

    zstate = json.loads((data_dir / "zeitgeist_state.json").read_text(encoding="utf-8"))
    modern = zstate.get("schema_version") == 3
    assert modern or zstate.get("schema_version") == "2.0"
    if modern:
        from src.zeitgeist.publication import verify_publication_bundle, verify_current_pulse
        manifest = verify_publication_bundle(repo_root, require_current_pulse=True)
        verify_current_pulse(zstate, manifest)
        assert zstate["publication_id"] == manifest["publication_id"]
        _assert_valid_pulse_votes(zstate["mood_votes"])
    else:
        _assert_valid_daily_pulse_state(zstate, require_v2=True)
    pub_pulse = zstate.get("published_pulse")
    assert isinstance(pub_pulse, dict)
    assert pub_pulse.get("schema_version") == "2.0"
    assert pub_pulse.get("image_data_uri") == ""
    # Modern public JSON carries release provenance in addition to the legacy
    # history DTO. Keep legacy wire/rules checks strict and validate the exact
    # typed public extension rather than accepting arbitrary new source fields.
    allowed = FIRESTORE_PULSE_HISTORY_ALLOWED_FIELDS | (
        {"publication_id", "release_republication", "photo_selection_policy",
         "photo_selection_policy_date"} if modern else set()
    )
    assert set(pub_pulse).issubset(allowed)
    def assert_public_provenance(payload):
        from src.zeitgeist.public_schema import photo_policy_projection
        photo_policy_projection(payload)
        if "release_republication" in payload:
            from scripts.export_public_release import _public_republication
            record = payload["release_republication"]
            assert modern
            assert _public_republication(record) == record
            assert record["original_publication_date"] <= payload["date"][:10]
    assert_public_provenance(zstate)
    assert_public_provenance(pub_pulse)

    pfeed = json.loads((data_dir / "pulse_feed.json").read_text(encoding="utf-8"))
    assert pfeed.get("version") == "2.0"
    assert pfeed.get("schema_version") == "2.0"
    assert isinstance(pfeed.get("updated_at"), str) and pfeed["updated_at"]
    entries = pfeed.get("entries")
    assert isinstance(entries, list) and len(entries) >= 1
    for entry in entries:
        assert entry.get("schema_version") == "2.0"
        assert entry.get("image_data_uri") == ""
        extra = set(entry) - allowed
        assert not extra, f"Disallowed public feed keys: {extra}"
        assert_public_provenance(entry)
        assert SECTION_7_1_FEED_KEYS.issubset(entry)
        _assert_valid_pulse_votes(entry["mood_votes"])
        if modern and entry["date"][:10] == manifest["date"]:
            assert entry["publication_id"] == manifest["publication_id"]
