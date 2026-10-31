"""TDD test suite for Workstream 1 (WS1): StateStore binary serialization, date-sharded cache, monotonic state_version, rate limits, and stdlib isolation."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.state_store import (
    FirestoreStateStore,
    InMemoryStateStore,
    JsonFileStateStore,
    _firestore_value_to_py,
    _FirestoreRestClient,
    _py_to_firestore_value,
)


class _MockHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _MockHttpResponse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


class TestStateStoreSeam1BinaryBytesRoundTrip(unittest.TestCase):
    """Seam 1: Binary bytesValue round-trip for 1,536-byte float16 embedding vectors."""

    def test_py_to_firestore_value_and_back_preserves_1536_byte_float16_vector(self) -> None:
        rng = np.random.default_rng(42)
        vec_f32 = rng.standard_normal(768).astype(np.float32)
        vec_f32 /= np.linalg.norm(vec_f32)
        vec_f16 = vec_f32.astype(np.float16)
        raw_bytes = vec_f16.tobytes()
        self.assertEqual(len(raw_bytes), 1536)

        encoded = _py_to_firestore_value(raw_bytes)
        self.assertIn("bytesValue", encoded)
        self.assertNotIn("stringValue", encoded)
        self.assertEqual(encoded["bytesValue"], base64.b64encode(raw_bytes).decode("ascii"))

        # Also verify bytearray support
        encoded_ba = _py_to_firestore_value(bytearray(raw_bytes))
        self.assertEqual(encoded_ba, encoded)

        decoded = _firestore_value_to_py(encoded)
        self.assertIsInstance(decoded, bytes)
        self.assertEqual(decoded, raw_bytes)

        recovered_f16 = np.frombuffer(decoded, dtype=np.float16)
        np.testing.assert_array_equal(vec_f16, recovered_f16)

    def test_save_and_load_ingestion_cache_day_round_trip_across_adapters(self) -> None:
        now_utc = datetime(2026, 10, 26, 6, 0, 0, tzinfo=timezone.utc)
        vec_f16 = np.linspace(-1.0, 1.0, 768, dtype=np.float16)
        raw_bytes = vec_f16.tobytes()

        day_items = {
            "hash_orbit_1": {
                "url": "https://example.com/orbit-essay",
                "source_id": "o1_paul_graham",
                "bucket": "O1",
                "origin": "curated_O1_O3",
                "title": "Maker Schedule Craft",
                "summary": "Deep reflection on uninterrupted blocks of time.",
                "published_at": (now_utc - timedelta(hours=12)).isoformat(),
                "embedding_f16": raw_bytes,
                "mentions": ["karpathy"],
                "anchored_word": None,
            }
        }

        # 1. InMemoryStateStore
        mem_store = InMemoryStateStore()
        mem_store.save_ingestion_cache_day("2026-10-26", day_items)
        loaded_mem = mem_store.load_ingestion_cache(now_utc)
        self.assertIn("hash_orbit_1", loaded_mem)
        self.assertEqual(loaded_mem["hash_orbit_1"]["embedding_f16"], raw_bytes)
        self.assertEqual(loaded_mem["hash_orbit_1"]["embedding_f32"].dtype, np.float32)
        self.assertEqual(loaded_mem["hash_orbit_1"]["embedding_f32"].shape, (768,))
        np.testing.assert_allclose(
            loaded_mem["hash_orbit_1"]["embedding_f32"],
            vec_f16.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
        )

        # 2. JsonFileStateStore
        with tempfile.TemporaryDirectory() as tmpdir:
            json_store = JsonFileStateStore(repo_root=Path(tmpdir))
            json_store.save_ingestion_cache_day("2026-10-26", day_items)
            loaded_json = json_store.load_ingestion_cache(now_utc)
            self.assertIn("hash_orbit_1", loaded_json)
            self.assertEqual(loaded_json["hash_orbit_1"]["embedding_f16"], raw_bytes)
            np.testing.assert_allclose(
                loaded_json["hash_orbit_1"]["embedding_f32"],
                vec_f16.astype(np.float32),
                rtol=1e-3,
                atol=1e-3,
            )


class TestStateStoreSeam2AsymmetricCacheRetentionAndPruning(unittest.TestCase):
    """Seam 2: Asymmetric 14d Personal Orbit vs 96h Macro Velocity retention, pruning, and paginated REST client."""

    def test_asymmetric_cache_retention_and_prune_expired_cache_items(self) -> None:
        now_utc = datetime(2026, 10, 26, 6, 0, 0, tzinfo=timezone.utc)
        raw_bytes = np.zeros(768, dtype=np.float16).tobytes()

        # Shard 1: 5 days ago (2026-10-21) -> Personal Orbit (120h old <= 336h) kept; Macro (120h old > 96h) pruned
        day_5d_ago = "2026-10-21"
        ts_5d_ago = (now_utc - timedelta(days=5)).isoformat()
        items_5d_ago = {
            "orbit_5d": {
                "url": "https://example.com/orbit-5d",
                "source_id": "o2_faith_essay",
                "bucket": "O2",
                "origin": "curated_O1_O3",
                "title": "Contemplative Rhythm",
                "summary": "Sabbath reflection.",
                "published_at": ts_5d_ago,
                "embedding_f16": raw_bytes,
            },
            "macro_5d": {
                "url": "https://example.com/macro-5d",
                "source_id": "m1_tech_wire",
                "bucket": "M1",
                "origin": "curated_M1_M3",
                "title": "Breaking Chip News",
                "summary": "Daily wire update.",
                "published_at": ts_5d_ago,
                "embedding_f16": raw_bytes,
            },
        }

        # Shard 2: 13 days ago (2026-10-13) -> Personal Orbit (312h old <= 336h) kept
        day_13d_ago = "2026-10-13"
        ts_13d_ago = (now_utc - timedelta(days=13)).isoformat()
        items_13d_ago = {
            "orbit_13d": {
                "url": "https://example.com/orbit-13d",
                "source_id": "o1_simon_willison",
                "bucket": "O1",
                "origin": "curated_O1_O3",
                "title": "Biweekly Builder Notes",
                "summary": "Notes on local tooling.",
                "published_at": ts_13d_ago,
                "embedding_f16": raw_bytes,
            }
        }

        # Shard 3: 16 days ago (2026-10-10) -> Entire shard expired (> 14d)
        day_16d_ago = "2026-10-10"
        ts_16d_ago = (now_utc - timedelta(days=16)).isoformat()
        items_16d_ago = {
            "orbit_16d": {
                "url": "https://example.com/orbit-16d",
                "source_id": "o1_old",
                "bucket": "O1",
                "origin": "curated_O1_O3",
                "title": "Expired Essay",
                "summary": "Older than 14 days.",
                "published_at": ts_16d_ago,
                "embedding_f16": raw_bytes,
            }
        }

        mem = InMemoryStateStore()
        mem.save_ingestion_cache_day(day_5d_ago, items_5d_ago)
        mem.save_ingestion_cache_day(day_13d_ago, items_13d_ago)
        mem.save_ingestion_cache_day(day_16d_ago, items_16d_ago)

        loaded = mem.load_ingestion_cache(now_utc)
        self.assertIn("orbit_5d", loaded)
        self.assertIn("orbit_13d", loaded)
        self.assertNotIn("macro_5d", loaded)
        self.assertNotIn("orbit_16d", loaded)

        pruned_count = mem.prune_expired_cache_items(now_utc)
        self.assertGreaterEqual(pruned_count, 2)
        self.assertNotIn(day_16d_ago, mem._ingestion_cache_shards)
        self.assertNotIn("macro_5d", mem._ingestion_cache_shards[day_5d_ago]["items"])
        self.assertIn("orbit_5d", mem._ingestion_cache_shards[day_5d_ago]["items"])

    def test_firestore_rest_client_paginated_list_documents_and_delete_document(self) -> None:
        client = _FirestoreRestClient(project_id="test-proj", access_token="fake-token")
        page1 = {
            "documents": [
                {
                    "name": "projects/test-proj/databases/(default)/documents/ingestion_cache/2026-10-25",
                    "fields": {"date": {"stringValue": "2026-10-25"}},
                }
            ],
            "nextPageToken": "tok_page_2",
        }
        page2 = {
            "documents": [
                {
                    "name": "projects/test-proj/databases/(default)/documents/ingestion_cache/2026-10-26",
                    "fields": {"date": {"stringValue": "2026-10-26"}},
                }
            ]
        }
        seen_urls: list[str] = []
        seen_methods: list[str] = []

        def fake_urlopen(req, timeout=10):
            seen_urls.append(req.full_url)
            seen_methods.append(req.get_method())
            if req.get_method() == "DELETE":
                return _MockHttpResponse({})
            if "pageToken=tok_page_2" in req.full_url:
                return _MockHttpResponse(page2)
            return _MockHttpResponse(page1)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            snaps = client.list_documents("ingestion_cache", page_size=1)
            self.assertEqual(len(snaps), 2)
            self.assertEqual(snaps[0].id, "2026-10-25")
            self.assertEqual(snaps[0].to_dict()["date"], "2026-10-25")
            self.assertEqual(snaps[1].id, "2026-10-26")
            self.assertTrue(any("pageToken=tok_page_2" in u for u in seen_urls))

            # Also verify .delete() on _FirestoreRestDocRef
            client.collection("ingestion_cache").document("2026-10-10").delete()
            self.assertIn("DELETE", seen_methods)


class _FakeSnap:
    def __init__(self, doc_id: str, data: dict | None) -> None:
        self.id = doc_id
        self.exists = data is not None
        self._data = dict(data) if data is not None else {}

    def to_dict(self) -> dict:
        return dict(self._data)


class _FakeDocRef:
    def __init__(self, col_dict: dict[str, dict], doc_id: str) -> None:
        self._col = col_dict
        self._id = doc_id

    def get(self) -> _FakeSnap:
        return _FakeSnap(self._id, self._col.get(self._id))

    def set(self, data: dict) -> None:
        self._col[self._id] = dict(data)

    def delete(self) -> None:
        self._col.pop(self._id, None)


class _FakeCol:
    def __init__(self, db: dict[str, dict[str, dict]], name: str) -> None:
        self._col = db.setdefault(name, {})

    def document(self, doc_id: str) -> _FakeDocRef:
        return _FakeDocRef(self._col, doc_id)

    def stream(self) -> list[_FakeSnap]:
        return [_FakeSnap(k, self._col[k]) for k in sorted(self._col.keys())]


class _FakeClient:
    def __init__(self) -> None:
        self.db: dict[str, dict[str, dict]] = {}

    def collection(self, name: str) -> _FakeCol:
        return _FakeCol(self.db, name)


class TestStateStoreSeam3MonotonicVersionAndReconcile(unittest.TestCase):
    """Seam 3: Monotonic state_version resolution, transient _needs_git_reconcile flag, and Git reconciliation."""

    def test_state_version_resolution_and_git_reconcile_across_firestore_and_local_json(self) -> None:
        fake_client = _FakeClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            json_store = JsonFileStateStore(repo_root=tmp_root)
            fs_store = FirestoreStateStore(client=fake_client, fallback_store=json_store)

            # Seed local JSON pulse state at state_version=2, Firestore at state_version=5
            json_store.save_pulse_state(
                {
                    "state_version": 2,
                    "date": "2026-10-25",
                    "word": "ANCHOR",
                    "gloss": "Earlier local state.",
                }
            )
            fake_client.collection("profile_state").document("daily_pulse").set(
                {
                    "state_version": 5,
                    "date": "2026-10-26",
                    "word": "STILLNESS",
                    "gloss": "Newer Firestore state.",
                }
            )

            loaded_pulse = fs_store.load_pulse_state()
            self.assertEqual(loaded_pulse["state_version"], 5)
            self.assertEqual(loaded_pulse["word"], "STILLNESS")
            self.assertTrue(fs_store.needs_git_reconcile)
            self.assertTrue(loaded_pulse.get("_needs_git_reconcile"))

            # Saving back to Firestore must strip transient _needs_git_reconcile key
            fs_store.save_pulse_state(loaded_pulse)
            persisted_doc = (
                fake_client.collection("profile_state").document("daily_pulse").get().to_dict()
            )
            self.assertNotIn("_needs_git_reconcile", persisted_doc)

            # Now test game_state lag: Firestore at state_version=9, local JSON at state_version=4
            local_game = json_store.load_game_state()
            local_game["state_version"] = 4
            local_game["turn"] = 4
            json_store.save_game_state(local_game)

            fs_game = dict(local_game)
            fs_game["state_version"] = 9
            fs_game["turn"] = 9
            fake_client.collection("profile_state").document("active_game").set(fs_game)

            loaded_game = fs_store.load_game_state()
            self.assertEqual(loaded_game["state_version"], 9)
            self.assertEqual(loaded_game["turn"], 9)
            self.assertTrue(fs_store.needs_git_reconcile)
            self.assertTrue(loaded_game.get("_needs_git_reconcile"))

            # Reconcile local JSON mirrors from authoritative Firestore state
            reconciled = fs_store.reconcile_git_from_firestore_if_needed()
            self.assertTrue(reconciled)
            self.assertFalse(fs_store.needs_git_reconcile)

            synced_local_game = json_store.load_game_state()
            self.assertEqual(synced_local_game["state_version"], 9)
            self.assertEqual(synced_local_game["turn"], 9)
            self.assertNotIn("_needs_git_reconcile", synced_local_game)

            # Explicit set_needs_git_reconcile(True) flags store and triggers reconcile
            fs_store.set_needs_git_reconcile(True)
            self.assertTrue(fs_store.needs_git_reconcile)
            self.assertTrue(fs_store.reconcile_git_from_firestore_if_needed())
            self.assertFalse(fs_store.needs_git_reconcile)


class TestStateStoreSeam4GameRateLimitsAndCooldowns(unittest.TestCase):
    """Seam 4: Game rate-limit & cooldown enforcement (15s solo VS_BOT, 45s shared/PVP_DUEL, 24/hr, 3 games/day)."""

    def test_game_rate_limits_solo_15s_shared_45s_24_per_hour_and_3_games_per_day(self) -> None:
        fake_client = _FakeClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_store = JsonFileStateStore(repo_root=Path(tmpdir))
            fs_store = FirestoreStateStore(client=fake_client, fallback_store=json_store)

            t0 = 1761458400  # 2025-10-26T06:00:00Z
            ok1, reason1 = fs_store.check_and_increment_game_rate_limit(
                "alice", now_epoch=t0, mode="VS_BOT"
            )
            self.assertTrue(ok1)
            self.assertEqual(reason1, "ok")

            # 10s later in solo VS_BOT (< 15s) -> rejected for cooldown
            ok_10s, reason_10s = fs_store.check_and_increment_game_rate_limit(
                "alice", now_epoch=t0 + 10, mode="VS_BOT"
            )
            self.assertFalse(ok_10s)
            self.assertIn("cooldown", reason_10s)

            # 16s later in solo VS_BOT (>= 15s, no other human within 120s) -> allowed!
            ok_16s, _ = fs_store.check_and_increment_game_rate_limit(
                "alice", now_epoch=t0 + 16, mode="VS_BOT"
            )
            self.assertTrue(ok_16s)

            # Now distinct human 'bob' moves at t0 + 20
            ok_bob, _ = fs_store.check_and_increment_game_rate_limit(
                "bob", now_epoch=t0 + 20, mode="VS_BOT"
            )
            self.assertTrue(ok_bob)

            # Alice tries again at t0 + 36 (20s after her t0+16 move, >= 15s solo, but < 45s because bob moved within 120s)
            ok_shared, reason_shared = fs_store.check_and_increment_game_rate_limit(
                "alice", now_epoch=t0 + 36, mode="VS_BOT"
            )
            self.assertFalse(ok_shared)
            self.assertIn("cooldown", reason_shared)

            # Alice at t0 + 62 (46s after her t0+16 move, >= 45s) -> allowed
            ok_after_45s, _ = fs_store.check_and_increment_game_rate_limit(
                "alice", now_epoch=t0 + 62, mode="VS_BOT"
            )
            self.assertTrue(ok_after_45s)

            # In PVP_DUEL mode, 45s cooldown applies even without another recent mover
            ok_pvp1, _ = fs_store.check_and_increment_game_rate_limit(
                "charlie", now_epoch=t0 + 500, mode="PVP_DUEL"
            )
            self.assertTrue(ok_pvp1)
            ok_pvp_fast, reason_pvp_fast = fs_store.check_and_increment_game_rate_limit(
                "charlie", now_epoch=t0 + 525, mode="PVP_DUEL"
            )
            self.assertFalse(ok_pvp_fast)
            self.assertIn("cooldown", reason_pvp_fast)

            # 24 moves/hour cap test for 'dave'
            for i in range(24):
                ok_m, _ = fs_store.check_and_increment_game_rate_limit(
                    "dave", now_epoch=t0 + 1000 + i * 60, mode="VS_BOT"
                )
                self.assertTrue(ok_m)
            ok_25th, reason_25th = fs_store.check_and_increment_game_rate_limit(
                "dave", now_epoch=t0 + 1000 + 24 * 60, mode="VS_BOT"
            )
            self.assertFalse(ok_25th)
            self.assertIn("hourly", reason_25th)

            # 3 completed games/day cap test for 'erin'
            for g in range(3):
                ok_g, _ = fs_store.check_and_increment_game_rate_limit(
                    "erin",
                    now_epoch=t0 + 5000 + g * 60,
                    mode="VS_BOT",
                    game_completed=True,
                )
                self.assertTrue(ok_g)
            ok_g4, reason_g4 = fs_store.check_and_increment_game_rate_limit(
                "erin", now_epoch=t0 + 5000 + 3 * 60, mode="VS_BOT", is_new_round=True
            )
            self.assertFalse(ok_g4)
            self.assertIn("daily", reason_g4)


class TestStateStoreSeam5HelpersAndStdlibIsolation(unittest.TestCase):
    """Seam 5: Orbit roster, 365-day pulse feed archive, bot memory, ELO ledger & stdlib import isolation."""

    def test_orbit_roster_bot_memory_elo_ledger_and_365_day_pulse_feed(self) -> None:
        fake_client = _FakeClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_store = JsonFileStateStore(repo_root=root)
            fs_store = FirestoreStateStore(client=fake_client, fallback_store=json_store)

            # 1. Orbit roster
            roster = {
                "handles": {
                    "sama": {
                        "vitality_score": 1.25,
                        "last_scraped_at": "2025-10-26T06:00:00Z",
                        "discovered_from": "karpathy",
                        "mention_count_24h": 2,
                    }
                },
                "curated_o3_cursor": {"designboom": "2025-10-26T06:00:00Z"},
                "state_version": 1,
            }
            fs_store.save_orbit_roster(roster)
            loaded_roster = fs_store.load_orbit_roster()
            self.assertEqual(loaded_roster["handles"]["sama"]["vitality_score"], 1.25)

            # 2. Bot memory (profile_state/bot_memory + data/bot_brain.json)
            bot_mem = {
                "state_version": 2,
                "games_learned": 14,
                "states": {"0000000000000:0000000000000": {"v": 0.12, "n": 10, "round": 1}},
            }
            fs_store.save_bot_memory(bot_mem)
            loaded_mem = fs_store.load_bot_memory()
            self.assertEqual(loaded_mem["games_learned"], 14)
            self.assertTrue((root / "data" / "bot_brain.json").exists())

            # 3. ELO ledger (profile_state/elo_ledger + data/hall_of_records.json)
            ledger = {
                "state_version": 3,
                "bot_elo": 1200,
                "players": {"alice": {"elo": 824, "wins": 1, "losses": 0, "draws": 0, "moves_made": 4}},
                "recent_rounds": [{"round_id": 1, "winner": "alice"}],
            }
            fs_store.save_elo_ledger(ledger)
            loaded_ledger = fs_store.load_elo_ledger()
            self.assertEqual(loaded_ledger["bot_elo"], 1200)
            self.assertEqual(loaded_ledger["players"]["alice"]["elo"], 824)
            self.assertTrue((root / "data" / "hall_of_records.json").exists())

            # 4. Pulse feed entry (pulse_history/{YYYY-MM-DD} + data/pulse_feed.json 365-day cap)
            base_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
            for day_idx in range(370):
                d_str = (base_dt + timedelta(days=day_idx)).strftime("%Y-%m-%d")
                fs_store.save_pulse_feed_entry(
                    d_str,
                    {
                        "date": d_str,
                        "word": f"WORD{day_idx}",
                        "gloss": "Daily test gloss",
                        "visual_medium": "PHOTOGRAPHY",
                    },
                )
            feed_path = root / "data" / "pulse_feed.json"
            self.assertTrue(feed_path.exists())
            feed_data = json.loads(feed_path.read_text(encoding="utf-8"))
            entries = feed_data.get("entries", feed_data if isinstance(feed_data, list) else [])
            self.assertEqual(len(entries), 365)
            # Newest entry first
            self.assertEqual(entries[0]["word"], "WORD369")

    def test_zero_install_stdlib_import_isolation_does_not_load_numpy(self) -> None:
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        code = (
            "import sys; "
            "import src.state_store as ss; "
            "store = ss.InMemoryStateStore(); "
            "store.check_and_increment_game_rate_limit('alice', 1761458400, 'VS_BOT'); "
            "assert 'numpy' not in sys.modules, 'numpy was imported on stdlib path!'"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)


if __name__ == "__main__":
    unittest.main()
