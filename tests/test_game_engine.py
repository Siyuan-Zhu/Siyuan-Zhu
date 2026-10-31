"""Unit & integration tests for the Rotatable Community Game Engine (`src/game_engine.py`) and Persistence Seam (`src/state_store.py`)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from deploy.cloud_run_service import dispatch_get_request
from scripts.process_visitor_event import process_event
from src.game_engine import (
    COOLDOWN_SECONDS,
    MAX_MOVES_PER_USER_PER_HOUR,
    VALID_GAMES,
    MoveRequest,
    build_initial_game_state,
    build_midgame_showcase_state,
    parse_move_issue_title,
    process_move,
    render_board_svg,
    render_readme_game_section,
)
from src.state_store import (
    FirestoreStateStore,
    InMemoryStateStore,
    JsonFileStateStore,
    compute_voter_hmac,
)


class _FakeFirestoreDocSnapshot:
    def __init__(self, data: dict | None) -> None:
        self._data = copy.deepcopy(data)
        self.exists = data is not None
        self.update_time = json.dumps(data, sort_keys=True) if self.exists else None

    def to_dict(self) -> dict | None:
        return copy.deepcopy(self._data)


class _FakeFirestoreDocRef:
    def __init__(self, storage: dict[str, dict], key: str) -> None:
        self._storage = storage
        self._key = key

    def get(self) -> _FakeFirestoreDocSnapshot:
        return _FakeFirestoreDocSnapshot(self._storage.get(self._key))

    def set(self, data: dict, merge: bool = False) -> None:
        if merge and self._key in self._storage:
            merged = dict(self._storage[self._key])
            merged.update(data)
            self._storage[self._key] = merged
        else:
            self._storage[self._key] = dict(data)


class _FakeFirestoreCollection:
    def __init__(self, db: dict[str, dict[str, dict]], name: str) -> None:
        self._col = db.setdefault(name, {})

    def document(self, doc_id: str) -> _FakeFirestoreDocRef:
        return _FakeFirestoreDocRef(self._col, doc_id)

    def stream(self) -> list[_FakeFirestoreDocSnapshot]:
        # Return documents ordered by reverse key (newest date first)
        return [
            _FakeFirestoreDocSnapshot(self._col[k])
            for k in sorted(self._col.keys(), reverse=True)
        ]


class _FakeFirestoreClient:
    def __init__(self) -> None:
        self.db: dict[str, dict[str, dict]] = {}

    def commit_documents(self, writes) -> None:
        from src.state_store import PublicationConflict
        for col, doc, _data, snapshot in writes:
            current = self.db.get(col, {}).get(doc)
            if (current is not None) != snapshot.exists or (snapshot.exists and current != snapshot.to_dict()):
                raise PublicationConflict("Concurrent fake write")
        for col, doc, data, _snapshot in writes:
            target = self.db.setdefault(col, {})
            target[doc] = dict(target.get(doc, {}), **copy.deepcopy(data))

    def collection(self, name: str) -> _FakeFirestoreCollection:
        return _FakeFirestoreCollection(self.db, name)


class TestRotatableGameEngineMechanics(unittest.TestCase):
    """Verify Connect 4 mechanics, SVG output, and README rendering."""

    def test_connect4_initial_and_showcase_svg_validity(self) -> None:
        self.assertEqual(VALID_GAMES, ("connect4",))
        for game_key in VALID_GAMES:
            init_state = build_initial_game_state(game_key, round_id=1)
            self.assertEqual(init_state["active_game"], game_key)
            svg_init = render_board_svg(init_state)
            self.assertIn('viewBox="0 0 580 360"', svg_init)
            ET.fromstring(svg_init)

            mid_state = build_midgame_showcase_state(game_key, round_id=2)
            self.assertEqual(mid_state["active_game"], game_key)
            svg_mid = render_board_svg(mid_state)
            ET.fromstring(svg_mid)

            md_section = render_readme_game_section(mid_state)
            self.assertIn("./assets/game_board.svg", md_section)
            self.assertIn("<details>", md_section)
            self.assertNotIn("##### How it works", md_section)

    def test_connect4_move_minimax_bot_reply_and_win_auto_reset(self) -> None:
        state = build_initial_game_state("connect4", round_id=1)
        req = MoveRequest(
            game_key="connect4",
            move_id="col4",
            player_login="alice-dev",
            expected_turn=state["turn"],
            timestamp_epoch=1000.0,
            issue_number=10,
        )
        res = process_move(state, req)
        self.assertTrue(res.accepted)
        self.assertEqual(res.status_code, "accepted")
        self.assertEqual(res.new_state["turn"], 2)
        self.assertEqual(res.new_state["last_mover"], "alice-dev")
        ET.fromstring(res.svg_board)

        # Set up 3 challenger discs in bottom row (cols 1..3) and win with col4
        win_state = build_initial_game_state("connect4", round_id=3)
        win_state["board"][5][0] = 1
        win_state["board"][5][1] = 1
        win_state["board"][5][2] = 1
        win_req = MoveRequest(
            game_key="connect4",
            move_id="col4",
            player_login="bob-quant",
            expected_turn=win_state["turn"],
            timestamp_epoch=2000.0,
            issue_number=11,
        )
        win_res = process_move(win_state, win_req)
        self.assertTrue(win_res.accepted)
        self.assertTrue(win_res.round_over)
        self.assertEqual(win_res.winner, "community")
        self.assertEqual(win_res.new_state["round_id"], 4)
        self.assertEqual(win_res.new_state["turn"], 1)
        self.assertEqual(len(win_res.new_state["past_rounds"]), 1)


class TestAntiAbuseGuardrailsAndHotSwap(unittest.TestCase):
    """Verify all 5 deterministic Anti-Abuse Guardrails."""

    def test_all_five_anti_abuse_guardrails(self) -> None:
        state = build_midgame_showcase_state("connect4", round_id=1)
        turn = state["turn"]

        # 1. Guardrail 1: invalid_login (bot or malformed handle)
        r_bot = process_move(
            state,
            MoveRequest(
                game_key="connect4",
                move_id="col3",
                player_login="spammer[bot]",
                expected_turn=turn,
                timestamp_epoch=1000.0,
            ),
        )
        self.assertFalse(r_bot.accepted)
        self.assertEqual(r_bot.status_code, "invalid_login")

        # 2. Guardrail 2: wrong_game
        r_wrong = process_move(
            state,
            MoveRequest(
                game_key="invalid_game",
                move_id="col3",
                player_login="octocat",
                expected_turn=turn,
                timestamp_epoch=1000.0,
            ),
        )
        self.assertFalse(r_wrong.accepted)
        self.assertEqual(r_wrong.status_code, "wrong_game")

        # 3. Guardrail 3: stale_turn
        r_stale = process_move(
            state,
            MoveRequest(
                game_key="connect4",
                move_id="col3",
                player_login="octocat",
                expected_turn=turn - 1,
                timestamp_epoch=1000.0,
            ),
        )
        self.assertFalse(r_stale.accepted)
        self.assertEqual(r_stale.status_code, "stale_turn")

        # Valid move by alice at t=1000.0
        r_ok = process_move(
            state,
            MoveRequest(
                game_key="connect4",
                move_id="col3",
                player_login="alice",
                expected_turn=turn,
                timestamp_epoch=1000.0,
            ),
        )
        self.assertTrue(r_ok.accepted)

        # 4a. Guardrail 4: cooldown_active for same player within 300s
        r_cd = process_move(
            r_ok.new_state,
            MoveRequest(
                game_key="connect4",
                move_id="col2",
                player_login="alice",
                expected_turn=r_ok.new_state["turn"],
                timestamp_epoch=1000.0 + (COOLDOWN_SECONDS - 10),
            ),
        )
        self.assertFalse(r_cd.accepted)
        self.assertEqual(r_cd.status_code, "cooldown_active")

        # Different player ('bob') can play 2 seconds later with ZERO cooldown wait
        r_bob = process_move(
            r_ok.new_state,
            MoveRequest(
                game_key="connect4",
                move_id="col2",
                player_login="bob",
                expected_turn=r_ok.new_state["turn"],
                timestamp_epoch=1002.0,
            ),
        )
        self.assertTrue(r_bob.accepted)

        # 4b. Guardrail 4: rate_limited (>6 moves per hour by same player)
        rl_state = build_initial_game_state("connect4", round_id=1)
        rl_state["rate_limits"] = {"charlie": [100.0, 500.0, 900.0, 1300.0, 1700.0, 2100.0]}
        self.assertEqual(len(rl_state["rate_limits"]["charlie"]), MAX_MOVES_PER_USER_PER_HOUR)
        r_rl = process_move(
            rl_state,
            MoveRequest(
                game_key="connect4",
                move_id="col4",
                player_login="charlie",
                expected_turn=rl_state["turn"],
                timestamp_epoch=2500.0,
            ),
        )
        self.assertFalse(r_rl.accepted)
        self.assertEqual(r_rl.status_code, "rate_limited")

        # 5. Guardrail 5: invalid_move (whitelist enforcement, blocks free-text XSS)
        r_xss = process_move(
            state,
            MoveRequest(
                game_key="connect4",
                move_id="<script>alert(1)</script>",
                player_login="mallory",
                expected_turn=turn,
                timestamp_epoch=3000.0,
            ),
        )
        self.assertFalse(r_xss.accepted)
        self.assertEqual(r_xss.status_code, "invalid_move")
        self.assertNotIn("<script>", r_xss.svg_board)
        self.assertNotIn("<script>", r_xss.readme_section)

    def test_parse_move_issue_title_variants(self) -> None:
        self.assertEqual(
            parse_move_issue_title("game:move:connect4:col4:turn=8"),
            ("connect4", "col4", 8),
        )
        self.assertEqual(
            parse_move_issue_title("game:move:connect4:col:4:turn=8"),
            ("connect4", "col:4", 8),
        )
        self.assertEqual(
            parse_move_issue_title("game:move:connect4:mode:new_pvp:turn=1"),
            ("connect4", "mode:new_pvp", 1),
        )
        self.assertEqual(
            parse_move_issue_title("connect4: drop 5"),
            ("connect4", "col5", None),
        )
        self.assertIsNone(parse_move_issue_title("bug report: something else"))


class TestStateStoreAdaptersAndFirestoreSeam(unittest.TestCase):
    """Verify InMemoryStateStore, JsonFileStateStore, and FirestoreStateStore ports & adapters."""

    def test_in_memory_and_json_file_state_stores(self) -> None:
        mem = InMemoryStateStore()
        gstate = mem.load_game_state()
        self.assertEqual(gstate["active_game"], "connect4")

        v_hmac = compute_voter_hmac("192.0.2.10", "Mozilla/5.0", "2026-10-24", salt="test-salt")
        ok1, tally1 = mem.record_mood_vote("2026-10-24", "spot_on", v_hmac)
        self.assertTrue(ok1)
        self.assertEqual(tally1["spot_on"], 21)

        # Duplicate vote on same day by same HMAC is rejected
        ok2, tally2 = mem.record_mood_vote("2026-10-24", "too_heavy", v_hmac)
        self.assertFalse(ok2)
        self.assertEqual(tally2["spot_on"], 21)
        self.assertEqual(tally2["too_heavy"], 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            json_store = JsonFileStateStore(repo_root=tmp_root)
            loaded = json_store.load_game_state()
            self.assertEqual(loaded["active_game"], "connect4")
            self.assertTrue((tmp_root / "data" / "game_state.json").exists())

            json_store.save_pulse_state(
                {"date_iso": "2026-10-24", "analysis": {"word_of_the_day": "CONVERGENCE"}}
            )
            json_store.save_pulse_state(
                {"date_iso": "2026-10-25", "analysis": {"word_of_the_day": "ANCHORED"}}
            )
            recent = json_store.get_recent_pulse_words(days=30)
            self.assertEqual(recent[:2], ["ANCHORED", "CONVERGENCE"])

    def test_firestore_state_store_with_mock_client_and_cloud_run_vote_redirect(self) -> None:
        fake_client = _FakeFirestoreClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = JsonFileStateStore(repo_root=Path(tmpdir))
            fs_store = FirestoreStateStore(client=fake_client, fallback_store=fallback)

            # 1. Save & load game state via Firestore
            c4_state = build_midgame_showcase_state("connect4", round_id=3)
            fs_store.save_game_state(c4_state)
            loaded_game = fs_store.load_game_state()
            self.assertEqual(loaded_game["active_game"], "connect4")
            self.assertIn("active_game", fake_client.db["profile_state"])

            # 2. Save pulse states across multiple days and verify 30-day rolling word history in Firestore
            fs_store.save_pulse_state(
                {"date_iso": "2026-10-23", "analysis": {"word_of_the_day": "CALIBRATION"}}
            )
            fs_store.save_pulse_state(
                {"date_iso": "2026-10-24", "analysis": {"word_of_the_day": "CONVERGENCE"}}
            )
            words = fs_store.get_recent_pulse_words(days=30)
            self.assertIn("CONVERGENCE", words)
            self.assertIn("CALIBRATION", words)
            self.assertIn("2026-10-24", fake_client.db["pulse_history"])

            # 3. Cloud Run GET /vote?dir=spot_on backed by FirestoreStateStore
            code1, body1, headers1 = dispatch_get_request(
                "/vote?dir=spot_on",
                client_ip="198.51.100.7",
                user_agent="TestBrowser/1.0",
                store=fs_store,
                date_str="2026-10-24",
            )
            self.assertEqual(code1, 302)
            self.assertTrue(body1["accepted"])
            self.assertEqual(headers1["Location"], "https://github.com/Siyuan-Zhu#mood-spot_on")

            # Duplicate click from same IP + User-Agent on same day still redirects 302 but does not increment
            code2, body2, _ = dispatch_get_request(
                "/vote?dir=spot_on",
                client_ip="198.51.100.7",
                user_agent="TestBrowser/1.0",
                store=fs_store,
                date_str="2026-10-24",
            )
            self.assertEqual(code2, 302)
            self.assertFalse(body2["accepted"])
            self.assertEqual(body2["tally"]["spot_on"], body1["tally"]["spot_on"])

    def test_end_to_end_process_event_connect4(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            store = JsonFileStateStore(repo_root=tmp_root)
            current = store.load_game_state()
            turn = current["turn"]

            res = process_event(
                repo_root=tmp_root,
                visitor_login="octocat",
                issue_title=f"game:move:connect4:col4:turn={turn}",
                issue_number=42,
                timestamp_epoch=5000.0,
                store=store,
            )
            self.assertEqual(res["status"], "move_applied")
            self.assertTrue(res["accepted"])
            self.assertEqual(res["turn"], turn + 1)
            self.assertTrue((tmp_root / "assets" / "game_board.svg").exists())


if __name__ == "__main__":
    unittest.main()
