"""Unit tests for Connect 4 64-bit bitboard, 8-ply Negamax, Tabular TD(lambda), and Dual-Mode PvP/PvB."""

from __future__ import annotations

import time
import unittest

from src.game_engine import (
    apply_visitor_move,
    build_initial_game_state,
    canonical_board_key,
    check_and_heal_stale_turn,
    choose_bot_move,
    encode_bitboard,
    evaluate_tactical_window,
    find_winning_cells,
    has_won_bitboard,
    update_td_lambda_memory,
)


def _empty_board() -> list[list[int]]:
    return [[0 for _ in range(7)] for _ in range(6)]


class TestConnect4BitboardAndSolver(unittest.TestCase):
    """Seam 1: 64-bit bitboard, 8-ply Negamax, and Tabular TD(lambda)."""

    def test_bitboard_win_detection_all_four_directions_and_sentinel(self) -> None:
        # 1. Horizontal win on bottom row (row=5, cols 1..4) for player 1
        board_h = _empty_board()
        for c in (1, 2, 3, 4):
            board_h[5][c] = 1
        pos_h, mask_h = encode_bitboard(board_h, 1)
        self.assertTrue(has_won_bitboard(pos_h))
        self.assertEqual(pos_h & mask_h, pos_h)
        self.assertEqual(
            find_winning_cells(board_h, 1),
            [(5, 1), (5, 2), (5, 3), (5, 4)],
        )

        # 2. Vertical win in col 3 (rows 2..5) for player 2
        board_v = _empty_board()
        for r in (2, 3, 4, 5):
            board_v[r][3] = 2
        pos_v, _ = encode_bitboard(board_v, 2)
        self.assertTrue(has_won_bitboard(pos_v))
        self.assertEqual(
            find_winning_cells(board_v, 2),
            [(2, 3), (3, 3), (4, 3), (5, 3)],
        )

        # 3. Rising diagonal (/) win: (5,1), (4,2), (3,3), (2,4)
        board_d1 = _empty_board()
        for r, c in ((5, 1), (4, 2), (3, 3), (2, 4)):
            board_d1[r][c] = 1
        pos_d1, _ = encode_bitboard(board_d1, 1)
        self.assertTrue(has_won_bitboard(pos_d1))
        self.assertEqual(
            find_winning_cells(board_d1, 1),
            [(5, 1), (4, 2), (3, 3), (2, 4)],
        )

        # 4. Falling diagonal (\) win: (2,1), (3,2), (4,3), (5,4)
        board_d2 = _empty_board()
        for r, c in ((2, 1), (3, 2), (4, 3), (5, 4)):
            board_d2[r][c] = 2
        pos_d2, _ = encode_bitboard(board_d2, 2)
        self.assertTrue(has_won_bitboard(pos_d2))
        self.assertEqual(
            find_winning_cells(board_d2, 2),
            [(2, 1), (3, 2), (4, 3), (5, 4)],
        )

        # 5. Sentinel guard: 3 at top of col 0 + 1 at bottom of col 1 must NOT wrap around
        board_wrap = _empty_board()
        board_wrap[0][0] = 1
        board_wrap[1][0] = 1
        board_wrap[2][0] = 1
        board_wrap[5][1] = 1
        pos_wrap, _ = encode_bitboard(board_wrap, 1)
        self.assertFalse(has_won_bitboard(pos_wrap))
        self.assertEqual(find_winning_cells(board_wrap, 1), [])

    def test_canonical_board_key_horizontal_mirror_symmetry(self) -> None:
        board_left = _empty_board()
        board_left[5][0] = 1
        board_left[5][1] = 2
        board_left[4][1] = 1
        board_left[5][3] = 2

        board_right = [list(reversed(row)) for row in board_left]

        key_left = canonical_board_key(board_left)
        key_right = canonical_board_key(board_right)
        self.assertEqual(key_left, key_right)
        self.assertEqual(len(key_left), 26)

        # Distinct non-symmetric board has a different canonical key
        board_other = _empty_board()
        board_other[5][2] = 1
        self.assertNotEqual(key_left, canonical_board_key(board_other))

    def test_evaluate_tactical_window_incorporates_bot_memory_value(self) -> None:
        board = _empty_board()
        board[5][3] = 2
        board[5][2] = 1
        key = canonical_board_key(board)

        empty_mem = {"states": {}}
        base_score = evaluate_tactical_window(board, empty_mem)

        boosted_mem = {"states": {key: {"v": 0.5, "n": 4}}}
        boosted_score = evaluate_tactical_window(board, boosted_mem)
        self.assertAlmostEqual(boosted_score - base_score, 60.0, places=3)

    def test_choose_bot_move_takes_immediate_win_and_blocks_threat_under_150ms(self) -> None:
        bot_memory = {"states": {}, "games_learned": 0}

        # Case A: Bot (player 2) has 3-in-a-row at row 5, cols 1, 2, 3 -> must take col 0 or 4
        win_board = _empty_board()
        win_board[5][1] = 2
        win_board[5][2] = 2
        win_board[5][3] = 2
        win_board[5][0] = 1  # block col 0 so col 4 is the unique winning move
        win_board[4][1] = 1
        win_board[4][2] = 1
        t0 = time.perf_counter()
        move_win = choose_bot_move(win_board, bot_memory, depth=8)
        elapsed_win_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(move_win, 4)
        self.assertLess(elapsed_win_ms, 150.0)

        # Case B: Opponent (player 1) has 3-in-a-row in col 2 (rows 3, 4, 5) -> Bot must block col 2
        block_board = _empty_board()
        block_board[5][2] = 1
        block_board[4][2] = 1
        block_board[3][2] = 1
        block_board[5][3] = 2
        block_board[4][3] = 2
        t1 = time.perf_counter()
        move_block = choose_bot_move(block_board, bot_memory, depth=8)
        elapsed_block_ms = (time.perf_counter() - t1) * 1000.0
        self.assertEqual(move_block, 2)
        self.assertLess(elapsed_block_ms, 150.0)

        # Case C: Midgame search at depth=8 completes in < 150ms
        mid_board = _empty_board()
        mid_board[5][3] = 1
        mid_board[4][3] = 2
        mid_board[5][2] = 1
        mid_board[5][4] = 2
        t2 = time.perf_counter()
        move_mid = choose_bot_move(mid_board, bot_memory, depth=8)
        elapsed_mid_ms = (time.perf_counter() - t2) * 1000.0
        self.assertIn(move_mid, range(7))
        self.assertLess(elapsed_mid_ms, 150.0)

    def test_update_td_lambda_memory_shifts_values_and_caps_at_5000_states(self) -> None:
        mem: dict = {"states": {}, "games_learned": 0}
        trace = ["00000000000010000000000000", "00000000000030000000000002"]

        # Positive reward (+1.0) shifts state values > 0
        mem = update_td_lambda_memory(mem, trace, 1.0, alpha=0.08, gamma=0.95, lam=0.7)
        self.assertEqual(mem["games_learned"], 1)
        v_s0 = mem["states"][trace[0]]["v"]
        v_s1 = mem["states"][trace[1]]["v"]
        self.assertGreater(v_s0, 0.0)
        self.assertGreater(v_s1, v_s0)
        self.assertEqual(mem["states"][trace[0]]["n"], 1)

        # Negative reward (-1.0) on fresh state shifts value < 0
        loss_key = "00000000000050000000000004"
        mem = update_td_lambda_memory(mem, [loss_key], -1.0, alpha=0.08, gamma=0.95, lam=0.7)
        self.assertEqual(mem["games_learned"], 2)
        self.assertLess(mem["states"][loss_key]["v"], 0.0)

        # State cap enforcement at 5,000 states and strict ['v', 'n', 'round'] schema
        big_states = {
            f"{i:026x}": {"v": 0.1, "n": (1 if i < 100 else 5), "round": 1}
            for i in range(5010)
        }
        big_mem = {"states": big_states, "games_learned": 10}
        big_mem = update_td_lambda_memory(
            big_mem, [f"{5020:026x}"], 1.0, alpha=0.08, gamma=0.95, lam=0.7
        )
        self.assertLessEqual(len(big_mem["states"]), 5000)
        self.assertEqual(
            set(big_mem["states"][f"{5020:026x}"].keys()),
            {"v", "n", "round"},
        )


class TestConnect4DualModeStateMachine(unittest.TestCase):
    """Seam 2: Dual-Mode (VS_BOT / PVP_DUEL) transitions, team affinity, stall takeover, stale turn."""

    def test_auto_reset_on_first_move_after_completed_round(self) -> None:
        state = build_initial_game_state("connect4", round_id=3)
        state["status"] = "COMPLETED"
        state["winner"] = "red"
        state["turn"] = 9
        state["board"][5][0] = 1
        state["board"][5][1] = 1
        state["board"][5][2] = 1
        state["board"][5][3] = 1
        state["winning_cells"] = [(5, 0), (5, 1), (5, 2), (5, 3)]

        bot_mem = {"states": {}, "games_learned": 0}
        elo_ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}

        res = apply_visitor_move(
            state=state,
            actor_handle="alice",
            action="col:4",
            now_epoch=2000,
            bot_memory=bot_mem,
            elo_ledger=elo_ledger,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(state["round_id"], 4)
        self.assertEqual(state["status"], "ACTIVE")
        # Previous winning row cleared, new move + bot reply placed
        self.assertEqual(state["board"][5][0], 0)
        self.assertEqual(state["board"][5][3], 1)

    def test_mode_switch_to_pvp_preserves_early_discs_and_enforces_team_affinity(self) -> None:
        state = build_initial_game_state("connect4", round_id=1)
        bot_mem = {"states": {}, "games_learned": 0}
        elo_ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}

        # Switch to PVP_DUEL on turn 1
        res_mode = apply_visitor_move(
            state, "alice", "mode:new_pvp", 1000, bot_mem, elo_ledger
        )
        self.assertTrue(res_mode.accepted)
        self.assertEqual(state["mode"], "PVP_DUEL")

        # Turn 1 (Red): alice drops in col 4
        res_t1 = apply_visitor_move(
            state, "alice", "col:4", 1010, bot_mem, elo_ledger
        )
        self.assertTrue(res_t1.accepted)
        self.assertEqual(state["turn"], 2)
        self.assertIn("alice", state["red_contributors"])

        # Turn 2 (Yellow): alice attempts to play Yellow -> rejected by team affinity!
        res_self = apply_visitor_move(
            state, "alice", "col:3", 1020, bot_mem, elo_ledger
        )
        self.assertFalse(res_self.accepted)
        self.assertEqual(state["turn"], 2)

        # Turn 2 (Yellow): bob plays Yellow in col 3 -> accepted!
        res_t2 = apply_visitor_move(
            state, "bob", "col:3", 1030, bot_mem, elo_ledger
        )
        self.assertTrue(res_t2.accepted)
        self.assertEqual(state["turn"], 3)
        self.assertIn("bob", state["yellow_contributors"])

        # Turn 3 (Red): bob attempts to play Red -> rejected by team affinity!
        res_bob_red = apply_visitor_move(
            state, "bob", "col:5", 1040, bot_mem, elo_ledger
        )
        self.assertFalse(res_bob_red.accepted)
        self.assertEqual(state["turn"], 3)

    def test_pvp_stall_900s_allows_bot_takeover(self) -> None:
        state = build_initial_game_state("connect4", round_id=1)
        state["mode"] = "PVP_DUEL"
        bot_mem = {"states": {}, "games_learned": 0}
        elo_ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}

        # Alice plays Turn 1 (Red) at t=1000 -> now Turn 2 (waiting on Yellow)
        apply_visitor_move(state, "alice", "col:4", 1000, bot_mem, elo_ledger)
        self.assertEqual(state["turn"], 2)

        # Before 900s (at t=1500), mode:new_vs_bot takeover is rejected
        res_early = apply_visitor_move(
            state, "alice", "mode:new_vs_bot", 1500, bot_mem, elo_ledger
        )
        self.assertFalse(res_early.accepted)
        self.assertEqual(state["mode"], "PVP_DUEL")

        # At t=1905 (>= 900s elapsed), mode:new_vs_bot converts to VS_BOT and Bot plays Yellow!
        res_takeover = apply_visitor_move(
            state, "alice", "mode:new_vs_bot", 1905, bot_mem, elo_ledger
        )
        self.assertTrue(res_takeover.accepted)
        self.assertEqual(state["mode"], "VS_BOT")
        self.assertEqual(state["turn"], 3)
        # Board now has both Alice's Red disc (1) and Bot's Yellow disc (2)
        flat = [cell for row in state["board"] for cell in row]
        self.assertEqual(flat.count(1), 1)
        self.assertEqual(flat.count(2), 1)

    def test_check_and_heal_stale_turn(self) -> None:
        active_state = {"status": "ACTIVE", "turn": 5}
        self.assertEqual(check_and_heal_stale_turn(active_state, None), (True, "ok"))
        self.assertEqual(check_and_heal_stale_turn(active_state, 5), (True, "ok"))
        self.assertEqual(
            check_and_heal_stale_turn(active_state, 3), (False, "stale_turn")
        )

        completed_state = {"status": "COMPLETED", "turn": 9}
        self.assertEqual(
            check_and_heal_stale_turn(completed_state, 1), (True, "ok")
        )
        self.assertEqual(
            check_and_heal_stale_turn(completed_state, 4), (False, "stale_turn")
        )

    def test_apply_visitor_move_updates_move_log_last_move_summary_and_caps_at_12(self) -> None:
        state = build_initial_game_state("connect4", round_id=1)
        bot_mem = {"states": {}, "games_learned": 0}
        elo_ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}

        # 1. VS_BOT move appends human move + bot reply to move_log and updates last_move_summary
        res_vb = apply_visitor_move(
            state, "alice", "col:4", 1000, bot_mem, elo_ledger
        )
        self.assertTrue(res_vb.accepted)
        self.assertIn("alice", state["last_move_summary"])
        self.assertIn("col 4", state["last_move_summary"])
        self.assertIn(state["last_move_summary"], res_vb.readme_section)
        self.assertEqual(len(state["move_log"]), 2)
        self.assertEqual(state["move_log"][0]["player"], "@alice")
        self.assertEqual(state["move_log"][0]["side"], "RED")
        self.assertEqual(state["move_log"][0]["col"], 4)
        self.assertEqual(state["move_log"][1]["player"], "House-Bot")
        self.assertEqual(state["move_log"][1]["side"], "YELLOW")
        self.assertIsInstance(state["move_log"][1]["col"], int)
        self.assertEqual(len(state["move_history"]), 2)

        # 2. Switch to PVP_DUEL on turn 2 records mode switch in move_log and last_move_summary
        res_mode = apply_visitor_move(
            state, "bob", "mode:new_pvp", 1010, bot_mem, elo_ledger
        )
        self.assertTrue(res_mode.accepted)
        self.assertIn("@bob", state["last_move_summary"])
        self.assertIn("PvP Duel", state["last_move_summary"])
        self.assertEqual(state["move_log"][-1]["side"], "MODE")
        self.assertEqual(state["move_log"][-1]["player"], "@bob")

        # 3. Capping at 12 entries in move_log when prior log has 12 entries
        state["move_log"] = [
            {"turn": i, "player": f"@u{i}", "side": "RED", "col": 1, "summary": "Dropped in col 1"}
            for i in range(1, 13)
        ]
        res_pvp = apply_visitor_move(
            state, "charlie", "col:3", 1020, bot_mem, elo_ledger
        )
        self.assertTrue(res_pvp.accepted)
        self.assertEqual(len(state["move_log"]), 12)
        self.assertEqual(state["move_log"][-1]["player"], "@charlie")
        self.assertEqual(state["move_log"][-1]["col"], 3)
        self.assertIn("@charlie", state["last_move_summary"])


class TestPrewarmAndVisitorEventPipeline(unittest.TestCase):
    """Seam 5: Cold-start self-play prewarming and process_visitor_event end-to-end integration."""

    def test_prewarm_connect4_book_generates_states(self) -> None:
        from scripts.prewarm_connect4_book import prewarm_connect4_book

        mem = prewarm_connect4_book(episodes=12, depth=4, epsilon=0.18, seed=42)
        self.assertEqual(mem["games_learned"], 12)
        self.assertGreater(len(mem["states"]), 10)
        self.assertLessEqual(len(mem["states"]), 5000)

    def test_process_visitor_event_handles_col_and_mode_actions_and_mood_votes(self) -> None:
        import tempfile
        from pathlib import Path
        from scripts.process_visitor_event import parse_cli_args, process_event
        from src.state_store import InMemoryStateStore

        cli_ns = parse_cli_args(
            [
                "--login",
                "alice-dev",
                "--title",
                "game:move:connect4:col:4:turn=1",
                "--issue-number",
                "77",
                "--event-name",
                "issues",
            ]
        )
        self.assertEqual(cli_ns.login, "alice-dev")
        self.assertEqual(cli_ns.title, "game:move:connect4:col:4:turn=1")
        self.assertEqual(cli_ns.issue_number, 77)
        self.assertEqual(cli_ns.event_name, "issues")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = InMemoryStateStore(
                initial_game_state=build_initial_game_state("connect4", round_id=1)
            )
            # 1. Mode switch to PVP_DUEL
            res_mode = process_event(
                repo_root=root,
                visitor_login="alice",
                issue_title="game:move:connect4:mode:new_pvp",
                issue_number=10,
                timestamp_epoch=1000.0,
                store=store,
            )
            self.assertEqual(res_mode["status"], "move_applied")
            self.assertEqual(store.load_game_state()["mode"], "PVP_DUEL")

            # 2. Drop in col:4 by alice (Red) at t=1050 (past 45s PvP cooldown)
            res_col = process_event(
                repo_root=root,
                visitor_login="alice",
                issue_title="game:move:connect4:col:4:turn=1",
                issue_number=11,
                timestamp_epoch=1050.0,
                store=store,
            )
            self.assertEqual(res_col["status"], "move_applied")
            self.assertEqual(store.load_game_state()["turn"], 2)
            self.assertTrue((root / "assets" / "game_board.svg").exists())

            # 3. Mood vote uses user_agent="github-issue"
            res_vote = process_event(
                repo_root=root,
                visitor_login="alice",
                issue_title="pulse:vote:spot_on",
                issue_number=12,
                timestamp_epoch=1060.0,
                store=store,
            )
            self.assertEqual(res_vote["status"], "recorded_vote")
            # Duplicate mood vote from same user on same day even from a different issue number is rejected!
            res_dup = process_event(
                repo_root=root,
                visitor_login="alice",
                issue_title="pulse:vote:too_sunny",
                issue_number=99,
                timestamp_epoch=1070.0,
                store=store,
            )
            self.assertEqual(res_dup["status"], "ignored")

            # Legacy 4-key mood vote alias ('resonant' -> 'spot_on') from a second user succeeds
            res_legacy_vote = process_event(
                repo_root=root,
                visitor_login="bob",
                issue_title="pulse:vote:resonant",
                issue_number=100,
                timestamp_epoch=1080.0,
                store=store,
            )
            self.assertEqual(res_legacy_vote["status"], "recorded_vote")
            self.assertEqual(res_legacy_vote["vote"], "spot_on")

    def test_process_visitor_event_rejected_move_does_not_burn_rate_limit_and_enforces_is_new_round(self) -> None:
        import tempfile
        from pathlib import Path
        from scripts.process_visitor_event import process_event
        from src.state_store import InMemoryStateStore

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Fill column 1 completely on turn 1 so dropping in col:1 is rejected as column_full
            initial_state = build_initial_game_state("connect4", round_id=1)
            for r in range(6):
                initial_state["board"][r][0] = 1 if r % 2 == 0 else 2
            store = InMemoryStateStore(initial_game_state=initial_state)

            # 1. Visitor 'carol' attempts to drop in full column 1 at t=2000 -> rejected (column_full)
            res_invalid = process_event(
                repo_root=root,
                visitor_login="carol",
                issue_title="game:move:connect4:col:1:turn=1",
                issue_number=201,
                timestamp_epoch=2000.0,
                store=store,
            )
            self.assertFalse(res_invalid["accepted"])
            self.assertEqual(res_invalid["status"], "column_full")

            # Rate limit must NOT have been burned by the rejected move:
            # 'carol' can immediately play a valid move in col:4 just 1 second later (t=2001)
            res_valid = process_event(
                repo_root=root,
                visitor_login="carol",
                issue_title="game:move:connect4:col:4:turn=1",
                issue_number=202,
                timestamp_epoch=2001.0,
                store=store,
            )
            self.assertEqual(res_valid["status"], "move_applied")
            self.assertTrue(res_valid["accepted"])
            rl_after_round1 = store.load_game_rate_limits()
            self.assertEqual(
                rl_after_round1["users"]["carol"]["daily_games_count"], 1
            )

            # 2. Enforce 3 new rounds/day cap via is_new_round in process_event
            for round_idx in (2, 3):
                store.save_game_state(
                    build_initial_game_state("connect4", round_id=round_idx)
                )
                res_r = process_event(
                    repo_root=root,
                    visitor_login="carol",
                    issue_title="game:move:connect4:col:3:turn=1",
                    issue_number=202 + round_idx,
                    timestamp_epoch=2001.0 + round_idx * 60.0,
                    store=store,
                )
                self.assertEqual(res_r["status"], "move_applied")

            # Starting a 4th new round on the same UTC day must be blocked by daily_game_limit_exceeded
            store.save_game_state(build_initial_game_state("connect4", round_id=4))
            res_r4 = process_event(
                repo_root=root,
                visitor_login="carol",
                issue_title="game:move:connect4:col:2:turn=1",
                issue_number=210,
                timestamp_epoch=2300.0,
                store=store,
            )
            self.assertFalse(res_r4["accepted"])
            self.assertEqual(res_r4["status"], "daily_game_limit_exceeded")


if __name__ == "__main__":
    unittest.main()
