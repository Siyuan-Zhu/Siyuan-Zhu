"""Unit tests for Persistent ELO Rating & Finisher Bonus Ledger (WS4 Seam 4)."""

from __future__ import annotations

import unittest

from src.leaderboard.elo_ledger import (
    BOT_INITIAL_ELO,
    FINISHER_BONUS,
    INITIAL_ELO,
    K_FACTOR,
    MIN_ELO_FLOOR,
    expected_score,
    settle_round,
)


class TestEloLedger(unittest.TestCase):
    """Verify expected_score, VS_BOT single/multi-contributor splits, PVP_DUEL zero-sum pool, and floor clamp."""

    def test_constants_and_expected_score_curve(self) -> None:
        self.assertEqual(INITIAL_ELO, 800)
        self.assertEqual(BOT_INITIAL_ELO, 1200)
        self.assertEqual(K_FACTOR, 32)
        self.assertEqual(FINISHER_BONUS, 8)
        self.assertEqual(MIN_ELO_FLOOR, 100)

        # Equal ratings -> 0.50
        self.assertAlmostEqual(expected_score(1000, 1000), 0.5, places=6)
        # 800 vs 1200 -> 1 / (1 + 10^1) = 1 / 11 ~= 0.090909
        self.assertAlmostEqual(expected_score(800, 1200), 1.0 / 11.0, places=5)
        self.assertAlmostEqual(
            expected_score(800, 1200) + expected_score(1200, 800), 1.0, places=6
        )

    def test_vs_bot_single_player_win_and_finisher_bonus(self) -> None:
        ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}
        updated, deltas = settle_round(
            ledger=ledger,
            mode="VS_BOT",
            winner="red",
            red_contributors={"alice": 4},
            yellow_contributors={},
            finisher_handle="alice",
            moves_count=7,
            now_iso="2026-10-27T00:00:00Z",
        )
        # Pool delta = 32 * (1 - 1/11) = 29.09 -> round(29.09) = 29, plus +8 finisher bonus = +37
        self.assertEqual(deltas["alice"], 37)
        self.assertEqual(updated["players"]["alice"]["elo"], 837)
        self.assertEqual(updated["players"]["alice"]["wins"], 1)
        self.assertEqual(updated["players"]["alice"]["finishes"], 1)
        # Bot ELO decreases by round(29.09) = 29 -> 1171
        self.assertEqual(updated["bot_elo"], 1171)

    def test_vs_bot_multi_contributor_proportional_split_and_finisher_bonus(self) -> None:
        ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}
        # Alice played 3 moves (75%), Bob played 1 winning move (25%)
        updated, deltas = settle_round(
            ledger=ledger,
            mode="VS_BOT",
            winner="red",
            red_contributors={"alice": 3, "bob": 1},
            yellow_contributors={},
            finisher_handle="bob",
            moves_count=7,
            now_iso="2026-10-27T00:05:00Z",
        )
        # Pool delta = 29.0909:
        # Alice share = round(29.0909 * 0.75) = round(21.818) = 22
        # Bob share = round(29.0909 * 0.25) + 8 = round(7.272) + 8 = 7 + 8 = 15
        self.assertEqual(deltas["alice"], 22)
        self.assertEqual(deltas["bob"], 15)
        self.assertEqual(updated["players"]["alice"]["elo"], 822)
        self.assertEqual(updated["players"]["bob"]["elo"], 815)
        self.assertEqual(updated["players"]["bob"]["finishes"], 1)

    def test_pvp_duel_zero_sum_pool_plus_finisher_bonus(self) -> None:
        ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}
        updated, deltas = settle_round(
            ledger=ledger,
            mode="PVP_DUEL",
            winner="yellow",
            red_contributors={"alice": 4},
            yellow_contributors={"bob": 4},
            finisher_handle="bob",
            moves_count=8,
            now_iso="2026-10-27T00:10:00Z",
        )
        # Both start at 800 -> expected_score = 0.5 -> pool delta = 32 * 0.5 = 16
        # Alice (Red loser) gets -16 -> 784
        # Bob (Yellow winner + finisher) gets +16 + 8 = +24 -> 824
        self.assertEqual(deltas["alice"], -16)
        self.assertEqual(deltas["bob"], 24)
        self.assertEqual(updated["players"]["alice"]["elo"], 784)
        self.assertEqual(updated["players"]["bob"]["elo"], 824)
        # Bot ELO untouched in PVP_DUEL
        self.assertEqual(updated["bot_elo"], 1200)

    def test_min_elo_floor_clamping_at_100(self) -> None:
        ledger = {
            "bot_elo": 800,
            "players": {
                "low-player": {
                    "elo": 102,
                    "wins": 0,
                    "losses": 10,
                    "draws": 0,
                    "finishes": 0,
                },
                "equal-opp": {
                    "elo": 102,
                    "wins": 0,
                    "losses": 10,
                    "draws": 0,
                    "finishes": 0,
                },
            },
            "recent_rounds": [],
        }
        updated, _ = settle_round(
            ledger=ledger,
            mode="PVP_DUEL",
            winner="yellow",
            red_contributors={"low-player": 4},
            yellow_contributors={"equal-opp": 4},
            finisher_handle="equal-opp",
            moves_count=8,
            now_iso="2026-10-27T00:15:00Z",
        )
        self.assertGreaterEqual(updated["players"]["low-player"]["elo"], MIN_ELO_FLOOR)
        self.assertEqual(updated["players"]["low-player"]["elo"], 100)

    def test_vs_bot_house_bot_win_deducts_red_elo_and_respects_player_schema(self) -> None:
        ledger = {"bot_elo": 1200, "players": {}, "recent_rounds": []}
        updated, deltas = settle_round(
            ledger=ledger,
            mode="VS_BOT",
            winner="bot",
            red_contributors={"alice": 3, "bob": 1},
            yellow_contributors={},
            finisher_handle="House-Bot",
            moves_count=8,
            now_iso="2026-10-27T00:20:00Z",
        )
        # Expected score for 800 vs 1200 is 1/11 (~0.0909). Actual score = 0.0 -> pool_delta_red = -2.909
        self.assertLess(deltas["alice"], 0)
        self.assertLess(deltas["bob"], 0)
        self.assertEqual(updated["players"]["alice"]["losses"], 1)
        self.assertEqual(updated["players"]["bob"]["losses"], 1)
        self.assertEqual(updated["players"]["alice"]["draws"], 0)
        self.assertGreater(updated["bot_elo"], 1200)
        # Verify Firestore schema compliance on player records
        self.assertEqual(
            set(updated["players"]["alice"].keys()),
            {"elo", "wins", "losses", "draws", "finishes", "last_played_at"},
        )


if __name__ == "__main__":
    unittest.main()
