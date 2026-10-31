"""Tests for Identity-Staked Reputation (ELO) Leaderboard & Hall of Records Engine."""

from __future__ import annotations
import unittest

from src.leaderboard.identity_verifier import PlayerIdentity, verify_github_player
from src.leaderboard.elo_ledger import (
    DEFAULT_LEDGER_PATH,
    INITIAL_ELO,
    STAKE_TIERS,
    build_seeded_ledger,
    compute_reputation_rank,
    create_empty_ledger,
    get_active_stakes,
    get_sorted_leaderboard,
    load_ledger,
    record_move,
    settle_round,
)
from src.leaderboard.render_markdown import render_game_section_md


class IdentityVerifierTest(unittest.TestCase):
    def test_offline_valid_handle_constructs_official_avatar_and_verifies(self) -> None:
        identity = verify_github_player("@Siyuan-Zhu", fetch_online=False)
        self.assertIsInstance(identity, PlayerIdentity)
        self.assertEqual(identity.github_login, "Siyuan-Zhu")
        self.assertEqual(identity.avatar_url, "https://github.com/Siyuan-Zhu.png?size=32")
        self.assertTrue(identity.verified)
        self.assertIsNone(identity.rejection_reason)

    def test_invalid_handle_syntax_is_rejected(self) -> None:
        for bad_handle in ["", "   ", "-leading-hyphen", "trailing-hyphen-", "double--hyphen", "bad space", "depend-bot[bot]"]:
            identity = verify_github_player(bad_handle, fetch_online=False)
            self.assertFalse(identity.verified, f"Expected {bad_handle!r} to be rejected")
            self.assertIsNotNone(identity.rejection_reason)

    def test_online_verification_accepts_established_profile_and_rejects_throwaway(self) -> None:
        def fake_http_get_established(url: str) -> dict:
            self.assertEqual(url, "https://api.github.com/users/octocat")
            return {
                "login": "octocat",
                "type": "User",
                "public_repos": 8,
                "followers": 1200,
                "created_at": "2011-01-25T18:44:36Z",
            }

        verified = verify_github_player(
            "octocat",
            fetch_online=True,
            http_get=fake_http_get_established,
            current_year=2026,
        )
        self.assertTrue(verified.verified)
        self.assertEqual(verified.public_repos, 8)
        self.assertEqual(verified.followers, 1200)
        self.assertEqual(verified.account_created_year, 2011)

        def fake_http_get_throwaway(url: str) -> dict:
            return {
                "login": "throwaway2026",
                "type": "User",
                "public_repos": 0,
                "followers": 0,
                "created_at": "2026-10-23T19:55:00Z",
            }

        rejected = verify_github_player(
            "throwaway2026",
            fetch_online=True,
            http_get=fake_http_get_throwaway,
            current_year=2026,
        )
        self.assertFalse(rejected.verified)
        self.assertIn("zero", (rejected.rejection_reason or "").lower())

    def test_online_verification_falls_back_gracefully_on_network_error(self) -> None:
        def failing_http_get(url: str) -> dict:
            raise TimeoutError("GitHub API rate limit or network timeout")

        fallback = verify_github_player(
            "karpathy",
            fetch_online=True,
            http_get=failing_http_get,
            current_year=2026,
        )
        self.assertTrue(fallback.verified)
        self.assertEqual(fallback.github_login, "karpathy")
        self.assertEqual(fallback.avatar_url, "https://github.com/karpathy.png?size=32")
        self.assertIsNone(fallback.rejection_reason)


class EloLedgerTest(unittest.TestCase):
    def test_reputation_rank_thresholds(self) -> None:
        self.assertEqual(compute_reputation_rank(1150), "Grandmaster Strategist")
        self.assertEqual(compute_reputation_rank(1100), "Grandmaster Strategist")
        self.assertEqual(compute_reputation_rank(1099), "Principal Tactician")
        self.assertEqual(compute_reputation_rank(1030), "Principal Tactician")
        self.assertEqual(compute_reputation_rank(1029), "Verified Challenger")
        self.assertEqual(compute_reputation_rank(1000), "Verified Challenger")
        self.assertEqual(compute_reputation_rank(980), "Verified Challenger")
        self.assertEqual(compute_reputation_rank(979), "Underdog Contender (Staked & Exposed)")
        self.assertEqual(compute_reputation_rank(900), "Underdog Contender (Staked & Exposed)")

    def test_new_verified_player_starts_at_800_and_records_standard_and_conviction_stakes(self) -> None:
        ledger = create_empty_ledger(game_key="connect4")
        self.assertEqual(INITIAL_ELO, 800)
        self.assertEqual(STAKE_TIERS["standard"], 25)
        self.assertEqual(STAKE_TIERS["conviction"], 50)

        alice = verify_github_player("alice-dev")
        bob = verify_github_player("bob-quant")

        entry1 = record_move(
            ledger,
            player=alice,
            game_key="connect4",
            move_label="Drop Col 4",
            stake_tier="standard",
            bot_reply="Bot dropped Col 4",
            issue_number=101,
            timestamp_utc="2026-10-23T18:00:00Z",
        )
        entry2 = record_move(
            ledger,
            player=bob,
            game_key="connect4",
            move_label="Drop Col 3",
            stake_tier="conviction",
            bot_reply="Bot dropped Col 5",
            issue_number=102,
            timestamp_utc="2026-10-23T18:05:00Z",
        )

        self.assertEqual(entry1["turn_number"], 1)
        self.assertEqual(entry1["elo_staked"], 25)
        self.assertEqual(entry1["stake_tier"], "standard")
        self.assertEqual(entry2["turn_number"], 2)
        self.assertEqual(entry2["elo_staked"], 50)
        self.assertEqual(entry2["stake_tier"], "conviction")

        self.assertEqual(ledger["players"]["alice-dev"]["elo"], 800)
        self.assertEqual(ledger["players"]["bob-quant"]["elo"], 800)

        active = get_active_stakes(ledger)
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["github_login"], "bob-quant")
        self.assertEqual(active[0]["elo_staked"], 50)

    def test_settle_round_awards_and_deducts_staked_elo_and_archives_round(self) -> None:
        ledger = create_empty_ledger(game_key="connect4")
        record_move(
            ledger,
            player="alice-dev",
            game_key="connect4",
            move_label="Drop Col 4",
            stake_tier="conviction",
            bot_reply="Bot dropped Col 4",
            issue_number=101,
            timestamp_utc="2026-10-23T18:00:00Z",
        )
        record_move(
            ledger,
            player="bob-quant",
            game_key="connect4",
            move_label="Drop Col 3",
            stake_tier="standard",
            bot_reply="Visitor connected 4!",
            issue_number=102,
            timestamp_utc="2026-10-23T18:02:00Z",
        )

        settle_round(ledger, winner="visitors", settled_at_utc="2026-10-23T18:02:05Z")

        self.assertEqual(ledger["players"]["alice-dev"]["elo"], 850)
        self.assertEqual(ledger["players"]["alice-dev"]["wins"], 1)
        self.assertEqual(ledger["players"]["alice-dev"]["losses"], 0)
        self.assertEqual(ledger["players"]["alice-dev"]["moves_played"], 1)

        self.assertEqual(ledger["players"]["bob-quant"]["elo"], 825)
        self.assertEqual(ledger["players"]["bob-quant"]["wins"], 1)
        self.assertEqual(ledger["players"]["bob-quant"]["moves_played"], 1)

        self.assertEqual(len(ledger["current_round_log"]), 0)
        self.assertEqual(len(ledger["past_rounds"]), 1)
        self.assertEqual(ledger["past_rounds"][0]["winner"], "visitors")

        # Next round: charlie stakes conviction (50 ELO) and loses to bot
        record_move(
            ledger,
            player="charlie-ml",
            game_key="connect4",
            move_label="Drop Col 1",
            stake_tier="conviction",
            bot_reply="Bot connected 4 in Col 4!",
            issue_number=103,
            timestamp_utc="2026-10-23T19:00:00Z",
        )
        settle_round(ledger, winner="bot", settled_at_utc="2026-10-23T19:00:05Z")

        self.assertEqual(ledger["players"]["charlie-ml"]["elo"], 750)
        self.assertEqual(
            ledger["players"]["charlie-ml"]["rank_title"],
            "Underdog Contender (Staked & Exposed)",
        )
        self.assertEqual(ledger["players"]["charlie-ml"]["losses"], 1)
        self.assertEqual(ledger["players"]["charlie-ml"]["moves_played"], 1)

        board = get_sorted_leaderboard(ledger)
        self.assertEqual([p["github_login"] for p in board], ["alice-dev", "bob-quant", "charlie-ml"])


class RenderMarkdownTest(unittest.TestCase):
    def test_render_game_section_md_includes_move_table_stakes_and_hall_of_records(self) -> None:
        ledger = create_empty_ledger(game_key="connect4")
        record_move(
            ledger,
            player="torvalds",
            game_key="connect4",
            move_label="Column 4 (Center)",
            stake_tier="conviction",
            bot_reply="Minimax 4-ply countered in Col 4",
            issue_number=88,
            timestamp_utc="2026-10-23T18:30:00Z",
        )
        settle_round(ledger, winner="visitors", settled_at_utc="2026-10-23T18:35:00Z")

        # Active round move
        record_move(
            ledger,
            player="karpathy",
            game_key="connect4",
            move_label="Column 3",
            stake_tier="standard",
            bot_reply="Minimax 4-ply countered in Col 5",
            issue_number=89,
            timestamp_utc="2026-10-23T19:10:00Z",
        )

        valid_moves = [
            {"id": "1", "label": "Column 1", "note": "Left flank"},
            {"id": "4", "label": "Column 4", "note": "Center file"},
        ]

        md = render_game_section_md(
            ledger=ledger,
            valid_moves=valid_moves,
            repo="Siyuan-Zhu/Siyuan-Zhu",
        )

        # 1. Check 1-Click Move Table links for standard (25 ELO) and conviction (50 ELO)
        self.assertIn("[Play Move (Stake 25 ELO)]", md)
        self.assertIn("title=game:move:1:standard", md)
        self.assertIn("title=game:move:4:standard", md)
        self.assertIn("[🔥 Double Down (Stake 50 ELO)]", md)
        self.assertIn("title=game:move:1:conviction", md)
        self.assertIn("title=game:move:4:conviction", md)

        # 2. Check explicit Identity-Staked Play note
        self.assertIn(
            "Identity-Staked Play: Submitting a move authenticates your GitHub profile (`@username` + avatar) and stakes your ELO on the Hall of Records.",
            md,
        )

        # 3. Check collapsible Hall of Records summary and subsections
        self.assertIn(
            "<details><summary><b>📜 Hall of Records — Active Stakes, Move Log & All-Time ELO Leaderboard</b></summary>",
            md,
        )
        self.assertIn("Active Stakes in Current Round", md)
        self.assertIn("Recent Move Log", md)
        self.assertIn("All-Time Verified GitHub Leaderboard", md)

        # 4. Check 16px avatars and player entries
        self.assertIn('width="16" height="16"', md)
        self.assertIn("https://github.com/karpathy.png?size=32", md)
        self.assertIn("https://github.com/torvalds.png?size=32", md)
        self.assertIn("850", md)
        self.assertNotIn("\x24", md)

    def test_seeded_hall_of_records_json_and_zero_forbidden_patterns(self) -> None:
        self.assertTrue(DEFAULT_LEDGER_PATH.exists())
        ledger = load_ledger(DEFAULT_LEDGER_PATH)
        self.assertIsInstance(ledger["players"], dict)
        # Rank coverage belongs to an explicit fixture, not fabricated public
        # visitor activity. A legitimate launch ledger may be empty.
        rank_fixture = create_empty_ledger()
        rank_fixture["players"] = {
            f"fixture-rank-{index}": {"elo": rating}
            for index, rating in enumerate((1100, 1030, 980, 800))
        }
        board = get_sorted_leaderboard(rank_fixture)
        titles = {row["rank_title"] for row in board}
        self.assertEqual(
            titles,
            {
                "Grandmaster Strategist",
                "Principal Tactician",
                "Verified Challenger",
                "Underdog Contender (Staked & Exposed)",
            },
        )
        rendered = render_game_section_md(ledger)
        self.assertNotIn("\x24", rendered)


if __name__ == "__main__":
    unittest.main()
