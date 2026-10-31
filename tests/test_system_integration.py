"""End-to-end integration and compliance test suite for Siyuan-Zhu Living Digital Artifact."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from deploy.cloud_run_service import (
    CloudRunHandler,
    dispatch_request,
    parse_json_body,
)
from scripts.build_interactive_prototype import build_prototype_html
from scripts.build_readme import (
    FLAGSHIP_PROJECTS,
    VARIANT_FILES,
    VARIANT_ORDERS,
    build_all_readmes,
    main as build_readme_main,
)
from scripts.orbit import (
    cmd_add_source,
    cmd_clear_interjection,
    cmd_interject,
    cmd_status,
    cmd_switch_game,
    main as orbit_main,
)
from scripts.process_visitor_event import process_event
from src.game_engine import (
    VALID_GAMES,
    MoveRequest,
    build_initial_game_state,
    build_midgame_showcase_state,
    process_move,
    render_board_svg,
)
from src.leaderboard.elo_ledger import (
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
from src.leaderboard.identity_verifier import verify_github_player
from src.leaderboard.render_markdown import render_game_section_md
from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator, run_daily_zeitgeist
from src.zeitgeist.methodology import (
    analyze_seven_bucket_orbit,
    default_curated_corpus,
    load_orbit_config,
)
from src.zeitgeist.visual_router import route_and_render_visual


REPO_ROOT = Path(__file__).resolve().parents[1]


def _clone_repo_to_tempdir() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Create an isolated temporary copy of config/, data/, assets/, and variants/ for stateful tests."""
    tmp = tempfile.TemporaryDirectory()
    tmp_root = Path(tmp.name)
    for folder in ("config", "data", "assets", "variants"):
        src_dir = REPO_ROOT / folder
        if src_dir.exists():
            shutil.copytree(src_dir, tmp_root / folder)
    return tmp, tmp_root


class TestSubsystem1ZeitgeistAndVisualRouter(unittest.TestCase):
    """Subsystem 1: Weighted Orbit ADK & Hybrid Visual Router."""

    def test_orbit_sources_weights_and_schema(self) -> None:
        config_path = REPO_ROOT / "config" / "orbit_sources.json"
        self.assertTrue(config_path.exists())
        config = load_orbit_config(config_path)

        lenses = config.get("lenses", {})
        expected_weights = {
            "personal_orbit": 0.40,
            "global_macro": 0.25,
            "markets_finops": 0.20,
            "ai_builders": 0.15,
        }
        self.assertEqual(set(lenses.keys()), set(expected_weights.keys()))
        total_weight = 0.0
        for key, expected in expected_weights.items():
            actual = float(lenses[key]["weight"])
            self.assertAlmostEqual(actual, expected, places=4)
            self.assertGreaterEqual(len(lenses[key].get("sources", [])), 1)
            total_weight += actual
        self.assertAlmostEqual(total_weight, 1.00, places=4)

    def test_analyze_seven_bucket_orbit_determinism_and_fields(self) -> None:
        config = load_orbit_config(REPO_ROOT / "config" / "orbit_sources.json")
        config["active_interjection"]["enabled"] = False
        corpus = default_curated_corpus()

        first = analyze_seven_bucket_orbit(config, corpus, date_iso="2026-10-23")
        second = analyze_seven_bucket_orbit(config, corpus, date_iso="2026-10-23")

        self.assertEqual(first.composite_score, second.composite_score)
        self.assertGreaterEqual(first.composite_score, 0.0)
        self.assertLessEqual(first.composite_score, 100.0)
        self.assertEqual(first.word_of_the_day, second.word_of_the_day)
        self.assertTrue(first.word_of_the_day)
        self.assertTrue(first.explanation_phrase)
        self.assertTrue(first.visual_concrete_anchor)
        self.assertEqual(len(first.lens_breakdown), 4)
        self.assertGreaterEqual(len(first.dominant_keywords), 3)

    def test_visual_router_all_three_modes_and_svg_output(self) -> None:
        base_config = load_orbit_config(REPO_ROOT / "config" / "orbit_sources.json")
        base_config["active_interjection"]["enabled"] = False
        corpus = default_curated_corpus()
        analysis = analyze_seven_bucket_orbit(base_config, corpus, date_iso="2026-10-23")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. CURATED_PHOTO mode
            svg_photo = tmp_path / "pulse_photo.svg"
            dec_photo = route_and_render_visual(
                analysis=analysis,
                config=base_config,
                svg_output_path=svg_photo,
                prefer_mode="photo",
            )
            self.assertEqual(dec_photo.selected_mode, "CURATED_PHOTO")
            self.assertTrue(dec_photo.safety_pass)
            photo_text = svg_photo.read_text(encoding="utf-8")
            self.assertIn("<svg", photo_text)
            self.assertIn("</svg>", photo_text)
            self.assertIn(analysis.word_of_the_day, photo_text)
            ET.fromstring(photo_text)

            # 2. ARCHITECTURAL_BLUEPRINT_SVG mode
            svg_blueprint = tmp_path / "pulse_blueprint.svg"
            dec_blueprint = route_and_render_visual(
                analysis=analysis,
                config=base_config,
                svg_output_path=svg_blueprint,
                prefer_mode="blueprint",
            )
            self.assertEqual(dec_blueprint.selected_mode, "ARCHITECTURAL_BLUEPRINT_SVG")
            self.assertTrue(dec_blueprint.safety_pass)
            bp_text = svg_blueprint.read_text(encoding="utf-8")
            self.assertIn("<svg", bp_text)
            self.assertIn("</svg>", bp_text)
            self.assertIn("40% PERSONAL ORBIT", bp_text)
            ET.fromstring(bp_text)

            # 3. AI_GENERATED_FALLBACK mode (non-builder day + high alignment threshold)
            fallback_config = copy.deepcopy(base_config)
            fallback_config.setdefault("budget_guardrails", {})["ai_image_fallback_threshold"] = 0.99
            non_builder_analysis = replace(analysis, is_builder_systems_day=False)
            svg_fallback = tmp_path / "pulse_fallback.svg"
            dec_fallback = route_and_render_visual(
                analysis=non_builder_analysis,
                config=fallback_config,
                svg_output_path=svg_fallback,
                prefer_mode=None,
            )
            self.assertEqual(dec_fallback.selected_mode, "AI_GENERATED_FALLBACK")
            self.assertTrue(dec_fallback.safety_pass)
            fb_text = svg_fallback.read_text(encoding="utf-8")
            self.assertIn("<svg", fb_text)
            self.assertIn("</svg>", fb_text)
            ET.fromstring(fb_text)

    def test_adk_orchestrator_budget_guardrails(self) -> None:
        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)

        orchestrator = ZeitgeistADKOrchestrator(repo_root=tmp_root)
        analysis, decision, telemetry = orchestrator.run(date_iso="2026-10-23", prefer_visual_mode=None)

        self.assertFalse(telemetry.within_budget)
        self.assertIsNone(telemetry.estimated_monthly_cost_index)
        self.assertEqual(telemetry.spending_status, "unknown_pending_billing_reconciliation")
        self.assertEqual(decision.selected_mode, "BESPOKE_SVG")
        self.assertEqual(decision.word_of_the_day, orchestrator.last_published_pulse.word)
        self.assertTrue((tmp_root / "assets" / "daily_pulse.svg").exists())
        self.assertTrue((tmp_root / "data" / "zeitgeist_state.json").exists())
        self.assertTrue(analysis.word_of_the_day)

    def test_collect_96h_signals_live_collect_wires_stage1_collectors_and_persists_cache(self) -> None:
        from unittest.mock import patch
        from src.state_store import InMemoryStateStore
        from src.zeitgeist.adk_pipeline import collect_96h_signals
        from src.zeitgeist.collectors import compute_url_hash

        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)
        store = InMemoryStateStore()

        curated_item = {
            "url_hash": compute_url_hash("https://simonwillison.net/2026/Oct/27/live-1"),
            "url": "https://simonwillison.net/2026/Oct/27/live-1",
            "title": "Live Curated Feed Item with @scout_alpha and @scout_beta",
            "summary": "Systems architecture synthesis with @scout_alpha",
            "source_id": "o1_simon_willison",
            "origin": "curated_O1_O3",
            "tier": "O1",
            "source_tier": "O1",
            "bucket": "ORBIT",
            "weight": 2.0,
            "orbit_weight": 2.0,
            "published_at": "2026-10-27T10:00:00Z",
            "mentioned_handles": ["scout_alpha", "scout_beta"],
        }
        ig_item = {
            "url_hash": compute_url_hash("https://www.instagram.com/p/live_ig_1"),
            "url": "https://www.instagram.com/p/live_ig_1",
            "title": "Instagram Orbit Dispatch",
            "summary": "Craft notes with @scout_alpha",
            "source_id": "o3_ig_hubermanlab",
            "origin": "curated_O1_O3",
            "tier": "O3",
            "source_tier": "O3",
            "bucket": "ORBIT",
            "weight": 1.3,
            "orbit_weight": 1.3,
            "published_at": "2026-10-27T11:00:00Z",
            "fetched_at": "2026-10-27T11:00:00Z",
            "mentioned_handles": ["scout_alpha"],
        }
        macro_item = {
            "url_hash": compute_url_hash("https://news.ycombinator.com/item?id=999"),
            "url": "https://news.ycombinator.com/item?id=999",
            "title": "Macro Scout Breakthrough",
            "summary": "High velocity front-page signal",
            "source_id": "m4_scout_hn_frontpage",
            "tier": "M4_SCOUT",
            "source_tier": "MACRO",
            "bucket": "MACRO",
            "weight": 1.0,
            "orbit_weight": 1.0,
            "published_at": "2026-10-27T11:30:00Z",
        }

        # A successful live fixture must satisfy the actual diversified source contract.
        curated_pool = []
        for bucket, prefix, tiers, template in (
            ("ORBIT", "o", ("O1", "O2", "O3"), curated_item),
            ("MACRO", "m", ("M1", "M2", "M3"), macro_item),
        ):
            for i in range(6):
                row = dict(template, source_id=f"{prefix}_fixture_{i}", bucket=bucket,
                           tier=tiers[i % 3], source_tier=tiers[i % 3],
                           url=f"https://example.org/{prefix}/entry-{i}")
                row["url_hash"] = compute_url_hash(row["url"])
                curated_pool.append(row)

        from types import SimpleNamespace
        embedding_client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kwargs:
            SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * 768) for _ in kwargs["contents"]])))
        with (
            patch.dict(os.environ, {"APIFY_TOKEN": "fixture-token-no-network"}),
            patch("src.zeitgeist.adk_pipeline.fetch_all_curated_feeds", return_value=curated_pool) as mock_feeds,
            patch("src.zeitgeist.adk_pipeline.fetch_instagram_batch", return_value=[ig_item]) as mock_ig,
            patch("src.zeitgeist.adk_pipeline.fetch_macro_scout_signals", return_value=[macro_item]) as mock_macro,
        ):
            signals = collect_96h_signals(
                tmp_root,
                date_iso="2026-10-27",
                store=store,
                live_collect=True,
                genai_client=embedding_client,
                now_epoch=1793102400,
            )

        self.assertEqual(mock_feeds.call_count, 1)
        self.assertEqual(mock_ig.call_count, 1)
        self.assertEqual(mock_macro.call_count, 1)
        self.assertEqual(len(signals), 14)
        for sig in signals:
            self.assertEqual(len(sig.url_hash), 32)
            self.assertEqual(len(sig.embedding_f16), 1536)

        # Verify the live one-hop roster admitted the curated mention and saved its cache shard.
        saved_roster = store.load_orbit_roster()
        self.assertIn("scout_alpha", saved_roster.get("handles", {}))
        self.assertIn("2026-10-27", store._ingestion_cache_shards)
        self.assertEqual(len(store._ingestion_cache_shards["2026-10-27"]["items"]), 14)


class TestSubsystem2Connect4GameEngine(unittest.TestCase):
    """Subsystem 2: Single-Engine Connect 4 Arcade & Strategy Verification."""

    def test_connect4_engine_lifecycle_and_svg(self) -> None:
        self.assertEqual(VALID_GAMES, ("connect4",))
        state = build_initial_game_state("connect4", round_id=1)
        self.assertEqual(state["active_game"], "connect4")

        outcome = process_move(
            state,
            MoveRequest(
                game_key="connect4",
                move_id="col4",
                player_login="octocat",
                expected_turn=state["turn"],
                timestamp_epoch=1000.0,
            ),
        )
        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.new_state["turn"], 2)

        svg_content = render_board_svg(outcome.new_state)
        self.assertIn('viewBox="0 0 580 360"', svg_content)
        self.assertIn("<svg", svg_content)
        self.assertIn("</svg>", svg_content)
        ET.fromstring(svg_content)

    def test_game_completion_settles_round_and_auto_resets_via_event_processor(self) -> None:
        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)

        # Configure Connect 4 with 3 challenger discs in bottom row (cols 1..3)
        c4_state = build_initial_game_state("connect4", round_id=4)
        c4_state["board"][5][0] = 1
        c4_state["board"][5][1] = 1
        c4_state["board"][5][2] = 1
        turn = c4_state["turn"]
        (tmp_root / "data" / "game_state.json").write_text(
            json.dumps(c4_state, indent=2) + "\n",
            encoding="utf-8",
        )

        # Challenger plays winning move col4
        res = process_event(
            repo_root=tmp_root,
            visitor_login="torvalds",
            issue_title=f"game:move:connect4:col4:turn={turn}",
            issue_number=201,
            fetch_online=False,
        )
        self.assertEqual(res["status"], "move_applied")
        self.assertTrue(res["round_over"])

        # Verify Victory Board Showcase preserves the 4 winning discs until next opening move
        updated_game = json.loads(
            (tmp_root / "data" / "game_state.json").read_text(encoding="utf-8")
        )
        new_board = updated_game["board"]
        total_pieces = sum(1 for r in range(6) for c in range(7) if new_board[r][c] not in (0, "."))
        self.assertEqual(total_pieces, 4)
        self.assertEqual(len(updated_game["winning_cells"]), 4)
        self.assertEqual(updated_game["status"], "COMPLETED")


class TestSubsystem3IdentityEloAndPulsePoll(unittest.TestCase):
    """Subsystem 3: Verified GitHub Identity, Elo Staking & Mood Poll."""

    def test_identity_verifier_gates_throwaway_and_bot_accounts(self) -> None:
        # Bot & malformed handles rejected
        bot_id = verify_github_player("depend-bot[bot]", fetch_online=False)
        self.assertFalse(bot_id.verified)

        # Throwaway zero-activity account rejected in online check
        def fake_throwaway(_url: str) -> dict:
            return {
                "login": "newbie2026",
                "type": "User",
                "public_repos": 0,
                "followers": 0,
                "created_at": "2026-10-20T00:00:00Z",
            }

        young_id = verify_github_player(
            "newbie2026",
            fetch_online=True,
            http_get=fake_throwaway,
            current_year=2026,
        )
        self.assertFalse(young_id.verified)

        # Established human account accepted
        def fake_human(_url: str) -> dict:
            return {
                "login": "octocat",
                "type": "User",
                "public_repos": 12,
                "followers": 950,
                "created_at": "2011-01-25T18:44:36Z",
            }

        human_id = verify_github_player(
            "octocat",
            fetch_online=True,
            http_get=fake_human,
            current_year=2026,
        )
        self.assertTrue(human_id.verified)
        self.assertEqual(human_id.avatar_url, "https://github.com/octocat.png?size=32")

    def test_elo_ledger_staked_tiers_settlement_and_markdown(self) -> None:
        ledger = create_empty_ledger(game_key="connect4")
        self.assertEqual(INITIAL_ELO, 800)
        self.assertEqual(STAKE_TIERS["standard"], 25)
        self.assertEqual(STAKE_TIERS["conviction"], 50)

        record_move(
            ledger,
            player="alice-quant",
            game_key="connect4",
            move_label="Drop Col 3",
            stake_tier="conviction",
            bot_reply="Bot dropped Col 4",
            issue_number=11,
        )
        record_move(
            ledger,
            player="bob-systems",
            game_key="connect4",
            move_label="Drop Col 4",
            stake_tier="standard",
            bot_reply="Bot dropped Col 5",
            issue_number=12,
        )
        self.assertEqual(len(get_active_stakes(ledger)), 2)

        settle_round(ledger, winner="challengers")
        self.assertEqual(ledger["players"]["alice-quant"]["elo"], 850)
        self.assertEqual(compute_reputation_rank(1050), "Principal Tactician")
        self.assertEqual(ledger["players"]["bob-systems"]["elo"], 825)

        board = get_sorted_leaderboard(ledger)
        self.assertEqual(board[0]["github_login"], "alice-quant")

        md = render_game_section_md(
            ledger=ledger,
            valid_moves=[{"id": "col4", "label": "Col IV", "note": "Center"}],
            repo="Siyuan-Zhu/Siyuan-Zhu",
        )
        self.assertIn("Hall of Records", md)
        self.assertIn("title=game:move:col4:standard", md)
        self.assertIn("title=game:move:col4:conviction", md)
        self.assertIn("alice-quant", md)


class TestSubsystem4MobileControlPlaneCloudRunAndAssembler(unittest.TestCase):
    """Subsystem 4: Mobile Control Plane, Cloud Run Service & Master Assembler."""

    def test_orbit_cli_interject_clear_add_source_and_switch_game(self) -> None:
        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)

        interject_args = argparse.Namespace(
            word="ANCHORED",
            phrase="Coffee in Williamsburg before a systems review.",
            note="Williamsburg field dispatch",
            mode="photo",
            image_url=None,
            channel="unit_test_mobile",
        )
        # The photo-provider success contract is covered in test_manual_publication_control.
        # This integration path exercises a real deterministic publication without network.
        interject_args.mode = "blueprint"
        rc = cmd_interject(tmp_root, interject_args)
        self.assertEqual(rc, 0)
        cfg = load_orbit_config(tmp_root / "config" / "orbit_sources.json")
        self.assertTrue(cfg["active_interjection"]["enabled"])
        self.assertEqual(cfg["active_interjection"]["override_word"], "ANCHORED")
        self.assertEqual(cfg["active_interjection"]["override_visual_mode"], "blueprint")

        zstate = json.loads((tmp_root / "data" / "zeitgeist_state.json").read_text(encoding="utf-8"))
        self.assertEqual(zstate["analysis"]["word_of_the_day"], "ANCHORED")
        self.assertEqual(zstate["visual_decision"]["selected_mode"], "BESPOKE_SVG")

        rc_clear = cmd_clear_interjection(tmp_root)
        self.assertEqual(rc_clear, 0)
        cfg_after = load_orbit_config(tmp_root / "config" / "orbit_sources.json")
        self.assertFalse(cfg_after["active_interjection"]["enabled"])

        add_args = argparse.Namespace(
            lens="personal_orbit",
            id="test_builder_feed",
            name="Test Builder Feed",
            uri="https://example.com/feed.xml",
            category="builder_creator",
            tags="architecture,craft",
        )
        self.assertEqual(cmd_add_source(tmp_root, add_args), 0)
        self.assertEqual(cmd_status(tmp_root), 0)

        # switch-game connect4 (--mode pvp and --mode ai)
        self.assertEqual(
            cmd_switch_game(
                tmp_root,
                argparse.Namespace(game_key="connect4", mode="pvp"),
            ),
            0,
        )
        pvp_state = json.loads(
            (tmp_root / "data" / "game_state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(pvp_state["active_game"], "connect4")
        self.assertEqual(pvp_state["mode"], "PVP_DUEL")

        self.assertEqual(
            cmd_switch_game(
                tmp_root,
                argparse.Namespace(game_key="connect4", mode="ai"),
            ),
            0,
        )
        active_state = json.loads(
            (tmp_root / "data" / "game_state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(active_state["active_game"], "connect4")
        self.assertEqual(active_state["mode"], "VS_BOT")

        # Reject retired games via both cmd_switch_game and CLI parser
        for retired_game in ("equilibrium", "chess_tactics", "word_guess", "twenty_questions", "battleship"):
            with self.assertRaises(SystemExit):
                cmd_switch_game(
                    tmp_root,
                    argparse.Namespace(game_key=retired_game, mode=None),
                )
            with self.assertRaises(SystemExit):
                orbit_main(["switch-game", retired_game])

    def test_github_workflows_mobile_and_community_guards(self) -> None:
        mobile_wf = (REPO_ROOT / ".github" / "workflows" / "mobile_interject.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("pip install -r requirements.txt", mobile_wf)
        self.assertIn("python3 scripts/build_readme.py", mobile_wf)
        self.assertIn("GCP_PROJECT_ID", mobile_wf)
        self.assertIn("FIRESTORE_DATABASE_ID", mobile_wf)
        self.assertIn("STATE_STORE_BACKEND", mobile_wf)
        for retired in ("word_guess", "twenty_questions", "equilibrium", "chess_tactics", "battleship", "wordle"):
            self.assertNotIn(retired, mobile_wf)

        comm_wf = (REPO_ROOT / ".github" / "workflows" / "community_game_move.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("pip install -r requirements.txt", comm_wf)
        self.assertIn("python3 scripts/build_readme.py", comm_wf)
        self.assertIn("orbit:", comm_wf)

    def test_staging_zeitgeist_dryrun_workflow_isolation_and_guards(self) -> None:
        staging_wf_path = REPO_ROOT / ".github" / "workflows" / "staging_zeitgeist_dryrun.yml"
        self.assertTrue(staging_wf_path.is_file(), "staging_zeitgeist_dryrun.yml must exist")
        staging_wf = staging_wf_path.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", staging_wf)
        self.assertNotIn("schedule:", staging_wf)
        self.assertNotIn("push:", staging_wf)
        for inp in ("live_collect", "live_llm", "live_unsplash", "live_firestore"):
            self.assertIn(inp, staging_wf)
        self.assertIn("APIFY_INSTAGRAM_BUILD", staging_wf)
        self.assertIn("contents: read", staging_wf)
        self.assertNotIn("contents: write", staging_wf)
        self.assertIn("/tmp/zeitgeist-staging-sandbox", staging_wf)
        self.assertIn("ZEITGEIST_STATE_DIR: /tmp/zeitgeist-staging-state", staging_wf)
        self.assertIn('STATE_STORE_BACKEND: "json"', staging_wf)
        self.assertNotIn("MAX_DAILY_COST_USD", staging_wf)
        self.assertIn('MAX_STAGE_RETRIES: "1"', staging_wf)
        self.assertIn("pytest -q", staging_wf)
        self.assertIn("scripts/run_daily_zeitgeist.py", staging_wf)
        self.assertIn("scripts/build_readme.py --check", staging_wf)
        self.assertIn("staging-zeitgeist-dryrun-bundle", staging_wf)
        self.assertNotIn("git push", staging_wf)
        self.assertNotIn("git_commit_and_push_with_retry", staging_wf)
        self.assertNotIn('STATE_STORE_BACKEND: "firestore"', staging_wf)

        daily_script_path = REPO_ROOT / "scripts" / "run_daily_zeitgeist.py"
        self.assertTrue(daily_script_path.is_file(), "scripts/run_daily_zeitgeist.py must exist")

        from scripts.run_daily_zeitgeist import main as run_daily_main

        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)
        staging_state_dir = tmp_root / "staging_state_out"
        prev_state_dir = os.environ.get("ZEITGEIST_STATE_DIR")
        prev_backend = os.environ.get("STATE_STORE_BACKEND")
        prev_live_llm = os.environ.get("ZEITGEIST_LIVE_LLM")
        prev_live_collect = os.environ.get("ZEITGEIST_LIVE_COLLECT")
        os.environ["ZEITGEIST_STATE_DIR"] = str(staging_state_dir)
        os.environ["STATE_STORE_BACKEND"] = "json"
        os.environ["ZEITGEIST_LIVE_LLM"] = "0"
        os.environ["ZEITGEIST_LIVE_COLLECT"] = "0"
        try:
            rc = run_daily_main(["--repo-root", str(tmp_root), "--date", "2026-04-20"])
            self.assertEqual(rc, 0)
            self.assertTrue((staging_state_dir / "zeitgeist_state.json").is_file())
            self.assertTrue((staging_state_dir / "game_state.json").is_file())
        finally:
            for k, prev_v in (
                ("ZEITGEIST_STATE_DIR", prev_state_dir),
                ("STATE_STORE_BACKEND", prev_backend),
                ("ZEITGEIST_LIVE_LLM", prev_live_llm),
                ("ZEITGEIST_LIVE_COLLECT", prev_live_collect),
            ):
                if prev_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev_v

    def test_process_visitor_event_end_to_end(self) -> None:
        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)

        c4_state = build_midgame_showcase_state("connect4", round_id=4)
        (tmp_root / "data" / "game_state.json").write_text(
            json.dumps(c4_state, indent=2) + "\n",
            encoding="utf-8",
        )

        move_result = process_event(
            repo_root=tmp_root,
            visitor_login="octocat",
            issue_title=f"game:move:connect4:col4:turn={c4_state['turn']}",
            issue_number=101,
            fetch_online=False,
        )
        self.assertEqual(move_result["status"], "move_applied")
        self.assertEqual(move_result["player"], "octocat")

        vote_result = process_event(
            repo_root=tmp_root,
            visitor_login="octocat",
            issue_title="pulse:vote:spot_on",
            issue_number=102,
            fetch_online=False,
        )
        self.assertEqual(vote_result["status"], "recorded_vote")
        self.assertEqual(vote_result["vote"], "spot_on")

        # Owner orbit:add/remove:instagram:<handle> + handle_blocklist enforcement
        blocked_res = process_event(
            repo_root=tmp_root,
            visitor_login="Siyuan-Zhu",
            issue_title="orbit:add:instagram:instagram",
            issue_number=103,
            fetch_online=False,
        )
        self.assertEqual(blocked_res["status"], "ignored")

        add_ig_res = process_event(
            repo_root=tmp_root,
            visitor_login="Siyuan-Zhu",
            issue_title="orbit:add:instagram:studio_curation_test",
            issue_number=104,
            fetch_online=False,
        )
        self.assertEqual(add_ig_res["status"], "orbit_add_instagram")
        self.assertEqual(add_ig_res["handle"], "studio_curation_test")
        cfg_after_add = json.loads(
            (tmp_root / "config" / "orbit_sources.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "studio_curation_test", cfg_after_add["curated_o3_instagram_handles"]
        )
        roster_after_add = json.loads(
            (tmp_root / "data" / "orbit_roster.json").read_text(encoding="utf-8")
        )
        self.assertIn("studio_curation_test", roster_after_add["handles"])

        rem_ig_res = process_event(
            repo_root=tmp_root,
            visitor_login="Siyuan-Zhu",
            issue_title="orbit:remove:instagram:studio_curation_test",
            issue_number=105,
            fetch_online=False,
        )
        self.assertEqual(rem_ig_res["status"], "orbit_remove_instagram")
        cfg_after_rem = json.loads(
            (tmp_root / "config" / "orbit_sources.json").read_text(encoding="utf-8")
        )
        self.assertNotIn(
            "studio_curation_test", cfg_after_rem["curated_o3_instagram_handles"]
        )
        roster_after_rem = json.loads(
            (tmp_root / "data" / "orbit_roster.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("studio_curation_test", roster_after_rem["handles"])

    def test_cloud_run_service_dispatch_request(self) -> None:
        from unittest.mock import patch
        tmp, isolated_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)
        self.enterContext(patch("deploy.cloud_run_service.REPO_ROOT", isolated_root))
        self.enterContext(patch("scripts.orbit.REPO_ROOT", isolated_root))

        # 1. /healthz
        code_h, body_h = dispatch_request("/healthz", {}, "", "")
        self.assertEqual(code_h, 200)
        self.assertEqual(body_h["status"], "ok")

        # 2a. Fail-closed when ORBIT_WEBHOOK_TOKEN / expected_token is unset -> 401/503
        for protected_path in ("/interject", "/daily-cron", "/daily-pulse"):
            code_unset, body_unset = dispatch_request(
                protected_path,
                {"word": "FOCUS"},
                "",
                "",
            )
            self.assertIn(code_unset, (401, 503))
            self.assertEqual(body_unset["status"], "error")
            self.assertEqual(
                body_unset.get("error"),
                "unauthorized_or_unconfigured_webhook_token",
            )

        # 2b. Unauthorized /interject (wrong Bearer token) -> 401
        code_unauth, body_unauth = dispatch_request(
            "/interject",
            {"word": "FOCUS"},
            "Bearer wrong-token",
            "secret-orbit-token",
        )
        self.assertEqual(code_unauth, 401)
        self.assertEqual(body_unauth["status"], "error")
        self.assertEqual(
            body_unauth.get("error"),
            "unauthorized_or_unconfigured_webhook_token",
        )

        # 2c. Payload size cap (> 65536 bytes -> 413) and malformed JSON (-> 400)
        code_large, body_large = parse_json_body(b"x" * 65537, content_length=65537)
        self.assertEqual(code_large, 413)
        self.assertEqual(body_large["status"], "error")
        self.assertEqual(body_large["error"], "payload_too_large")

        code_bad_json, body_bad_json = parse_json_body(b"{malformed json", content_length=15)
        self.assertEqual(code_bad_json, 400)
        self.assertEqual(body_bad_json["status"], "error")
        self.assertEqual(body_bad_json["error"], "invalid_json")

        # Verify CloudRunHandler.do_POST directly for 413 and 400 without crashing
        import io
        from unittest.mock import MagicMock

        def _invoke_do_post(raw_body: bytes, content_length: int) -> tuple[int, dict]:
            handler = CloudRunHandler.__new__(CloudRunHandler)
            handler.path = "/interject"
            handler.headers = {
                "Content-Length": str(content_length),
                "Authorization": "Bearer secret-orbit-token",
            }
            handler.rfile = io.BytesIO(raw_body)
            handler.wfile = io.BytesIO()
            sent_codes: list[int] = []
            handler.send_response = lambda code, *_a: sent_codes.append(code)  # type: ignore[assignment]
            handler.send_header = MagicMock()  # type: ignore[assignment]
            handler.end_headers = MagicMock()  # type: ignore[assignment]
            handler.do_POST()
            return sent_codes[0], json.loads(handler.wfile.getvalue().decode("utf-8"))

        post_413_code, post_413_body = _invoke_do_post(b"{}", 65537)
        self.assertEqual(post_413_code, 413)
        self.assertEqual(post_413_body["error"], "payload_too_large")

        post_400_code, post_400_body = _invoke_do_post(b"{not-valid-json", 15)
        self.assertEqual(post_400_code, 400)
        self.assertEqual(post_400_body["error"], "invalid_json")

        # 3. Authorized /interject -> 200
        code_int, body_int = dispatch_request(
            "/interject",
            {
                "word": "CONVERGENCE",
                "phrase": "Systems balance across four lenses.",
                "mode": "blueprint",
            },
            "Bearer secret-orbit-token",
            "secret-orbit-token",
        )
        self.assertEqual(code_int, 200)
        self.assertEqual(body_int["word_of_the_day"], "CONVERGENCE")

        # 4. Authorized /daily-pulse and /daily-cron -> 200
        code_dp, body_dp = dispatch_request(
            "/daily-pulse",
            {},
            "Bearer secret-orbit-token",
            "secret-orbit-token",
        )
        self.assertEqual(code_dp, 200)
        self.assertEqual(body_dp["status"], "ok")
        self.assertFalse(body_dp["within_budget"])

    def test_build_readme_and_role_variants(self) -> None:
        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)

        build_all_readmes(tmp_root)
        readme_path = tmp_root / "README.md"
        self.assertTrue(readme_path.exists())
        readme_text = readme_path.read_text(encoding="utf-8")
        self.assertNotIn("Selected Work (Role-Tailored)", readme_text)

        proj_by_id = {p["id"]: p for p in FLAGSHIP_PROJECTS}
        role_texts: dict[str, str] = {}

        for var_key in ("ai-engineering", "fpa-analytics", "bi-data"):
            self.assertIn(var_key, VARIANT_ORDERS)
            v_path = tmp_root / "variants" / f"README-{var_key}.md"
            self.assertTrue(v_path.exists(), f"Missing variant {var_key}")
            v_text = v_path.read_text(encoding="utf-8")
            self.assertGreater(len(v_text), 500)
            self.assertIn("Selected Work (Role-Tailored)", v_text)
            positions = [v_text.index(proj_by_id[pid]["name"]) for pid in VARIANT_ORDERS[var_key]]
            self.assertEqual(
                positions,
                sorted(positions),
                f"Projects in {var_key} variant do not match VARIANT_ORDERS[{var_key}]",
            )
            role_texts[var_key] = v_text

        self.assertEqual(
            len(set(role_texts.values())),
            3,
            "Each role variant must render a distinct role-ordered project sequence",
        )

        for short_key, filename in VARIANT_FILES.items():
            v_path = tmp_root / "variants" / filename
            self.assertTrue(v_path.exists(), f"Missing variant {short_key}: {filename}")
            v_text = v_path.read_text(encoding="utf-8")
            self.assertGreater(len(v_text), 500)
            order_key = filename.replace("README.", "").replace(".md", "")
            self.assertIn("Selected Work (Role-Tailored)", v_text)
            positions = [v_text.index(proj_by_id[pid]["name"]) for pid in VARIANT_ORDERS[order_key]]
            self.assertEqual(
                positions,
                sorted(positions),
                f"Projects in {order_key} variant do not match VARIANT_ORDERS[{order_key}]",
            )

        self.assertIn("./assets/daily_pulse.svg", readme_text)
        self.assertIn("./assets/game_board.svg", readme_text)
        self.assertIn("title=game:move:", readme_text)
        self.assertTrue("/vote?dir=" in readme_text or "title=pulse:vote:" in readme_text)
        self.assertEqual(build_readme_main(["--check", "--repo-root", str(tmp_root)]), 0)
        (tmp_root / "variants" / "README-ai-engineering.md").write_text("stale", encoding="utf-8")
        self.assertEqual(build_readme_main(["--check", "--repo-root", str(tmp_root)]), 1)

    def test_build_interactive_prototype_html(self) -> None:
        tmp, tmp_root = _clone_repo_to_tempdir()
        self.addCleanup(tmp.cleanup)

        out_path = build_prototype_html(tmp_root)
        self.assertTrue(out_path.exists())
        html_text = out_path.read_text(encoding="utf-8")

        self.assertIn("CONVERGENCE", html_text)
        self.assertIn("README.md", html_text)
        self.assertIn("PROJECTS.md", html_text)
        self.assertIn("Connect 4", html_text)
        self.assertIn("How it works", html_text)


class TestConfidentialityAndZeroDollarCompliance(unittest.TestCase):
    """Confidentiality & Zero-Dollar Compliance Audit across all repository text files."""

    def test_confidentiality_and_zero_dollar_compliance(self) -> None:
        dollar_char = chr(36)
        # Construct forbidden terms dynamically so this test file itself contains zero literal matches
        forbidden_terms = [
            "cross" + "walk",
            "chur" + "ch",
            "mis" + "sion",
            "do" + "nor",
            "pas" + "tor",
            "xe" + "ro",
            "bill" + ".com",
            "div" + "vy",
            "tru" + "ist",
            "over" + "flow",
            "planning " + "center",
            "pfi" + "zer",
        ]

        candidate_files: list[Path] = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "prototype-profile-readme.html",
            REPO_ROOT / "prototype-community-game.html",
        ]
        glob_patterns = [
            "variants/*.md",
            "config/*.json",
            "data/*.json",
            "assets/*.svg",
            "src/**/*.py",
            "scripts/*.py",
            "deploy/*",
            ".github/workflows/*.yml",
            "tests/*.py",
        ]
        for pattern in glob_patterns:
            candidate_files.extend(sorted(REPO_ROOT.glob(pattern)))

        checked_count = 0
        for file_path in candidate_files:
            if not file_path.is_file() or "__pycache__" in file_path.parts:
                continue
            checked_count += 1
            rel_path = file_path.relative_to(REPO_ROOT)
            content = file_path.read_text(encoding="utf-8")

            check_text = content
            if file_path.suffix == ".yml":
                gh_expr = re.escape(dollar_char) + r"\{\{[^}]+\}\}"
                check_text = re.sub(gh_expr, "", check_text)

            self.assertNotIn(
                dollar_char,
                check_text,
                f"Forbidden literal dollar character found in {rel_path}",
            )

            lowered = content.lower()
            if rel_path == Path("config/orbit_sources.json"):
                lowered = lowered.replace("life." + "chur" + "ch", "life_ig_seed")
                approved_seed = "cross" + "walk." + "chur" + "ch"
                lowered = lowered.replace(json.dumps(approved_seed), '"approved_ig_seed"')
            for term in forbidden_terms:
                pattern = rf"\b{re.escape(term)}s?\b"
                self.assertIsNone(
                    re.search(pattern, lowered),
                    f"Forbidden confidential string '{term}' found in {rel_path}",
                )

        self.assertGreaterEqual(checked_count, 25)


if __name__ == "__main__":
    unittest.main()
