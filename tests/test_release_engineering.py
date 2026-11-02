"""Release engineering tests for Ticket R-01:
- Flagship repository link remediation (`finance-analytics-warehouse`, `oecd-aid-finance-llm-classifier`, `computer-vision-document-motion-convnet`)
- Configurable repository slug routing (`resolve_repo_slug`, `TARGET_REPO_SLUG`, `--repo-slug`)
- Automated release link and asset validator CLI (`scripts/validate_release_links.py`)
"""

from __future__ import annotations

from pathlib import Path
import shutil
import pytest

from scripts import build_readme, process_visitor_event
from scripts.build_readme import FLAGSHIP_PROJECTS, VARIANT_ORDERS, render_profile_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_flagship_projects_and_docs_replace_dead_gl_variance_link() -> None:
    by_id = {p["id"]: p for p in FLAGSHIP_PROJECTS}
    assert "gl-variance" in by_id
    gl_entry = by_id["gl-variance"]
    assert gl_entry["name"] == "finance-analytics-warehouse"
    assert gl_entry["url"] == "https://github.com/Siyuan-Zhu/finance-analytics-warehouse"
    assert "DuckDB" in gl_entry["domain"] and "dbt" in gl_entry["domain"]
    assert "34 monthly closes" in gl_entry["summary"]
    assert "116 automated SQL and Excel checks" in gl_entry["summary"]
    assert gl_entry["stack"] == "Python · DuckDB · dbt · SQL · openpyxl"

    for order_list in VARIANT_ORDERS.values():
        assert "gl-variance" in order_list

    variant_md = render_profile_markdown(
        REPO_ROOT,
        variant="fpa-analytics",
        write_assets=False,
    )
    assert "https://github.com/Siyuan-Zhu/finance-analytics-warehouse" in variant_md
    assert "gl-variance-and-run-rate-pipeline" not in variant_md

    projects_md = (REPO_ROOT / "PROJECTS.md").read_text(encoding="utf-8")
    assert "https://github.com/Siyuan-Zhu/finance-analytics-warehouse" in projects_md
    assert "gl-variance-and-run-rate-pipeline" not in projects_md
    assert "FLAN-UL2" in projects_md
    assert "SBERT" in projects_md
    assert "Vote-K" in projects_md
    assert "96.84%" in projects_md
    assert "https://github.com/Siyuan-Zhu/computer-vision-document-motion-convnet" in projects_md

    proto_py = (REPO_ROOT / "scripts" / "build_interactive_prototype.py").read_text(encoding="utf-8")
    assert "gl-variance-and-run-rate-pipeline" not in proto_py
    assert "https://github.com/Siyuan-Zhu/finance-analytics-warehouse" in proto_py


def test_resolve_repo_slug_precedence_and_build_check_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TARGET_REPO_SLUG", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert build_readme.resolve_repo_slug(None) == "Siyuan-Zhu/Siyuan-Zhu"
    assert build_readme.resolve_repo_slug("") == "Siyuan-Zhu/Siyuan-Zhu"

    monkeypatch.setenv("GITHUB_REPOSITORY", "Siyuan-Zhu/readme-live-sandbox")
    assert build_readme.resolve_repo_slug(None) == "Siyuan-Zhu/readme-live-sandbox"

    monkeypatch.setenv("TARGET_REPO_SLUG", "Siyuan-Zhu/Siyuan-Zhu")
    assert build_readme.resolve_repo_slug(None) == "Siyuan-Zhu/Siyuan-Zhu"
    assert build_readme.resolve_repo_slug("Custom-Org/Custom-Repo") == "Custom-Org/Custom-Repo"

    # Set up isolated workspace copy for build_all_readmes and check_all_readmes
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    shutil.copytree(REPO_ROOT / "assets", tmp_path / "assets")

    # Even with GITHUB_REPOSITORY=Siyuan-Zhu/readme-live-sandbox, explicit --repo-slug wins
    monkeypatch.delenv("TARGET_REPO_SLUG", raising=False)
    rc_build = build_readme.main(["--repo-root", str(tmp_path), "--repo-slug", "Siyuan-Zhu/Siyuan-Zhu"])
    assert rc_build == 0
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new" in readme_text
    assert "readme-live-sandbox" not in readme_text

    # check_all_readmes passes when checking against the same slug and fails on a different slug
    assert build_readme.check_all_readmes(tmp_path, repo_slug="Siyuan-Zhu/Siyuan-Zhu") is True
    assert build_readme.check_all_readmes(tmp_path, repo_slug="Siyuan-Zhu/readme-live-sandbox") is False
    assert build_readme.main(["--check", "--repo-root", str(tmp_path), "--repo-slug", "Siyuan-Zhu/Siyuan-Zhu"]) == 0
    assert build_readme.main(["--check", "--repo-root", str(tmp_path), "--repo-slug", "Siyuan-Zhu/readme-live-sandbox"]) == 1


def test_process_visitor_event_honors_repo_slug_and_target_repo_slug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    shutil.copytree(REPO_ROOT / "assets", tmp_path / "assets")
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")

    monkeypatch.setenv("GITHUB_REPOSITORY", "Staging-Owner/readme-live-sandbox")
    monkeypatch.setenv("TARGET_REPO_SLUG", "Siyuan-Zhu/Siyuan-Zhu")

    # Owner check for orbit:* uses resolved slug owner ("Siyuan-Zhu"), not Staging-Owner
    rejected = process_visitor_event.process_event(
        tmp_path,
        visitor_login="Staging-Owner",
        issue_title="orbit:reset",
    )
    assert rejected["status"] == "rejected_unauthorized"

    # CLI main forwards --repo-slug into build_all_readmes
    rc = process_visitor_event.main(
        [
            "--repo-root",
            str(tmp_path),
            "--repo-slug",
            "Custom-Owner/Custom-Profile",
            "--login",
            "octocat",
            "--title",
            "pulse:vote:spot_on",
        ]
    )
    assert rc == 0
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/Custom-Owner/Custom-Profile/issues/new" in readme_text


def test_validate_release_bundle_passes_on_valid_bundle_and_catches_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import validate_release_links

    # Set up a valid bundle in tmp_path
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    shutil.copytree(REPO_ROOT / "assets", tmp_path / "assets")
    shutil.copy2(REPO_ROOT / "PROJECTS.md", tmp_path / "PROJECTS.md")
    from src.game_engine import create_initial_game_state
    import json
    (tmp_path / "data/game_state.json").write_text(json.dumps(create_initial_game_state()))
    legacy = tmp_path / "data/connect4_state.json"
    if legacy.exists():
        legacy.write_text(json.dumps(create_initial_game_state()))
    build_readme.build_all_readmes(tmp_path, repo_slug="Siyuan-Zhu/Siyuan-Zhu")

    report = validate_release_links.validate_release_bundle(
        tmp_path,
        expected_repo_slug="Siyuan-Zhu/Siyuan-Zhu",
    )
    assert report["ok"] is True, f"Expected valid bundle, got errors: {report.get('errors')}"
    assert report["errors"] == []
    assert report["checked_files_count"] == 10
    assert report["issue_links_count"] == 11

    # CLI exits 0 and supports --json
    rc_ok = validate_release_links.main(
        [
            "--repo-root",
            str(tmp_path),
            "--expected-repo-slug",
            "Siyuan-Zhu/Siyuan-Zhu",
            "--json",
        ]
    )
    assert rc_ok == 0
    out_json = capsys.readouterr().out
    assert '"ok": true' in out_json

    # 1. Wrong expected_repo_slug fails
    report_wrong_slug = validate_release_links.validate_release_bundle(
        tmp_path,
        expected_repo_slug="Siyuan-Zhu/readme-live-sandbox",
    )
    assert report_wrong_slug["ok"] is False
    assert any("issues/new" in err or "slug" in err for err in report_wrong_slug["errors"])

    # 2. Turn mismatch in connect4_state.json / game_state.json fails
    c4_state_path = tmp_path / "data" / "connect4_state.json"
    c4_state_path.write_text('{"turn": 99}', encoding="utf-8")
    report_turn = validate_release_links.validate_release_bundle(
        tmp_path,
        expected_repo_slug="Siyuan-Zhu/Siyuan-Zhu",
    )
    assert report_turn["ok"] is False
    assert any("turn" in err.lower() for err in report_turn["errors"])
    c4_state_path.unlink()

    # 3. Dead gl-variance-and-run-rate-pipeline link fails
    projects_path = tmp_path / "PROJECTS.md"
    orig_projects = projects_path.read_text(encoding="utf-8")
    projects_path.write_text(
        orig_projects + "\nhttps://github.com/Siyuan-Zhu/gl-variance-and-run-rate-pipeline\n",
        encoding="utf-8",
    )
    report_dead_link = validate_release_links.validate_release_bundle(
        tmp_path,
        expected_repo_slug="Siyuan-Zhu/Siyuan-Zhu",
    )
    assert report_dead_link["ok"] is False
    assert any("gl-variance-and-run-rate-pipeline" in err for err in report_dead_link["errors"])
    projects_path.write_text(orig_projects, encoding="utf-8")

    # 4. Corrupted WebP header fails
    dark_webp = tmp_path / "assets" / "daily_pulse_dark.webp"
    orig_webp = dark_webp.read_bytes()
    dark_webp.write_bytes(b"NOT_A_WEBP_HEADER_123456")
    report_bad_webp = validate_release_links.validate_release_bundle(
        tmp_path,
        expected_repo_slug="Siyuan-Zhu/Siyuan-Zhu",
    )
    assert report_bad_webp["ok"] is False
    assert any("daily_pulse_dark.webp" in err for err in report_bad_webp["errors"])
    dark_webp.write_bytes(orig_webp)

    # 5. Malformed SVG fails and CLI returns exit code 1
    board_svg = tmp_path / "assets" / "game_board.svg"
    orig_svg = board_svg.read_text(encoding="utf-8")
    board_svg.write_text("<svg><unclosed>", encoding="utf-8")
    rc_fail = validate_release_links.main(
        [
            "--repo-root",
            str(tmp_path),
            "--expected-repo-slug",
            "Siyuan-Zhu/Siyuan-Zhu",
        ]
    )
    assert rc_fail == 1
    board_svg.write_text(orig_svg, encoding="utf-8")


def _repository_issue_slug(root: Path) -> str:
    from scripts.validate_release_links import ISSUE_NEW_URL_RE
    slugs = {slug for slug, _query in ISSUE_NEW_URL_RE.findall((root / "README.md").read_text())}
    approved = {"Siyuan-Zhu/Siyuan-Zhu", "Siyuan-Zhu/readme-live-sandbox"}
    assert len(slugs) == 1 and slugs <= approved, "Expected one actual approved Issue destination"
    return slugs.pop()


def test_live_repository_bundle_passes_release_validator_and_check_all_readmes() -> None:
    from scripts import validate_release_links

    slug = _repository_issue_slug(REPO_ROOT)
    # Round-trip the committed fixture's presentation mode. This does not verify
    # deployment enablement; live repository variables/readiness do that separately.
    paused = "<sub>Game paused." in (REPO_ROOT / "README.md").read_text()
    with pytest.MonkeyPatch.context() as fixture_env:
        fixture_env.setenv("COMMUNITY_GAME_ENABLED", "false" if paused else "true")
        assert build_readme.check_all_readmes(REPO_ROOT, repo_slug=slug) is True
        report = validate_release_links.validate_release_bundle(
            REPO_ROOT,
            expected_repo_slug=slug,
        )
        assert report["ok"] is True, f"Live repository failed release validation: {report['errors']}"


def test_firestore_collection_prefix_isolation_and_sanitization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from datetime import datetime, timezone
    from src.state_store import (
        FirestoreStateStore,
        JsonFileStateStore,
        JsonStateStore,
        compute_voter_hmac,
        get_state_store,
    )
    from tests.test_firestore_store import _RecordingFirestoreClient

    assert JsonStateStore is JsonFileStateStore

    rec_client = _RecordingFirestoreClient()

    # 1. Default (empty) collection_prefix preserves exact top-level collection names
    monkeypatch.delenv("FIRESTORE_COLLECTION_PREFIX", raising=False)
    default_store = FirestoreStateStore(client=rec_client, project_id="test-proj")
    assert default_store.collection_prefix == ""
    assert default_store._col("profile_state") == "profile_state"

    # 2. Invalid prefix characters fail closed with ValueError
    with pytest.raises(ValueError, match="collection_prefix"):
        FirestoreStateStore(client=rec_client, collection_prefix="bad/prefix")
    with pytest.raises(ValueError, match="collection_prefix"):
        FirestoreStateStore(client=rec_client, collection_prefix="bad prefix!")

    # 3. Environment variable and explicit collection_prefix route all collections under prefix
    monkeypatch.setenv("FIRESTORE_COLLECTION_PREFIX", "  staging_  ")
    env_store = get_state_store(
        repo_root=tmp_path,
        backend="firestore",
        firestore_client=rec_client,
    )
    assert isinstance(env_store, FirestoreStateStore)
    assert env_store.collection_prefix == "staging_"
    assert env_store._col("profile_state") == "staging_profile_state"

    explicit_store = get_state_store(
        repo_root=tmp_path,
        backend="firestore",
        firestore_client=rec_client,
        collection_prefix="sandbox_r02_",
    )
    assert isinstance(explicit_store, FirestoreStateStore)
    assert explicit_store.collection_prefix == "sandbox_r02_"

    # Exercise all read/write/prune/reconcile methods on staging_ store
    env_store.save_game_state({"round_id": 2, "turn": 3, "mode": "VS_BOT", "status": "ACTIVE"})
    assert "staging_profile_state/active_game" in rec_client.docs
    assert "profile_state/active_game" not in rec_client.docs
    assert env_store.load_game_state()["round_id"] == 2

    env_store.save_pulse_state(
        {
            "schema_version": "2.0",
            "state_version": 2,
            "date": "2026-10-28",
            "word": "BULKHEAD",
            "gloss": "Structural partition isolating fault domains.",
            "commissioned_thesis": "Cross-lens synthesis.",
            "visual_medium": "PHOTOGRAPHY",
            "fallback_webp_path": "assets/pulse/2026-10-28.webp",
            "mood_votes": {"spot_on": 20, "too_sunny": 3, "too_heavy": 4},
            "published_at": "2026-10-28T06:00:00Z",
            "analysis": {
                "date_iso": "2026-10-28",
                "word_of_the_day": "BULKHEAD",
                "explanation_phrase": "Structural partition isolating fault domains.",
                "composite_score": 70.4,
            },
        }
    )
    assert "staging_profile_state/daily_pulse" in rec_client.docs
    assert "staging_pulse_history/2026-10-28" in rec_client.docs
    assert "profile_state/daily_pulse" not in rec_client.docs
    assert "pulse_history/2026-10-28" not in rec_client.docs
    assert env_store.load_pulse_state()["word"] == "BULKHEAD"
    assert "BULKHEAD" in env_store.get_recent_pulse_words(days=5)

    env_store.save_pulse_feed_entry(
        "2026-10-28",
        {
            "date": "2026-10-28",
            "word": "BULKHEAD",
            "gloss": "Structural partition isolating fault domains.",
        },
    )
    assert "staging_pulse_history/2026-10-28" in rec_client.docs

    vhmac = compute_voter_hmac("198.51.100.7", "TestAgent/1.0", "2026-10-28")
    accepted, tally = env_store.record_mood_vote("2026-10-28", "spot_on", vhmac)
    assert accepted is True
    assert tally["spot_on"] == 21
    assert "staging_pulse_votes/2026-10-28" in rec_client.docs
    assert f"staging_vote_rate_limits/2026-10-28_{vhmac}" in rec_client.docs
    assert "pulse_votes/2026-10-28" not in rec_client.docs

    env_store.save_game_rate_limits({"users": {"octocat": {"last_move_ts": 100}}, "state_version": 1})
    assert "staging_profile_state/game_rate_limits" in rec_client.docs
    assert "octocat" in env_store.load_game_rate_limits()["users"]

    env_store.save_orbit_roster({"handles": {"nowness": {}}, "curated_o3_cursor": {}, "state_version": 1})
    assert "staging_profile_state/orbit_roster" in rec_client.docs
    assert "nowness" in env_store.load_orbit_roster()["handles"]

    env_store.save_bot_memory({"games_learned": 12, "states": {}, "state_version": 2})
    assert "staging_profile_state/bot_memory" in rec_client.docs
    assert env_store.load_bot_memory()["games_learned"] == 12

    env_store.save_elo_ledger({"bot_elo": 1215, "players": {}, "recent_rounds": [], "state_version": 2})
    assert "staging_profile_state/elo_ledger" in rec_client.docs
    assert env_store.load_elo_ledger()["bot_elo"] == 1215

    now_utc = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    env_store.save_ingestion_cache_day(
        "2026-10-28",
        {"h1": {"bucket": "O1", "published_at": "2026-10-28T05:00:00Z"}},
    )
    assert "staging_ingestion_cache/2026-10-28" in rec_client.docs
    assert "h1" in env_store.load_ingestion_cache(now_utc)
    env_store.prune_expired_cache_items(now_utc)

    env_store.set_needs_git_reconcile(True)
    assert env_store.reconcile_git_from_firestore_if_needed() is True

    # Ensure zero unprefixed collections were touched in Firestore
    unprefixed = [k for k in rec_client.docs if not k.startswith("staging_")]
    assert unprefixed == []


def test_backup_and_rollback_manifest_verification_and_restore(
    tmp_path: Path,
) -> None:
    import hashlib
    from src.state_store import FirestoreStateStore
    from tests.test_firestore_store import _RecordingFirestoreClient
    from scripts.migrate_release_state import (
        create_release_backup,
        rollback_release_state,
    )

    repo_root = tmp_path / "repo"
    (repo_root / "variants").mkdir(parents=True)
    (repo_root / "data").mkdir(parents=True)
    (repo_root / "state").mkdir(parents=True)
    (repo_root / "assets" / "pulse").mkdir(parents=True)

    (repo_root / "README.md").write_text("# Original README\n", encoding="utf-8")
    (repo_root / "PROJECTS.md").write_text("# Original PROJECTS\n", encoding="utf-8")
    (repo_root / "variants" / "README.swe_ai.md").write_text("# Variant A\n", encoding="utf-8")
    (repo_root / "data" / "game_state.json").write_text('{"round_id": 1, "turn": 3}\n', encoding="utf-8")
    (repo_root / "data" / "zeitgeist_state.json").write_text('{"word": "ORIGINAL"}\n', encoding="utf-8")
    (repo_root / "state" / "pulse_feed.json").write_text('{"version": 1, "entries": []}\n', encoding="utf-8")
    (repo_root / "assets" / "header.svg").write_text("<svg>orig</svg>\n", encoding="utf-8")
    (repo_root / "assets" / "hero.webp").write_bytes(b"RIFForigWEBPVP8 ")
    (repo_root / "assets" / "pulse" / "2026-10-28.webp").write_bytes(b"RIFFpulseWEBPVP8 ")

    rec_client = _RecordingFirestoreClient()
    fs_store = FirestoreStateStore(
        client=rec_client,
        project_id="test-proj",
        collection_prefix="staging_",
    )
    rec_client.docs["staging_profile_state/daily_pulse"] = {"word": "ORIGINAL_FS", "state_version": 5}
    rec_client.docs["staging_profile_state/active_game"] = {"round_id": 1, "turn": 3, "state_version": 5}
    rec_client.docs["staging_pulse_feed/v1"] = {"version": 1, "entries": [{"date": "2026-10-28", "word": "ORIGINAL_FS"}]}
    rec_client.docs["staging_pulse_history/2026-10-28"] = {"date": "2026-10-28", "word": "ORIGINAL_FS"}
    rec_client.docs["staging_pulse_votes/2026-10-28"] = {"spot_on": 20, "too_sunny": 3, "too_heavy": 4}

    backup_dir = tmp_path / "backups" / "pre-v3"
    manifest = create_release_backup(
        repo_root=repo_root,
        backup_dir=backup_dir,
        firestore_store=fs_store,
    )

    manifest_path = backup_dir / "manifest.json"
    assert manifest_path.exists()
    assert (backup_dir / "firestore_export.json").exists()
    assert "README.md" in manifest["files"]
    assert "PROJECTS.md" in manifest["files"]
    assert "variants/README.swe_ai.md" in manifest["files"]
    assert "data/game_state.json" in manifest["files"]
    assert "state/pulse_feed.json" in manifest["files"]
    assert "assets/header.svg" in manifest["files"]
    assert "assets/hero.webp" in manifest["files"]
    assert "assets/pulse/2026-10-28.webp" in manifest["files"]

    expected_readme_sha = hashlib.sha256(b"# Original README\n").hexdigest()
    assert manifest["files"]["README.md"] == expected_readme_sha

    # Mutate working tree and Firestore state
    (repo_root / "README.md").write_text("# Corrupted README\n", encoding="utf-8")
    (repo_root / "data" / "game_state.json").write_text('{"round_id": 99, "turn": 1}\n', encoding="utf-8")
    rec_client.docs["staging_profile_state/daily_pulse"] = {"word": "MUTATED_FS", "state_version": 99}

    # Tamper with a backup file and verify rollback fails closed BEFORE restoring anything
    backed_up_game = backup_dir / "files" / "data" / "game_state.json"
    original_backed_up_bytes = backed_up_game.read_bytes()
    backed_up_game.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        rollback_release_state(
            repo_root=repo_root,
            backup_dir=backup_dir,
            firestore_store=fs_store,
        )
    # Working tree should still have the post-mutation state because rollback aborted pre-flight
    assert (repo_root / "README.md").read_text(encoding="utf-8") == "# Corrupted README\n"

    # Restore valid backup file bytes and test unprefixed Firestore production guard on rollback
    backed_up_game.write_bytes(original_backed_up_bytes)
    prod_fs_store = FirestoreStateStore(
        client=rec_client,
        project_id="test-proj",
        collection_prefix="",
    )
    with pytest.raises(PermissionError, match="--confirm-production"):
        rollback_release_state(
            repo_root=repo_root,
            backup_dir=backup_dir,
            firestore_store=prod_fs_store,
            confirm_production=False,
        )

    # Unknown remote writes must be reconciled before local or remote restore.
    with pytest.raises(ValueError, match="unexpected"):
        rollback_release_state(repo_root, backup_dir, firestore_store=fs_store)
    rec_client.docs["staging_profile_state/daily_pulse"] = {"word": "ORIGINAL_FS", "state_version": 5}
    # The canonical snapshot is unchanged again; local rollback can now proceed.
    restored = rollback_release_state(
        repo_root=repo_root,
        backup_dir=backup_dir,
        firestore_store=fs_store,
    )
    assert restored["restored_file_count"] == len(manifest["files"])
    assert (repo_root / "README.md").read_text(encoding="utf-8") == "# Original README\n"
    assert (repo_root / "data" / "game_state.json").read_text(encoding="utf-8") == '{"round_id": 1, "turn": 3}\n'
    assert rec_client.docs["staging_profile_state/daily_pulse"]["word"] == "ORIGINAL_FS"


def test_migrate_release_state_normalization_idempotency_and_production_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy
    import json
    from src.state_store import FirestoreStateStore
    from tests.test_firestore_store import (
        _RecordingFirestoreClient,
        _assert_valid_active_game_state,
        _assert_valid_bot_memory_state,
        _assert_valid_daily_pulse_state,
        _assert_valid_elo_ledger_state,
        _assert_valid_game_rate_limits_state,
        _assert_valid_orbit_roster_state,
    )
    from scripts.migrate_release_state import (
        main as migrate_cli_main,
        migrate_release_state,
        create_release_backup,
    )

    repo_root = tmp_path / "repo"
    (repo_root / "data").mkdir(parents=True)
    (repo_root / "state").mkdir(parents=True)

    # Seed legacy v1/v2 state with legacy mood aliases and base64 image bloat
    legacy_pulse = {
        "schema_version": "1.0",
        "state_version": 4,
        "date_iso": "2026-10-28",
        "analysis": {
            "date_iso": "2026-10-28",
            "word_of_the_day": "resilience",
            "explanation_phrase": "Adaptive fault tolerance under load.",
        },
        "visitor_mood_tally": {
            "resonant": 14,
            "electric": 5,
            "skeptical": 2,
        },
        "image_data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
        "published_at": "2026-10-28T06:00:00Z",
        "updated_at": "2026-10-28T06:00:00Z",
    }
    (repo_root / "data" / "zeitgeist_state.json").write_text(
        json.dumps(legacy_pulse, indent=2) + "\n", encoding="utf-8"
    )

    legacy_game = {
        "schema_version": "1.0",
        "state_version": 7,
        "round_id": 2,
        "turn": 5,
        "mode": "VS_BOT",
        "status": "ACTIVE",
        "last_voter": "alice",
        "winning_line": [],
        "round_participants": {"red": {"alice": 3}, "yellow": {"House-Bot": 2}},
        "updated_at": "2026-10-28T06:00:00Z",
    }
    (repo_root / "data" / "game_state.json").write_text(
        json.dumps(legacy_game, indent=2) + "\n", encoding="utf-8"
    )

    legacy_feed = [
        {
            "date": "2026-10-28",
            "word": "resilience",
            "gloss": "Adaptive fault tolerance under load.",
            "mood_votes": {"resonant": 14, "electric": 5, "skeptical": 2},
            "image_data_uri": "data:image/webp;base64,UklGRg==",
            "published_at": "2026-10-28T06:00:00Z",
        }
    ]
    (repo_root / "data" / "pulse_feed.json").write_text(
        json.dumps(legacy_feed, indent=2) + "\n", encoding="utf-8"
    )

    # 1. Dry-run normalizes and validates without mutating disk
    before_pulse_bytes = (repo_root / "data" / "zeitgeist_state.json").read_bytes()
    dry_report = migrate_release_state(
        repo_root=repo_root,
        target_backend="json",
        apply=False,
    )
    assert dry_report["applied"] is False
    assert (repo_root / "data" / "zeitgeist_state.json").read_bytes() == before_pulse_bytes

    docs = dry_report["documents"]
    assert set(docs["profile_state"].keys()) == {
        "daily_pulse",
        "active_game",
        "elo_ledger",
        "bot_memory",
        "orbit_roster",
        "game_rate_limits",
    }
    dp = docs["profile_state"]["daily_pulse"]
    ag = docs["profile_state"]["active_game"]
    el = docs["profile_state"]["elo_ledger"]
    bm = docs["profile_state"]["bot_memory"]
    or_doc = docs["profile_state"]["orbit_roster"]
    grl = docs["profile_state"]["game_rate_limits"]
    pf = docs["pulse_feed"]["v1"]

    assert dp["schema_version"] == 3
    assert "image_data_uri" not in dp
    assert dp["mood_votes"] == {"spot_on": 14, "too_sunny": 5, "too_heavy": 2}
    assert dp["visitor_mood_tally"] == {"spot_on": 14, "too_sunny": 5, "too_heavy": 2}
    _assert_valid_daily_pulse_state(dp, require_v2=True)

    assert ag["schema_version"] == 3
    assert isinstance(ag["board_state"], str) and len(ag["board_state"]) == 42
    _assert_valid_active_game_state(ag)
    _assert_valid_elo_ledger_state(el)
    _assert_valid_bot_memory_state(bm)
    _assert_valid_orbit_roster_state(or_doc)
    _assert_valid_game_rate_limits_state(grl)

    assert pf["schema_version"] == 3
    assert len(pf["entries"]) == 1
    assert pf["entries"][0]["image_data_uri"] == ""
    assert pf["entries"][0]["mood_votes"] == {"spot_on": 14, "too_sunny": 5, "too_heavy": 2}

    # 2. Apply to JSON backend and verify idempotency on second run
    apply_json_1 = migrate_release_state(
        repo_root=repo_root,
        target_backend="json",
        apply=True,
    )
    assert apply_json_1["applied"] is True
    assert len(apply_json_1["changed_documents"]) > 0
    snapshot_after_first = {
        p.relative_to(repo_root).as_posix(): p.read_bytes()
        for p in sorted((repo_root / "data").glob("*.json"))
    }

    apply_json_2 = migrate_release_state(
        repo_root=repo_root,
        target_backend="json",
        apply=True,
    )
    assert apply_json_2["applied"] is True
    assert apply_json_2["changed_documents"] == []
    snapshot_after_second = {
        p.relative_to(repo_root).as_posix(): p.read_bytes()
        for p in sorted((repo_root / "data").glob("*.json"))
    }
    assert snapshot_after_first == snapshot_after_second

    # 3. Production safety gate on unprefixed Firestore backend
    rec_client = _RecordingFirestoreClient()
    prod_store = FirestoreStateStore(
        client=rec_client,
        project_id="test-proj",
        collection_prefix="",
    )
    with pytest.raises(PermissionError, match="--confirm-production"):
        migrate_release_state(
            repo_root=repo_root,
            target_backend="firestore",
            firestore_store=prod_store,
            apply=True,
            confirm_production=False,
        )
    assert rec_client.docs == {}

    monkeypatch.delenv("FIRESTORE_COLLECTION_PREFIX", raising=False)
    exit_code = migrate_cli_main(
        [
            "migrate",
            "--repo-root",
            str(repo_root),
            "--target-backend",
            "firestore",
            "--apply",
        ]
    )
    assert exit_code == 2

    # 4. Apply to staging_ prefixed Firestore backend and verify idempotency
    staging_store = FirestoreStateStore(
        client=rec_client,
        project_id="test-proj",
        collection_prefix="staging_",
    )
    staging_backup = tmp_path / "staging-seed-backup"
    staging_plan = tmp_path / "staging-seed-plan.json"
    create_release_backup(repo_root, staging_backup, firestore_store=staging_store)
    migrate_release_state(repo_root, target_backend="firestore", firestore_store=staging_store,
                          backup_dir=staging_backup, plan_file=staging_plan, seed_from_json=True)
    apply_fs_1 = migrate_release_state(
        repo_root=repo_root,
        target_backend="firestore",
        firestore_store=staging_store,
        backup_dir=staging_backup,
        plan_file=staging_plan,
        seed_from_json=True,
        apply=True,
    )
    assert apply_fs_1["applied"] is True
    assert len(apply_fs_1["changed_documents"]) > 0
    assert "staging_profile_state/daily_pulse" in rec_client.docs
    assert "staging_profile_state/active_game" in rec_client.docs
    assert "staging_profile_state/elo_ledger" in rec_client.docs
    assert "staging_profile_state/bot_memory" in rec_client.docs
    assert "staging_profile_state/orbit_roster" in rec_client.docs
    assert "staging_profile_state/game_rate_limits" in rec_client.docs
    assert "staging_pulse_feed/v1" in rec_client.docs
    assert "staging_pulse_history/2026-10-28" in rec_client.docs
    assert "staging_pulse_votes/2026-10-28" in rec_client.docs
    assert [k for k in rec_client.docs if not k.startswith("staging_")] == []

    docs_snapshot_1 = copy.deepcopy(rec_client.docs)
    apply_fs_2 = migrate_release_state(
        repo_root=repo_root,
        target_backend="firestore",
        firestore_store=staging_store,
        backup_dir=staging_backup,
        plan_file=staging_plan,
        seed_from_json=True,
        apply=True,
    )
    assert apply_fs_2["applied"] is True
    assert apply_fs_2["changed_documents"] == []
    assert rec_client.docs == docs_snapshot_1

    # 5. CLI subcommands end-to-end (backup -> migrate --apply -> rollback)
    cli_backup_dir = tmp_path / "cli-backup"
    (repo_root / "README.md").write_text("# CLI Pre-Cutover README\n", encoding="utf-8")
    assert (
        migrate_cli_main(
            [
                "backup",
                "--repo-root",
                str(repo_root),
                "--backup-dir",
                str(cli_backup_dir),
            ]
        )
        == 0
    )
    assert (cli_backup_dir / "manifest.json").exists()

    (repo_root / "README.md").write_text("# Mutated During Cutover\n", encoding="utf-8")
    assert (
        migrate_cli_main(
            [
                "migrate",
                "--repo-root",
                str(repo_root),
                "--target-backend",
                "json",
                "--dry-run",
            ]
        )
        == 0
    )
    assert (
        migrate_cli_main(
            [
                "migrate",
                "--repo-root",
                str(repo_root),
                "--target-backend",
                "json",
                "--apply",
            ]
        )
        == 0
    )
    assert (
        migrate_cli_main(
            [
                "rollback",
                "--repo-root",
                str(repo_root),
                "--backup-dir",
                str(cli_backup_dir),
            ]
        )
        == 0
    )
    assert (repo_root / "README.md").read_text(encoding="utf-8") == "# CLI Pre-Cutover README\n"


def test_firestore_oidc_and_inline_sa_credential_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock, patch
    from src.state_store import FirestoreStateStore, _FirestoreRestClient

    # 1. FirestoreStateStore.__init__ database_id resolution from FIRESTORE_DATABASE_ID
    monkeypatch.delenv("FIRESTORE_DATABASE_ID", raising=False)
    store_default = FirestoreStateStore(project_id="git-readme-features")
    assert store_default._database_id == "(default)"

    monkeypatch.setenv("FIRESTORE_DATABASE_ID", "git-readme-features")
    store_legacy_default = FirestoreStateStore(project_id="git-readme-features")
    assert store_legacy_default._database_id == "(default)"

    monkeypatch.setenv("FIRESTORE_DATABASE_ID", "   ")
    store_empty_env = FirestoreStateStore(project_id="git-readme-features")
    assert store_empty_env._database_id == "(default)"

    monkeypatch.setenv("FIRESTORE_DATABASE_ID", "named-staging-db")
    store_named_env = FirestoreStateStore(project_id="git-readme-features")
    assert store_named_env._database_id == "named-staging-db"

    store_explicit = FirestoreStateStore(
        project_id="git-readme-features",
        database_id="explicit-db",
    )
    assert store_explicit._database_id == "explicit-db"

    # 2. _FirestoreRestClient._resolve_token checks GOOGLE_APPLICATION_CREDENTIALS_JSON and FIREBASE_SERVICE_ACCOUNT
    monkeypatch.delenv("FIRESTORE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GCP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_URL", raising=False)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", raising=False)
    monkeypatch.delenv("GCP_SA_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT", raising=False)

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", '{"client_email":"sa1@test.iam.gserviceaccount.com"}')
    rest_gac = _FirestoreRestClient(project_id="git-readme-features")
    with patch.object(
        _FirestoreRestClient,
        "_mint_sa_access_token",
        return_value="minted-from-gac-json",
    ) as mock_mint_gac:
        assert rest_gac._resolve_token() == "minted-from-gac-json"
        mock_mint_gac.assert_called_once_with('{"client_email":"sa1@test.iam.gserviceaccount.com"}')

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", '{"client_email":"sa2@test.iam.gserviceaccount.com"}')
    rest_fb = _FirestoreRestClient(project_id="git-readme-features")
    with patch.object(
        _FirestoreRestClient,
        "_mint_sa_access_token",
        return_value="minted-from-firebase-sa",
    ) as mock_mint_fb:
        assert rest_fb._resolve_token() == "minted-from-firebase-sa"
        mock_mint_fb.assert_called_once_with('{"client_email":"sa2@test.iam.gserviceaccount.com"}')

    # 3. FirestoreStateStore._get_client() prefers _FirestoreRestClient when GitHub OIDC or inline SA env vars are set
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT", raising=False)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://token.actions.githubusercontent.com/id")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "gh-oidc-req-token")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    oidc_store = FirestoreStateStore(project_id="git-readme-features")
    with (
        patch.object(
            _FirestoreRestClient,
            "_resolve_token",
            return_value="oidc-exchanged-token",
        ) as mock_resolve,
        patch("google.cloud.firestore.Client", side_effect=AssertionError("Should not call gRPC firestore.Client when OIDC env is active")),
    ):
        client = oidc_store._get_client()
        assert isinstance(client, _FirestoreRestClient)
        assert mock_resolve.call_count == 1


def test_validate_release_links_output_json_and_migrate_include_firestore_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from scripts import validate_release_links
    from scripts.migrate_release_state import main as migrate_cli_main
    from src.state_store import FirestoreStateStore
    from tests.test_firestore_store import _RecordingFirestoreClient

    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    shutil.copytree(REPO_ROOT / "assets", tmp_path / "assets")
    shutil.copy2(REPO_ROOT / "PROJECTS.md", tmp_path / "PROJECTS.md")
    build_readme.build_all_readmes(tmp_path, repo_slug="Siyuan-Zhu/Siyuan-Zhu")

    out_json_path = tmp_path / "state_out" / "link_audit_production_slug.json"
    rc = validate_release_links.main(
        [
            "--repo-root",
            str(tmp_path),
            "--expected-repo-slug",
            "Siyuan-Zhu/Siyuan-Zhu",
            "--output-json",
            str(out_json_path),
        ]
    )
    assert rc == 0
    assert out_json_path.is_file()
    audit_payload = json.loads(out_json_path.read_text(encoding="utf-8"))
    assert audit_payload["ok"] is True
    assert audit_payload["expected_repo_slug"] == "Siyuan-Zhu/Siyuan-Zhu"

    rec_client = _RecordingFirestoreClient()
    monkeypatch.setattr(FirestoreStateStore, "_get_client", lambda self: rec_client)

    from scripts.migrate_release_state import create_release_backup
    store = FirestoreStateStore(project_id="test-project", collection_prefix="staging_")
    backup = tmp_path / "firestore-backup"
    plan = tmp_path / "firestore-plan.json"
    create_release_backup(tmp_path, backup, firestore_store=store)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    common = ["migrate", "--repo-root", str(tmp_path), "--target-backend", "firestore",
              "--collection-prefix", "staging_", "--backup-dir", str(backup),
              "--plan-file", str(plan), "--seed-from-json"]
    assert migrate_cli_main(common + ["--dry-run"]) == 0
    rc_mig = migrate_cli_main(common + ["--apply"])
    assert rc_mig == 0
    assert "staging_profile_state/daily_pulse" in rec_client.docs
    assert "staging_profile_state/active_game" in rec_client.docs
    assert [k for k in rec_client.docs if not k.startswith("staging_")] == []


def test_single_writer_workflow_guards_and_remote_staging_pipeline_contracts() -> None:
    gh_open = chr(36) + "{{"
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    daily_wf = (workflows_dir / "daily_zeitgeist.yml").read_text(encoding="utf-8")
    comm_wf = (workflows_dir / "community_game_move.yml").read_text(encoding="utf-8")
    mobile_wf = (workflows_dir / "mobile_interject.yml").read_text(encoding="utf-8")
    staging_wf = (workflows_dir / "staging_zeitgeist_dryrun.yml").read_text(encoding="utf-8")

    # 1. Single-writer guard and standardized Firestore/target-slug env vars across all 3 writer workflows
    for name, wf_text in (
        ("daily_zeitgeist.yml", daily_wf),
        ("community_game_move.yml", comm_wf),
        ("mobile_interject.yml", mobile_wf),
    ):
        assert "vars.ZEITGEIST_WRITER_ENABLED == 'true'" in wf_text, f"{name} missing single-writer guard"
        assert f"TARGET_REPO_SLUG: {gh_open} vars.TARGET_REPO_SLUG || github.repository }}}}" in wf_text, (
            f"{name} missing TARGET_REPO_SLUG env"
        )
        assert (
            f"FIRESTORE_COLLECTION_PREFIX: {gh_open} vars.FIRESTORE_COLLECTION_PREFIX || '' }}}}" in wf_text
        ), f"{name} missing FIRESTORE_COLLECTION_PREFIX env"
        assert (
            f"GCP_PROJECT_ID: {gh_open} secrets.GCP_PROJECT_ID || secrets.GOOGLE_CLOUD_PROJECT || secrets.FIRESTORE_PROJECT_ID || 'git-readme-features' }}}}"
            in wf_text
        ), f"{name} missing standardized GCP_PROJECT_ID fallback"
        assert (
            f"FIRESTORE_DATABASE_ID: {gh_open} vars.FIRESTORE_DATABASE_ID || '(default)' }}}}" in wf_text
        ), f"{name} missing standardized FIRESTORE_DATABASE_ID default"
        assert (
            f"FIRESTORE_DATABASE_ID: {gh_open} vars.FIRESTORE_DATABASE_ID || 'git-readme-features' }}}}" not in wf_text
        ), f"{name} still has broken FIRESTORE_DATABASE_ID default"

    assert "vars.COMMUNITY_GAME_ENABLED == 'true' && (" in comm_wf
    assert "vars.ZEITGEIST_WRITER_ENABLED == 'true' && (" in comm_wf
    # Visitor availability must not implicitly activate paid scheduled publishing.
    assert "github.event_name != 'schedule' || vars.ZEITGEIST_SCHEDULE_ENABLED == 'true'" in daily_wf
    assert "ZEITGEIST_SCHEDULE_ENABLED" not in comm_wf
    assert "ZEITGEIST_SCHEDULE_ENABLED" not in mobile_wf
    for prefix in ("game:move:", "game:mode:", "connect4: drop ", "pulse:vote:", "orbit:"):
        assert f"startsWith(github.event.issue.title, '{prefix}')" in comm_wf

    # 2. Remote staging dry-run workflow contract
    assert "contents: read" in staging_wf
    assert "id-token: write" in staging_wf
    assert "contents: write" not in staging_wf
    assert "live_firestore:" in staging_wf
    assert "Verify live Firestore read/write under isolated staging_ prefix" in staging_wf
    assert f'FIRESTORE_COLLECTION_PREFIX: "staging_{gh_open} github.run_id }}}}_{gh_open} github.run_attempt }}}}_"' in staging_wf
    assert 'GCP_PROJECT_ID: "git-readme-features"' in staging_wf
    assert 'FIRESTORE_DATABASE_ID: "(default)"' in staging_wf
    assert f"TARGET_REPO_SLUG: {gh_open} github.repository }}}}" in staging_wf
    assert f"APIFY_API_TOKEN: {gh_open} secrets.APIFY_API_TOKEN || secrets.APIFY_TOKEN }}}}" in staging_wf
    assert (
        "Validate production-slug and sandbox-slug README rendering, links, and state migration dry-run"
        in staging_wf
    )
    assert "/tmp/zeitgeist-staging-state/link_audit_production_slug.json" in staging_wf
    assert "/tmp/zeitgeist-staging-state/link_audit_sandbox_slug.json" in staging_wf
    assert "/tmp/private-staging-validation-backup" in staging_wf
    artifact_paths = staging_wf.split("      - name: Upload isolated staging dry-run bundle artifact", 1)[1]
    assert "private-staging-validation-backup" not in artifact_paths
    assert "scripts/rehearse_staging_application.py" in staging_wf
    assert "/tmp/zeitgeist-staging-state/migration_dry_run.json" in staging_wf
    assert "Rehearse unique staging Firestore migration, readback and rollback" in staging_wf
    assert "if: inputs.live_firestore == true" in staging_wf
    assert "scripts/rehearse_staging_recovery.py" in staging_wf
    assert "--backup-dir /tmp/private-staging-recovery" in staging_wf
    assert "scripts/verify_staging_interactions.py" in staging_wf
    assert "--require-publication-manifest" in staging_wf










@pytest.mark.parametrize("enabled", ["true", "false"])
@pytest.mark.parametrize("slug", ["Siyuan-Zhu/Siyuan-Zhu", "Siyuan-Zhu/readme-live-sandbox"])
def test_repository_bundle_audit_uses_actual_single_approved_issue_slug(tmp_path, monkeypatch, slug, enabled):
    """The same audit must accept the real staging tree and the public export."""
    import sys
    from src.state_store import JsonFileStateStore
    for folder in ("data", "assets", "variants", "config"):
        shutil.copytree(REPO_ROOT / folder, tmp_path / folder)
    shutil.copy2(REPO_ROOT / "PROJECTS.md", tmp_path / "PROJECTS.md")
    monkeypatch.setenv("COMMUNITY_GAME_ENABLED", enabled)
    build_readme.build_all_readmes(tmp_path, repo_slug=slug, state_store=JsonFileStateStore(tmp_path))
    content = (tmp_path / "README.md").read_text()
    if enabled == "false":
        assert "<sub>Game paused." in content
        assert "title=game:move:" not in content
        assert "title=pulse:vote:spot_on" in content
    else:
        assert "<sub>Game paused." not in content
        assert "title=game:move:" in content
    # CI's ambient setting may differ from the recorded fixture. The audit must
    # round-trip the fixture itself, without changing the wider test environment.
    ambient = "false" if enabled == "true" else "true"
    monkeypatch.setenv("COMMUNITY_GAME_ENABLED", ambient)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    test_live_repository_bundle_passes_release_validator_and_check_all_readmes()
    import os
    assert os.environ["COMMUNITY_GAME_ENABLED"] == ambient
    # Matching the recorded mode must not excuse malformed interactive controls.
    (tmp_path / "README.md").write_text(content.replace("title=pulse:vote:spot_on", "title=pulse:vote:unexpected"))
    with pytest.raises(AssertionError):
        test_live_repository_bundle_passes_release_validator_and_check_all_readmes()


@pytest.mark.parametrize("slugs", [[], ["Unapproved/Repository"], ["Siyuan-Zhu/Siyuan-Zhu", "Siyuan-Zhu/readme-live-sandbox"]])
def test_repository_bundle_audit_rejects_missing_unapproved_or_mixed_issue_destinations(tmp_path, slugs):
    (tmp_path / "README.md").write_text("\n".join("https://github.com/" + slug + "/issues/new?title=fixture" for slug in slugs))
    with pytest.raises(AssertionError):
        _repository_issue_slug(tmp_path)
