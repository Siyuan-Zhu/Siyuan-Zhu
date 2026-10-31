"""The exported tree is the release artifact; private checkout tests are insufficient."""

import copy
import json
from pathlib import Path
import subprocess
import shutil
import os

from PIL import Image
import pytest

from scripts import export_public_release as export
from scripts import promote_release_state as promotion
from src.zeitgeist.publication import write_publication_manifest
from tests.test_promotion_package import inputs


@pytest.fixture(scope="session")
def source_checkout(tmp_path_factory):
    # The clean exported tree deliberately has no Git metadata. Build an owned
    # fixture commit so these acceptance tests run there too, with no skips.
    source = tmp_path_factory.mktemp("owned-export-source")
    for path in export.ROOT.rglob("*"):
        relative = str(path.relative_to(export.ROOT))
        if path.is_file() and export.selected_source(relative):
            assert not path.is_symlink()
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    environment = dict(
        os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1"
    )
    command = [
        "git",
        "-C",
        str(source),
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "user.name=Export Fixture",
        "-c",
        "user.email=export-fixture@example.invalid",
    ]
    for args in (
        ("init", "-q"),
        ("add", "."),
        ("commit", "-qm", "Owned offline source fixture"),
    ):
        subprocess.run(
            command + list(args), env=environment, check=True, capture_output=True
        )
    return source


@pytest.fixture
def accepted(tmp_path, source_checkout, monkeypatch):
    monkeypatch.setattr(export, "ROOT", source_checkout)
    root, source, target, previous = inputs.__wrapped__(tmp_path)
    for path in root.glob("assets/**/*.webp"):
        Image.new("RGB", (48, 32), "#556677").save(path, format="WEBP")
    (root / "assets/daily_pulse.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" data-publication-layout="curatorial" '
        'viewBox="0 0 48 32"><metadata id="mood-votes">old</metadata><rect width="48" height="32" fill="#556677"/></svg>'
    )
    manifest = write_publication_manifest(
        root,
        date=previous["date"],
        word=previous["word"],
        gloss=previous["gloss"],
        archetype=previous["archetype"],
        medium=previous["medium"],
        archive_path="assets/pulse/" + previous["date"] + ".webp",
    )
    from dataclasses import replace
    from tests.test_firestore_store import _make_sample_published_pulse

    sample = replace(
        _make_sample_published_pulse(),
        date=previous["date"],
        word=previous["word"],
        gloss=previous["gloss"],
        webp_asset_path="assets/pulse/" + previous["date"] + ".webp",
    )
    entry = sample.to_feed_entry(include_curatorial_metadata=True)
    entry["publication_id"] = manifest["publication_id"]
    pulse = dict(
        entry,
        schema_version=3,
        state_version=5,
        visitor_mood_tally=entry["mood_votes"],
        published_pulse=copy.deepcopy(entry),
    )
    (root / "data/zeitgeist_state.json").write_text(json.dumps(pulse))
    (root / "data/pulse_feed.json").write_text(json.dumps({"entries": [entry]}))
    source["documents"]["profile_state/daily_pulse"]["data"] = pulse
    source["documents"]["pulse_history/" + previous["date"]]["data"] = entry
    source["documents"]["pulse_feed/v1"]["data"] = {"entries": [entry]}
    commit = subprocess.check_output(
        ["git", "-C", str(export.ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    plan = promotion.prepare_plan(
        source,
        target,
        bundle_root=root,
        source_sha=commit,
        source_run="41000000000",
        source_snapshot_sha=promotion.digest(source),
        target_snapshot_sha=promotion.digest(target),
        manifest_sha=promotion.digest(manifest),
        game_policy="fresh",
        operation_id="b" * 32,
        source_backup_sha="c" * 64,
        target_backup_sha="d" * 64,
        journal_path=tmp_path / "private-journal.json",
    )
    return root, plan, commit


def perform(accepted, destination, **options):
    bundle, plan, commit = accepted
    return export.export_release(
        export.ROOT, commit, bundle, plan, plan["plan_sha256"], destination, **options
    )


def test_actual_export_is_closed_history_free_complete_and_deterministic(
    accepted, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    before = {
        str(path): path.read_bytes()
        for path in accepted[0].rglob("*")
        if path.is_file()
    }
    one = perform(accepted, first)
    two = perform(accepted, second)
    assert one == two
    assert not (first / ".git").exists()
    assert sorted(p.name for p in (first / "docs").iterdir()) == ["resonance.md"]
    for private in (
        "AGENTS.md",
        "GLOSSARY.md",
        "work",
        "data/ingestion_cache",
        "data/spend_control.json",
    ):
        assert not (first / private).exists()
    assert set(one["source_files"]) == set(
        export.source_inventory(export.ROOT, accepted[2])
    )
    assert export.STATIC_FILES | export.WORKFLOWS <= one["export_files"].keys()
    assert {
        str(path.relative_to(first)) for path in (first / "data").iterdir()
    } == export.PUBLIC_JSON
    assert len(list((first / "variants").glob("*.md"))) == 8
    assert "Siyuan-Zhu/Siyuan-Zhu/issues/new" in (first / "README.md").read_text()
    assert "readme-live-sandbox/issues/new" not in (first / "README.md").read_text()
    assert (
        json.loads((first / "data/game_state.json").read_text())["moves_this_round"]
        == 0
    )
    assert (
        json.loads((first / "data/hall_of_records.json").read_text())["players"] == {}
    )
    pulse = json.loads((first / "data/zeitgeist_state.json").read_text())
    assert pulse["mood_votes"] == promotion.ZERO
    assert (
        one["accepted_bundle_files"]["assets/daily_pulse.svg"]
        != one["export_files"]["assets/daily_pulse.svg"]
    )
    assert before == {
        str(path): path.read_bytes()
        for path in accepted[0].rglob("*")
        if path.is_file()
    }
    # Import from the ACTUAL export and run its strict gates, without source PYTHONPATH.
    result = subprocess.run(
        [
            __import__("sys").executable,
            "scripts/validate_release_links.py",
            "--expected-repo-slug",
            export.PUBLIC_SLUG,
            "--require-publication-manifest",
            "--json",
        ],
        cwd=first,
        capture_output=True,
        text=True,
        env={"PATH": __import__("os").environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert one["cloud_state_changed"] is False
    assert one["git_history_created"] is False


def test_dry_run_builds_and_checks_artifact_without_creating_destination(
    accepted, tmp_path
):
    destination = tmp_path / "not-created"
    report = perform(accepted, destination, dry_run=True)
    assert report["status"] == "validated_dry_run"
    assert not destination.exists()


@pytest.mark.parametrize("field", ["credit_url", "unrelated_private_field"])
def test_accepted_bundle_cannot_be_substituted(accepted, tmp_path, field):
    path = accepted[0] / "data/zeitgeist_state.json"
    body = json.loads(path.read_text())
    body[field] = "modified"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        perform(accepted, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_does_not_merge_into_existing_or_source_directory(accepted, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "private.json").write_text("private")
    with pytest.raises(ValueError):
        perform(accepted, destination)
    with pytest.raises(ValueError):
        perform(accepted, accepted[0] / "export")
    assert (destination / "private.json").read_text() == "private"


def test_fresh_public_dto_omits_private_operational_fields(accepted):
    envelope = copy.deepcopy(accepted[1])
    pulse = envelope["plan"]["after"]["profile_state/daily_pulse"]
    pulse.update(
        telemetry={"account_id": "PRIVATE"},
        raw_source_rows=["PRIVATE"],
        commission={"prompt": "PRIVATE"},
        vote_rate_limits={"PRIVATE": 1},
    )
    pulse["analysis"] = {"word_of_the_day": "REAL", "personal_note": "PRIVATE"}
    pulse["visual_decision"] = {
        "selected_mode": "PHOTOGRAPHY",
        "local_svg_path": "/tmp/PRIVATE/assets/daily_pulse.svg",
    }
    envelope["plan_sha256"] = promotion.digest(envelope["plan"])
    projected = export.public_projections(envelope)
    assert "PRIVATE" not in json.dumps(projected)
    assert (
        projected["data/zeitgeist_state.json"]["visual_decision"]["local_svg_path"]
        == "assets/daily_pulse.svg"
    )


def test_generated_projection_rejects_private_paths_and_secret_like_values():
    for value in (
        "/tmp/private.json",
        "file:///private/tmp/private",
        "AIza" + "Q" * 35,
    ):
        with pytest.raises(ValueError):
            export._audit_public_text(
                "data/zeitgeist_state.json", json.dumps({"image_url": value}).encode()
            )


def test_public_inventory_stays_outside_export_and_cli_is_offline(
    accepted, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(accepted[1]))
    destination = tmp_path / "export"
    args = [
        "--source-root",
        str(export.ROOT),
        "--source-sha",
        accepted[2],
        "--bundle-root",
        str(accepted[0]),
        "--plan",
        str(path),
        "--expected-plan-sha",
        accepted[1]["plan_sha256"],
        "--output-dir",
        str(destination),
        "--private-inventory",
        str(destination / "private.json"),
    ]
    monkeypatch.setattr(
        export,
        "export_release",
        lambda *a, **k: pytest.fail("Private report location must block before export"),
    )
    assert export.main(args) == 1
    assert json.loads(capsys.readouterr().out)["error_type"] == "ValueError"
    assert not destination.exists()


def test_exact_plan_digest_and_missing_bundle_block_before_demo_bootstrap(
    accepted, tmp_path, monkeypatch
):
    from src.zeitgeist import adk_pipeline

    monkeypatch.setattr(
        adk_pipeline,
        "run_daily_zeitgeist",
        lambda **kwargs: pytest.fail("Export must never bootstrap a publication"),
    )
    bundle, plan, commit = accepted
    with pytest.raises(ValueError):
        export.export_release(
            export.ROOT, commit, bundle, plan, "0" * 64, tmp_path / "wrong-plan"
        )
    (bundle / "data/zeitgeist_state.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        perform(accepted, tmp_path / "missing-bundle")


def test_final_code_and_commissioning_commits_are_reported_separately(
    accepted, tmp_path
):
    envelope = copy.deepcopy(accepted[1])
    envelope["plan"]["source_sha"] = "a" * 40
    envelope["plan_sha256"] = promotion.digest(envelope["plan"])
    report = perform(
        (accepted[0], envelope, accepted[2]), tmp_path / "export", dry_run=True
    )
    assert report["source_sha"] == accepted[2]
    assert report["commissioning_source_sha"] == "a" * 40
    assert report["commissioning_run"] == "41000000000"


def test_cli_private_report_is_separate_and_mode_600(accepted, tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(accepted[1]))
    destination = tmp_path / "not-created"
    report_path = tmp_path / "private/inventory.json"
    args = [
        "--source-root",
        str(export.ROOT),
        "--source-sha",
        accepted[2],
        "--bundle-root",
        str(accepted[0]),
        "--plan",
        str(path),
        "--expected-plan-sha",
        accepted[1]["plan_sha256"],
        "--output-dir",
        str(destination),
        "--private-inventory",
        str(report_path),
        "--dry-run",
    ]
    assert export.main(args) == 0
    assert not destination.exists()
    assert report_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(report_path.read_text())["export_files"]
    assert json.loads(capsys.readouterr().out)["production_authorized"] is False


def test_identical_json_symlink_outside_accepted_bundle_is_rejected(accepted, tmp_path):
    original = accepted[0] / "data/zeitgeist_state.json"
    outside = tmp_path / "outside-accepted-bundle.json"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(ValueError):
        perform(accepted, tmp_path / "export")
