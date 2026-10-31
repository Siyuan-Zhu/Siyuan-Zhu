"""A fresh pulse must project the canonical game, regardless of Git mirror age."""

from argparse import Namespace
import copy
import json
import re

import pytest

from src import state_store as stores
from src.game_engine import (
    create_initial_game_state,
    normalize_game_state,
    render_board_svg,
)
from src.zeitgeist import adk_pipeline
from tests.test_publication_atomic_commit import _publication_fixture


def _stale_checkout(root, wire, *, local_version):
    canonical = create_initial_game_state()
    canonical.update(
        turn=6, state_version=4, moves_this_round=5, current_turn_color="YELLOW"
    )
    canonical["board"][5][:5] = ["C", "B", "C", "B", "C"]
    canonical = normalize_game_state(canonical)
    local = create_initial_game_state()
    local.update(turn=4, state_version=local_version, moves_this_round=3)
    local["board"][5][:3] = ["C", "B", "C"]
    local = normalize_game_state(local)
    canonical_elo = {
        "state_version": 4,
        "bot_elo": 1200,
        "players": {"canonical-player": {"elo": 1575}},
        "recent_rounds": [],
    }
    local_elo = {
        "state_version": local_version,
        "bot_elo": 1200,
        "players": {"checkout-player": {"elo": 1999}},
        "recent_rounds": [],
    }
    local_store = stores.JsonFileStateStore(root)
    local_store.save_game_state(local)
    local_store.save_elo_ledger(local_elo)
    (root / "assets/game_board.svg").write_text(
        render_board_svg(local, elo_ledger=local_elo)
    )
    wire.docs["staging_profile_state/active_game"] = copy.deepcopy(canonical)
    wire.docs["staging_profile_state/elo_ledger"] = copy.deepcopy(canonical_elo)
    return canonical, canonical_elo


def _manual(root):
    from scripts import orbit

    return orbit.cmd_interject(
        root,
        Namespace(
            word="ANCHORED",
            phrase="Steady craft.",
            mode="blueprint",
            image_url=None,
            note=None,
            channel="test",
        ),
    )


@pytest.mark.parametrize("entrypoint", ["orchestrator", "manual"])
@pytest.mark.parametrize("local_version", [3, 4, 5])
def test_canonical_game_and_elo_replace_stale_mirrors_and_existing_svg(
    tmp_path, monkeypatch, entrypoint, local_version
):
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    canonical, elo = _stale_checkout(tmp_path, wire, local_version=local_version)
    before_versions = {
        key: wire.versions[key]
        for key in (
            "staging_profile_state/active_game",
            "staging_profile_state/elo_ledger",
        )
    }
    monkeypatch.setattr(stores, "get_state_store", lambda **kwargs: store)
    if entrypoint == "orchestrator":
        adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
            "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
        )
    else:
        assert _manual(tmp_path) == 0
    for path in [tmp_path / "README.md", *(tmp_path / "variants").glob("*.md")]:
        turns = re.findall(r"game:(?:move|mode):[^\s)>]+?turn=(\d+)", path.read_text())
        assert turns and set(turns) == {"6"}
    assert (tmp_path / "assets/game_board.svg").read_text() == render_board_svg(
        canonical, elo_ledger=elo
    )
    assert json.loads((tmp_path / "data/game_state.json").read_text()) == canonical
    assert json.loads((tmp_path / "data/hall_of_records.json").read_text()) == elo
    assert wire.docs["staging_profile_state/active_game"] == canonical
    assert wire.docs["staging_profile_state/elo_ledger"] == elo
    assert {key: wire.versions[key] for key in before_versions} == before_versions


def test_game_projection_is_staged_before_collection_or_paid_providers(
    tmp_path, monkeypatch
):
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    canonical, elo = _stale_checkout(tmp_path, wire, local_version=4)
    collect = adk_pipeline.collect_96h_signals
    observed = []

    def inspect(root, **kwargs):
        assert json.loads((root / "data/game_state.json").read_text()) == canonical
        assert json.loads((root / "data/hall_of_records.json").read_text()) == elo
        observed.append(True)
        return collect(root, **kwargs)

    monkeypatch.setattr(adk_pipeline, "collect_96h_signals", inspect)
    adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
        "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
    )
    assert observed == [True]


def test_missing_canonical_game_fails_before_provider_and_never_seeds_checkout(
    tmp_path, monkeypatch
):
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    wire.docs.pop("staging_profile_state/active_game")
    wire.versions.pop("staging_profile_state/active_game")
    before = copy.deepcopy(wire.docs)
    monkeypatch.setattr(
        adk_pipeline,
        "collect_96h_signals",
        lambda *a, **k: pytest.fail("No providers when canonical game is absent"),
    )
    with pytest.raises(RuntimeError, match="[Cc]anonical game"):
        adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
            "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
        )
    assert wire.docs == before and not wire.commits


@pytest.mark.parametrize("boundary", ["readme", "commit"])
def test_rejected_publication_keeps_existing_game_files_and_remote_game_untouched(
    tmp_path, monkeypatch, boundary
):
    from scripts import build_readme

    wire, store = _publication_fixture(tmp_path, monkeypatch)
    _stale_checkout(tmp_path, wire, local_version=4)
    paths = (
        "README.md",
        "data/game_state.json",
        "data/hall_of_records.json",
        "assets/game_board.svg",
    )
    before_files = {path: (tmp_path / path).read_bytes() for path in paths}
    before_remote = copy.deepcopy(wire.docs)
    if boundary == "readme":

        def fail(*args, **kwargs):
            raise ValueError("README invalid")

        monkeypatch.setattr(build_readme, "build_all_readmes", fail)
    else:
        wire.reject = 403
    with pytest.raises((RuntimeError, ValueError)):
        adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
            "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
        )
    assert {path: (tmp_path / path).read_bytes() for path in paths} == before_files
    assert wire.docs == before_remote


def test_absent_canonical_ratings_render_empty_instead_of_checkout_ratings(
    tmp_path, monkeypatch
):
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    _stale_checkout(tmp_path, wire, local_version=5)
    wire.docs.pop("staging_profile_state/elo_ledger")
    wire.versions.pop("staging_profile_state/elo_ledger")
    adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
        "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
    )
    svg = (tmp_path / "assets/game_board.svg").read_text()
    assert "No recorded ratings" in (tmp_path/"README.md").read_text() and "checkout-player" not in (tmp_path/"README.md").read_text()
    assert "elo-leaderboard" not in svg
    assert (
        json.loads((tmp_path / "data/hall_of_records.json").read_text())["players"]
        == {}
    )
    assert "staging_profile_state/elo_ledger" not in wire.docs


def test_invalid_canonical_board_fails_closed_before_collection(tmp_path, monkeypatch):
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    wire.docs["staging_profile_state/active_game"] = {"turn": 6, "state_version": 4}
    monkeypatch.setattr(
        adk_pipeline,
        "collect_96h_signals",
        lambda *a, **k: pytest.fail("No providers for invalid canonical game"),
    )
    with pytest.raises(RuntimeError, match="Canonical game invalid"):
        adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
            "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
        )
    assert not wire.commits


def test_existing_legacy_game_mirror_is_projected_with_the_same_canonical_turn(
    tmp_path, monkeypatch
):
    from scripts.validate_release_links import _load_active_game_lock

    wire, store = _publication_fixture(tmp_path, monkeypatch)
    canonical, elo = _stale_checkout(tmp_path, wire, local_version=4)
    (tmp_path / "data/connect4_state.json").write_text('{"turn":4}')
    adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
        "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
    )
    errors = []
    assert _load_active_game_lock(tmp_path, errors) == (6, canonical["round_id"]) and not errors
    assert json.loads((tmp_path / "data/connect4_state.json").read_text()) == canonical
