"""Future Kyoto finishing must not alter already accepted publications."""
import copy
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from src import state_store
from src.zeitgeist import curatorial_compositor as compositor
from src.zeitgeist.publication import verify_publication_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("word", ["Assemblage", "Counterbalance", "Transformation"])
def test_new_kyoto_has_unmarked_spine_foot(dark, word):
    plate = compositor._render_kyoto_tankobon_plate(
        Image.new("RGB", (1200, 800), (110, 120, 130)), word=word,
        gloss="A shared call to gather in communal sanctuary.", date_str="2026-11-02",
        attribution={"photographer_name": "Retained creator"},
        visual_medium="PHOTOGRAPHY", dark_mode=dark,
    )
    background = (22, 28, 36) if dark else (244, 240, 232)
    assert plate.crop((937, 668, 1200, 800)).getextrema() == tuple(
        (channel, channel) for channel in background
    )


@pytest.mark.parametrize("art_location", ["checkout", "retained_recovery"])
def test_rebuild_reuses_accepted_bundle_without_running_compositor(
    tmp_path, monkeypatch, art_location
):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    for name in ("assets", "data", "config", "variants", "docs"):
        if (ROOT / name).exists():
            shutil.copytree(ROOT / name, source / name)
    for name in ("README.md", "PROJECTS.md"):
        shutil.copy2(ROOT / name, source / name)
    manifest = verify_publication_bundle(source, require_current_pulse=True)
    documents = {
        name: json.loads((source / "data" / name).read_text())
        for name in ("zeitgeist_state.json", "pulse_feed.json", "game_state.json",
                     "hall_of_records.json", "bot_brain.json", "orbit_roster.json")
    }
    before = {path: (source / path).read_bytes()
              for path in ("assets/pulse_manifest.json", *manifest["files"])}
    shutil.copytree(source, target)
    if art_location == "retained_recovery":
        retained = source / ".publication-recovery" / manifest["publication_id"]
        shutil.copytree(target, retained)
        for root in (source, target):
            (root / "assets/pulse_manifest.json").unlink()
    monkeypatch.setattr(state_store, "_canonical_publication_snapshot",
                        lambda store: copy.deepcopy(documents))

    def no_render(*args, **kwargs):
        pytest.fail("Rebuilding an accepted publication must reuse its artwork")

    monkeypatch.setattr(compositor, "render_curatorial_plate_pair", no_render)
    monkeypatch.setattr(compositor, "render_curatorial_plates", no_render)
    monkeypatch.setenv("TARGET_REPO_SLUG", "Siyuan-Zhu/Siyuan-Zhu")
    state_store._rebuild_publication(
        target, source, object(), expected_publication_id=manifest["publication_id"]
    )
    assert verify_publication_bundle(target, require_current_pulse=True) == manifest
    assert {path: (target / path).read_bytes() for path in before} == before
