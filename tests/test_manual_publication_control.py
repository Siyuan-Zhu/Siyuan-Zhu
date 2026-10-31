"""Owner control must produce one coherent v3 publication or preserve the old one."""
from argparse import Namespace
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import orbit
from src.zeitgeist import adk_pipeline as pipeline, visual_router


def _workspace(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text(json.dumps({"active_interjection": {"enabled": False}}))
    return tmp_path


def _args(**changes):
    values = dict(word="ANCHORED", phrase="Steady craft across shifting currents.", mode="auto",
                  image_url=None, note="A quiet workshop beside the harbor.", channel="test")
    return Namespace(**(values | changes))


@pytest.mark.parametrize("changes", [
    {"word": "<b>FOCUS</b>"}, {"phrase": '<img src=x onerror="bad">'},
    {"phrase": "word " * 13}, {"mode": "video"}, {"word": "কিভাবে"},
])
def test_invalid_owner_input_never_writes_configuration(tmp_path, changes):
    root = _workspace(tmp_path)
    original = (root / "config/orbit_sources.json").read_bytes()
    with pytest.raises(ValueError):
        orbit.cmd_interject(root, _args(**changes))
    assert (root / "config/orbit_sources.json").read_bytes() == original
    assert not (root / "assets").exists()


def test_provider_failure_restores_owner_configuration_and_last_publication(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    (root / "assets").mkdir()
    (root / "assets/daily_pulse.svg").write_text("old-publication")
    original = (root / "config/orbit_sources.json").read_bytes()
    def fail(**kwargs):
        raise RuntimeError("Provider unavailable")
    monkeypatch.setattr(orbit, "run_daily_zeitgeist", fail)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        orbit.cmd_interject(root, _args())
    assert (root / "config/orbit_sources.json").read_bytes() == original
    assert (root / "assets/daily_pulse.svg").read_text() == "old-publication"


def _grounded_director(monkeypatch):
    original = pipeline.commission_daily_zeitgeist
    def llm(prompt, temperature):
        assert "ANCHORED" in prompt
        assert "Steady craft across shifting currents." in prompt
        return dict(word="ANCHORED", gloss="Steady craft across shifting currents.",
                    commissioned_thesis="A harbor anchor gives craft stability under changing tides.",
                    visual_anchor="Weathered iron anchor beside a quiet harbor workshop",
                    unsplash_queries=["harbor iron anchor", "workshop weathered timber", "calm mooring harbor"],
                    imagen_commission_prompt="Weathered iron anchor beside quiet harbor timber, no text.")
    def commission(*args, **kwargs):
        kwargs["llm_callable"] = llm
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", commission)


def _photo():
    out = BytesIO()
    Image.new("RGB", (1200, 800), (90, 110, 120)).save(out, format="JPEG")
    return {"id": "fixture", "image_bytes": out.getvalue(), "semantic_alignment": .9,
            "compositional_clarity": .9, "editorial_memorability": .9, "safety_pass": True,
            "photographer_name": "Test fixture", "unsplash_url": "https://unsplash.com/photos/fixture"}


@pytest.mark.parametrize("mode,medium", [("auto", "PHOTOGRAPHY"), ("photo", "PHOTOGRAPHY"), ("blueprint", "BESPOKE_SVG")])
def test_phone_override_is_commissioned_before_all_representations(mode, medium, tmp_path, monkeypatch):
    from src.zeitgeist.publication import verify_publication_bundle
    root = _workspace(tmp_path)
    monkeypatch.setenv("ZEITGEIST_OFFLINE_DEMO", "1")
    monkeypatch.setenv("STATE_STORE_BACKEND", "json")
    _grounded_director(monkeypatch)
    select = pipeline.select_and_render_pulse_assets
    def render(**kwargs):
        kwargs["candidates"] = [_photo()]
        return select(**kwargs)
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    assert orbit.cmd_interject(root, _args(mode=mode)) == 0
    state = json.loads((root / "data/zeitgeist_state.json").read_text())
    manifest = verify_publication_bundle(root)
    assert manifest["word"] == state["analysis"]["word_of_the_day"] == "ANCHORED"
    assert manifest["medium"] == medium
    assert manifest["date"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert state["commission"]["visual_anchor"].startswith("Weathered iron anchor")
    assert "ANCHORED" in (root / "README.md").read_text()
    assert "Steady craft across shifting currents." in (root / "README.md").read_text()


def test_forced_photo_failure_never_generates_or_overwrites(tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/daily_pulse.svg").write_text("old-publication")
    def forbidden(*a, **k):
        pytest.fail("Photo mode must not call a generative provider")
    monkeypatch.setattr(visual_router, "_generate_live_imagen3_bytes", forbidden)
    monkeypatch.setattr(visual_router, "_generate_live_flux_bytes", forbidden)
    with pytest.raises(ValueError, match="photograph"):
        visual_router.select_and_render_pulse_assets(
            {"word": "ANCHORED", "gloss": "Steady craft."}, repo_root=tmp_path,
            candidates=[], requested_medium="photo",
        )
    assert (tmp_path / "assets/daily_pulse.svg").read_text() == "old-publication"


def test_owner_can_repeat_word_but_director_cannot_ignore_constraints():
    override = {"enabled": True, "override_word": "ANCHORED", "override_phrase": "Steady craft."}
    def wrong(prompt):
        return dict(word="UNRELATED", gloss="An unrelated story.", commissioned_thesis="A different story entirely.")
    with pytest.raises(ValueError, match="Owner-constrained"):
        pipeline.commission_daily_zeitgeist([], ["ANCHORED"], owner_override=override, llm_callable=wrong)


def test_owner_explicit_repeat_is_allowed_with_grounded_new_commission():
    def grounded(prompt):
        return dict(word="ANCHORED", gloss="Steady craft.", commissioned_thesis="Harbor anchors steady the workshop under changing tides.",
                    visual_anchor="Weathered iron anchor beside quiet harbor timber",
                    unsplash_queries=["harbor iron anchor", "weathered workshop timber", "quiet harbor mooring"])
    result = pipeline.commission_daily_zeitgeist([], ["ANCHORED"],
        owner_override={"enabled": True, "override_word": "ANCHORED", "override_phrase": "Steady craft."},
        llm_callable=grounded)
    assert result.word == "ANCHORED"
    assert result.fallback_used is False


def test_blueprint_never_calls_an_image_or_vision_provider(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("Deterministic blueprint must not call an image or vision provider")
    for name in ("_fetch_live_unsplash_candidates", "fetch_curated_library_candidates",
                 "_generate_live_imagen3_bytes", "_generate_live_flux_bytes", "_score_candidate_with_gemini_vision"):
        monkeypatch.setattr(visual_router, name, forbidden)
    pulse = visual_router.select_and_render_pulse_assets(
        {"word": "ANCHORED", "gloss": "Steady craft."}, repo_root=tmp_path,
        requested_medium="blueprint", require_semantic_rescore=True,
    )
    assert pulse.visual_medium == "BESPOKE_SVG"
    assert pulse.curatorial_archetype == "CONCRETE_POETRY"


def test_manual_photo_does_not_trust_unrelated_static_semantic_score(tmp_path, monkeypatch):
    monkeypatch.setattr(visual_router, "_score_candidate_with_gemini_vision",
        lambda **kwargs: {"semantic_alignment": 0, "compositional_clarity": 0, "editorial_memorability": 0, "safety_pass": False})
    with pytest.raises(ValueError, match="photograph"):
        visual_router.select_and_render_pulse_assets(
            {"word": "ANCHORED", "gloss": "Steady craft."}, repo_root=tmp_path,
            candidates=[_photo()], requested_medium="photo", require_semantic_rescore=True,
        )
    assert not (tmp_path / "assets").exists()
