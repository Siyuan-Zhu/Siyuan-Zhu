"""Release acceptance for editorial presentation and publication asset consistency."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from src.game_engine import create_initial_game_state, render_board_svg, render_readme_game_section
from src.zeitgeist.curatorial_compositor import ALL_ARCHETYPES, render_curatorial_plate_pair
from src.zeitgeist.visual_router import select_and_render_pulse_assets, update_svg_mood_vote_bar


def test_editorial_game_controls_preserve_turn_lock_and_accessibility():
    state = create_initial_game_state()
    section = render_readme_game_section(state, repo_slug="Example/profile")
    assert "connect_four_title_light.webp" in section
    assert "How it works" not in section
    for col, numeral in enumerate(("I", "II", "III", "IV", "V", "VI", "VII"), 1):
        assert f"[{numeral}](https://github.com/Example/profile/issues/new?" in section
        assert f"col:{col}:turn={state['turn']}" in section
    svg = render_board_svg(state)
    root = ET.fromstring(svg)
    assert root.get("role") == "img"
    assert "#FAF8F5" in svg
    assert "#C85A32" in svg
    assert "#1C1917" in svg
    assert "#38bdf8" not in svg


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_every_archetype_renders_both_themes_with_bundled_fonts(archetype, tmp_path):
    from src.zeitgeist.curatorial_compositor import _load_font, _SERIF_FONT_CANDIDATES
    font = _load_font(_SERIF_FONT_CANDIDATES, 48)
    assert "assets/fonts/CormorantGaramond" in str(font.path)
    light, dark = render_curatorial_plate_pair(
        Image.new("RGB", (1200, 800), (90, 110, 120)), word="Counterpoise",
        gloss="A steady balance across changing currents.", archetype=archetype, repo_root=tmp_path,
    )
    assert light.size == dark.size == (1200, 800)
    assert light.tobytes() != dark.tobytes() or archetype == "POETIC_CINEMA"


def test_publication_rejects_stale_or_swapped_theme_asset(tmp_path):
    from src.zeitgeist.publication import verify_publication_bundle
    pulse = select_and_render_pulse_assets(
        commission={"word": "Counterpoise", "gloss": "A steady balance.", "commissioned_thesis": "A steady balance."},
        medium_history_30d=[], repo_root=tmp_path, date_str="2026-10-29", candidates=[],
        imagen_bytes=_jpeg(), curatorial_archetype_override="LITERARY_JOURNAL",
    )
    manifest = verify_publication_bundle(tmp_path)
    assert manifest["word"] == pulse.word
    assert manifest["date"] == pulse.date
    svg = ET.parse(tmp_path / pulse.svg_asset_path).getroot()
    assert svg.get("viewBox") == "0 0 1200 800"
    image = next(n for n in svg if n.tag.endswith("image"))
    assert image.get("href").startswith("data:image/webp;base64,")
    # Votes remain external to the image; updating accessible metadata preserves artwork.
    update_svg_mood_vote_bar(tmp_path / pulse.svg_asset_path, {"spot_on": 31, "too_sunny": 3, "too_heavy": 4})
    assert verify_publication_bundle(tmp_path)["publication_id"] == manifest["publication_id"]
    (tmp_path / "assets/daily_pulse_light.webp").write_bytes((tmp_path / "assets/daily_pulse_dark.webp").read_bytes())
    with pytest.raises(ValueError, match="daily_pulse_light"):
        verify_publication_bundle(tmp_path)


def _jpeg():
    from io import BytesIO
    out = BytesIO()
    Image.new("RGB", (1200, 800), (90, 110, 120)).save(out, format="JPEG")
    return out.getvalue()


def test_photo_download_failure_never_becomes_synthetic_unsplash(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from src.zeitgeist.visual_router import fetch_curated_library_candidates
    client = MagicMock()
    client.__enter__.return_value.get.return_value.status_code = 404
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    assert fetch_curated_library_candidates("Counterpoise") == []
    pulse = select_and_render_pulse_assets(
        commission={"word": "Counterpoise", "gloss": "A steady balance."},
        medium_history_30d=[], repo_root=tmp_path, date_str="2026-10-29",
        candidates=[{"id": "missing", "semantic_alignment": .9, "compositional_clarity": .9,
                     "editorial_memorability": .9, "safety_pass": True}],
    )
    assert pulse.visual_medium == "BESPOKE_SVG"
    assert not pulse.attribution.get("unsplash_url")


def test_budget_denial_propagates_before_bfl_request_and_through_tournament(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    from unittest.mock import MagicMock
    from src.zeitgeist import visual_router
    from src.zeitgeist.spend_guard import SpendBlocked
    client = MagicMock()
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    guard = MagicMock()
    guard.begin_provider_request.return_value = None  # Explicit legacy policy fixture.
    guard.call.side_effect = SpendBlocked("Unverified credit balance")
    monkeypatch.setattr(visual_router, "get_spend_guard", lambda **kwargs: guard)
    with pytest.raises(SpendBlocked, match="Unverified"):
        visual_router.run_flux_3_tournament(["one", "two"], api_key="dummy")
    client.__enter__.return_value.post.assert_not_called()


def test_budget_denial_never_becomes_a_generated_fallback(monkeypatch, tmp_path):
    from src.zeitgeist import visual_router
    from src.zeitgeist.spend_guard import SpendBlocked
    from google import genai
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: object())
    def deny(_client, **kwargs):
        raise SpendBlocked("Monthly cap reached")
    monkeypatch.setattr(visual_router, "guard_genai_client", deny)
    with pytest.raises(SpendBlocked, match="Monthly cap"):
        select_and_render_pulse_assets(
            commission={"word": "Counterpoise", "gloss": "A steady balance."},
            medium_history_30d=[], repo_root=tmp_path, date_str="2026-10-29", candidates=[],
        )
    assert not (tmp_path / "assets/pulse_manifest.json").exists()


def test_rendered_publication_records_actual_utc_time_instead_of_a_fixed_hour(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    from src.zeitgeist import visual_router
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 10, 30, 5, 3, 21, tzinfo=timezone.utc)
    monkeypatch.setattr(visual_router, "datetime", Clock)
    pulse = select_and_render_pulse_assets(
        {"word": "Anchored", "gloss": "Steady craft."}, repo_root=tmp_path,
        date_str="2026-10-30", candidates=[], requested_medium="blueprint")
    assert pulse.published_at == "2026-10-30T05:03:21Z"
    assert pulse.to_feed_entry()["published_at"] == pulse.published_at
