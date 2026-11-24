"""Tests for the S1 edition look (SIY-204): planning, drawing and the router switch."""

from __future__ import annotations

import datetime as dt
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.zeitgeist import edition_look as L


def _photo(kind: str) -> Image.Image:
    img = Image.new("RGB", (1200, 800), (30, 34, 40))
    d = ImageDraw.Draw(img)
    if kind == "light":
        img.paste((236, 230, 220), (0, 0, 1200, 800))
        d.rectangle([500, 300, 560, 700], fill=(120, 60, 30))
    elif kind == "dark":
        d.ellipse([480, 300, 720, 540], fill=(200, 80, 60))
    elif kind == "grey":
        img.paste((128, 128, 128), (0, 0, 1200, 800))
        d.rectangle([300, 200, 900, 600], fill=(60, 60, 60))
    elif kind == "busy":
        for x in range(0, 1200, 6):
            d.line([(x, 0), (x, 800)], fill=(250, 248, 244) if x % 12 else (10, 10, 10), width=2)
    return img


def _dates(n: int, start: str = "2026-12-01") -> list[str]:
    d0 = dt.date.fromisoformat(start)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)]


CREDIT = "Photograph by Test Photographer"
GLOSS = "Mythmaking woven through modern real-life drama."


def test_plan_never_repeats_previous_template_and_uses_all() -> None:
    history: list[str] = []
    seen = []
    for i, day in enumerate(_dates(30)):
        photo = _photo(["light", "dark", "grey"][i % 3])
        plan = L.plan_night(day, history, photo, word="Fabulation", gloss=GLOSS, credit=CREDIT)
        assert plan.template in L.TEMPLATES
        if history:
            assert plan.template != history[0]
        history.insert(0, plan.template)
        seen.append(plan.template)
    off_photo = set(L.TEMPLATES) - set(L.ON_PHOTO)
    assert off_photo <= set(seen)


def test_colour_on_half_the_nights_and_never_on_grey() -> None:
    colour = [L.plan_night(d, [], _photo("dark"), word="Relic", gloss=GLOSS, credit=CREDIT).colour_on for d in _dates(8)]
    assert colour.count(True) == 4
    assert all(a != b for a, b in zip(colour, colour[1:]))
    grey = [L.plan_night(d, [], _photo("grey"), word="Relic", gloss=GLOSS, credit=CREDIT).colour_on for d in _dates(8)]
    assert not any(grey)
    assert L.analyse_photo(_photo("grey")).colour is None


def test_on_photo_templates_need_calm() -> None:
    history: list[str] = []
    for day in _dates(14):
        plan = L.plan_night(day, history, _photo("busy"), word="Breach", gloss=GLOSS, credit=CREDIT)
        assert plan.template not in L.ON_PHOTO
        history.insert(0, plan.template)


@pytest.mark.parametrize("template", L.TEMPLATES)
@pytest.mark.parametrize("voice", L.VOICES)
def test_every_template_renders_both_themes(template: str, voice: str) -> None:
    photo = _photo("dark")
    face = L.PAIRS[voice][0]
    placement = None
    if template in L.ON_PHOTO:
        placement = L.place_on_photo(photo, template, voice, face, "Climacteric", GLOSS, "2026-12-02", CREDIT)
        if placement is None:
            pytest.skip("synthetic photograph has no calm placement for this lettering")
    plan = L.NightPlan(template, voice, face, True, "rule" if template not in L.ON_PHOTO else "word", False, placement)
    light, dark = L.render_edition_pair(photo, word="Climacteric", gloss=GLOSS, date_iso="2026-12-02", credit=CREDIT, plan=plan)
    for img in (light, dark):
        assert img.size[0] == 1200
        assert 800 <= img.size[1] <= 1200


@pytest.mark.parametrize("voice", L.VOICES)
def test_long_word_shrinks_to_its_measure(voice: str) -> None:
    for limit in (290, 560, 640, 936):
        st = L.fit_word(voice, "Climacteric", limit)
        assert st.width("Climacteric") <= limit


def test_gloss_breaks_balanced_without_small_word_endings() -> None:
    st = L.gloss_style("italic")
    text = "Claiming a future triumph before proof arrives in town."
    lines = L.break_gloss(text, st, 600)
    assert len(lines) == 2
    assert " ".join(lines) == text
    for line in lines[:-1]:
        assert line.split()[-1].lower() not in L.STOPWORDS
    for line in L.wrap_greedy(text, st, 290)[:-1]:
        assert line.split()[-1].lower() not in L.STOPWORDS


def test_colophon_carries_number_date_and_credit() -> None:
    assert L.edition_number("2026-11-24") == 25
    full = L.colophon_text("2026-11-24", CREDIT, "full")[0]
    assert full == "No. 25 · 24 November 2026 · Photograph by Test Photographer"
    assert L.colophon_text("2026-11-24", CREDIT, "museum") == ["Test Photographer", "No. 25", "24 November 2026"]


def _router_run(tmp_path: Path, monkeypatch=None, switch: bool = True, archetype_history=None):
    from src.zeitgeist.adk_pipeline import DirectorCommission
    from src.zeitgeist.visual_router import select_and_render_pulse_assets

    if switch:
        (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "assets" / "edition_look.json").write_text(json.dumps({"look": "s1"}), encoding="utf-8")
    buf = BytesIO()
    _photo("light").save(buf, format="JPEG", quality=90)
    commission = DirectorCommission(
        word="Solastalgia",
        gloss="Homesickness felt while still at home.",
        commissioned_thesis="Quiet coastal bluff horizon under evening mist.",
        unsplash_queries=["coastal bluff mist", "minimalist horizon", "quiet sea"],
        imagen_commission_prompt="Minimalist 3:2 coastal horizon in twilight fog",
    )
    candidates = [{
        "id": "bluff-1",
        "exif": {"make": "Fixture Camera", "model": "Fixture Camera Model"},
        "photographer_name": "Elena Rostova",
        "photographer_url": "https://unsplash.com/@erostova",
        "unsplash_url": "https://unsplash.com/photos/bluff-1",
        "download_location": "https://api.unsplash.com/photos/bluff-1/download",
        "semantic_alignment": 0.75,
        "compositional_clarity": 0.72,
        "editorial_memorability": 0.70,
        "safety_pass": True,
        "image_bytes": buf.getvalue(),
    }]
    return select_and_render_pulse_assets(
        commission,
        medium_history_30d=["FLUX_PRO"] * 6 + ["PHOTOGRAPHY"] * 24,
        repo_root=tmp_path,
        date_str="2026-12-03",
        candidates=candidates,
        download_ping_callback=lambda *_: None,
        archetype_history_30d=archetype_history or ["LITERARY_JOURNAL"],
    )


def test_router_uses_edition_look_when_switched_on(tmp_path: Path) -> None:
    published = _router_run(tmp_path)
    assert published.curatorial_archetype in L.TEMPLATES
    with Image.open(tmp_path / "assets" / "daily_pulse_dark.webp") as dark:
        assert dark.size[0] == 1200
        size = dark.size
    svg = (tmp_path / "assets" / "daily_pulse.svg").read_text(encoding="utf-8")
    assert f'width="{size[0]}" height="{size[1]}"' in svg
    assert f'viewBox="0 0 {size[0]} {size[1]}"' in svg


def test_router_keeps_current_look_without_switch(tmp_path: Path) -> None:
    published = _router_run(tmp_path, switch=False)
    assert published.curatorial_archetype not in L.TEMPLATES


def test_router_falls_back_on_error(tmp_path: Path, monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(L, "render_edition_pair", boom)
    published = _router_run(tmp_path)
    assert published.curatorial_archetype not in L.TEMPLATES
    assert (tmp_path / "assets" / "daily_pulse_light.webp").stat().st_size > 1000
    assert (tmp_path / "assets" / "daily_pulse_dark.webp").stat().st_size > 1000
