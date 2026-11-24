"""Tests for the S1 edition look (SIY-204): planning, drawing and the router switch."""

from __future__ import annotations

import datetime as dt
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

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
        assert img.size[0] == 2400
        assert 1400 <= img.size[1] <= 2400


def _plan_for(template: str, voice: str, photo: Image.Image, word: str = "Climacteric"):
    face = L.PAIRS[voice][0]
    placement = None
    if template in L.ON_PHOTO:
        placement = L.place_on_photo(photo, template, voice, face, word, GLOSS, "2026-12-02", CREDIT)
        if placement is None:
            pytest.skip("synthetic photograph has no calm placement for this lettering")
    return L.NightPlan(template, voice, face, True, "rule" if template not in L.ON_PHOTO else "word", False, placement)


def _pair(photo, plan, word="Climacteric", **kw):
    return L.render_edition_pair(photo, word=word, gloss=GLOSS, date_iso="2026-12-02", credit=CREDIT, plan=plan, **kw)


def _text_lines(plan, photo, word="Climacteric"):
    """(text, style, align, x, baseline) for every line of type, on the 1200 px grid."""
    if plan.template in L.ON_PHOTO:
        return list(plan.placement["lines"]) + list(plan.placement["colophon"])
    ops, _, _ = LAYOUTS[plan.template](plan, word, GLOSS, "2026-12-02", CREDIT)
    return [(op[2], op[3], op[4], op[5], op[6]) for op in ops if op[0] == "text"]


def _best_shift(a, b, rect, reach: int = 3):
    """The (dx, dy) that best lays b's line of type over a's, searched within reach pixels."""
    x, y, w, h = (int(v) for v in rect)
    y0, y1 = max(reach, y - 4), min(a.shape[0] - reach, y + h + 4)
    x0, x1 = max(reach, x - 4), min(a.shape[1] - reach, x + w + 4)
    ref = a[y0:y1, x0:x1]
    cost = {(dx, dy): np.abs(ref - b[y0 + dy:y1 + dy, x0 + dx:x1 + dx]).mean()
            for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)}
    return min(cost, key=cost.get)


@pytest.mark.parametrize("template", L.TEMPLATES)
def test_plates_are_drawn_at_double_resolution(template: str) -> None:
    photo = _photo("dark")
    plan = _plan_for(template, "italic", photo)
    one = _pair(photo, plan, scale=1)
    two = _pair(photo, plan)
    assert L.SCALE == 2
    for a, b in zip(one, two):
        assert b.size == (a.size[0] * 2, a.size[1] * 2)
        assert b.size[0] == 2400


@pytest.mark.parametrize("template", L.TEMPLATES)
@pytest.mark.parametrize("voice", L.VOICES)
def test_double_resolution_matches_the_layout(template: str, voice: str) -> None:
    photo = _photo("dark")
    plan = _plan_for(template, voice, photo)
    one, two = _pair(photo, plan, scale=1)[0], _pair(photo, plan, scale=2)[0]
    down = two.resize(one.size, Image.Resampling.BOX)
    lines = _text_lines(plan, photo)
    assert lines
    a, b = np.asarray(one, dtype=np.float64), np.asarray(down, dtype=np.float64)
    assert np.abs(a - b).mean() < 2                       # photograph and plate in the same place
    # 1x glyphs are hinted to the pixel grid and 2x glyphs are not, so the bands are compared
    # softened to display sharpness; placement itself is held by the best-shift search below
    a = np.asarray(one.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float64)
    b = np.asarray(down.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float64)
    for _x, y, _w, h in L._rects(lines):
        band = slice(max(0, int(y) - 4), int(y + h) + 4)   # the full-width band the line sits in
        assert np.abs(a[band] - b[band]).mean() < 4
    for line, rect in zip(lines, L._rects(lines)):
        dx, dy = _best_shift(a, b, rect)
        assert abs(dx) <= 1 and abs(dy) <= 1, (line[0], dx, dy)


def test_small_hero_still_renders() -> None:
    small = _photo("dark").resize((600, 400))
    for template in ("S1_PLATE", "S1_BLEED", "S1_WALL"):
        plan = _plan_for(template, "wide", small)
        for img in _pair(small, plan):
            assert img.size[0] == 2400


def test_scaled_style_keeps_its_face_and_tracking() -> None:
    from src.zeitgeist import caesura as C

    for st in (L.voice_style("wide"), L.voice_style("condensed"), L.gloss_style("sans"), L.col_style(),
               C.word_style("wide", "Solastalgia"), C.word_style("italic", "RELIC")):
        s2 = st.scaled(2)
        assert (s2.key, s2.weight, s2.upper) == (st.key, st.weight, st.upper)
        assert s2.size == 2 * st.size
        assert s2.tracking == pytest.approx(2 * st.tracking)
        assert s2.line_height == pytest.approx(2 * st.line_height)
        for text in ("Solastalgia", "No. 25 · 24.11.2026"):
            assert s2.width(text) == pytest.approx(2 * st.width(text))
    tracked = C.word_style("wide", "Solastalgia")
    assert tracked.tracking == pytest.approx(C.LONG_TRACK * tracked.size)
    assert tracked.scaled(2).tracking == pytest.approx(C.LONG_TRACK * tracked.scaled(2).size)


def test_text_leaving_the_double_resolution_plate_raises() -> None:
    plan = L.NightPlan("S1_WALL", "wide", "italic", False, "none", False)
    with pytest.raises(ValueError):
        L.render_edition_pair(_photo("dark"), word="Relic", gloss=GLOSS, date_iso="2026-12-02",
                              credit="Photograph by " + "Averyveryverylongphotographername" * 2, plan=plan, scale=2)


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
        assert dark.size[0] == 2400
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


def test_text_leaving_the_plate_raises_so_the_router_falls_back() -> None:
    plan = L.NightPlan("S1_WALL", "wide", "italic", False, "none", False)
    with pytest.raises(ValueError):
        L.render_edition_pair(_photo("dark"), word="Relic", gloss=GLOSS, date_iso="2026-12-02",
                              credit="Photograph by " + "Averyveryverylongphotographername" * 2, plan=plan)


def test_sleeve_gloss_never_collides_with_credit() -> None:
    plan = L.NightPlan("S1_SLEEVE", "italic", "roman", False, "none", False)
    ops, _, _ = L._layout_sleeve(plan, "Communion", "Shared presence and collective warmth in person.", "2026-11-17", "Photograph by Nainoa Shizuru")
    texts = [op for op in ops if op[0] == "text"]
    credit = [op for op in texts if op[1] == "col" and op[2].startswith("Photograph")][0]
    credit_left = credit[5] - credit[3].width(credit[2])
    gloss_ops = [op for op in texts if op[1] == "gloss"]
    assert len(gloss_ops) <= 2
    for op in gloss_ops:
        same_line = abs(op[6] - credit[6]) < 1
        if same_line:
            assert op[5] + op[3].width(op[2]) <= credit_left - 40


def test_sleeve_gloss_stays_on_two_lines_with_a_long_credit() -> None:
    plan = L.NightPlan("S1_SLEEVE", "italic", "roman", False, "none", False)
    ops, _, _ = L._layout_sleeve(plan, "Lattice", "Civic space reborn in steel and monumental light.",
                                 "2026-11-06", "Photograph by Klara Kulikova Abernathy")
    gl = [op for op in ops if op[0] == "text" and op[1] == "gloss"]
    assert len(gl) == 2
    assert all(op[2].split()[-1].lower() not in L.STOPWORDS for op in gl[:-1])


LAYOUTS = {"S1_PLATE": L._layout_plate, "S1_SLEEVE": L._layout_sleeve, "S1_BLEED": L._layout_bleed,
           "S1_ENTRY": L._layout_entry, "S1_WALL": L._layout_wall}


@pytest.mark.parametrize("template", sorted(LAYOUTS))
@pytest.mark.parametrize("centred", [False, True])
def test_date_and_credit_only_on_caption_templates(template: str, centred: bool) -> None:
    plan = L.NightPlan(template, "italic", "roman", False, "none", centred)
    ops, _, _ = LAYOUTS[template](plan, "Relic", GLOSS, "2026-11-24", CREDIT)
    has_col = any(op[0] == "text" and op[1] == "col" for op in ops)
    assert has_col == (template in L.COLOPHON_TEMPLATES)


@pytest.mark.parametrize("template", L.ON_PHOTO)
def test_on_photo_templates_carry_no_date_or_credit(template: str) -> None:
    photo = _photo("dark")
    for voice in L.VOICES:
        pl = L.place_on_photo(photo, template, voice, L.PAIRS[voice][0], "Numen", GLOSS, "2026-11-24", CREDIT)
        if pl is not None:
            assert pl["colophon"] == []


def test_caption_templates_are_wall_sleeve_entry() -> None:
    assert set(L.COLOPHON_TEMPLATES) == {"S1_WALL", "S1_SLEEVE", "S1_ENTRY"}


def test_feed_entry_does_not_carry_the_template_name(tmp_path: Path) -> None:
    published = _router_run(tmp_path)
    assert "curatorial_archetype" not in published.to_feed_entry()


def test_history_starts_with_the_last_published_template_when_the_feed_lacks_names(tmp_path: Path) -> None:
    from src.zeitgeist.visual_router import _edition_history

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "zeitgeist_state.json").write_text(json.dumps({"date": "2026-11-24", "curatorial_archetype": "S1_BLEED"}), encoding="utf-8")
    assert _edition_history(tmp_path, []) == ["S1_BLEED"]
    assert _edition_history(tmp_path, ["POETIC_CINEMA"]) == ["S1_BLEED", "POETIC_CINEMA"]
    assert _edition_history(tmp_path, ["S1_FRAME", "S1_BLEED"]) == ["S1_FRAME", "S1_BLEED"]
    assert _edition_history(tmp_path / "missing", ["S1_FRAME"]) == ["S1_FRAME"]


def test_history_ignores_a_state_from_the_same_date(tmp_path: Path) -> None:
    from src.zeitgeist.visual_router import _edition_history

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "zeitgeist_state.json").write_text(json.dumps({"date": "2026-11-25", "curatorial_archetype": "S1_BLEED"}), encoding="utf-8")
    assert _edition_history(tmp_path, [], "2026-11-25") == []
    assert _edition_history(tmp_path, [], "2026-11-26") == ["S1_BLEED"]
