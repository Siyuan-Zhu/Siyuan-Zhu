"""Tests for Caesura, the eighth plate template (SIY-204): the subject breaks the line of type."""

from __future__ import annotations

import datetime as dt

import pytest
from PIL import Image, ImageDraw

from src.zeitgeist import caesura as C
from src.zeitgeist import edition_look as L

SKY = (200, 215, 230)
BAR = (520, 580)
WORD = "Relic"
GLOSS = "The sudden quietude of an obsolete monument."
CREDIT = "Photograph by Test Photographer"


def _monolith() -> Image.Image:
    """Calm sky with one upright dark bar."""
    img = Image.new("RGB", (1200, 800), SKY)
    ImageDraw.Draw(img).rectangle([BAR[0], 150, BAR[1] - 1, 750], fill=(70, 40, 25))
    return img


def _busy() -> Image.Image:
    img = Image.new("RGB", (1200, 800), SKY)
    d = ImageDraw.Draw(img)
    for x in range(0, 1200, 6):
        d.line([(x, 0), (x, 800)], fill=(250, 248, 244) if x % 12 else (10, 10, 10), width=2)
    d.rectangle([BAR[0], 150, BAR[1] - 1, 750], fill=(70, 40, 25))
    return img


def _italic_date(sans_face: bool) -> str:
    """A date whose word voice is italic, and whose production gloss face is sans (or roman)."""
    d = dt.date(2026, 12, 1)
    while True:
        o = d.toordinal()
        if L.VOICE_CYCLE[o % 4] == "italic" and (L.PAIRS["italic"][(o // 4) % 2] == "sans") == sans_face:
            return d.isoformat()
        d += dt.timedelta(days=1)


def _ink_boxes(pl):
    out = []
    for text, st, align, x, base in pl["lines"]:
        assert align == "left"
        l, t, r, b = C.ink_extent(st, text)
        out.append((x + l, base + t, x + r, base + b))
    return out


@pytest.mark.parametrize("voice", L.VOICES)
def test_word_left_gloss_right_on_one_baseline_with_equal_gutters(voice: str) -> None:
    face = "roman" if voice == "italic" else L.PAIRS[voice][0]
    pl = C.place_caesura(_monolith(), voice, face, WORD, GLOSS)
    assert pl is not None
    word, *gloss = pl["lines"]
    assert word[0] == WORD
    assert " ".join(g[0] for g in gloss) == GLOSS
    assert 2 <= len(gloss) <= 3
    assert pl["ink"] == "k"
    boxes = _ink_boxes(pl)
    word_right = boxes[0][2]
    gloss_left = min(b[0] for b in boxes[1:])
    assert word_right < BAR[0] and gloss_left > BAR[1]
    assert abs((BAR[0] - word_right) - (gloss_left - BAR[1])) <= 3
    assert all(abs(b[0] - gloss_left) <= 1 for b in boxes[1:])   # each gloss line flush on its ink
    assert word[4] in [g[4] for g in gloss]                       # one shared baseline
    assert pl["colophon"] == []


@pytest.mark.parametrize("voice", L.VOICES)
def test_text_stays_inside_the_plate_and_off_the_subject(voice: str) -> None:
    face = "roman" if voice == "italic" else L.PAIRS[voice][0]
    pl = C.place_caesura(_monolith(), voice, face, "Numen", GLOSS)
    assert pl is not None
    for x0, y0, x1, y1 in _ink_boxes(pl):
        assert 0 <= x0 and x1 <= L.WIDTH and 0 <= y0 and y1 <= 800
        assert x1 < BAR[0] or x0 > BAR[1]
    word = _ink_boxes(pl)[0]
    assert word[2] < BAR[0]


def test_busy_photograph_is_ineligible() -> None:
    for voice in L.VOICES:
        assert C.place_caesura(_busy(), voice, "roman", WORD, GLOSS) is None


def test_photograph_without_a_subject_is_ineligible() -> None:
    flat = Image.new("RGB", (1200, 800), SKY)
    for voice in L.VOICES:
        assert C.place_caesura(flat, voice, "roman", WORD, GLOSS) is None


def test_plan_night_picks_caesura_when_eligible_and_not_previous() -> None:
    day = _italic_date(sans_face=True)
    plan = L.plan_night(day, ["S1_FRAME"], _monolith(), word=WORD, gloss=GLOSS, credit=CREDIT)
    assert plan.template == "S1_CAESURA"
    assert plan.voice == "italic"
    assert plan.gloss_face == "roman"          # beside the italic word the gloss is the roman
    assert plan.placement is not None
    again = L.plan_night(day, ["S1_CAESURA"], _monolith(), word=WORD, gloss=GLOSS, credit=CREDIT)
    assert again.template != "S1_CAESURA"


def test_plan_night_never_picks_caesura_without_a_placement() -> None:
    flat = Image.new("RGB", (1200, 800), SKY)
    for day in ("2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04"):
        assert L.plan_night(day, [], flat, word=WORD, gloss=GLOSS, credit=CREDIT).template != "S1_CAESURA"
        assert L.plan_night(day, [], _busy(), word=WORD, gloss=GLOSS, credit=CREDIT).template != "S1_CAESURA"


def test_caesura_is_an_on_photo_template_and_the_rarest() -> None:
    assert "S1_CAESURA" in L.TEMPLATES and "S1_CAESURA" in L.ON_PHOTO
    assert L.ORDER[0] == "S1_CAESURA"
    assert set(L.ORDER) == set(L.TEMPLATES)


def test_caesura_renders_with_the_shared_photo_path() -> None:
    day = _italic_date(sans_face=False)
    photo = _monolith()
    plan = L.plan_night(day, [], photo, word=WORD, gloss=GLOSS, credit=CREDIT)
    assert plan.template == "S1_CAESURA"
    light, dark = L.render_edition_pair(photo, word=WORD, gloss=GLOSS, date_iso=day, credit=CREDIT, plan=plan)
    assert light.size == dark.size == (1200, 800)
    px = light.load()
    x0, y0, x1, y1 = (int(v) for v in _ink_boxes(plan.placement)[0])
    assert any(px[x, y] != SKY for x in range(x0, x1) for y in range(y0, y1))


def test_place_on_photo_dispatches_to_caesura() -> None:
    pl = L.place_on_photo(_monolith(), "S1_CAESURA", "italic", "sans", WORD, GLOSS, "2026-12-01", CREDIT)
    assert pl is not None
    assert pl["lines"][1][1].font.path.endswith("SortsMillGoudy-Regular.ttf")


def test_caesura_takes_every_night_it_fits_unless_it_ran_the_night_before() -> None:
    day = "2026-12-03"
    recent = ["S1_PLATE", "S1_CAESURA", "S1_FRAME"]  # Caesura ran two nights ago
    plan = L.plan_night(day, recent, _monolith(), word=WORD, gloss=GLOSS, credit=CREDIT)
    assert plan.template == "S1_CAESURA"
    plan = L.plan_night(day, ["S1_CAESURA", "S1_PLATE"], _monolith(), word=WORD, gloss=GLOSS, credit=CREDIT)
    assert plan.template != "S1_CAESURA"
