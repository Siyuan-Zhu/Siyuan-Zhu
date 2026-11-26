"""Tests for the art director (SIY-244): options, variety guard, calls and decision. No network."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from src.zeitgeist import art_director as AD
from src.zeitgeist import edition_look as L

CREDIT = "Photograph by Test Photographer"
GLOSS = "Mythmaking woven through modern real-life drama."
DAY = "2026-12-02"


def _photo(kind: str = "dark") -> Image.Image:
    img = Image.new("RGB", (1200, 800), (30, 34, 40))
    d = ImageDraw.Draw(img)
    if kind == "dark":
        d.ellipse([480, 300, 720, 540], fill=(200, 80, 60))
    elif kind == "busy":
        for x in range(0, 1200, 6):
            d.line([(x, 0), (x, 800)], fill=(250, 248, 244) if x % 12 else (10, 10, 10), width=2)
    return img


def _build(history=(), kind="dark"):
    return AD.build_options(_photo(kind), word="Relic", gloss=GLOSS, date_iso=DAY, credit=CREDIT, history=list(history))


def _templates(options) -> set:
    return {o.template for o in options}


def test_options_cover_templates_voices_and_settings() -> None:
    options, required = _build()
    ids = [o.id for o in options]
    assert len(ids) == len(set(ids))
    assert "plate.italic.large" in ids and "plate.wide.standard" in ids
    assert {o.setting for o in options if o.template == "S1_WALL"} == {"standard", "large"}
    assert required in ids


def test_previous_template_is_never_offered() -> None:
    options, _ = _build(["S1_PLATE"])
    assert "S1_PLATE" not in _templates(options)


def test_template_used_three_times_in_seven_nights_is_never_offered() -> None:
    history = ["S1_WALL", "S1_PLATE", "S1_WALL", "S1_SLEEVE", "S1_WALL", "S1_ENTRY", "S1_BLEED"]
    options, _ = _build(history)
    assert "S1_WALL" not in _templates(options)
    older = ["S1_PLATE", "S1_SLEEVE", "S1_ENTRY", "S1_BLEED", "S1_PLATE", "S1_SLEEVE", "S1_ENTRY", "S1_WALL", "S1_WALL", "S1_WALL"]
    assert "S1_WALL" in _templates(_build(older)[0])      # the three uses are outside the last seven


def test_busy_photograph_offers_no_photo_templates() -> None:
    options, _ = _build(kind="busy")
    assert options and not (_templates(options) & set(L.ON_PHOTO))


def test_required_is_the_least_recently_used_eligible_template() -> None:
    history = ["S1_PLATE", "S1_SLEEVE", "S1_ENTRY", "S1_BLEED", "S1_WALL"]
    options, required = _build(history, kind="busy")
    chosen = next(o for o in options if o.id == required)
    assert chosen.template == "S1_WALL"      # eligible off-photo templates: the one used longest ago
    options, required = _build(["S1_WALL"], kind="busy")
    never_used = {"S1_PLATE", "S1_SLEEVE", "S1_ENTRY", "S1_BLEED"}
    assert next(o for o in options if o.id == required).template in never_used


def test_bleed_options_are_centred_or_split() -> None:
    options, _ = _build()
    assert {o.centred for o in options if o.template == "S1_BLEED"} == {True, False}
