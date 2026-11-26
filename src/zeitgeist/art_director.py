"""The art director (SIY-244): chooses each night's template, lettering and in-template setting by
judgement, within hard rules; any failure returns None so rotation (edition_look.plan_night) decides.

The director never touches the photograph or the words. It reads them, shortlists treatments, compares
drawn plates in both orders and checks the winner at phone size. Models answer only with ids from the
offered list and values from fixed enums.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Sequence

from PIL import Image

from . import edition_look as L

SHORT = {
    "S1_FRAME": "frame", "S1_SUBTITLE": "subtitle", "S1_PLATE": "plate", "S1_SLEEVE": "sleeve",
    "S1_BLEED": "bleed", "S1_ENTRY": "entry", "S1_WALL": "wall", "S1_CAESURA": "caesura",
}
SETTINGS = {
    "S1_PLATE": ("standard", "large"), "S1_SLEEVE": ("standard", "large"),
    "S1_ENTRY": ("standard", "large"), "S1_WALL": ("standard", "large"),
}
MAX_IN_SEVEN = 2  # a template used three times in the last seven nights is not offered


@dataclass(frozen=True)
class Option:
    id: str
    template: str
    voice: str
    gloss_face: str
    setting: str
    centred: bool
    placement: Optional[dict]


def _option_id(template: str, voice: str, setting: str, centred: bool) -> str:
    tail = ("centred" if centred else "split") if template == "S1_BLEED" else (
        "" if template in L.ON_PHOTO else setting)
    return ".".join(p for p in (SHORT[template], voice, tail) if p)


def to_plan(option: Option, *, colour_on: bool, device: str) -> L.NightPlan:
    return L.NightPlan(option.template, option.voice, option.gloss_face, colour_on, device, option.centred,
                       option.placement, option.setting)


def build_options(photo: Image.Image, *, word: str, gloss: str, date_iso: str, credit: str,
                  history: Sequence[str]) -> tuple[list[Option], Optional[str]]:
    """The night's options and the id the shortlist must include (None when there are no options)."""
    o = dt.date.fromisoformat(date_iso[:10]).toordinal()
    facts = L.analyse_photo(photo)
    colour_on = facts.colour is not None and o % 2 == 0
    recent = [h for h in history if h in L.TEMPLATES]
    banned = {recent[0]} if recent else set()
    banned |= {t for t in L.TEMPLATES if recent[:7].count(t) > MAX_IN_SEVEN}
    rotation_voice = L.VOICE_CYCLE[o % 4]

    def face_for(voice: str) -> str:
        faces = L.PAIRS[voice]
        return faces[(o // 4) % len(faces)]

    options: list[Option] = []
    required_voice: dict[str, str] = {}
    for template in L.ORDER:
        if template in banned:
            continue
        for voice in L.VOICES:
            face = face_for(voice)
            if template == "S1_CAESURA":
                face = L.caesura_face(voice, face)
            placement = None
            if template in L.ON_PHOTO:
                placement = L.place_on_photo(photo, template, voice, face, word, gloss, date_iso, credit, facts)
                if placement is None:
                    continue
            if template == "S1_BLEED":
                variants = [("standard", True), ("standard", False)]
            else:
                variants = [(s, False) for s in SETTINGS.get(template, ("standard",))]
            for setting, centred in variants:
                option = Option(_option_id(template, voice, setting, centred), template, voice, face, setting,
                                centred, placement)
                plan = to_plan(option, colour_on=colour_on, device=L.device_for(template, colour_on, o))
                if L.plan_fits(plan, word, gloss, date_iso, credit):
                    options.append(option)
    if not options:
        return [], None

    def last_use(t: str) -> int:
        return recent.index(t) if t in recent else 10_000

    templates = sorted({op.template for op in options}, key=lambda t: (-last_use(t), L.ORDER.index(t)))
    lru = templates[0]
    bleed_centred = sum(1 for h in recent if h == "S1_BLEED") % 2 == 0
    pool = [op for op in options if op.template == lru]
    pick = next((op for op in pool if op.voice == rotation_voice and op.setting == "standard"
                 and (lru != "S1_BLEED" or op.centred == bleed_centred)), pool[0])
    return options, pick.id
