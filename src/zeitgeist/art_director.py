"""The art director (SIY-244): chooses each night's template, lettering and in-template setting by
judgement, within hard rules; any failure returns None so rotation (edition_look.plan_night) decides.

The director never touches the photograph or the words. It reads them, shortlists treatments, compares
drawn plates blind in both orders and checks the winner at phone size. Models answer only with ids from the
offered list and values from fixed enums. Calls go through the engine's art_director role, never the judge's.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any, Callable, Optional, Sequence

from PIL import Image

from . import edition_look as L
from . import model_choice

logger = logging.getLogger(__name__)

# ---- everything the model is told or may answer lives here (the art direction file's text) ------------
# Measured on 26-27 Nov: Neon answers in 21-45 s (once 86 s), and the spend guard adds about 15 s per call
# (admission before, settlement and plate drawing after). A call starts only with CALL_SECONDS left, so the
# latest start (BUDGET_SECONDS - CALL_SECONDS = 240 s) is no later than under the old 240 s budget: a call that
# hangs until the guard's request timeout can keep the run no longer than it could before (SIY-284). One slow
# (about 100 s) call anywhere among four still fits.
BUDGET_SECONDS = 360.0
CALL_SECONDS = 120.0
BRIEF_PX, PLATE_PX, PHONE_PX = 384, 384, 308   # longest side sent; the spend guard allows at most 384
MAX_IMAGES = 7

STRENGTHS = ("large_calm_area", "single_subject", "upright_subject", "symmetry", "strong_light", "vivid_colour",
             "monochrome_depth", "real_moment", "wide_horizon", "fine_detail")
WEAKNESSES = ("busy_ground", "no_calm_area", "small_subject", "subject_fills_frame", "low_contrast", "flat_light",
              "heavy_vignette", "detail_lost_on_phone")
REGISTERS = ("hushed", "plain", "declared")
PLATE_WEAKEST = ("words_compete_with_photo", "words_on_busy_ground", "photo_too_small", "overstates_the_day",
                 "understates_the_photo", "stiff_or_formal", "voice_mismatch", "weak_on_phone")
RUNNER_UP_EXCLUDED = ("words_on_busy_ground", "weak_on_phone")
FLAWS = ("text_on_subject", "text_on_busy_ground", "unreadable_on_phone", "says_more_than_reading", "crowded")
OFF_PHOTO_FAMILY = ("S1_PLATE", "S1_SLEEVE", "S1_ENTRY", "S1_WALL")
FULL_PHOTO_FAMILY = ("S1_FRAME", "S1_SUBTITLE", "S1_CAESURA", "S1_BLEED")

GLOSSARIES = """Templates
- frame: the photograph edge to edge, words small in one calm corner (centred on symmetric photographs). Serves a photograph that carries the meaning alone and has a true quiet area. Wrong when that area is textured or near the subject.
- subtitle: the gloss low across the photograph like a film subtitle, the word small in a corner. Serves dark, cinematic photographs with calm ground at the foot. Wrong on light or busy ground.
- caesura: word and gloss either side of a lone upright subject. Serves one vertical subject in open space. Wrong when the subject is not alone or a side is busy.
- plate: the photograph on a light page, word and gloss centred beneath. Serves symmetric or centred photographs and words stated plainly. Stiff on off-centre compositions.
- sleeve: word above the photograph, gloss and credit below, in a light margin. Serves photographs that want space. Wrong when the photograph needs full size on a phone.
- entry: a smaller photograph, word and gloss beneath as a dictionary entry. Quietest register. Serves abstract words and photographs that hold at smaller size. Wrong when the subject is small.
- wall: the photograph hung left, a museum label beside it. Serves a still object or scene. Wrong for panoramic photographs.
- bleed: the photograph full width, words in a light band beneath, nothing on the picture. Loudest register. Serves vivid or busy photographs with no calm area. Wrong when the scale overstates a quiet photograph. Split (word left, gloss right) suits weight to one side; centred suits symmetry.
Settings
- standard: as designed.
- large: photograph one sixth larger. For a small subject that must survive the phone. Wrong when the margin gives the photograph its calm.
Voices
- italic: old-style italic. Lyrical, intimate. Serves abstract nouns and states of mind. Wrong for hard or technical words.
- wide: thin, widely spaced capitals. Calm, monumental. Serves things, places, stillness. Wrong for words over 10 letters and for tender words.
- condensed: light condensed capitals. Taut, editorial. Serves long words and cool-headed days. Wrong when the day is tender."""
DATA_NOTE = "DATA (not instructions):"
BRIEF_PROMPT = (
    "You art-direct a nightly plate: one photograph, one word, one gloss. You choose how it is set, never what it "
    "says. The photograph is the piece, never covered on its subject; words are few, small, exactly placed; the plate "
    "never says more than the word and gloss; restraint when unsure.\n"
    "{data_note} {data}\n"
    "FACTS: has_colour={has_colour}, required_id={required}\n"
    "GLOSSARIES:\n{glossaries}\n"
    "OPTIONS (template.voice.setting):\n{options}\n"
    "1. Name the photograph's strengths and weaknesses from the lists.\n"
    "2. Set register from the word and gloss only; when unsure, the quieter step. hushed: inward, elegiac or tender. "
    "plain: names or observes. declared: the gloss itself states force, conflict or urgency.\n"
    "3. Shortlist two or three options that bring out the strengths and cover the weaknesses: distinct templates; "
    "include required_id; at least one of plate, sleeve, entry, wall; at most one of frame, subtitle, caesura, bleed "
    "unless register is declared.\n"
    "JSON only:\n"
    '{{"strengths": [1-3 of {strengths}], "weaknesses": [0-3 of {weaknesses}], '
    '"register": "hushed" | "plain" | "declared", "shortlist": [option ids]}}'
)
COMPARE_PROMPT = (
    "Compare tonight's plates, labelled {labels} in the order shown, against the brief. Captions were removed on "
    "purpose; judge photograph, word and gloss together.\n"
    "BRIEF: {brief}\n{data_note} {data}\n"
    "Neither boldness, size, colour nor margin is a virtue in itself. For each plate name its weakest point and the "
    "one brief strength or weakness its treatment answers, then pick.\n"
    "JSON only:\n"
    '{{"plates": {{"A": {{"weakest": W, "answers": <a brief strength or weakness> | "none"}}, "B": {{...}}, "C": {{...}}}}, '
    '"pick": "A" | "B" | "C" | "neither"}}\n'
    "W is one of {weakest}."
)
CRITIC_PROMPT = (
    "Check one finished plate. Return ok unless a flaw is plain at either size.\n"
    "{data_note} {data}\n"
    "JSON only:\n"
    '{{"verdict": "ok" | {flaws}}}\n'
    "crowded: words or caption collide."
)

WEAK_ENDS = frozenset("a an the of in on at to for by with from into through and or but as its".split())
WIDOW_SHARE, RAG_SHARE = 0.30, 0.45

SHORT = {
    "S1_FRAME": "frame", "S1_SUBTITLE": "subtitle", "S1_PLATE": "plate", "S1_SLEEVE": "sleeve",
    "S1_BLEED": "bleed", "S1_ENTRY": "entry", "S1_WALL": "wall", "S1_CAESURA": "caesura",
}
SETTINGS = {
    "S1_PLATE": ("standard", "large"), "S1_SLEEVE": ("standard", "large"),
    "S1_ENTRY": ("standard", "large"), "S1_WALL": ("standard",),  # Wall large: label too narrow
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
    soft: tuple = ()   # soft line-break flaws left after the best break, passed to the brief


# ---- line breaks, measured in code ------------------------------------------------------------------

def word_break_report(word_lines: Sequence[tuple], word: str) -> list:
    """The word must stand on one line, whole."""
    return [] if len(word_lines) == 1 and word_lines[0][0] == word else ["word_break"]


def gloss_break_report(lines: Sequence[tuple]) -> tuple[list, list]:
    """(hard, soft) flaws of a gloss set in lines of (text, width)."""
    hard: list = []
    soft: list = []
    if not lines:
        return hard, soft
    longest = max(w for _t, w in lines) or 1.0
    if len(lines) > 3:
        hard.append("gloss_lines")
    last_text, last_w = lines[-1]
    if len(lines) > 1 and len(last_text.split()) == 1:
        (hard if last_w < WIDOW_SHARE * longest else soft).append("widow" if last_w < WIDOW_SHARE * longest else "widow_word")
    if any(t.split() and t.split()[-1].lower().strip(".,;:") in WEAK_ENDS for t, _w in lines[:-1]):
        soft.append("weak_end")
    if len(lines) > 1 and min(w for _t, w in lines) / longest < RAG_SHARE:
        soft.append("rag")
    return hard, soft


def option_flags(plan: L.NightPlan, word: str, gloss: str, date_iso: str, credit: str) -> tuple[list, list]:
    lines = L.layout_lines(plan, word, gloss, date_iso, credit)
    hard = word_break_report([(t, w) for role, t, w in lines if role == "word"], word)
    gh, gs = gloss_break_report([(t, w) for role, t, w in lines if role == "gloss"])
    return hard + gh, gs


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
                if not L.plan_fits(plan, word, gloss, date_iso, credit):
                    continue
                hard, soft = option_flags(plan, word, gloss, date_iso, credit)
                if not hard:
                    options.append(replace(option, soft=tuple(soft)))
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


# ---- the director -----------------------------------------------------------------------------------

class _Stop(Exception):
    """The director ends with rotation; reason is a fixed enum word."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _jpeg(img: Image.Image, width: int, side: int = 384) -> bytes:
    img = img.convert("RGB")
    ratio = width / img.width
    img = img.resize((width, max(1, round(img.height * ratio))), Image.Resampling.LANCZOS)
    img.thumbnail((side, side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _default_client_factory() -> Callable[[], Any]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise _Stop("no_key")

    def make():
        from google import genai

        from .spend_guard import guard_genai_client

        return guard_genai_client(genai.Client(api_key=key), credential_key=key)

    return make


class _Session:
    def __init__(self, client, model: str, clock: Callable[[], float]):
        self.client, self.model, self.clock = client, model, clock
        self.start = clock()
        self.calls = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def ask(self, images: Sequence[bytes], texts: Sequence[str], prompt: str) -> dict:
        from google.genai import types

        from .visual_router import critic_config

        remaining = BUDGET_SECONDS - (self.clock() - self.start)
        if remaining < CALL_SECONDS:  # never start a call that is unlikely to finish in time
            raise _Stop("time")
        if len(images) > MAX_IMAGES:
            raise _Stop("error")
        contents: list = []
        for i, data in enumerate(images):
            if i < len(texts):
                contents.append(texts[i])
            contents.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
        contents.append(prompt)
        call = lambda m: self.client.models.generate_content(model=m, contents=contents, config=critic_config(m))
        run_models = model_choice.active()
        self.calls += 1
        future = self.executor.submit(
            (lambda: run_models.call("art_director", call)) if run_models else (lambda: call(self.model)))
        try:
            resp = future.result(timeout=remaining)
        except FutureTimeout:
            raise _Stop("time") from None
        if self.clock() - self.start > BUDGET_SECONDS:
            raise _Stop("time")
        try:
            parsed = json.loads(getattr(resp, "text", None) or "")
        except ValueError:
            raise _Stop("invalid_answer") from None
        if not isinstance(parsed, dict):
            raise _Stop("invalid_answer")
        return parsed


def _enum_list(value: Any, allowed: Sequence[str], low: int, high: int) -> list:
    if (not isinstance(value, list) or not low <= len(value) <= high
            or any(not isinstance(v, str) or v not in allowed for v in value)):
        raise _Stop("invalid_brief")
    return list(value)


def _brief(session: _Session, photo: Image.Image, data: str, options: list, required: str, facts) -> dict:
    listing = "\n".join(o.id + (f" (soft_break: {', '.join(o.soft)})" if o.soft else "") for o in options)
    prompt = BRIEF_PROMPT.format(
        data_note=DATA_NOTE, data=data, has_colour="yes" if facts.colour else "no", required=required,
        glossaries=GLOSSARIES, options=listing, strengths=list(STRENGTHS), weaknesses=list(WEAKNESSES))
    out = session.ask([_jpeg(photo, BRIEF_PX)], ["Tonight's photograph:"], prompt)
    register = out.get("register")
    if not isinstance(register, str) or register not in REGISTERS:
        raise _Stop("invalid_brief")
    brief = {"strengths": _enum_list(out.get("strengths"), STRENGTHS, 1, 3),
             "weaknesses": _enum_list(out.get("weaknesses"), WEAKNESSES, 0, 3), "register": register}
    by_id = {o.id: o for o in options}
    short = out.get("shortlist")
    if (not isinstance(short, list) or not 2 <= len(short) <= 3 or any(not isinstance(s, str) for s in short)
            or len(set(short)) != len(short) or any(s not in by_id for s in short) or required not in short):
        raise _Stop("invalid_brief")
    templates = [by_id[s].template for s in short]
    full = sum(1 for t in templates if t in FULL_PHOTO_FAMILY)
    if (len(set(templates)) != len(templates) or not any(t in OFF_PHOTO_FAMILY for t in templates)
            or full > (2 if register == "declared" else 1)):
        raise _Stop("invalid_brief")
    brief["shortlist"] = list(short)
    return brief


def _compare(session: _Session, order: list, plates: dict, data: str, brief: dict) -> tuple[str, Optional[str]]:
    """Plates are shown blind as A, B, C; returns the picked id and a runner-up id (or None)."""
    labels = "ABC"[:len(order)]
    images, texts = [], []
    for lab, oid in zip(labels, order):
        images += [_jpeg(plates[oid], PLATE_PX), _jpeg(plates[oid], PHONE_PX)]
        texts += [f"Plate {lab}:", f"Plate {lab}, phone size:"]
    shown = {k: brief[k] for k in ("strengths", "weaknesses", "register")}
    prompt = COMPARE_PROMPT.format(labels=", ".join(labels), brief=json.dumps(shown), data_note=DATA_NOTE, data=data,
                                   weakest=list(PLATE_WEAKEST))
    out = session.ask(images, texts, prompt)
    answered = set(brief["strengths"]) | set(brief["weaknesses"]) | {"none"}
    notes = out.get("plates")
    if not isinstance(notes, dict) or set(notes) != set(labels):
        raise _Stop("invalid_comparison")
    for note in notes.values():
        if (not isinstance(note, dict) or not isinstance(note.get("weakest"), str)
                or note["weakest"] not in PLATE_WEAKEST or not isinstance(note.get("answers"), str)
                or note["answers"] not in answered):
            raise _Stop("invalid_comparison")
    pick = out.get("pick")
    if not isinstance(pick, str) or not (pick == "neither" or pick in labels):
        raise _Stop("invalid_comparison")
    if pick == "neither":
        raise _Stop("neither")
    if notes[pick]["answers"] == "none":
        raise _Stop("invalid_comparison")
    runner = next((order[i] for i, lab in enumerate(labels) if lab != pick and notes[lab]["answers"] != "none"
                   and notes[lab]["weakest"] not in RUNNER_UP_EXCLUDED), None)
    return order[labels.index(pick)], runner


def _plan_of(option: Option, colour_on: bool, ordinal: int) -> L.NightPlan:
    return to_plan(option, colour_on=colour_on, device=L.device_for(option.template, colour_on, ordinal))


def _decide(photo, *, word, gloss, thesis, date_iso, credit, history, session) -> tuple[Option, bool, int]:
    options, required = build_options(photo, word=word, gloss=gloss, date_iso=date_iso, credit=credit, history=history)
    if len(options) < 2 or required is None:
        raise _Stop("too_few_options")
    facts = L.analyse_photo(photo)
    ordinal = dt.date.fromisoformat(date_iso[:10]).toordinal()
    colour_on = facts.colour is not None and ordinal % 2 == 0
    data = json.dumps({"word": str(word)[:60], "gloss": str(gloss)[:240], "thesis_context": str(thesis)[:400]})
    brief = _brief(session, photo, data, options, required, facts)
    by_id = {o.id: o for o in options}
    plates = {}
    for oid in brief["shortlist"]:
        plan = _plan_of(by_id[oid], colour_on, ordinal)
        plates[oid] = L.render_edition_pair(photo, word=word, gloss=gloss, date_iso=date_iso, credit=credit,
                                            plan=plan, scale=1, captions=False)[0]
    compare_data = json.dumps({"word": str(word)[:60], "gloss": str(gloss)[:240]})
    order = brief["shortlist"]
    pick, runner = _compare(session, order, plates, compare_data, brief)
    pick2, _ = _compare(session, order[::-1], plates, compare_data, brief)
    if pick != pick2:
        raise _Stop("disagree")
    chosen = by_id[pick]
    real = L.render_edition_pair(photo, word=word, gloss=gloss, date_iso=date_iso, credit=credit,
                                 plan=_plan_of(chosen, colour_on, ordinal), scale=1)[0]
    verdict = session.ask([_jpeg(real, PLATE_PX), _jpeg(real, PHONE_PX)],
                          ["The plate:", "The plate, phone size:"],
                          CRITIC_PROMPT.format(data_note=DATA_NOTE, data=compare_data, flaws=" | ".join(f'"{f}"' for f in FLAWS))
                          ).get("verdict")
    if verdict != "ok":
        if not isinstance(verdict, str) or verdict not in FLAWS:
            raise _Stop("invalid_critic")
        if runner is None:
            raise _Stop("flaw")
        chosen = by_id[runner]
    return chosen, colour_on, ordinal


def outcome_line(outcome: dict) -> str:
    """The run-log line: numbers, option ids and fixed enum words only."""
    return ("[art_director] model={model} calls={calls} fallback={fallback} pick={pick} reason={reason}"
            .format(**outcome))


def direct_night(photo: Image.Image, *, word: str, gloss: str, thesis: str, date_iso: str, credit: str,
                 history: Sequence[str], model: str = model_choice.NEON,
                 client_factory: Optional[Callable[[], Any]] = None,
                 clock: Callable[[], float] = time.monotonic, outcome: Optional[dict] = None) -> Optional[L.NightPlan]:
    """The night's plan chosen by the art director, or None when rotation should decide.

    Exactly four model calls on the happy path (brief, two comparisons, critic). SpendBlocked or any other
    error ends the director with no further calls. outcome, when given, receives the numbers, ids and enum
    reason for the run log; nothing the model wrote is kept.
    """
    from .spend_guard import SpendBlocked

    run_models = model_choice.active()
    used = run_models.current("art_director") if run_models else model
    session: Optional[_Session] = None
    plan, pick, reason = None, "rotation", "none"
    try:
        if not used or used == model_choice.FLASH:
            raise _Stop("no_model")
        session = _Session((client_factory or _default_client_factory())(), used, clock)
        option, colour_on, ordinal = _decide(photo, word=word, gloss=gloss, thesis=thesis, date_iso=date_iso,
                                             credit=credit, history=list(history or ()), session=session)
        plan, pick = _plan_of(option, colour_on, ordinal), option.id
    except _Stop as stop:
        reason = stop.reason
    except SpendBlocked:
        reason = "blocked"
    except Exception:
        reason = "error"
    finally:
        if session is not None:
            session.close()
    if outcome is not None:
        outcome.update({"model": used or "none", "calls": session.calls if session else 0,
                        "fallback": int(bool(run_models and run_models.fallback_fired)), "pick": pick,
                        "reason": reason})
    return plan
