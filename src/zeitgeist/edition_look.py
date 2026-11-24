"""Edition look S1: the nightly plate drawn in the templates David chose (SIY-204).

Pure functions. plan_night decides the template, lettering, gloss face, colour and
alignment for a night; render_edition_pair draws the light and dark plates.
The photograph is always shown whole; only the plate around it is designed here.
"""

from __future__ import annotations

import colorsys
import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
WIDTH = 1200
LOOK_NAME = "s1"

TEMPLATES: tuple[str, ...] = (
    "S1_FRAME", "S1_SUBTITLE", "S1_PLATE", "S1_SLEEVE", "S1_BLEED", "S1_ENTRY", "S1_WALL",
)
ON_PHOTO: tuple[str, ...] = ("S1_FRAME", "S1_SUBTITLE")
ORDER: tuple[str, ...] = (
    "S1_SUBTITLE", "S1_FRAME", "S1_BLEED", "S1_PLATE", "S1_WALL", "S1_SLEEVE", "S1_ENTRY",
)
VOICES: tuple[str, ...] = ("italic", "wide", "condensed")
VOICE_CYCLE: tuple[str, ...] = ("italic", "wide", "condensed", "italic")
PAIRS: dict[str, tuple[str, ...]] = {"italic": ("sans", "roman"), "wide": ("italic",), "condensed": ("italic",)}
FIRST_EDITION = dt.date(2026, 10, 31)
STOPWORDS = frozenset(
    "a an the of in on to and for with by at into upon through before across from its their".split()
)
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
THEMES = {
    "light": {"page": (255, 255, 255), "ink": (27, 30, 34)},
    "dark": {"page": (13, 17, 23), "ink": (238, 242, 246)},
}
PHOTO_INK = {"w": (255, 255, 255), "k": (20, 17, 15)}
# (font key, weight, size px, tracking em, uppercase, line height)
VOICE_SPEC = {
    "italic": ("goudy_i", None, 98, 0.0, False, 1.02),
    "wide": ("darker", 500, 64, 0.30, True, 1.10),
    "condensed": ("antonio", 300, 58, 0.10, True, 1.05),
}
GLOSS_SPEC = {
    "italic": ("goudy_i", None, 43, 1.30),
    "roman": ("goudy", None, 41, 1.30),
    "sans": ("darker", 450, 46, 1.20),
}
SK = {"italic": 0.82, "wide": 1.0, "condensed": 0.9}
EK = {"italic": 0.62, "wide": 0.95, "condensed": 0.74}
COL_SPEC = ("darker", 500, 23, 0.13, 1.30)
COL_ALPHA = 0.72
FONT_FILES = {
    "goudy": "SortsMillGoudy-Regular.ttf",
    "goudy_i": "SortsMillGoudy-Italic.ttf",
    "darker": "DarkerGrotesque-Variable.ttf",
    "antonio": "Antonio-Variable.ttf",
}


@lru_cache(maxsize=256)
def _font(key: str, weight: Optional[int], size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONT_DIR / FONT_FILES[key]), size)
    if weight is not None:
        font.set_variation_by_axes([weight])
    return font


def _lin(v):
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _rel_lum(rgb) -> float:
    r, g, b = (float(_lin(c)) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(la: float, lb: float) -> float:
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _towards(rgb, bg_lum: float, target: float) -> tuple[int, int, int]:
    """Lighten on dark grounds or darken on light grounds until the colour reaches the target contrast."""
    r, g, b = (float(c) for c in rgb)
    up = bg_lum < 0.3
    for _ in range(120):
        if _contrast(_rel_lum((r, g, b)), bg_lum) >= target:
            break
        if up:
            r, g, b = (min(255.0, c + 3 + (255 - c) * 0.03) for c in (r, g, b))
        else:
            r, g, b = (max(0.0, c - 3 - c * 0.03) for c in (r, g, b))
    return (round(r), round(g), round(b))


@dataclass(frozen=True)
class PhotoFacts:
    colour: Optional[tuple[int, int, int]]
    symmetry: float


def _fit_photo(img: Image.Image, size) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def analyse_photo(img: Image.Image) -> PhotoFacts:
    """The night's colour (None for a black-and-white photograph) and a symmetry measure."""
    full = img.convert("RGB")
    small = full.resize((160, 107))
    q = small.quantize(colors=12, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[:36]
    total = 160 * 107
    best = None
    for n, idx in q.getcolors() or []:
        r, g, b = (v / 255 for v in pal[idx * 3: idx * 3 + 3])
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        share = n / total
        if s < 0.22 or l < 0.16 or l > 0.84 or share < 0.025:
            continue
        score = (s ** 1.6) * (share ** 0.4)
        if best is None or score > best[0]:
            best = (score, h, s)
    colour = None
    if best is not None:
        rr, gg, bb = colorsys.hls_to_rgb(best[1], 0.53, max(0.62, min(0.9, best[2] * 1.5)))
        colour = (round(rr * 255), round(gg * 255), round(bb * 255))
    thumb = np.asarray(full.resize((60, 40)), dtype=np.float64)
    lum = 0.2126 * _lin(thumb[..., 0]) + 0.7152 * _lin(thumb[..., 1]) + 0.0722 * _lin(thumb[..., 2])
    symmetry = float(np.mean(np.abs(lum[:, :30] - lum[:, ::-1][:, :30])))
    return PhotoFacts(colour=colour, symmetry=symmetry)


@dataclass
class Style:
    font: ImageFont.FreeTypeFont
    size: float
    tracking: float
    upper: bool
    line_height: float

    def prep(self, text: str) -> str:
        return text.upper() if self.upper else text

    def width(self, text: str) -> float:
        text = self.prep(text)
        if not text:
            return 0.0
        if not self.tracking:
            return float(self.font.getlength(text))
        return sum(float(self.font.getlength(ch)) for ch in text) + self.tracking * (len(text) - 1)

    def ascent_descent(self):
        return self.font.getmetrics()

    def baseline_in_line(self) -> float:
        asc, desc = self.ascent_descent()
        return (self.line_height - (asc + desc)) / 2 + asc


def _style(key, weight, size, tracking_em, upper, lh) -> Style:
    s = max(8, round(size))
    return Style(_font(key, weight, s), s, tracking_em * s, upper, lh * s)


def voice_style(voice: str, factor: float = 1.0) -> Style:
    key, weight, size, tr, up, lh = VOICE_SPEC[voice]
    return _style(key, weight, size * factor, tr, up, lh)


def gloss_style(face: str, factor: float = 1.0, lh: Optional[float] = None) -> Style:
    key, weight, size, line = GLOSS_SPEC[face]
    return _style(key, weight, size * factor, 0.0, False, lh or line)


def col_style(factor: float = 1.0, tracking_em: Optional[float] = None) -> Style:
    key, weight, size, tr, lh = COL_SPEC
    return _style(key, weight, size * factor, tr if tracking_em is None else tracking_em, True, lh)


def fit_word(voice: str, word: str, limit: float, factor: float = 1.0) -> Style:
    st = voice_style(voice, factor)
    w = st.width(word)
    if w > limit:
        f = factor * limit / w
        st = voice_style(voice, f)
        while st.width(word) > limit and st.size > 8:
            f *= 0.98
            st = voice_style(voice, f)
    return st


def _small(word: str) -> bool:
    return word.lower().strip(".,;") in STOPWORDS


def break_gloss(text: str, st: Style, max_width: float) -> list:
    """One line if it fits, else two balanced lines never ending on a small word."""
    words = text.split()
    if st.width(text) <= max_width or len(words) < 2:
        return [text]
    best, best_cost = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        wa, wb = st.width(a), st.width(b)
        if wa > max_width or wb > max_width or _small(words[i - 1]):
            continue
        cost = abs(wa - wb)
        if best_cost is None or cost < best_cost:
            best, best_cost = [a, b], cost
    return best if best is not None else wrap_greedy(text, st, max_width)


def wrap_greedy(text: str, st: Style, max_width: float) -> list:
    lines, cur = [], []
    for w in text.split():
        if cur and st.width(" ".join(cur + [w])) > max_width:
            if len(cur) > 1 and _small(cur[-1]):
                carry = cur.pop()
                lines.append(" ".join(cur))
                cur = [carry, w]
            else:
                lines.append(" ".join(cur))
                cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _draw_text(draw, x, baseline, text, st: Style, fill) -> None:
    text = st.prep(text)
    if not st.tracking:
        draw.text((x, baseline), text, font=st.font, fill=fill, anchor="ls")
        return
    for ch in text:
        draw.text((x, baseline), ch, font=st.font, fill=fill, anchor="ls")
        x += float(st.font.getlength(ch)) + st.tracking


def _draw_line(draw, x, baseline, text, st, fill, align="left", stop_fill=None):
    w = st.width(text)
    left = x if align == "left" else (x - w / 2 if align == "center" else x - w)
    if stop_fill is not None and text.endswith("."):
        body = text[:-1]
        _draw_text(draw, left, baseline, body, st, fill)
        _draw_text(draw, left + st.width(body) + st.tracking, baseline, ".", st, stop_fill)
    else:
        _draw_text(draw, left, baseline, text, st, fill)


@dataclass(frozen=True)
class NightPlan:
    template: str
    voice: str
    gloss_face: str
    colour_on: bool
    device: str
    centred: bool
    placement: Optional[dict] = field(default=None, compare=False)


def edition_number(date_iso: str) -> int:
    return (dt.date.fromisoformat(date_iso[:10]) - FIRST_EDITION).days + 1


def plan_night(date_iso: str, history: Sequence[str], photo: Image.Image, *, word: str, gloss: str,
               credit: str) -> NightPlan:
    """Choose the night's template and treatment. history is most recent first."""
    o = dt.date.fromisoformat(date_iso[:10]).toordinal()
    facts = analyse_photo(photo)
    voice = VOICE_CYCLE[o % 4]
    faces = PAIRS[voice]
    face = faces[(o // 4) % len(faces)]
    colour_on = facts.colour is not None and o % 2 == 0
    recent = [h for h in history if h in TEMPLATES]
    previous = recent[0] if recent else None
    placements = {t: place_on_photo(photo, t, voice, face, word, gloss, date_iso, credit, facts) for t in ON_PHOTO}
    eligible = [t for t in ORDER if t != previous and (t not in ON_PHOTO or placements[t] is not None)]

    def last_use(t: str) -> int:
        return recent.index(t) if t in recent else 10_000

    eligible.sort(key=lambda t: (-last_use(t), ORDER.index(t)))
    template = eligible[0]
    centred = template == "S1_BLEED" and sum(1 for h in recent if h == template) % 2 == 0
    if not colour_on:
        device = "none"
    elif template in ON_PHOTO:
        device = "word"
    else:
        device = "word" if (o // 2) % 2 == 0 else "rule"
    return NightPlan(template, voice, face, colour_on, device, centred, placements.get(template))


class _PhotoProbe:
    def __init__(self, photo: Image.Image):
        arr = np.asarray(photo.convert("RGB"), dtype=np.float64)
        self.lum = 0.2126 * _lin(arr[..., 0]) + 0.7152 * _lin(arr[..., 1]) + 0.0722 * _lin(arr[..., 2])
        self.h, self.w = self.lum.shape

    def region(self, x, y, w, h):
        x0, y0 = max(0, int(x - 6)), max(0, int(y - 2))
        x1, y1 = min(self.w, int(np.ceil(x + w + 6))), min(self.h, int(np.ceil(y + h + 2)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return self.lum[y0:y1, x0:x1]

    def score(self, rects, ink: str):
        worst, max_edge, med = 99.0, 0.0, 0.0
        for (x, y, w, h) in rects:
            reg = self.region(x, y, w, h)
            if reg is None:
                return None
            lo, hi = np.percentile(reg, 2.5), np.percentile(reg, 97.5)
            worst = min(worst, _contrast(1.0, hi) if ink == "w" else _contrast(lo, 0.006))
            hh, ww = (reg.shape[0] - 1) // 2 * 2, (reg.shape[1] - 1) // 2 * 2
            a = reg[0:hh:2, 0:ww:2]
            e = np.abs(a - reg[0:hh:2, 1:ww + 1:2]) + np.abs(a - reg[1:hh + 1:2, 0:ww:2])
            max_edge = max(max_edge, float(e.mean()) if e.size else 0.0)
            med = float(np.median(reg))
        return worst, max_edge, med


def _block_lines(template, voice, face, word, gloss, pos):
    if template == "S1_FRAME":
        wst = fit_word(voice, word, 1000, 0.85 if pos in ("tc", "bc") else 1.0)
        gst = gloss_style(face)
        glines = break_gloss(gloss, gst, 1000)
    else:
        wst = fit_word(voice, word, 600, SK[voice])
        gst, glines = None, []
    gap = 14
    height = wst.line_height + ((gap + gst.line_height * len(glines)) if glines else 0)
    top = 50 if pos[0] == "t" else 800 - 50 - height
    align = {"l": "left", "r": "right", "c": "center"}[pos[1]]
    x = {"left": 58, "right": WIDTH - 58, "center": WIDTH / 2}[align]
    lines = [(word, wst, align, x, top + wst.baseline_in_line())]
    y = top + wst.line_height + gap
    for g in glines:
        lines.append((g, gst, align, x, y + gst.baseline_in_line()))
        y += gst.line_height
    return lines


def _rects(lines):
    out = []
    for text, st, align, x, base in lines:
        w = st.width(text)
        left = x if align == "left" else (x - w / 2 if align == "center" else x - w)
        asc, desc = st.ascent_descent()
        out.append((left, base - asc, w, asc + desc))
    return out


def _subtitle_lines(face, gloss):
    gst = gloss_style(face, 1.12, 1.24)
    glines = break_gloss(gloss, gst, WIDTH - 180)
    top = 800 - 52 - gst.line_height * len(glines)
    return [(g, gst, "center", WIDTH / 2, top + gst.line_height * i + gst.baseline_in_line()) for i, g in enumerate(glines)]


def colophon_text(date_iso: str, credit: str, kind: str) -> list:
    d = dt.date.fromisoformat(date_iso[:10])
    num = f"No. {edition_number(date_iso)}"
    long = f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    short = f"{d.day:02d}.{d.month:02d}.{d.year}"
    if kind == "two":
        return [f"{num} · {short}", credit]
    if kind == "short":
        return [f"{num} · {short}"]
    if kind == "cred":
        return [credit]
    if kind == "museum":
        return [credit.replace("Photograph by ", ""), num, long]
    return [f"{num} · {long} · {credit}"]


def _colophon_corner(probe, text_rects, ink, prefer_top, date_iso, credit):
    cst = col_style()
    lines = colophon_text(date_iso, credit, "two")
    h = cst.line_height * len(lines)
    best = None
    for k in ("br", "bl", "tr", "tl"):
        top = 50 if k[0] == "t" else 800 - 46 - h
        align = "left" if k[1] == "l" else "right"
        x = 58 if align == "left" else WIDTH - 58
        cl = [(t, cst, align, x, top + cst.line_height * i + cst.baseline_in_line()) for i, t in enumerate(lines)]
        crs = _rects(cl)
        clash = any(not (a[0] + a[2] < b[0] - 28 or a[0] > b[0] + b[2] + 28 or a[1] + a[3] < b[1] - 20 or a[1] > b[1] + b[3] + 20)
                    for a in crs for b in text_rects)
        if clash:
            continue
        s = probe.score(crs, ink)
        if s is None or s[0] < 4.5:
            continue
        rank = s[0] + (100 if (k[0] == "t") == prefer_top else 0)
        if best is None or rank > best[0]:
            best = (rank, cl)
    return best[1] if best else None


def place_on_photo(photo, template, voice, face, word, gloss, date_iso, credit, facts=None):
    """Where the words go on the photograph, or None when no placement passes the calm test."""
    facts = facts or analyse_photo(photo)
    probe = _PhotoProbe(_fit_photo(photo, (WIDTH, 800)))
    if template == "S1_FRAME":
        min_c, max_e, inks = 7.0, 0.009, ("w", "k")
        positions = ["tc", "bc", "tl", "bl", "tr", "br"] if facts.symmetry < 0.05 else ["tl", "bl", "tr", "br"]
    else:
        min_c, max_e, inks, positions = 6.0, 0.011, ("w",), ["tl", "tr"]
    sub = _subtitle_lines(face, gloss) if template == "S1_SUBTITLE" else []
    best = None
    for pos in positions:
        lines = _block_lines(template, voice, face, word, gloss, pos) + sub
        rects = _rects(lines)
        if any(r[0] < 0 or r[0] + r[2] > WIDTH or r[1] < 0 or r[1] + r[3] > 800 for r in rects):
            continue
        for ink in inks:
            s = probe.score(rects, ink)
            if s is None or s[0] < min_c or s[1] > max_e:
                continue
            v = s[0] - s[1] * 120 + (50 if pos in ("tc", "bc") else 0)
            if best is None or v > best["v"] + 0.6:
                best = {"v": v, "pos": pos, "ink": ink, "lines": lines, "rects": rects}
    if best is None:
        return None
    col = _colophon_corner(probe, best["rects"], best["ink"], best["pos"][0] != "t", date_iso, credit)
    if col is None:
        return None
    best["colophon"] = col
    best["word_bg"] = float(np.median(probe.region(*best["rects"][0])))
    return best


def _rgba(rgb, alpha: float = 1.0):
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(round(255 * alpha)))


def _composite(base: Image.Image, painter) -> Image.Image:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    painter(ImageDraw.Draw(overlay))
    out = base.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB")


def render_edition_pair(photo, *, word, gloss, date_iso, credit, plan: NightPlan):
    """Light and dark plates for the planned night."""
    facts = analyse_photo(photo)
    return tuple(_render(photo, word, gloss, date_iso, credit, plan, facts, th) for th in ("light", "dark"))


def _render(photo, word, gloss, date_iso, credit, plan, facts, theme):
    if plan.template in ON_PHOTO:
        return _render_on_photo(photo, word, gloss, date_iso, credit, plan, facts)
    page, ink = THEMES[theme]["page"], THEMES[theme]["ink"]
    page_lum = _rel_lum(page)
    colour = facts.colour if plan.colour_on else None
    word_fill = _towards(colour, page_lum, 4.5) if (colour and plan.device == "word") else ink
    stop_fill = _towards(colour, page_lum, 3.0) if colour else ink
    rule_fill = _towards(colour, page_lum, 3.0) if (colour and plan.device == "rule") else None
    builder = {"S1_PLATE": _layout_plate, "S1_SLEEVE": _layout_sleeve, "S1_BLEED": _layout_bleed,
               "S1_ENTRY": _layout_entry, "S1_WALL": _layout_wall}[plan.template]
    ops, height, (px, py, pw, ph) = builder(plan, word, gloss, date_iso, credit)
    canvas = Image.new("RGB", (WIDTH, int(round(height))), page)
    canvas.paste(_fit_photo(photo, (pw, ph)), (int(px), int(py)))

    def paint(draw):
        for op in ops:
            if op[0] == "rule":
                if rule_fill is not None:
                    draw.rectangle([op[1], op[2], op[1] + 46, op[2] + 4], fill=_rgba(rule_fill))
                continue
            _, role, text, st, align, x, base = op
            fill = {"word": _rgba(word_fill), "gloss": _rgba(ink), "col": _rgba(ink, COL_ALPHA)}[role]
            _draw_line(draw, x, base, text, st, fill, align, _rgba(stop_fill) if role == "gloss" else None)

    return _composite(canvas, paint)


def _render_on_photo(photo, word, gloss, date_iso, credit, plan, facts):
    pl = plan.placement or place_on_photo(photo, plan.template, plan.voice, plan.gloss_face, word, gloss,
                                          date_iso, credit, facts)
    if pl is None:
        raise ValueError("no calm placement for an on-photo template")
    base = _fit_photo(photo, (WIDTH, 800))
    ink = PHOTO_INK[pl["ink"]]
    colour = facts.colour if plan.colour_on else None
    word_fill = _towards(colour, pl["word_bg"], 7.0) if colour else ink
    stop_fill = _towards(colour, pl["word_bg"], 3.0) if colour else ink

    def paint(draw):
        for i, (text, st, align, x, b) in enumerate(pl["lines"]):
            if i == 0:
                _draw_line(draw, x, b, text, st, _rgba(word_fill), align)
            else:
                _draw_line(draw, x, b, text, st, _rgba(ink), align, _rgba(stop_fill))
        for (text, st, align, x, b) in pl["colophon"]:
            _draw_line(draw, x, b, text, st, _rgba(ink, COL_ALPHA), align)

    return _composite(base, paint)


def _txt(role, text, st, align, x, base):
    return ("text", role, text, st, align, x, base)


def _layout_plate(plan, word, gloss, date_iso, credit):
    ops, cx, y = [], WIDTH / 2, 72
    photo_box = (132, y, 936, 624)
    y += 624 + 38
    wst = fit_word(plan.voice, word, 936)
    ops.append(_txt("word", word, wst, "center", cx, y + wst.baseline_in_line()))
    y += wst.line_height
    if plan.device == "rule":
        y += 24
        ops.append(("rule", cx - 23, y))
        y += 26
    else:
        y += 28
    gst = gloss_style(plan.gloss_face)
    for g in break_gloss(gloss, gst, 936):
        ops.append(_txt("gloss", g, gst, "center", cx, y + gst.baseline_in_line()))
        y += gst.line_height
    y += 30
    cst = col_style()
    for c in colophon_text(date_iso, credit, "full"):
        ops.append(_txt("col", c, cst, "center", cx, y + cst.baseline_in_line()))
        y += cst.line_height
    return ops, y + 88, photo_box


def _layout_sleeve(plan, word, gloss, date_iso, credit):
    ops, left, right, y = [], 132, WIDTH - 132, 72
    if plan.device == "rule":
        ops.append(("rule", left, y))
        y += 24
    wst = fit_word(plan.voice, word, 640)
    base = y + wst.baseline_in_line()
    ops.append(_txt("word", word, wst, "left", left, base))
    cst = col_style()
    ops.append(_txt("col", colophon_text(date_iso, credit, "short")[0], cst, "right", right, base))
    y += wst.line_height + 30
    photo_box = (left, y, 936, 624)
    y += 624 + 28
    gst = gloss_style(plan.gloss_face)
    first = y + gst.baseline_in_line()
    for g in break_gloss(gloss, gst, 600):
        ops.append(_txt("gloss", g, gst, "left", left, y + gst.baseline_in_line()))
        y += gst.line_height
    ops.append(_txt("col", colophon_text(date_iso, credit, "cred")[0], cst, "right", right, first))
    return ops, y + 84, photo_box


def _layout_bleed(plan, word, gloss, date_iso, credit):
    ops, y = [], 800
    photo_box = (0, 0, WIDTH, 800)
    if plan.device == "rule":
        ops.append(("rule", (WIDTH / 2 - 23) if plan.centred else 58, y))
    y += 30
    gst, cst = gloss_style(plan.gloss_face), col_style()
    if plan.centred:
        wst = fit_word(plan.voice, word, 1084, SK[plan.voice])
        ops.append(_txt("word", word, wst, "center", WIDTH / 2, y + wst.baseline_in_line()))
        y += wst.line_height + 12
        for g in break_gloss(gloss, gst, 1084):
            ops.append(_txt("gloss", g, gst, "center", WIDTH / 2, y + gst.baseline_in_line()))
            y += gst.line_height
        y += 14
        for c in colophon_text(date_iso, credit, "full"):
            ops.append(_txt("col", c, cst, "center", WIDTH / 2, y + cst.baseline_in_line()))
            y += cst.line_height
    else:
        wst = fit_word(plan.voice, word, 520, SK[plan.voice])
        glines = break_gloss(gloss, gst, 1084 - wst.width(word) - 56)
        ops.append(_txt("word", word, wst, "left", 58, y + wst.baseline_in_line()))
        gy = y + wst.baseline_in_line() - gst.baseline_in_line()
        for g in glines:
            ops.append(_txt("gloss", g, gst, "right", WIDTH - 58, gy + gst.baseline_in_line()))
            gy += gst.line_height
        y = max(y + wst.line_height, gy) + 14
        for c in colophon_text(date_iso, credit, "full"):
            ops.append(_txt("col", c, cst, "left", 58, y + cst.baseline_in_line()))
            y += cst.line_height
    return ops, y + 44, photo_box


def _layout_entry(plan, word, gloss, date_iso, credit):
    ops, left, width, y = [], 192, 816, 76
    photo_box = (left, y, width, 544)
    y += 544
    if plan.device == "rule":
        y += 34
        ops.append(("rule", left, y))
        y += 22
    else:
        y += 38
    wst = fit_word(plan.voice, word, 560, EK[plan.voice])
    gst = gloss_style(plan.gloss_face, lh=1.36)
    indent, gap = 1.1 * gst.size, 0.6 * gst.size
    words = gloss.split()
    head_w = wst.width(word) + gap
    best_i, best_cost = None, None
    for i in range(-1, len(words) - 1):
        if 0 <= i < 2 or (i >= 0 and _small(words[i])):
            continue
        l1 = head_w + gst.width(" ".join(words[: i + 1])) if i >= 0 else head_w - gap
        l2 = gst.width(" ".join(words[i + 1:]))
        if l1 > width or l2 > width - indent:
            continue
        cost = abs(l1 - (l2 + indent)) + (60 if i == -1 else 0)
        if best_cost is None or cost < best_cost:
            best_i, best_cost = i, cost
    line_h = max(gst.line_height, wst.line_height * 0.9)
    base = y + gst.baseline_in_line()
    ops.append(_txt("word", word, wst, "left", left, base))
    if best_i is None:
        first, rest = [], wrap_greedy(gloss, gst, width - indent)
    else:
        first, rest = (words[: best_i + 1] if best_i >= 0 else []), [" ".join(words[best_i + 1:])]
    if first:
        ops.append(_txt("gloss", " ".join(first), gst, "left", left + head_w, base))
    for r in rest:
        base += line_h
        ops.append(_txt("gloss", r, gst, "left", left + indent, base))
    y = base + (gst.line_height - gst.baseline_in_line()) + 18
    cst = col_style()
    for c in colophon_text(date_iso, credit, "two"):
        ops.append(_txt("col", c, cst, "right", left + width, y + cst.baseline_in_line()))
        y += cst.line_height
    return ops, y + 84, photo_box


def _layout_wall(plan, word, gloss, date_iso, credit):
    ops, x, width, foot = [], 872, 290, 600
    photo_box = (96, 110, 735, 490)
    wst = fit_word(plan.voice, word, width, 0.6)
    gst = gloss_style(plan.gloss_face, 33 / GLOSS_SPEC[plan.gloss_face][2], 1.32)
    glines = wrap_greedy(gloss, gst, width)
    cst = col_style(26 / 23, 0.09)
    clines = colophon_text(date_iso, credit, "museum")
    height = wst.line_height + 12 + gst.line_height * len(glines) + 18 + cst.line_height * len(clines)
    if plan.device == "rule":
        height += 20
    y = foot - height
    ops.append(_txt("word", word, wst, "left", x, y + wst.baseline_in_line()))
    y += wst.line_height + 12
    if plan.device == "rule":
        ops.append(("rule", x, y))
        y += 20
    for g in glines:
        ops.append(_txt("gloss", g, gst, "left", x, y + gst.baseline_in_line()))
        y += gst.line_height
    y += 18
    for c in clines:
        ops.append(_txt("col", c, cst, "left", x, y + cst.baseline_in_line()))
        y += cst.line_height
    return ops, 710, photo_box
