"""Caesura, the eighth plate template (SIY-204): one upright subject breaks the line of type.

The photograph is shown full bleed. The search finds one upright subject (a figure, a monolith,
a shaft of light, or a calm upright form such as a beam, a doorway or an aisle) and lets it
break the line: the word stands to the subject's left, the gloss to its right, on one shared
baseline, each set flush toward the subject at one gutter of 1.5 cap heights measured from the
ink. When a real straight horizontal edge (a horizon, a ridge, a stage floor) runs under the
word, the shared baseline may stand on it. Nights without such a subject are not eligible;
there is no one-sided fallback.

place_caesura returns a placement in the shape edition_look._render_on_photo draws, or None.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image

from . import edition_look as L

W, H = L.WIDTH, 800
MARGIN, EDGE = 58, 50          # Frame's margins: sides, top and bottom

MIN_C, MAX_E = 7.0, 0.009      # production on-photo floor (Frame)
WORD_FACTOR = {"italic": 0.68, "wide": 1.0, "condensed": 0.9}   # italic below Frame's 0.82: it shares a line
GLOSS_FACTOR = 0.95
TEX_MAX = 0.014                # texture in perceptual luma under the type
CLEAR = 18                     # no busy pixel within this distance of a line of type
SUBJECT_DL = 0.15              # the subject differs from the type's ground by this much luminance
SUBJECT_MAX = 0.45 * W         # widest subject accepted
GUTTER_CAPS = 1.5              # gutter, in cap heights of the word, ink to subject
LONG_TRACK = 0.12              # em, capitals words over eight letters
W_HI = 1.05 / MIN_C - 0.05     # white ink: ground must stay below this luminance
K_LO = MIN_C * 0.056 - 0.05    # dark ink: ground must stay above this luminance


# ---- lettering ---------------------------------------------------------------

@lru_cache(maxsize=64)
def word_style(voice: str, word: str) -> L.Style:
    st = L.voice_style(voice, WORD_FACTOR[voice])
    caps = st.upper or word.isupper()
    if caps and len(word) > 8:
        st = L.Style(st.font, st.size, LONG_TRACK * st.size, st.upper, st.line_height)
    elif caps and not st.tracking:
        # an italic word that arrives in capitals is letterspaced lightly, as capitals need
        st = L.Style(st.font, st.size, 0.06 * st.size, st.upper, st.line_height)
    return st


def gloss_style(face: str) -> L.Style:
    return L.gloss_style(face, GLOSS_FACTOR)


def cap_height(st: L.Style) -> float:
    return float(-st.font.getbbox("H", anchor="ls")[1])


def ink_extent(st: L.Style, text: str):
    """(left, top, right, bottom) of the real ink, relative to the pen at (0, baseline)."""
    t = st.prep(text)
    if not st.tracking:
        return tuple(float(v) for v in st.font.getbbox(t, anchor="ls"))
    x, box = 0.0, None
    for ch in t:
        if ch.strip():
            l, tp, r, b = st.font.getbbox(ch, anchor="ls")
            cur = (x + l, tp, x + r, b)
            box = cur if box is None else (min(box[0], cur[0]), min(box[1], cur[1]),
                                           max(box[2], cur[2]), max(box[3], cur[3]))
        x += float(st.font.getlength(ch)) + st.tracking
    return box


def breaks(text: str, st: L.Style, n: int):
    """All n-line breaks that never end a line on a small word. The last line carries two words,
    or on three lines one long word."""
    words = text.split()
    cuts = [(i,) for i in range(1, len(words))] if n == 2 else \
        [(i, j) for i in range(1, len(words)) for j in range(i + 1, len(words))]
    out = []
    for c in cuts:
        idx = (0,) + c + (len(words),)
        if any(L._small(words[idx[k + 1] - 1]) for k in range(n - 1)):
            continue
        if idx[-1] - idx[-2] < 2 and (n == 2 or len(words[-1].strip(".,;")) < 8):
            continue    # a single last word only on three lines, and only a long one
        lines = [" ".join(words[idx[k]:idx[k + 1]]) for k in range(n)]
        ws = [ink_extent(st, t)[2] - ink_extent(st, t)[0] for t in lines]
        if min(ws[:-1]) < 0.5 * max(ws) or ws[-1] < 0.4 * max(ws):
            continue    # an even rag: no short middle line, no stub at the end
        cost = sum((max(ws) - w) ** 2 for w in ws[:-1]) + 0.5 * (max(ws) - ws[-1]) ** 2
        if ws[-1] > max(ws[:-1]) + 4:
            cost += 4 * (ws[-1] - max(ws[:-1])) ** 2    # the last line no longer than the ones above
        out.append((cost, max(ws), lines))
    return sorted(out, key=lambda r: r[0])


def best_break(text: str, st: L.Style, measure: float):
    """Two balanced lines; three only when two will not fit the narrow side."""
    for n in (2, 3):
        fits = [r for r in breaks(text, st, n) if r[1] <= measure]
        if fits:
            return fits[0][2]
    return None


# ---- the photograph ----------------------------------------------------------------

def _integral(a):
    return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _box_mean(a, r):
    ii = _integral(a)
    h, w = a.shape
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    return s / ((y1 - y0)[:, None] * (x1 - x0)[None, :])


class Ground:
    """Luminance, perceptual luma, busy texture, and where each ink fails, for the fitted photograph."""

    def __init__(self, fitted: Image.Image):
        self.probe = L._PhotoProbe(fitted)
        self.lum = self.probe.lum
        g = np.asarray(fitted.convert("L"), dtype=np.float64) / 255
        self.gam = g
        e = np.zeros_like(g)
        e[:, :-1] += np.abs(np.diff(g, axis=1))
        e[:-1, :] += np.abs(np.diff(g, axis=0))
        busy = _box_mean(e, 4) > 2.2 * TEX_MAX
        self.bad = {"w": busy | (self.lum > W_HI), "k": busy | (self.lum < K_LO)}
        self.busy_ii = _integral(busy.astype(np.float64))

    def busy_count(self, x0, y0, x1, y1) -> float:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        a = self.busy_ii
        return float(a[y1, x1] - a[y0, x1] - a[y1, x0] + a[y0, x0])

    def texture(self, box) -> float:
        x0, y0, x1, y1 = box
        r = self.gam[max(0, int(y0 - 2)):int(y1 + 2), max(0, int(x0 - 6)):int(x1 + 6)]
        return float(np.abs(np.diff(r, axis=1)).mean() + np.abs(np.diff(r, axis=0)).mean())


def stands_on(gr: Ground, x0, x1, base) -> bool:
    """A real straight horizontal edge (a horizon, a ridge, a stage floor) runs just under the word."""
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    ys = range(int(base) + 4, min(H - 2, int(base) + 17))
    prof = [float(np.abs(gr.gam[y + 2, x0:x1] - gr.gam[y - 2, x0:x1]).mean()) for y in ys]
    return bool(prof) and max(prof) >= 0.08


# ---- the subject ---------------------------------------------------------------

def subject_runs(gr: Ground, ink: str, top, bot):
    """Stretches of the word's band where type would fail: candidate subjects (x0, x1)."""
    rows = gr.bad[ink][max(0, int(top)):int(bot)]
    bad = rows.mean(axis=0) > 0.04
    runs, x = [], 0
    while x < W:
        if bad[x]:
            s = x
            while x < W and bad[x]:
                x += 1
            if runs and s - runs[-1][1] < 24:
                runs[-1] = (runs[-1][0], x)
            else:
                runs.append((s, x))
        else:
            x += 1
    return [r for r in runs if 8 <= r[1] - r[0] <= SUBJECT_MAX]


def subject_ok(gr: Ground, sl: int, sr: int, base: int, cap: float):
    """One upright subject: it stands out from the type's ground by SUBJECT_DL in luminance, and it
    runs at least one cap height above the cap line and one below the baseline. Returns the visible
    edges (x0, x1) of the subject in the word's band, where the gutter is measured from, or None."""
    top, bot = int(base - 2 * cap), int(base + cap)
    if top < 0 or bot > H:
        return None
    ground = np.concatenate([gr.lum[int(base - cap):int(base), max(0, sl - 100):max(0, sl - 24)].ravel(),
                             gr.lum[int(base - cap):int(base), min(W, sr + 24):min(W, sr + 100)].ravel()])
    if ground.size < 50:
        return None
    gm = float(np.median(ground))
    hit = np.abs(gr.lum[top:bot, sl:sr] - gm) >= SUBJECT_DL
    need = max(3, 0.3 * (sr - sl))     # the same upright form in every row, not a wider mass at one height
    if (hit.sum(axis=1) >= need).mean() < 0.9:
        return None
    band = np.abs(gr.lum[int(base - cap):int(base), sl:sr] - gm) >= SUBJECT_DL
    seen = np.where(band.mean(axis=0) > 0.1)[0]
    return (sl + int(seen[0]), sl + int(seen[-1]) + 1) if seen.size else None


# ---- placement ------------------------------------------------------------------

def place_caesura(photo: Image.Image, voice: str, face: str, word: str, gloss: str,
                  why: Optional[dict] = None) -> Optional[dict]:
    """The best Caesura placement for the night, or None when the photograph has no subject with calm
    room on both sides at one height. face is the gloss face. why, a dict, counts failure reasons."""
    why = why if why is not None else {}

    def no(reason):
        why[reason] = why.get(reason, 0) + 1

    gr = Ground(L._fit_photo(photo, (W, H)))
    wst, gst = word_style(voice, word), gloss_style(face)
    cap = cap_height(wst)
    gut = round(GUTTER_CAPS * cap)
    wl, wt, wr, wb = ink_extent(wst, word)
    wink = wr - wl
    lh = gst.line_height
    best = None
    for base in range(int(EDGE + cap + 4), H - EDGE - 20, 3):
        for ink in ("w", "k"):
            for sl, sr in subject_runs(gr, ink, base + wt, base + max(wb, 0)):
                if sl - gut - wink < MARGIN:
                    no("word room")
                    continue
                if W - MARGIN - sr - gut < 200:
                    no("gloss room")
                    continue
                seen = subject_ok(gr, sl, sr, base, cap)
                if seen is None:
                    no("subject")
                    continue
                # the gutter runs from the visible edge of the subject; a fainter fringe beyond it is
                # still held off by the calm tests below
                xr_ink, xl_ink = seen[0] - gut, seen[1] + gut
                if xr_ink - wink < MARGIN:
                    no("word room")
                    continue
                gl = best_break(gloss, gst, W - MARGIN - xl_ink)
                if gl is None:
                    no("gloss room")
                    continue
                on_line = stands_on(gr, xr_ink - wink, xr_ink, base)
                for foot in ((False, True) if on_line else (False,)):
                    off = -(len(gl) - 1) if foot else 0
                    wx = xr_ink - wr                                    # pen x of the word
                    wbox = (wx + wl, base + wt, wx + wr, base + wb)
                    gpens, gboxes = [], []
                    for i, t in enumerate(gl):
                        l, tp, r, b = ink_extent(gst, t)
                        y = base + (off + i) * lh
                        gpens.append((xl_ink - l, y))
                        gboxes.append((xl_ink, y + tp, xl_ink + r - l, y + b))
                    boxes = [wbox] + gboxes
                    if min(b[1] for b in boxes) < EDGE or max(b[3] for b in boxes) > H - EDGE \
                            or max(b[2] for b in boxes) > W - MARGIN:
                        no("margins")
                        continue
                    # nothing busy within CLEAR of any line; a word standing on a real line may touch it below
                    below = [2 if (on_line and i == 0) else CLEAR for i in range(len(boxes))]
                    if any(gr.busy_count(b[0] - CLEAR, b[1] - CLEAR, b[2] + CLEAR, b[3] + below[i]) > 0
                           for i, b in enumerate(boxes)):
                        no("busy within 18 px")
                        continue
                    rects = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in boxes]
                    sw, sg = gr.probe.score(rects[:1], ink), gr.probe.score(rects[1:], ink)
                    if not sw or not sg or min(sw[0], sg[0]) < MIN_C or max(sw[1], sg[1]) > MAX_E:
                        no("contrast or edges")
                        continue
                    tex = max(gr.texture(b) for b in boxes)
                    if tex > TEX_MAX:
                        no("texture")
                        continue
                    no("pass")
                    v = 0.5 * (min(sw[0], 12) + min(sg[0], 12)) - 200 * max(sw[1], sg[1]) - 150 * tex
                    v -= 2.0 * (len(gl) - 2)
                    v += 6 if on_line else 0
                    v += 1.5 if foot else 0
                    v -= 4 * abs(base / H - 0.5)
                    if best is None or v > best["v"]:
                        best = {
                            "v": v, "ink": ink, "base": base, "subject": seen, "gutter": gut,
                            "on_line": on_line, "foot": foot, "boxes": boxes,
                            "lines": [(word, wst, "left", wx, base)]
                                     + [(t, gst, "left", x, y) for t, (x, y) in zip(gl, gpens)],
                            "word_bg": float(np.median(gr.probe.region(*rects[0]))),
                            "stop_bg": float(np.median(gr.probe.region(*rects[-1]))),
                            "colophon": [],
                        }
    return best
