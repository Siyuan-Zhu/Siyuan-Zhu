"""Curatorial 5-Archetype Compositor & Quadrant Edge-Density Detector (WS3 Seam 4).

Implements the Stitch Curatorial Editorial System:
  1. `compute_quietest_quadrant(image_bytes)` — evaluates 4 quadrants of a 1200x800 (3:2)
     plate for normalized Sobel/gradient edge density and WCAG relative luminance.
  2. `select_curatorial_archetype(...)` — selects today's curatorial archetype across
     the 5-style 30-day rotation (`POETIC_CINEMA`, `LITERARY_JOURNAL`, `STUDIO_MONOGRAPH`,
     `KYOTO_TANKOBON`, `CONCRETE_POETRY`) with consecutive-day anti-repetition.
  3. `render_curatorial_plates(...)` — renders dual-mode 1200x800 WebP plates
     (`assets/daily_pulse_light.webp` and `assets/daily_pulse_dark.webp`) across all 5
     archetypes for both real Unsplash photography (~80%) and commissioned AI/procedural
     plates (~20%).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import textwrap
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PLATE_WIDTH: int = 1200
PLATE_HEIGHT: int = 800

EDGE_GRADIENT_THRESHOLD: float = 0.15
CALM_QUADRANT_MAX_EDGE_DENSITY: float = 0.14
BREAKOUT_VELOCITY_THRESHOLD: float = 4.0

ARCHETYPE_TARGETS: dict[str, float] = {
    "POETIC_CINEMA": 0.35,
    "LITERARY_JOURNAL": 0.25,
    "STUDIO_MONOGRAPH": 0.20,
    "KYOTO_TANKOBON": 0.15,
    "CONCRETE_POETRY": 0.05,
}

VALID_ARCHETYPES: tuple[str, ...] = (
    "POETIC_CINEMA",
    "LITERARY_JOURNAL",
    "STUDIO_MONOGRAPH",
    "KYOTO_TANKOBON",
    "CONCRETE_POETRY",
)
ALL_ARCHETYPES: tuple[str, ...] = VALID_ARCHETYPES
QUADRANT_NAMES: tuple[str, ...] = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
)

VALID_QUADRANTS: tuple[str, ...] = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "margin_outside",
    "margin_bottom",
    "margin_right",
    "full_canvas",
)

QUADRANT_BOUNDS_1200X800: dict[str, tuple[int, int, int, int]] = {
    "upper_left": (0, 0, 600, 400),
    "upper_right": (600, 0, 1200, 400),
    "lower_left": (0, 400, 600, 800),
    "lower_right": (600, 400, 1200, 800),
}

_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_SERIF_FONT_CANDIDATES = (str(_FONT_DIR / "CormorantGaramond.ttf"),)
_SERIF_ITALIC_CANDIDATES = (str(_FONT_DIR / "CormorantGaramond-Italic.ttf"),)
_SANS_FONT_CANDIDATES = (str(_FONT_DIR / "IBMPlexMono-Regular.ttf"),)
_MONO_FONT_CANDIDATES = _SANS_FONT_CANDIDATES


def _load_font(candidates: Sequence[str], size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    raise RuntimeError("Bundled editorial font missing or unreadable; refusing typography drift")


def _coerce_image(image_source: bytes | Image.Image) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    with Image.open(BytesIO(image_source)) as im:
        return im.convert("RGB")


def compute_quietest_quadrant(
    image_source: bytes | Image.Image,
) -> tuple[str, float, float]:
    """Evaluate the 4 quadrants of a 1200x800 (3:2) image for edge density and luminance.

    Returns `(best_quadrant, min_edge_density, mean_relative_luminance)` where:
      - `best_quadrant` is one of `("upper_left", "upper_right", "lower_left", "lower_right")`
      - `min_edge_density` in `[0.0, 1.0]` is the fraction of pixels in `best_quadrant`
        with normalized gradient magnitude `> 0.15`
      - `mean_relative_luminance` in `[0.0, 1.0]` is WCAG relative luminance in `best_quadrant`
    """
    rgb_img = _coerce_image(image_source)
    fitted = ImageOps.fit(
        rgb_img,
        (PLATE_WIDTH, PLATE_HEIGHT),
        method=Image.Resampling.LANCZOS,
    )
    arr = np.asarray(fitted, dtype=np.float32) / 255.0
    # WCAG relative luminance per pixel
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    # Normalized gradient magnitude via central differences
    gy, gx = np.gradient(lum)
    grad_mag = np.hypot(gx, gy)
    max_grad = float(np.max(grad_mag))
    if max_grad > 1e-6:
        # Scale against theoretical max or empirical 99th percentile while preserving flat images
        norm_denom = max(0.35, min(1.0, max_grad))
        norm_grad = grad_mag / norm_denom
    else:
        norm_grad = grad_mag

    best_quadrant = "upper_left"
    min_edge_density = 1.0
    best_luminance = 0.5

    # Inspect inner 84% of each quadrant so frame borders don't skew edge density
    for q_name, (x0, y0, x1, y1) in QUADRANT_BOUNDS_1200X800.items():
        pad_x = int((x1 - x0) * 0.08)
        pad_y = int((y1 - y0) * 0.08)
        sub_grad = norm_grad[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
        sub_lum = lum[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
        edge_density = float(np.mean(sub_grad > EDGE_GRADIENT_THRESHOLD))
        mean_lum = float(np.mean(sub_lum))
        if edge_density < min_edge_density - 1e-6:
            min_edge_density = edge_density
            best_quadrant = q_name
            best_luminance = mean_lum

    return (
        best_quadrant,
        round(float(min_edge_density), 4),
        round(float(best_luminance), 4),
    )


def select_curatorial_archetype(
    min_edge_density: float,
    breakout_surge: bool = False,
    velocity: float = 0.0,
    archetype_history_30d: Sequence[str] = (),
) -> str:
    """Select today's curatorial archetype across the 5-archetype 30-day rotation.

    Rules:
      1. `CONCRETE_POETRY` is eligible only when `breakout_surge and velocity >= 4.0`.
      2. `POETIC_CINEMA` is eligible only when `min_edge_density < 0.14` (a calm quadrant exists).
      3. Consecutive-day anti-repetition: yesterday's archetype (`archetype_history_30d[0]`)
         is excluded from the candidate set when at least one other archetype is eligible.
      4. Among eligible archetypes, selects the archetype maximizing target deficit
         `T(A) - count_30d(A) / max(1, len(history_30d))`.
    """
    cleaned_history = [
        str(a).strip().upper()
        for a in (archetype_history_30d or ())
        if str(a).strip().upper() in ARCHETYPE_TARGETS
    ][:30]

    eligible: list[str] = []
    for arch in VALID_ARCHETYPES:
        if arch == "CONCRETE_POETRY" and not (breakout_surge and float(velocity) >= BREAKOUT_VELOCITY_THRESHOLD):
            continue
        if arch == "POETIC_CINEMA" and float(min_edge_density) >= CALM_QUADRANT_MAX_EDGE_DENSITY:
            continue
        eligible.append(arch)

    if not eligible:
        eligible = ["LITERARY_JOURNAL", "STUDIO_MONOGRAPH", "KYOTO_TANKOBON"]

    # When breakout surge triggers and CONCRETE_POETRY hasn't been used yesterday, prioritize it if under target
    yesterday = cleaned_history[0] if cleaned_history else None
    if yesterday and len(eligible) > 1 and yesterday in eligible:
        eligible = [a for a in eligible if a != yesterday]

    n_hist = max(1, len(cleaned_history))
    best_arch = eligible[0]
    best_deficit = -999.0

    for arch in eligible:
        target = ARCHETYPE_TARGETS[arch]
        actual_share = sum(1 for h in cleaned_history if h == arch) / float(n_hist) if cleaned_history else 0.0
        deficit = target - actual_share
        # Give CONCRETE_POETRY a boost on genuine breakout surge days when its 30d count is 0
        if arch == "CONCRETE_POETRY" and breakout_surge and float(velocity) >= BREAKOUT_VELOCITY_THRESHOLD:
            if "CONCRETE_POETRY" not in cleaned_history:
                deficit += 1.0
        if deficit > best_deficit + 1e-9:
            best_deficit = deficit
            best_arch = arch

    return best_arch


def resolve_effective_quadrant(archetype: str, detected_quadrant: str) -> str:
    """Return the canonical Firestore-validated `text_quadrant` for a given archetype."""
    arch = str(archetype or "POETIC_CINEMA").strip().upper()
    if arch == "POETIC_CINEMA":
        q = str(detected_quadrant or "upper_left").strip().lower()
        return q if q in QUADRANT_BOUNDS_1200X800 else "upper_left"
    if arch == "LITERARY_JOURNAL":
        return "margin_outside"
    if arch == "STUDIO_MONOGRAPH":
        return "margin_bottom"
    if arch == "KYOTO_TANKOBON":
        return "margin_right"
    if arch == "CONCRETE_POETRY":
        return "full_canvas"
    return "upper_left"


def _credit_line(attribution: Mapping[str, Any], visual_medium: str) -> str:
    medium = str(visual_medium or "PHOTOGRAPHY").upper()
    if medium == "PHOTOGRAPHY":
        photographer = str(attribution.get("photographer_name") or "Unsplash").strip()
        return f"Photograph · {photographer} / Unsplash"
    if medium == "FLUX_PRO":
        return "Commissioned Plate · FLUX Editorial Synthesis"
    if medium == "IMAGEN_3":
        return "Commissioned Plate · Gemini Imagen 3"
    return "Procedural Plate · Topographic Synthesis"


def _pixel_wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Wrap complete text by ink width, splitting an oversized token without omission."""
    lines: list[str] = []
    line = ''
    for token in text.strip().split():
        candidate = f'{line} {token}'.strip()
        if font.getbbox(candidate)[2] - font.getbbox(candidate)[0] <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
            line = ''
        for ch in token:
            candidate = line + ch
            box = font.getbbox(candidate)
            if box[2] - box[0] > width and line:
                lines.append(line)
                line = ch
            else:
                line = candidate
    if line:
        lines.append(line)
    return lines


def _relative_luminance(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def layout_poetic_cinema_text(hero_img: Image.Image, *, word: str, gloss: str, quadrant: str) -> dict[str, Any]:
    """Measure complete bundled-font text inside a 48px quadrant inset.

    Choose the most legible placement within the already selected quiet quadrant,
    using the local text backgrounds rather than its aggregate brightness. The
    single-line approved gloss wraps only when its actual glyphs require it.
    """
    q_key = quadrant if quadrant in QUADRANT_BOUNDS_1200X800 else 'upper_left'
    x0, y0, x1, y1 = QUADRANT_BOUNDS_1200X800[q_key]
    safe = (x0 + 48, y0 + 48, x1 - 48, y1 - 48)
    fonts = {False: _load_font(_SERIF_FONT_CANDIDATES, 60),
             True: _load_font(_SERIF_ITALIC_CANDIDATES, 21.25)}
    measured = []
    cursor = 0
    for italic, text in ((False, word), (True, gloss)):
        if italic:
            cursor += 16
        for line in _pixel_wrap(text, fonts[italic], safe[2] - safe[0]):
            bbox = fonts[italic].getbbox(line)
            measured.append((line, italic, bbox, cursor))
            cursor += bbox[3] - bbox[1] + 6
    height = cursor - 6
    width = max(b[2] - b[0] for _, _, b, _ in measured)
    if height > safe[3] - safe[1]:
        raise ValueError('Poetic text cannot fit its quiet quadrant without losing content')
    fitted = ImageOps.fit(hero_img.convert('RGB'), (PLATE_WIDTH, PLATE_HEIGHT), method=Image.Resampling.LANCZOS)
    pixels = np.asarray(fitted, dtype=float)
    # Try the existing outer-corner placement first, then move only as needed.
    xs = np.linspace(safe[0], safe[2] - width, 5).astype(int).tolist()
    ys = np.linspace(safe[1], safe[3] - height, 7).astype(int).tolist()
    if 'right' in q_key:
        xs.reverse()
    if 'lower' in q_key:
        ys.reverse()
    best = None
    for y in ys:
        for x in xs:
            runs = []
            for text, italic, bbox, offset in measured:
                line_width = bbox[2] - bbox[0]
                left = x + (width - line_width if 'right' in q_key else 0)
                top = y + offset
                runs.append({'text': text, 'italic': italic, 'font_size': 21.25 if italic else 60,
                    'position': (left - bbox[0], top - bbox[1]),
                    'bounds': (left, top, left + line_width, top + bbox[3] - bbox[1])})
            for ink, scrim_color in (((24,24,27), (251,249,245)), ((248,249,250), (12,16,22))):
                ink_lum = float(_relative_luminance(np.array(ink)))
                for opacity in (0.0, 0.18):
                    score = min(float(np.percentile(
                        (np.maximum(lum, ink_lum) + .05) / (np.minimum(lum, ink_lum) + .05), 5))
                        for run in runs
                        for lum in [_relative_luminance(
                            pixels[run['bounds'][1]:run['bounds'][3], run['bounds'][0]:run['bounds'][2]]
                            * (1-opacity) + np.array(scrim_color) * opacity)])
                    # Prefer unaltered photography and the corner when contrast is sufficient.
                    rank = (min(score, 7.0) - opacity, score)
                    if best is None or rank > best[0]:
                        best = (rank, {'runs': runs, 'ink': ink, 'scrim_color': scrim_color,
                            'scrim_opacity': opacity, 'scrim_bounds': (x-16,y-16,x+width+16,y+height+16),
                            'contrast_ratio': score})
    assert best is not None
    if best[1]['contrast_ratio'] < 4.5:
        raise ValueError('Poetic text lacks local contrast within the approved 18% scrim limit')
    return best[1]


def _render_poetic_cinema_plate(
    hero_img: Image.Image,
    *, word: str, gloss: str, quadrant: str, luminance: float, dark_mode: bool,
) -> Image.Image:
    """Full-bleed photography, complete measured type and no decorative chrome."""
    canvas = ImageOps.fit(hero_img, (PLATE_WIDTH, PLATE_HEIGHT), method=Image.Resampling.LANCZOS).convert('RGBA')
    layout = layout_poetic_cinema_text(canvas, word=word, gloss=gloss, quadrant=quadrant)
    if layout['scrim_opacity']:
        scrim = Image.new('RGBA', canvas.size, (0,0,0,0))
        ImageDraw.Draw(scrim).rectangle(layout['scrim_bounds'],
            fill=(*layout['scrim_color'], int(255 * layout['scrim_opacity'])))
        canvas = Image.alpha_composite(canvas, scrim)
    draw = ImageDraw.Draw(canvas)
    for run in layout['runs']:
        font = _load_font(_SERIF_ITALIC_CANDIDATES if run['italic'] else _SERIF_FONT_CANDIDATES, run['font_size'])
        draw.text(run['position'], run['text'], font=font, fill=(*layout['ink'],255))
    return canvas.convert('RGB')


def layout_literary_title(word):
    """Wrap actual glyph bounds inside the existing editorial column."""
    font = _load_font(_SERIF_FONT_CANDIDATES, 57.5)
    lines = _pixel_wrap(word, font, 360)
    runs = []
    y = 224
    for line in lines:
        bounds = font.getbbox(line)
        runs.append((line, (56-bounds[0], y-bounds[1])))
        y += bounds[3]-bounds[1]+12
    if y > 480:
        raise ValueError('Literary title exceeds its existing text column')
    return runs, max(294, y+12)


def _render_literary_journal_plate(
    hero_img: Image.Image,
    *,
    word: str,
    gloss: str,
    thesis: str,
    date_str: str,
    attribution: Mapping[str, Any],
    visual_medium: str,
    dark_mode: bool,
) -> Image.Image:
    """Render Archetype #2 (`LITERARY_JOURNAL`): asymmetric 40/60 spread with 100% untouched 3:2 plate."""
    bg = (15, 20, 25) if dark_mode else (251, 249, 245)
    ink_primary = (243, 244, 246) if dark_mode else (24, 24, 27)
    ink_secondary = (168, 162, 158) if dark_mode else (87, 83, 78)
    ink_muted = (120, 113, 108) if dark_mode else (120, 113, 108)
    rule_color = (42, 48, 56) if dark_mode else (228, 223, 215)

    canvas = Image.new("RGB", (PLATE_WIDTH, PLATE_HEIGHT), color=bg)
    draw = ImageDraw.Draw(canvas)

    # Right 60% column: untouched 3:2 photograph/art plate (660 x 440) centered vertically
    art_w, art_h = 660, 440
    art_x = 486
    art_y = (PLATE_HEIGHT - art_h) // 2
    fitted_art = ImageOps.fit(hero_img, (art_w, art_h), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted_art, (art_x, art_y))
    draw.rectangle([art_x - 1, art_y - 1, art_x + art_w, art_y + art_h], outline=rule_color, width=1)

    # Vertical hairline divider
    draw.line([(444, 96), (444, PLATE_HEIGHT - 96)], fill=rule_color, width=1)

    mono_font = _load_font(_MONO_FONT_CANDIDATES, 12)
    title_font = _load_font(_SERIF_FONT_CANDIDATES, 57.5)
    gloss_font = _load_font(_SERIF_ITALIC_CANDIDATES, 23.75)
    body_font = _load_font(_SANS_FONT_CANDIDATES, 14)

    left_x = 56
    draw.text((left_x, 168), f"THE DAILY ZEITGEIST  ·  {date_str}", font=mono_font, fill=ink_muted)
    draw.line([(left_x, 194), (left_x + 72, 194)], fill=rule_color, width=1)

    title_runs, gy = layout_literary_title(word)
    for text, position in title_runs:
        draw.text(position, text, font=title_font, fill=ink_primary)
    for line in _pixel_wrap(gloss.strip(), gloss_font, 330):
        draw.text((left_x, gy), line, font=gloss_font, fill=ink_secondary)
        gy += 27

    gy += 18
    draw.line([(left_x, gy), (left_x + 330, gy)], fill=rule_color, width=1)
    gy += 20

    for line in textwrap.wrap(thesis.strip(), width=42)[:5]:
        draw.text((left_x, gy), line, font=body_font, fill=ink_secondary)
        gy += 22

    credit = _credit_line(attribution, visual_medium)
    draw.text((art_x, art_y + art_h + 16), credit, font=mono_font, fill=ink_muted)

    return canvas


def _draw_studio_monograph_credit(draw, *, date_str, attribution, visual_medium):
    """Keep regular generation and display-only correction on the same typography."""
    font = _load_font(_MONO_FONT_CANDIDATES, 11)
    credit = f"{date_str}  ·  {_credit_line(attribution, visual_medium).upper()}"
    box = draw.textbbox((0, 0), credit, font=font)
    draw.text((PLATE_WIDTH // 2 - (box[2] - box[0]) // 2, 724), credit,
              font=font, fill=(120, 113, 108))


def replace_studio_monograph_footer(image, *, date_str, attribution, visual_medium,
                                    dark_mode):
    """Correct the credit strip without decoding/recomposing the accepted hero.

    Save the returned image losslessly to retain every decoded pixel outside
    this strip, including the original word, gloss and photograph/render.
    """
    if image.size != (PLATE_WIDTH, PLATE_HEIGHT):
        raise ValueError("Unexpected Studio Monograph dimensions")
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 716, PLATE_WIDTH - 1, 747),
                   fill=(15, 20, 25) if dark_mode else (251, 249, 245))
    _draw_studio_monograph_credit(draw, date_str=date_str, attribution=attribution,
                                  visual_medium=visual_medium)
    return canvas


def _render_studio_monograph_plate(
    hero_img: Image.Image,
    *,
    word: str,
    gloss: str,
    date_str: str,
    attribution: Mapping[str, Any],
    visual_medium: str,
    dark_mode: bool,
) -> Image.Image:
    """Render Archetype #3 (`STUDIO_MONOGRAPH`): centered 3:2 exhibition plate above wide museum plinth."""
    bg = (15, 20, 25) if dark_mode else (251, 249, 245)
    ink_primary = (243, 244, 246) if dark_mode else (24, 24, 27)
    ink_secondary = (168, 162, 158) if dark_mode else (87, 83, 78)
    ink_muted = (120, 113, 108) if dark_mode else (120, 113, 108)
    rule_color = (42, 48, 56) if dark_mode else (228, 223, 215)

    canvas = Image.new("RGB", (PLATE_WIDTH, PLATE_HEIGHT), color=bg)
    draw = ImageDraw.Draw(canvas)

    # Centered 3:2 plate (810 x 540) at top with generous mat margins
    art_w, art_h = 810, 540
    art_x = (PLATE_WIDTH - art_w) // 2
    art_y = 46
    fitted_art = ImageOps.fit(hero_img, (art_w, art_h), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted_art, (art_x, art_y))
    draw.rectangle([art_x - 1, art_y - 1, art_x + art_w, art_y + art_h], outline=rule_color, width=1)

    title_font = _load_font(_SERIF_FONT_CANDIDATES, 52.5)
    gloss_font = _load_font(_SERIF_ITALIC_CANDIDATES, 22.5)
    cx = PLATE_WIDTH // 2
    w_box = draw.textbbox((0, 0), word, font=title_font)
    w_w = w_box[2] - w_box[0]
    draw.text((cx - w_w // 2, 612), word, font=title_font, fill=ink_primary)

    gloss_line = gloss.strip()
    if len(gloss_line) > 84:
        gloss_line = gloss_line[:81].rstrip() + "..."
    g_box = draw.textbbox((0, 0), gloss_line, font=gloss_font)
    g_w = g_box[2] - g_box[0]
    draw.text((cx - g_w // 2, 672), gloss_line, font=gloss_font, fill=ink_secondary)

    _draw_studio_monograph_credit(draw, date_str=date_str, attribution=attribution,
                                  visual_medium=visual_medium)

    return canvas


def _render_kyoto_tankobon_plate(
    hero_img: Image.Image,
    *,
    word: str,
    gloss: str,
    date_str: str,
    attribution: Mapping[str, Any],
    visual_medium: str,
    dark_mode: bool,
) -> Image.Image:
    """Render Archetype #4 (`KYOTO_TANKOBON`): Japanese Tankobon jacket with vertical spine & vermilion Hanko seal."""
    bg = (15, 20, 25) if dark_mode else (251, 249, 245)
    spine_bg = (22, 28, 36) if dark_mode else (244, 240, 232)
    ink_primary = (243, 244, 246) if dark_mode else (24, 24, 27)
    ink_secondary = (168, 162, 158) if dark_mode else (87, 83, 78)
    rule_color = (48, 54, 62) if dark_mode else (220, 214, 204)
    vermilion = (200, 62, 43)

    canvas = Image.new("RGB", (PLATE_WIDTH, PLATE_HEIGHT), color=bg)
    draw = ImageDraw.Draw(canvas)

    # Right spine band
    spine_x = 936
    draw.rectangle([spine_x, 0, PLATE_WIDTH, PLATE_HEIGHT], fill=spine_bg)
    draw.line([(spine_x, 0), (spine_x, PLATE_HEIGHT)], fill=rule_color, width=1)

    # Left 3:2 art plate (840 x 560) untouched by text
    art_w, art_h = 840, 560
    art_x = 48
    art_y = 88
    fitted_art = ImageOps.fit(hero_img, (art_w, art_h), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted_art, (art_x, art_y))
    draw.rectangle([art_x - 1, art_y - 1, art_x + art_w, art_y + art_h], outline=rule_color, width=1)

    mono_font = _load_font(_MONO_FONT_CANDIDATES, 11)
    gloss_font = _load_font(_SERIF_ITALIC_CANDIDATES, 20)
    vert_font = _load_font(_SERIF_FONT_CANDIDATES, 32.5)
    seal_font = _load_font(_SERIF_FONT_CANDIDATES, 16)

    # Caption beneath plate
    gy = art_y + art_h + 22
    draw.text((art_x, gy), gloss.strip()[:92], font=gloss_font, fill=ink_secondary)
    draw.text((art_x, gy + 28), _credit_line(attribution, visual_medium), font=mono_font, fill=ink_secondary)

    # Vertical stacked uppercase word along spine center
    letters = [ch for ch in word.upper() if ch.strip()][:14]
    spine_cx = spine_x + (PLATE_WIDTH - spine_x) // 2
    draw.text((spine_cx - 34, 54), date_str, font=mono_font, fill=ink_secondary)
    draw.line([(spine_cx, 84), (spine_cx, 116)], fill=rule_color, width=1)

    vy = 132
    step_y = min(34, max(24, 420 // max(1, len(letters))))
    for ch in letters:
        c_box = draw.textbbox((0, 0), ch, font=vert_font)
        cw = c_box[2] - c_box[0]
        draw.text((spine_cx - cw // 2, vy), ch, font=vert_font, fill=ink_primary)
        vy += step_y

    # Vermilion Hanko seal at foot of spine
    seal_size = 44
    sx0 = spine_cx - seal_size // 2
    sy0 = PLATE_HEIGHT - 108
    draw.rectangle([sx0, sy0, sx0 + seal_size, sy0 + seal_size], outline=vermilion, width=2)
    draw.rectangle([sx0 + 4, sy0 + 4, sx0 + seal_size - 4, sy0 + seal_size - 4], fill=vermilion)
    draw.text((sx0 + 11, sy0 + 12), "DZ", font=seal_font, fill=(251, 249, 245))

    return canvas


def layout_concrete_title(word):
    """Keep the enlarged poster lettering inside its original structural grid."""
    text = word.upper()
    size = 97.5 if len(text)>12 else 130
    font = _load_font(_SERIF_FONT_CANDIDATES, size)
    lines = _pixel_wrap(text, font, 1080)
    runs = []
    y = 176
    for line in lines:
        bounds = font.getbbox(line)
        runs.append((line, (84-bounds[0],y-bounds[1])))
        y += bounds[3]-bounds[1]+12
    if y>410:
        raise ValueError('Concrete title exceeds its structural grid')
    return runs, max(336,y+12), size


def _render_concrete_poetry_plate(
    hero_img: Image.Image,
    *,
    word: str,
    gloss: str,
    thesis: str,
    date_str: str,
    velocity_score: float,
    dark_mode: bool,
) -> Image.Image:
    """Render Archetype #5 (`CONCRETE_POETRY`): Swiss Concrete Poetry / Constructivist poster for breakout surges."""
    bg = (14, 17, 22) if dark_mode else (247, 243, 235)
    ink_primary = (248, 249, 250) if dark_mode else (18, 18, 20)
    ink_secondary = (180, 176, 170) if dark_mode else (68, 64, 60)
    accent = (214, 58, 42)
    rule_color = (52, 58, 68) if dark_mode else (24, 24, 27)

    canvas = Image.new("RGB", (PLATE_WIDTH, PLATE_HEIGHT), color=bg)
    draw = ImageDraw.Draw(canvas)

    # Structural Swiss grid rules
    draw.rectangle([36, 36, PLATE_WIDTH - 36, PLATE_HEIGHT - 36], outline=rule_color, width=3)
    draw.line([(36, 118), (PLATE_WIDTH - 36, 118)], fill=rule_color, width=2)
    draw.line([(36, 510), (PLATE_WIDTH - 36, 510)], fill=rule_color, width=2)
    draw.line([(690, 510), (690, PLATE_HEIGHT - 36)], fill=rule_color, width=2)

    # Accent constructivist bar
    draw.rectangle([36, 118, 54, 510], fill=accent)

    mono_font = _load_font(_MONO_FONT_CANDIDATES, 14)
    display_font = _load_font(_SERIF_FONT_CANDIDATES, 130)
    gloss_font = _load_font(_SERIF_ITALIC_CANDIDATES, 30)
    body_font = _load_font(_SANS_FONT_CANDIDATES, 15)

    draw.text((64, 66), f"BREAKOUT SURGE EDITION  //  {date_str}", font=mono_font, fill=accent)
    draw.text((PLATE_WIDTH - 290, 66), f"VELOCITY +{velocity_score:.2f} SD", font=mono_font, fill=ink_primary)

    # Monumental word across center field, preserving full glyphs and the grid.
    title_runs, gy, title_size = layout_concrete_title(word)
    display_font = _load_font(_SERIF_FONT_CANDIDATES, title_size)
    for text, position in title_runs:
        draw.text(position, text, font=display_font, fill=ink_primary)

    for line in _pixel_wrap(gloss.strip(), gloss_font, 1016):
        draw.text((88, gy), line, font=gloss_font, fill=ink_secondary)
        gy += 34

    # Lower-left thesis block
    ty = 544
    for line in textwrap.wrap(thesis.strip(), width=56)[:6]:
        draw.text((68, ty), line, font=body_font, fill=ink_secondary)
        ty += 24

    # Lower-right 3:2 visual plate inset (330 x 220)
    inset_w, inset_h = 330, 220
    inset_x, inset_y = 780, 534
    fitted_inset = ImageOps.fit(hero_img, (inset_w, inset_h), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted_inset, (inset_x, inset_y))
    draw.rectangle([inset_x - 1, inset_y - 1, inset_x + inset_w, inset_y + inset_h], outline=accent, width=2)

    return canvas


def render_curatorial_plate_pair(
    hero_image_source: bytes | Image.Image,
    *,
    word: str,
    gloss: str,
    thesis: str = "",
    date_str: str = "2026-10-28",
    archetype: str = "POETIC_CINEMA",
    quadrant: str = "upper_left",
    luminance: float = 0.35,
    attribution: Optional[Mapping[str, Any]] = None,
    credit_line: Optional[str] = None,
    visual_medium: str = "PHOTOGRAPHY",
    velocity_score: float = 0.0,
    repo_root: Optional[Path] = None,
    light_out_path: Optional[Path] = None,
    dark_out_path: Optional[Path] = None,
) -> tuple[Image.Image, Image.Image]:
    """Render both light and dark 1200x800 WebP curatorial plates and optionally write them to disk."""
    hero_img = _coerce_image(hero_image_source)
    arch = str(archetype or "POETIC_CINEMA").strip().upper()
    if isinstance(attribution, str):
        attr_map: dict[str, Any] = {"photographer_name": attribution}
    else:
        attr_map = dict(attribution or {})
    if credit_line and not attr_map.get("photographer_name"):
        attr_map["photographer_name"] = credit_line
    effective_thesis = thesis or gloss

    rendered_pair: dict[str, Image.Image] = {}
    plates_bytes: dict[str, bytes] = {}
    for mode_name, is_dark in (("light", False), ("dark", True)):
        if arch == "POETIC_CINEMA":
            rendered = _render_poetic_cinema_plate(
                hero_img,
                word=word,
                gloss=gloss,
                quadrant=quadrant,
                luminance=luminance,
                dark_mode=is_dark,
            )
        elif arch == "LITERARY_JOURNAL":
            rendered = _render_literary_journal_plate(
                hero_img,
                word=word,
                gloss=gloss,
                thesis=effective_thesis,
                date_str=date_str,
                attribution=attr_map,
                visual_medium=visual_medium,
                dark_mode=is_dark,
            )
        elif arch == "STUDIO_MONOGRAPH":
            rendered = _render_studio_monograph_plate(
                hero_img,
                word=word,
                gloss=gloss,
                date_str=date_str,
                attribution=attr_map,
                visual_medium=visual_medium,
                dark_mode=is_dark,
            )
        elif arch == "KYOTO_TANKOBON":
            rendered = _render_kyoto_tankobon_plate(
                hero_img,
                word=word,
                gloss=gloss,
                date_str=date_str,
                attribution=attr_map,
                visual_medium=visual_medium,
                dark_mode=is_dark,
            )
        else:
            rendered = _render_concrete_poetry_plate(
                hero_img,
                word=word,
                gloss=gloss,
                thesis=effective_thesis,
                date_str=date_str,
                velocity_score=velocity_score,
                dark_mode=is_dark,
            )

        rendered_pair[mode_name] = rendered
        buf = BytesIO()
        rendered.save(buf, format="WEBP", quality=86, method=4)
        plates_bytes[mode_name] = buf.getvalue()

    if light_out_path is not None:
        Path(light_out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(light_out_path).write_bytes(plates_bytes["light"])
    if dark_out_path is not None:
        Path(dark_out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dark_out_path).write_bytes(plates_bytes["dark"])

    if repo_root is not None:
        assets_dir = Path(repo_root) / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "daily_pulse_light.webp").write_bytes(plates_bytes["light"])
        (assets_dir / "daily_pulse_dark.webp").write_bytes(plates_bytes["dark"])

    return rendered_pair["light"], rendered_pair["dark"]


def render_curatorial_plates(
    hero_image_source: bytes | Image.Image,
    *,
    word: str,
    gloss: str,
    thesis: str = "",
    date_str: str = "2026-10-28",
    archetype: str = "POETIC_CINEMA",
    quadrant: str = "upper_left",
    luminance: float = 0.35,
    attribution: Optional[Mapping[str, Any]] = None,
    credit_line: Optional[str] = None,
    visual_medium: str = "PHOTOGRAPHY",
    velocity_score: float = 0.0,
    repo_root: Optional[Path] = None,
    light_out_path: Optional[Path] = None,
    dark_out_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Render both light and dark 1200x800 WebP curatorial plates and return their output Paths."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    resolved_light = Path(light_out_path) if light_out_path is not None else (root / "assets" / "daily_pulse_light.webp")
    resolved_dark = Path(dark_out_path) if dark_out_path is not None else (root / "assets" / "daily_pulse_dark.webp")
    render_curatorial_plate_pair(
        hero_image_source,
        word=word,
        gloss=gloss,
        thesis=thesis,
        date_str=date_str,
        archetype=archetype,
        quadrant=quadrant,
        luminance=luminance,
        attribution=attribution,
        credit_line=credit_line,
        visual_medium=visual_medium,
        velocity_score=velocity_score,
        light_out_path=resolved_light,
        dark_out_path=resolved_dark,
    )
    return {"light": resolved_light, "dark": resolved_dark}

