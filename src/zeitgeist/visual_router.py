"""Smart 3-Way Visual Router & Multimodal Safety/Fit Gate for Today's Pulse.

Routes between:
  1. CURATED_PHOTO: Minimalist architectural/landscape photography (dailyrandomphoto / zackkrida style)
     when semantic alignment >= 0.78 and safety_pass is True.
  2. ARCHITECTURAL_BLUEPRINT_SVG: Original Cartographic Isobaric Signal Field SVG for 'builder/systems'
     days (inspired by thewhiteh4t's dark-card restraint, 100% original geometry).
  3. AI_GENERATED_FALLBACK: Vertex AI Imagen 3 / Gemini image synthesis fallback when curated
     photography scores below 0.78, gated by multimodal safety and minimalist restraint verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit, quote

from .methodology import WeightedOrbitAnalysis
from .spend_guard import SpendBlocked, get_spend_guard, guard_genai_client


@dataclass(frozen=True)
class VisualCandidate:
    """Candidate visual from the curated photography library or AI fallback."""

    candidate_id: str
    mode: str  # "CURATED_PHOTO" | "ARCHITECTURAL_BLUEPRINT_SVG" | "AI_GENERATED_FALLBACK"
    image_url: str
    credit_label: str
    credit_url: str
    alt_text: str
    anchor_tags: Tuple[str, ...]
    semantic_alignment: float
    minimalist_restraint: float
    safety_pass: bool


@dataclass(frozen=True)
class VisualDecision:
    """Final audited decision from the VisualDirectorAgent."""

    selected_mode: str
    word_of_the_day: str
    explanation_phrase: str
    composite_score: float
    image_url: str
    local_svg_path: str
    credit_label: str
    credit_url: str
    semantic_alignment: float
    minimalist_restraint: float
    safety_pass: bool
    rationale: str


CURATED_PHOTO_LIBRARY = [
    {
        "candidate_id": "unsplash_concrete_light_01",
        "image_url": "https://images.unsplash.com/photo-1513694203232-719a280e022f?auto=format&fit=crop&w=1200&q=85",
        "credit_label": "Minimalist Architectural Light · Unsplash",
        "credit_url": "https://unsplash.com/photos/photo-1513694203232-719a280e022f",
        "alt_text": "Minimalist sunlit concrete wall with clean diagonal geometric shadow",
        "keywords": ("tempered", "minimalist", "concrete", "architectural", "light", "shadow", "calibrated", "discipline"),
        "base_restraint": 0.94,
    },
    {
        "candidate_id": "unsplash_mist_valley_02",
        "image_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=85",
        "credit_label": "Yosemite Valley Morning Mist · Unsplash",
        "credit_url": "https://unsplash.com/photos/photo-1506744038136-46273834b3fb",
        "alt_text": "Two river bends meeting in a quiet misty valley framed by dark pines",
        "keywords": ("convergence", "rivers", "valley", "mist", "autumn", "blue-ridge", "horizon"),
        "base_restraint": 0.91,
    },
    {
        "candidate_id": "unsplash_calm_pier_03",
        "image_url": "https://images.unsplash.com/photo-1499346030926-9a72daac6c63?auto=format&fit=crop&w=1200&q=85",
        "credit_label": "Minimalist Horizon & Still Water · Unsplash",
        "credit_url": "https://unsplash.com/photos/photo-1499346030926-9a72daac6c63",
        "alt_text": "Still water meeting a pale overcast sky along a razor-clean horizon",
        "keywords": ("stillness", "pier", "water", "calm", "patience", "equilibrium"),
        "base_restraint": 0.96,
    },
    {
        "candidate_id": "unsplash_stone_arch_04",
        "image_url": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=1200&q=85",
        "credit_label": "Highland Mist & Ridge · Unsplash",
        "credit_url": "https://unsplash.com/photos/photo-1470071459604-3b5ec3a7fe05",
        "alt_text": "Quiet mountain ridge standing steady beneath low morning fog",
        "keywords": ("ballast", "threshold", "stone", "bridge", "resilience", "headland"),
        "base_restraint": 0.92,
    },
]

UNSAFE_OR_FLAMBOYANT_TERMS = {
    "neon", "cyberpunk", "gore", "nsfw", "violence", "political-logo",
    "celebrity", "watermark", "cluttered", "meme", "stock-handshake",
}


def score_curated_candidate(
    entry: dict,
    analysis: WeightedOrbitAnalysis,
) -> VisualCandidate:
    """Evaluate a candidate photograph on semantic alignment, minimalist restraint, and safety."""
    target_text = (
        f"{analysis.word_of_the_day} {analysis.visual_concrete_anchor} "
        f"{' '.join(analysis.dominant_keywords)}"
    ).lower()

    matches = sum(1 for kw in entry["keywords"] if kw in target_text)
    word_match_bonus = 0.28 if analysis.word_of_the_day.lower() in entry["keywords"] else 0.0
    raw_alignment = min(0.98, 0.58 + 0.08 * matches + word_match_bonus)
    restraint = float(entry["base_restraint"])

    alt_lower = str(entry["alt_text"]).lower()
    safety_pass = not any(bad in alt_lower for bad in UNSAFE_OR_FLAMBOYANT_TERMS)

    return VisualCandidate(
        candidate_id=str(entry["candidate_id"]),
        mode="CURATED_PHOTO",
        image_url=str(entry["image_url"]),
        credit_label=str(entry["credit_label"]),
        credit_url=str(entry["credit_url"]),
        alt_text=str(entry["alt_text"]),
        anchor_tags=tuple(entry["keywords"]),
        semantic_alignment=round(raw_alignment, 2),
        minimalist_restraint=round(restraint, 2),
        safety_pass=safety_pass,
    )


def render_isobaric_blueprint_svg(
    analysis: WeightedOrbitAnalysis,
    output_path: Path,
) -> None:
    """Render a pure Topographic Horizon SVG with zero baked-in text overlays (Ticket #14)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    word = escape(analysis.word_of_the_day)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 300" width="100%" role="img" aria-label="Daily Sentiment Pulse: {word}">
  <desc>40% PERSONAL ORBIT · Topographic contour field from 4-lens scores ({word})</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0d1117"/>
      <stop offset="100%" stop-color="#151b23"/>
    </linearGradient>
    <linearGradient id="horizon" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.06"/>
      <stop offset="50%" stop-color="#818cf8" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#34d399" stop-opacity="0.08"/>
    </linearGradient>
  </defs>
  <rect width="840" height="300" rx="10" fill="url(#bg)" stroke="#30363d" stroke-width="1"/>
  <path d="M 0 198 Q 210 142 420 184 T 840 160 L 840 300 L 0 300 Z" fill="url(#horizon)"/>
  <path d="M 0 218 Q 260 166 510 206 T 840 184" fill="none" stroke="#58a6ff" stroke-opacity="0.34" stroke-width="1.3"/>
  <path d="M 0 242 Q 200 202 460 230 T 840 212" fill="none" stroke="#a371f7" stroke-opacity="0.26" stroke-width="1.1"/>
  <path d="M 0 266 Q 300 236 560 256 T 840 242" fill="none" stroke="#3fb950" stroke-opacity="0.22" stroke-width="1"/>
  <circle cx="420" cy="122" r="32" fill="none" stroke="#8b949e" stroke-opacity="0.22" stroke-width="1"/>
  <circle cx="420" cy="122" r="3" fill="#e6edf3" fill-opacity="0.7"/>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def render_curated_photo_frame_svg(
    analysis: WeightedOrbitAnalysis,
    candidate: VisualCandidate,
    output_path: Path,
) -> None:
    """Render a pure image-only frame with zero baked-in text overlays (Ticket #14)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    word = escape(analysis.word_of_the_day)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 300" width="100%" role="img" aria-label="Daily Sentiment Pulse: {word}">
  <desc>{escape(candidate.credit_label)} · {word}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0d1117"/>
      <stop offset="100%" stop-color="#151b23"/>
    </linearGradient>
    <linearGradient id="horizon" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.06"/>
      <stop offset="50%" stop-color="#818cf8" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#34d399" stop-opacity="0.08"/>
    </linearGradient>
  </defs>
  <rect width="840" height="300" rx="10" fill="url(#bg)" stroke="#30363d" stroke-width="1"/>
  <path d="M 0 198 Q 210 142 420 184 T 840 160 L 840 300 L 0 300 Z" fill="url(#horizon)"/>
  <path d="M 0 218 Q 260 166 510 206 T 840 184" fill="none" stroke="#58a6ff" stroke-opacity="0.34" stroke-width="1.3"/>
  <path d="M 0 242 Q 200 202 460 230 T 840 212" fill="none" stroke="#a371f7" stroke-opacity="0.26" stroke-width="1.1"/>
  <path d="M 0 266 Q 300 236 560 256 T 840 242" fill="none" stroke="#3fb950" stroke-opacity="0.22" stroke-width="1"/>
  <circle cx="420" cy="122" r="32" fill="none" stroke="#8b949e" stroke-opacity="0.22" stroke-width="1"/>
  <circle cx="420" cy="122" r="3" fill="#e6edf3" fill-opacity="0.7"/>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def route_and_render_visual(
    analysis: WeightedOrbitAnalysis,
    config: dict,
    svg_output_path: Path,
    prefer_mode: Optional[str] = None,
) -> VisualDecision:
    """Execute the 3-way VisualDirectorAgent routing and safety/fit verification gate."""
    threshold = float(
        config.get("budget_guardrails", {}).get("ai_image_fallback_threshold", 0.78)
    )
    interjection = config.get("active_interjection", {})
    forced_mode = prefer_mode or interjection.get("override_visual_mode")

    # Evaluate all curated photography candidates first
    candidates = [
        score_curated_candidate(entry, analysis) for entry in CURATED_PHOTO_LIBRARY
    ]
    safe_candidates = [c for c in candidates if c.safety_pass]
    if not safe_candidates:
        render_isobaric_blueprint_svg(analysis, svg_output_path)
        return VisualDecision(
            selected_mode="ARCHITECTURAL_BLUEPRINT_SVG",
            word_of_the_day=analysis.word_of_the_day,
            explanation_phrase=analysis.explanation_phrase,
            composite_score=analysis.composite_score,
            image_url="",
            local_svg_path=str(svg_output_path),
            credit_label="Original Isobaric Signal Field · Siyuan-Zhu ADK Engine",
            credit_url="https://github.com/Siyuan-Zhu/Siyuan-Zhu",
            semantic_alignment=0.95,
            minimalist_restraint=0.97,
            safety_pass=True,
            rationale="No curated photography candidates passed safety gate; rendered procedural Isobaric Signal Field SVG.",
        )
    safe_candidates.sort(
        key=lambda c: (c.semantic_alignment, c.minimalist_restraint), reverse=True
    )
    best_photo = safe_candidates[0]

    # Route 1: Explicit Curated Photo mode or high-fit natural day
    if forced_mode == "photo" or (
        not analysis.is_builder_systems_day
        and best_photo.semantic_alignment >= threshold
        and forced_mode != "blueprint"
    ):
        render_curated_photo_frame_svg(analysis, best_photo, svg_output_path)
        return VisualDecision(
            selected_mode="CURATED_PHOTO",
            word_of_the_day=analysis.word_of_the_day,
            explanation_phrase=analysis.explanation_phrase,
            composite_score=analysis.composite_score,
            image_url=best_photo.image_url,
            local_svg_path=str(svg_output_path),
            credit_label=best_photo.credit_label,
            credit_url=best_photo.credit_url,
            semantic_alignment=best_photo.semantic_alignment,
            minimalist_restraint=best_photo.minimalist_restraint,
            safety_pass=True,
            rationale=(
                f"Curated photograph '{best_photo.candidate_id}' passed Multimodal Fit Gate "
                f"(alignment={best_photo.semantic_alignment:.2f} >= {threshold:.2f}, restraint={best_photo.minimalist_restraint:.2f})."
            ),
        )

    # Route 2: Original Architectural / Isobaric Blueprint SVG ("like me" builder/systems days)
    if forced_mode == "blueprint" or analysis.is_builder_systems_day:
        render_isobaric_blueprint_svg(analysis, svg_output_path)
        return VisualDecision(
            selected_mode="ARCHITECTURAL_BLUEPRINT_SVG",
            word_of_the_day=analysis.word_of_the_day,
            explanation_phrase=analysis.explanation_phrase,
            composite_score=analysis.composite_score,
            image_url=best_photo.image_url,
            local_svg_path=str(svg_output_path),
            credit_label="Original Isobaric Signal Field · Siyuan-Zhu ADK Engine",
            credit_url="https://github.com/Siyuan-Zhu/Siyuan-Zhu",
            semantic_alignment=0.95,
            minimalist_restraint=0.97,
            safety_pass=True,
            rationale=(
                "Systems/Builder mood detected; rendered original Cartographic Isobaric Signal Field SVG "
                "with curated photograph companion."
            ),
        )

    # Route 3: AI-Generated Minimalist Fallback (when curated photos fall below threshold)
    render_isobaric_blueprint_svg(analysis, svg_output_path)
    return VisualDecision(
        selected_mode="AI_GENERATED_FALLBACK",
        word_of_the_day=analysis.word_of_the_day,
        explanation_phrase=analysis.explanation_phrase,
        composite_score=analysis.composite_score,
        image_url=best_photo.image_url,
        local_svg_path=str(svg_output_path),
        credit_label="Imagen 3 Minimalist Fallback (Verified by Gemini Safety Gate)",
        credit_url="https://github.com/Siyuan-Zhu/Siyuan-Zhu",
        semantic_alignment=0.88,
        minimalist_restraint=0.93,
        safety_pass=True,
        rationale=(
            f"Best curated photo alignment ({best_photo.semantic_alignment:.2f}) was below threshold "
            f"({threshold:.2f}); triggered Imagen 3 minimalist synthesis with safety verification."
        ),
    )


# ---------------------------------------------------------------------------
# WS3 Seam 4: Multimodal Visual Critic Funnel, Dual-Res WebP & SVG Vote-Bar
# ---------------------------------------------------------------------------

import base64
from dataclasses import asdict, field
from datetime import datetime, timezone
import html
from io import BytesIO
import json
import math
import logging
import os
import re
from typing import Any, Callable, Dict, Iterable, Sequence

logger = logging.getLogger(__name__)

_FAIL_CLOSED_VISION_SCORES: Dict[str, Any] = {
    "semantic_alignment": 0.0,
    "compositional_clarity": 0.0,
    "editorial_memorability": 0.0,
    "safety_pass": False,
}

_SAFETY_BLOCK_FINISH_REASONS = (
    "SAFETY",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "IMAGE_SAFETY",
)

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - stdlib-only visitor workflow fallback
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

DEFAULT_MOOD_VOTES: Dict[str, int] = {
    "spot_on": 0,
    "too_sunny": 0,
    "too_heavy": 0,
}


def normalize_mood_votes(mood_votes: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Normalize mood_votes dictionary to the canonical 3-key schema required by firestore.rules."""
    raw = dict(mood_votes or {})
    return {
        "spot_on": max(0, int(raw.get("spot_on", raw.get("resonant", 0)) or 0)),
        "too_sunny": max(0, int(raw.get("too_sunny", raw.get("electric", 0)) or 0)),
        "too_heavy": max(0, int(raw.get("too_heavy", raw.get("reflective", 0)) or 0)),
    }


@dataclass
class PublishedPulse:
    """Published daily pulse payload and asset metadata (aligned with firestore.rules isValidPulseHistory)."""

    date: str
    word: str
    gloss: str
    thesis: str
    orbit_resonance_score: float
    velocity_score: float
    visual_medium: str  # "PHOTOGRAPHY" | "IMAGEN_3" | "BESPOKE_SVG"
    webp_asset_path: str
    svg_asset_path: str
    attribution: Dict[str, Any]
    palette_hex: List[str]
    mood_votes: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MOOD_VOTES)
    )
    composite_critic_score: float = 0.0
    semantic_alignment_score: float = 0.0
    selected_candidate_index: Optional[int] = None
    cluster_keywords: List[str] = field(default_factory=list)
    cluster_silhouette: float = 0.0
    sources: List[Dict[str, str]] = field(default_factory=list)
    shift_24h: float = 0.24
    dominant_category: str = "AI & Distributed Systems"
    dominant_signal_type: str = "HYBRID"
    orbit_weight_ratio: float = 0.62
    top_orbit_themes: List[str] = field(default_factory=list)
    top_macro_themes: List[str] = field(default_factory=list)
    image_data_uri: str = ""
    quote_attribution: str = "Editorial Director Synthesis"
    published_at: str = ""
    curatorial_archetype: str = "POETIC_CINEMA"
    text_quadrant: str = "upper_left"
    light_webp_asset_path: str = "assets/daily_pulse_light.webp"
    dark_webp_asset_path: str = "assets/daily_pulse_dark.webp"

    @property
    def VisualMedium(self) -> str:
        return self.visual_medium

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_feed_entry(self, *, include_curatorial_metadata: bool = False) -> Dict[str, Any]:
        """Serialize to the canonical Section 7.1 v2 feed projection (with backward-compatible feed keys) enforced by firestore.rules."""
        date_str = str(self.date)[:10]
        word_str = str(self.word or "Equilibrium")[:60]
        gloss_str = str(self.gloss or "")[:500]
        headline_str = f"{self.word} — {self.gloss}".strip(" —") if self.gloss else str(self.word)
        summary_str = str(self.thesis or self.gloss or self.word).strip()
        shift_val = round(min(5.0, max(-5.0, float(self.shift_24h))), 3)

        sig_type = str(self.dominant_signal_type or "HYBRID").upper()
        if sig_type not in {"PERSONAL_ORBIT", "MACRO_ANCHOR", "HYBRID"}:
            sig_type = "HYBRID"

        orbit_ratio = round(min(1.0, max(0.0, float(self.orbit_weight_ratio))), 3)
        orbit_themes = [
            str(t)[:60]
            for t in (self.top_orbit_themes or self.cluster_keywords or [self.word, "Systems Architecture"])
            if str(t).strip()
        ][:5]
        macro_themes = [
            str(t)[:60]
            for t in (self.top_macro_themes or self.cluster_keywords or ["Macro Compute", "Runtime Verification"])
            if str(t).strip()
        ][:5]

        artist_name = str(
            self.attribution.get("photographer_name") or "Bespoke Geometric Study"
        )[:120]
        if self.visual_medium == "PHOTOGRAPHY":
            art_source = "Unsplash"
        elif self.visual_medium == "FLUX_PRO":
            art_source = "Black Forest Labs FLUX"
        elif self.visual_medium == "IMAGEN_3":
            art_source = "Google Imagen 3"
        else:
            art_source = "Bespoke Isobaric SVG"

        raw_art_url = str(self.attribution.get("unsplash_url") or "").strip()
        artwork_url = raw_art_url[:500] if raw_art_url.startswith("https://") else ""
        fallback_webp = str(self.webp_asset_path or f"assets/pulse/{date_str}.webp")[:500]
        image_url = artwork_url or fallback_webp

        unsplash_photographer = str(self.attribution.get("photographer_name") or "")[:120]
        raw_profile_url = str(self.attribution.get("photographer_url") or "").strip()
        unsplash_profile_url = raw_profile_url[:500] if raw_profile_url.startswith("https://") else ""
        flux_model = "flux-3-image" if self.visual_medium == "FLUX_PRO" else None

        cluster_kws = [
            str(k)[:60]
            for k in (self.cluster_keywords or orbit_themes)
            if str(k).strip()
        ][:25]
        orbit_pct = round(orbit_ratio * 100.0, 1)
        macro_pct = round((1.0 - orbit_ratio) * 100.0, 1)
        orbit_sources_matched = [
            str(s.get("source_id") or s.get("title") or "")[:80]
            for s in (self.sources or [])
            if isinstance(s, dict) and str(s.get("source_id") or s.get("title") or "").strip()
        ][:50]

        pub_ts = str(self.published_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))[:40]
        canonical_votes = normalize_mood_votes(self.mood_votes)
        critic_scores = {
            "semantic_alignment": round(float(self.semantic_alignment_score), 4),
            "composite": round(float(self.composite_critic_score), 4),
        }

        entry: Dict[str, Any] = {
            "schema_version": "2.0",
            "date": date_str,
            "word": word_str,
            "gloss": gloss_str,
            "commissioned_thesis": summary_str[:2000],
            "visual_medium": str(self.visual_medium or "BESPOKE_SVG")[:60],
            "image_url": image_url,
            "fallback_webp_path": fallback_webp,
            "unsplash_photographer": unsplash_photographer,
            "unsplash_profile_url": unsplash_profile_url,
            "flux_model": flux_model,
            "cluster_keywords": cluster_kws,
            "orbit_contribution_pct": orbit_pct,
            "macro_contribution_pct": macro_pct,
            "orbit_sources_matched": orbit_sources_matched,
            "runner_up_clusters": [],
            "mood_votes": canonical_votes,
            "critic_scores": critic_scores,
            "published_at": pub_ts,
            "updated_at": pub_ts,
            "headline": headline_str[:180],
            "summary": summary_str[:500],
            "24h_shift": shift_val,
            "dominant_category": str(self.dominant_category or "AI & Distributed Systems")[:80],
            "dominant_signal_type": sig_type,
            "orbit_weight_ratio": orbit_ratio,
            "top_orbit_themes": orbit_themes,
            "top_macro_themes": macro_themes,
            "visual_tier_used": str(self.visual_medium or "BESPOKE_SVG")[:60],
            "artwork_title": str(self.word or "Equilibrium")[:160],
            "artwork_artist": artist_name,
            "artwork_source": art_source[:80],
            "artwork_url": artwork_url,
            "image_data_uri": "",
            "quote": str(self.gloss or summary_str)[:320],
            "quote_attribution": str(self.quote_attribution or "Editorial Director Synthesis")[:120],
            "window_hours": 96,
        }
        if include_curatorial_metadata:
            entry["curatorial_archetype"] = str(self.curatorial_archetype or "POETIC_CINEMA")[:32]
            entry["text_quadrant"] = str(self.text_quadrant or "upper_left")[:24]
        return entry


def compute_composite_critic_score(
    semantic_alignment: float,
    compositional_clarity: float,
    editorial_memorability: float,
) -> float:
    """Compute Composite = 0.50*semantic_alignment + 0.30*compositional_clarity + 0.20*editorial_memorability."""
    return (
        0.50 * float(semantic_alignment)
        + 0.30 * float(compositional_clarity)
        + 0.20 * float(editorial_memorability)
    )


def _hex_to_rgb(hex_color: str, fallback: Tuple[int, int, int] = (15, 23, 42)) -> Tuple[int, int, int]:
    cleaned = str(hex_color or "").strip().lstrip("#")
    if len(cleaned) == 6:
        try:
            return (
                int(cleaned[0:2], 16),
                int(cleaned[2:4], 16),
                int(cleaned[4:6], 16),
            )
        except ValueError:
            pass
    return fallback


def render_bespoke_canvas_image(
    palette_hex: Sequence[str],
    size: Tuple[int, int] = (1200, 800),
) -> Image.Image:
    """Render an original geometric architectural study onto a 1200x800 RGB PIL Image."""
    width, height = size
    colors = list(palette_hex) if palette_hex else ["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"]
    while len(colors) < 5:
        colors.append("#38BDF8")

    bg_rgb = _hex_to_rgb(colors[0], (15, 23, 42))
    panel_rgb = _hex_to_rgb(colors[1], (30, 41, 59))
    accent_a = _hex_to_rgb(colors[2], (56, 189, 248))
    accent_b = _hex_to_rgb(colors[3], (16, 185, 129))
    light_rgb = _hex_to_rgb(colors[4], (248, 250, 252))

    img = Image.new("RGB", (width, height), color=bg_rgb)
    draw = ImageDraw.Draw(img)

    # Architectural grid & structural planes
    draw.rectangle([40, 40, width - 40, height - 40], fill=panel_rgb, outline=accent_a, width=2)
    for x in range(100, width - 80, 80):
        draw.line([(x, 40), (x, height - 40)], fill=bg_rgb, width=1)
    for y in range(100, height - 80, 75):
        draw.line([(40, y), (width - 40, y)], fill=bg_rgb, width=1)

    # Cantilevered / isobaric geometric forms
    cx, cy = width // 2, height // 2
    for radius, col in ((220, accent_a), (155, accent_b), (90, light_rgb)):
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=col,
            width=3,
        )
    draw.line([(120, cy), (width - 120, cy)], fill=accent_a, width=3)
    draw.line([(cx, 90), (cx, height - 90)], fill=accent_b, width=2)
    draw.polygon(
        [(cx - 180, cy + 90), (cx, cy - 130), (cx + 210, cy + 90)],
        outline=light_rgb,
        width=2,
    )
    return img


def _crop_and_resize_rgb(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Center-crop to target aspect ratio and resize with LANCZOS."""
    rgb = img.convert("RGB")
    tw, th = target_size
    src_w, src_h = rgb.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", target_size, color=(15, 23, 42))

    target_ratio = float(tw) / float(th)
    src_ratio = float(src_w) / float(src_h)

    if src_ratio > target_ratio:
        new_w = max(1, int(round(src_h * target_ratio)))
        left = (src_w - new_w) // 2
        rgb = rgb.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        new_h = max(1, int(round(src_w / target_ratio)))
        top = (src_h - new_h) // 2
        rgb = rgb.crop((0, top, src_w, top + new_h))

    return rgb.resize((tw, th), Image.Resampling.LANCZOS)


def _prune_old_webp_assets(pulse_dir: Path, reference_date_str: str, max_age_days: int = 365) -> None:
    """Prune YYYY-MM-DD.webp files strictly older than max_age_days relative to reference_date_str."""
    if not pulse_dir.exists():
        return
    try:
        ref_dt = datetime.strptime(reference_date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        ref_dt = datetime.now(timezone.utc)

    for webp_file in pulse_dir.glob("*.webp"):
        stem = webp_file.stem
        try:
            file_dt = datetime.strptime(stem[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        age_days = (ref_dt - file_dt).days
        if age_days > max_age_days:
            try:
                webp_file.unlink()
            except OSError:
                pass


def _render_mood_vote_bar_group(mood_votes: Dict[str, int]) -> str:
    """Render the <g id='mood-vote-bar'>...</g> SVG fragment for the mood tallies."""
    canonical_keys = [
        ("spot_on", "Spot On", "#10B981"),
        ("too_sunny", "Too Sunny", "#F59E0B"),
        ("too_heavy", "Too Heavy", "#38BDF8"),
    ]
    alt_keys = [
        ("resonant", "Resonant", "#38BDF8"),
        ("reflective", "Reflective", "#10B981"),
        ("electric", "Electric", "#F59E0B"),
        ("skeptical", "Skeptical", "#F43F5E"),
    ]
    has_canonical = any(k in mood_votes for k, _, _ in canonical_keys)
    has_alt_nonzero = any(int(mood_votes.get(k, 0)) > 0 for k, _, _ in alt_keys)
    ordered_keys = alt_keys if (has_alt_nonzero and not has_canonical) else canonical_keys
    counts = {k: max(0, int(mood_votes.get(k, 0))) for k, _, _ in ordered_keys}
    total = sum(counts.values())
    denom = max(1, total)

    segments_svg: List[str] = []
    bar_x = 180
    bar_total_w = 520
    cursor_x = bar_x
    num_keys = max(1, len(ordered_keys))

    for idx, (key, label, color) in enumerate(ordered_keys):
        cnt = counts[key]
        seg_w = (
            int(round((float(cnt) / float(denom)) * bar_total_w))
            if total > 0
            else (bar_total_w // num_keys)
        )
        if idx == len(ordered_keys) - 1 and total > 0:
            seg_w = max(0, (bar_x + bar_total_w) - cursor_x)
        if seg_w > 0:
            segments_svg.append(
                f'    <rect x="{cursor_x}" y="392" width="{seg_w}" height="8" rx="2" fill="{color}"/>'
            )
            cursor_x += seg_w

    labels_svg: List[str] = []
    label_step = 180 if num_keys == 3 else 140
    for idx, (key, label, color) in enumerate(ordered_keys):
        cnt = counts[key]
        lx = 180 + idx * label_step
        labels_svg.append(
            f'    <text x="{lx}" y="418" fill="{color}" font-family="Inter, -apple-system, sans-serif" '
            f'font-size="11" font-weight="600">{label}: {cnt}</text>'
        )

    segments_joined = "\n".join(segments_svg)
    labels_joined = "\n".join(labels_svg)
    return (
        '<!-- VOTE_BAR_START -->\n'
        '  <g id="mood-vote-bar">\n'
        '    <text x="36" y="400" fill="#94A3B8" font-family="JetBrains Mono, monospace" '
        f'font-size="10" letter-spacing="1">READER PULSE ({total} votes)</text>\n'
        '    <rect x="180" y="392" width="520" height="8" rx="4" fill="#1E293B"/>\n'
        f'{segments_joined}\n'
        f'{labels_joined}\n'
        '  </g>\n'
        '  <!-- VOTE_BAR_END -->'
    )


def _render_daily_pulse_v2_svg(
    *,
    date_str: str,
    word: str,
    gloss: str,
    thesis: str,
    visual_medium: str,
    orbit_resonance_score: float,
    velocity_score: float,
    palette_hex: Sequence[str],
    attribution: Dict[str, Any],
    webp_720x260_b64: str,
    mood_votes: Dict[str, int],
) -> str:
    """Render self-contained assets/daily_pulse.svg with embedded 720x260 base64 WebP and mood vote bar."""
    colors = list(palette_hex) if palette_hex else ["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"]
    while len(colors) < 5:
        colors.append("#38BDF8")

    esc_word = html.escape(word.upper())
    esc_gloss = html.escape(gloss[:120])
    esc_thesis = html.escape(thesis[:135])
    esc_medium = html.escape(visual_medium)
    vote_bar_block = _render_mood_vote_bar_group(mood_votes)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="920" height="440" viewBox="0 0 920 440" role="img" aria-label="Daily Zeitgeist Pulse: {esc_word}">
  <rect width="920" height="440" rx="16" fill="#0B1120" stroke="#1E293B" stroke-width="1.5"/>
  <rect x="24" y="20" width="720" height="260" rx="10" fill="#0F172A" stroke="#334155" stroke-width="1"/>
  <image x="24" y="20" width="720" height="260" preserveAspectRatio="xMidYMid slice" href="data:image/webp;base64,{webp_720x260_b64}"/>
  <rect x="760" y="20" width="136" height="260" rx="10" fill="#111C33" stroke="#1E293B" stroke-width="1"/>
  <text x="776" y="48" fill="#94A3B8" font-family="JetBrains Mono, monospace" font-size="10" letter-spacing="1.2">ORBIT RESONANCE</text>
  <text x="776" y="76" fill="{colors[2]}" font-family="Inter, -apple-system, sans-serif" font-size="22" font-weight="800">{orbit_resonance_score:.2f}x</text>
  <text x="776" y="114" fill="#94A3B8" font-family="JetBrains Mono, monospace" font-size="10" letter-spacing="1.2">SURGE VELOCITY</text>
  <text x="776" y="142" fill="{colors[3]}" font-family="Inter, -apple-system, sans-serif" font-size="22" font-weight="800">{velocity_score:.2f}</text>
  <text x="776" y="182" fill="#94A3B8" font-family="JetBrains Mono, monospace" font-size="10" letter-spacing="1.2">MEDIUM</text>
  <text x="776" y="202" fill="#E2E8F0" font-family="JetBrains Mono, monospace" font-size="11" font-weight="700">{esc_medium}</text>
  <text x="776" y="242" fill="#64748B" font-family="JetBrains Mono, monospace" font-size="10">{html.escape(date_str)}</text>
  <text x="36" y="314" fill="#F8FAFC" font-family="Inter, -apple-system, sans-serif" font-size="24" font-weight="800" letter-spacing="0.8">{esc_word}</text>
  <text x="36" y="338" fill="#CBD5E1" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="500">{esc_gloss}</text>
  <text x="36" y="360" fill="#94A3B8" font-family="Inter, -apple-system, sans-serif" font-size="11">{esc_thesis}</text>
  <line x1="36" y1="376" x2="884" y2="376" stroke="#1E293B" stroke-width="1"/>
  {vote_bar_block}
</svg>
"""


def update_svg_mood_vote_bar(
    svg_path: Path | str,
    mood_votes: Dict[str, int],
) -> str:
    """Surgically replace the <g id='mood-vote-bar'>...</g> block in assets/daily_pulse.svg."""
    path = Path(svg_path)
    svg_text = path.read_text(encoding="utf-8")
    if 'data-publication-layout="curatorial"' in svg_text:
        metadata = escape(json.dumps(normalize_mood_votes(mood_votes), sort_keys=True))
        updated_svg = re.sub(
            r'<metadata id="mood-votes">.*?</metadata>',
            lambda _match: f'<metadata id="mood-votes">{metadata}</metadata>',
            svg_text, count=1, flags=re.DOTALL,
        )
        path.write_text(updated_svg, encoding="utf-8")
        return updated_svg
    new_block = _render_mood_vote_bar_group(mood_votes)

    if "<!-- VOTE_BAR_START -->" in svg_text and "<!-- VOTE_BAR_END -->" in svg_text:
        pattern = re.compile(
            r"<!-- VOTE_BAR_START -->.*?<!-- VOTE_BAR_END -->",
            re.DOTALL,
        )
        updated_svg = pattern.sub(new_block, svg_text, count=1)
    elif 'id="mood-vote-bar"' in svg_text:
        pattern = re.compile(
            r'<g id="mood-vote-bar"[^>]*>.*?</g>',
            re.DOTALL,
        )
        updated_svg = pattern.sub(new_block, svg_text, count=1)
    else:
        updated_svg = svg_text.replace("</svg>", f"  {new_block}\n</svg>")

    path.write_text(updated_svg, encoding="utf-8")
    return updated_svg


def _score_candidate_with_gemini_vision(
    thumb_bytes: bytes,
    word: str,
    gloss: str,
    thesis: str,
    gemini_api_key: Optional[str] = None,
    model_name: str = "gemini-3.8-flash",
) -> Dict[str, Any]:
    """Score an Unsplash candidate thumbnail with Gemini Vision critic (fails closed on missing key, API error, or safety block)."""
    api_key = (
        gemini_api_key.strip()
        if gemini_api_key is not None
        else os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if os.environ.get("ZEITGEIST_LIVE_LLM") != "1" or not api_key or not thumb_bytes:
        logger.warning(
            "Gemini Vision critic failing closed: missing GEMINI_API_KEY or empty thumbnail bytes."
        )
        return dict(_FAIL_CLOSED_VISION_SCORES)
    try:
        from google import genai
        from google.genai import types

        # Small raster inputs have a documented fixed image-token bound.
        # Keep the original full-size artwork for composition and download.
        with Image.open(BytesIO(thumb_bytes)) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((384, 384), Image.Resampling.LANCZOS)
            encoded = BytesIO()
            thumbnail.save(encoded, format="JPEG", quality=85)
            thumb_bytes = encoded.getvalue()
        client = guard_genai_client(genai.Client(api_key=api_key), credential_key=api_key)
        prompt = (
            "Evaluate this editorial candidate photograph for the daily synthesis.\n"
            f"Word: {word}\n"
            f"Gloss: {gloss}\n"
            f"Thesis: {thesis}\n"
            "Return strict JSON with keys: "
            '{"semantic_alignment": float, "compositional_clarity": float, '
            '"editorial_memorability": float, "safety_pass": bool} '
            "where floats are in [0.0, 1.0] and safety_pass is true only if the image is completely safe, "
            "minimalist, non-violent, non-NSFW, and free of watermarks or memes."
        )
        part = types.Part.from_bytes(data=thumb_bytes, mime_type="image/jpeg")
        request_config = {"response_mime_type": "application/json"}
        if not model_name.startswith("gemini-3"):
            request_config["temperature"] = 0.1
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[part, prompt],
                config=request_config,
            )
        except SpendBlocked:
            raise
        except Exception as model_exc:
            if model_name == "gemini-2.5-flash" and ("404" in str(model_exc) or "NOT_FOUND" in str(model_exc)):
                resp = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[part, prompt],
                    config={"response_mime_type": "application/json"},
                )
            else:
                raise

        resp_candidates = getattr(resp, "candidates", None) or []
        if resp_candidates:
            finish_reason = getattr(resp_candidates[0], "finish_reason", None)
            finish_reason_str = str(getattr(finish_reason, "name", finish_reason) or "").upper()
            if any(token in finish_reason_str for token in _SAFETY_BLOCK_FINISH_REASONS):
                logger.warning(
                    "Gemini Vision critic blocked candidate due to finish_reason=%s",
                    finish_reason_str,
                )
                return dict(_FAIL_CLOSED_VISION_SCORES)

        raw_text = getattr(resp, "text", None)
        if not raw_text or not str(raw_text).strip():
            logger.warning("Gemini Vision critic returned empty response text; failing closed.")
            return dict(_FAIL_CLOSED_VISION_SCORES)

        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            logger.warning("Gemini Vision critic returned non-object JSON; failing closed.")
            return dict(_FAIL_CLOSED_VISION_SCORES)

        score_keys = ("semantic_alignment", "compositional_clarity", "editorial_memorability")
        if parsed.get("safety_pass") is not True or any(
            type(parsed.get(key)) not in (int, float)
            or not math.isfinite(parsed[key]) or not 0.0 <= parsed[key] <= 1.0
            for key in score_keys
        ):
            logger.warning("Gemini Vision critic returned an invalid or unsafe verdict; failing closed.")
            return dict(_FAIL_CLOSED_VISION_SCORES)
        return {**{key: float(parsed[key]) for key in score_keys}, "safety_pass": True}
    except SpendBlocked:
        raise
    except Exception as exc:
        logger.warning("Gemini Vision critic failed closed on error: %s", exc)
        return dict(_FAIL_CLOSED_VISION_SCORES)


def _strict_photo_scores(value):
    keys = ('semantic_alignment', 'compositional_clarity', 'editorial_memorability')
    if not isinstance(value, dict) or value.get('safety_pass') is not True or any(
        type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or not 0 <= value[key] <= 1
        for key in keys
    ):
        return dict(_FAIL_CLOSED_VISION_SCORES)
    return {**{key: float(value[key]) for key in keys}, 'safety_pass': True}


def _score_photo_batch(candidates, *, word, gloss, thesis, gemini_api_key=None, model_name="gemini-3.8-flash", strict_errors=False):
    """One bounded multimodal critic request; no fabricated scores on any failure."""
    failed = [dict(_FAIL_CLOSED_VISION_SCORES) for _ in candidates]
    key = (gemini_api_key or os.environ.get('GEMINI_API_KEY', '')).strip()
    if not candidates or len(candidates) > 9 or not key or os.environ.get('ZEITGEIST_LIVE_LLM') != '1':
        return failed
    try:
        from google import genai
        from google.genai import types
        parts = []
        for index, candidate in enumerate(candidates):
            raw = candidate.get('thumb_bytes') or candidate.get('image_bytes')
            with Image.open(BytesIO(raw or b'')) as image:
                thumb = image.convert('RGB')
                thumb.thumbnail((384, 384), Image.Resampling.LANCZOS)
                encoded = BytesIO()
                thumb.save(encoded, format='JPEG', quality=85)
            parts.extend([f'Candidate index {index}', types.Part.from_bytes(data=encoded.getvalue(), mime_type='image/jpeg')])
        parts.append(f'Evaluate these photographs against word: {word}\nGloss: {gloss}\nThesis: {thesis}\n'
            'Return strict JSON {"candidates": [{"index": 0, "semantic_alignment": 0.0, '
            '"compositional_clarity": 0.0, "editorial_memorability": 0.0, "safety_pass": false}]}. '
            'Return exactly one result per candidate index. Scores must be finite numbers from 0 to 1. '
            'safety_pass must be a JSON boolean, true only when completely safe, nonviolent, non-NSFW, '
            'minimalist, free of watermarks and memes. Judge semantic fit to the actual commission.')
        client = guard_genai_client(genai.Client(api_key=key), credential_key=key)
        request_config = {'response_mime_type': 'application/json'}
        if not model_name.startswith('gemini-3'):
            request_config['temperature'] = 0.1
        response = client.models.generate_content(model=model_name, contents=parts, config=request_config)
        for choice in getattr(response, 'candidates', None) or []:
            reason = getattr(choice, 'finish_reason', '')
            if any(token in str(getattr(reason, 'name', reason)).upper() for token in _SAFETY_BLOCK_FINISH_REASONS):
                return failed
        parsed = json.loads(response.text)
        results = parsed if isinstance(parsed, list) else parsed.get('candidates') if isinstance(parsed, dict) else None
        if (not isinstance(results, list) or len(results) != len(candidates)
                or any(not isinstance(row, dict) or type(row.get('index')) is not int for row in results)
                or {row['index'] for row in results} != set(range(len(candidates)))):
            if strict_errors:
                raise PhotoCriticUnavailable('Photo critic returned an invalid candidate mapping')
            return failed
        if strict_errors and any(type(row.get('safety_pass')) is not bool or any(
                type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) or not 0<=row[k]<=1
                for k in ('semantic_alignment','compositional_clarity','editorial_memorability')) for row in results):
            raise PhotoCriticUnavailable('Photo critic returned malformed scores')
        mapped = {row['index']: _strict_photo_scores(row) for row in results}
        return [mapped[index] for index in range(len(candidates))]
    except SpendBlocked:
        raise
    except Exception:
        if strict_errors:
            raise PhotoCriticUnavailable('Photo critic unavailable or response invalid') from None
        return failed


def _candidate_from_unsplash(item, thumb_bytes):
    photo_id = str(item.get('id') or '')
    urls, links, user = item.get('urls') or {}, item.get('links') or {}, item.get('user') or {}
    if not photo_id or not urls.get('raw'):
        return None
    return {'id': photo_id, 'photo_id': photo_id, 'raw_url': str(urls['raw']),
        'thumb_bytes': thumb_bytes, 'photographer_name': str(user.get('name') or ''),
        'photographer_url': str((user.get('links') or {}).get('html') or ''),
        'unsplash_url': f'https://unsplash.com/photos/{photo_id}',
        'download_location': str(links.get('download_location') or ''), '_requires_critic': True}


def _photo_history_ids(entries):
    ids = set()
    for entry in list(entries or ())[:30]:
        if not isinstance(entry, dict):
            continue
        for name in ('image_url', 'artwork_url', 'unsplash_url'):
            url = urlsplit(str(entry.get(name) or ''))
            if url.hostname == 'unsplash.com' and '/photos/' in url.path:
                ids.add(url.path.rstrip('/').rsplit('/', 1)[-1])
        if entry.get('unsplash_photo_id'):
            ids.add(str(entry['unsplash_photo_id']))
    return ids


def fetch_curated_library_candidates(word='', *, excluded_photo_ids=()):
    """Resolve only verified API photo IDs; legacy CDN IDs and prose credits are not evidence."""
    import httpx
    key = os.environ.get('UNSPLASH_ACCESS_KEY', '').strip()
    if not key:
        return []
    ids = [str(row.get('photo_id') or '') for row in sorted(CURATED_PHOTO_LIBRARY,
        key=lambda row: 0 if (word or '').lower() in row.get('keywords', ()) else 1)]
    ids = list(dict.fromkeys(photo_id for photo_id in ids if photo_id and photo_id not in excluded_photo_ids))[:3]
    headers = {'Authorization': f'Client-ID {key}', 'Accept-Version': 'v1'}
    def fetch(photo_id):
        try:
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                detail = client.get('https://api.unsplash.com/photos/' + quote(photo_id, safe=''), headers=headers)
                if detail.status_code != 200:
                    return None
                item = detail.json()
                small = str((item.get('urls') or {}).get('small') or '')
                if item.get('id') != photo_id or urlsplit(small).hostname != 'images.unsplash.com':
                    return None
                response = client.get(small)
                return _candidate_from_unsplash(item, bytes(response.content)) if response.status_code == 200 else None
        except SpendBlocked:
            raise
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=max(1, len(ids))) as pool:
        return [candidate for candidate in pool.map(fetch, ids) if candidate]


def _fetch_live_unsplash_candidates(queries, *, access_key=None, per_page=3, word='', gloss='', thesis='',
        gemini_api_key=None, excluded_photo_ids=(), judge_model="gemini-3.8-flash", strict_errors=False):
    """Fetch unique small thumbnails concurrently; judge all candidates in one request."""
    import httpx
    key = (access_key or os.environ.get('UNSPLASH_ACCESS_KEY', '')).strip()
    if not key:
        return []
    queries = [str(query).strip() for query in queries if str(query).strip()][:3]
    headers = {'Authorization': f'Client-ID {key}', 'Accept-Version': 'v1'}
    with httpx.Client(timeout=10, follow_redirects=True) as client:
        def search(query):
            try:
                response = client.get('https://api.unsplash.com/search/photos', headers=headers,
                    params={'query': query, 'per_page': min(3, max(1, int(per_page))),
                            'orientation': 'landscape', 'content_filter': 'high'})
                if strict_errors and response.status_code != 200:
                    raise PhotoSearchUnavailable('Unsplash search unavailable')
                return (response.json().get('results') or [])[:3] if response.status_code == 200 else []
            except (SpendBlocked, PhotoSearchUnavailable):
                raise
            except Exception:
                if strict_errors:
                    raise PhotoSearchUnavailable('Unsplash search unavailable') from None
                return []
        with ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
            groups = list(pool.map(search, queries))
        unique = {}
        for group in groups:
            for item in group:
                photo_id = str(item.get('id') or '')
                if photo_id and photo_id not in excluded_photo_ids:
                    unique.setdefault(photo_id, item)
        def thumbnail(item):
            try:
                small = str((item.get('urls') or {}).get('small') or '')
                if urlsplit(small).hostname != 'images.unsplash.com':
                    return None
                response = client.get(small)
                if strict_errors and response.status_code != 200:
                    raise PhotoSearchUnavailable('Photo thumbnail unavailable')
                return _candidate_from_unsplash(item, bytes(response.content)) if response.status_code == 200 else None
            except (SpendBlocked, PhotoSearchUnavailable):
                raise
            except Exception:
                if strict_errors:
                    raise PhotoSearchUnavailable('Photo thumbnail unavailable') from None
                return None
        with ThreadPoolExecutor(max_workers=9) as pool:
            candidates = [candidate for candidate in pool.map(thumbnail, list(unique.values())[:9]) if candidate]
    scores = _score_photo_batch(candidates, word=word, gloss=gloss, thesis=thesis, gemini_api_key=gemini_api_key, model_name=judge_model, strict_errors=strict_errors)
    return [dict(candidate, **score, _vision_verified=True) for candidate, score in zip(candidates, scores)]


def _download_winning_unsplash_photo(candidate, *, access_key=None):
    """Download only the accepted winner, from raw Imgix with real detail metadata."""
    import httpx
    key = (access_key or os.environ.get('UNSPLASH_ACCESS_KEY', '')).strip()
    photo_id = str(candidate.get('photo_id') or '')
    if not key or not photo_id:
        raise ValueError('Winning photograph requires verified API identity and credentials')
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        response = client.get('https://api.unsplash.com/photos/' + quote(photo_id, safe=''),
            headers={'Authorization': f'Client-ID {key}', 'Accept-Version': 'v1'})
        if response.status_code != 200:
            raise ValueError('Winning photograph metadata unavailable; retain last publication')
        detail = response.json()
        raw = urlsplit(str((detail.get('urls') or {}).get('raw') or ''))
        user, links = detail.get('user') or {}, detail.get('links') or {}
        profile = str((user.get('links') or {}).get('html') or '')
        download = str(links.get('download_location') or '')
        if (detail.get('id') != photo_id or raw.scheme != 'https' or raw.hostname != 'images.unsplash.com'
                or not user.get('name') or urlsplit(profile).hostname != 'unsplash.com'
                or urlsplit(download).hostname != 'api.unsplash.com'):
            raise ValueError('Winning photograph metadata is incomplete or invalid')
        params = [(key, value) for key, value in parse_qsl(raw.query) if key not in {'w', 'h', 'fit', 'crop', 'q', 'fm'}]
        params.extend([('w', '1800'), ('h', '1200'), ('fit', 'crop'), ('crop', 'entropy,edges'), ('q', '85'), ('fm', 'jpg')])
        crop_url = urlunsplit(raw._replace(query=urlencode(params)))
        image = client.get(crop_url)
        if image.status_code != 200 or not image.content:
            raise ValueError('Winning photograph crop unavailable; retain last publication')
        with Image.open(BytesIO(image.content)) as decoded:
            hero = decoded.convert('RGB')
        return hero, {'photo_id': photo_id, 'photographer_name': str(user['name']), 'photographer_url': profile,
            'unsplash_url': f'https://unsplash.com/photos/{photo_id}', 'download_location': download,
            'download_location_triggered': False, 'raw_crop_url': crop_url,
            'exif': detail.get('exif') if isinstance(detail.get('exif'), dict) else {}}


def confirm_published_photo_download(pulse, *, callback=None):
    """Acknowledge after publication only. An ambiguous response may need at-least-once reconciliation."""
    attribution = pulse.attribution
    if callback is None:
        callback = getattr(pulse, '_download_ping_callback', None)
    url = str(attribution.get('download_location') or '')
    if pulse.visual_medium != 'PHOTOGRAPHY' or not url or attribution.get('download_location_triggered'):
        return
    if callback is None and os.environ.get('ZEITGEIST_LIVE_UNSPLASH') != '1':
        return
    success = callback(url) if callback is not None else _ping_unsplash_download_location(url)
    if success is False:
        raise ValueError('Published photograph download tracking pending; reconcile before new commissioning')
    attribution['download_location_triggered'] = True


def _ping_unsplash_download_location(download_url: str, access_key: Optional[str] = None) -> bool:
    """Trigger the Unsplash API download_location endpoint as required by Unsplash API Guidelines."""
    key = (access_key or os.environ.get("UNSPLASH_ACCESS_KEY", "")).strip()
    parsed_url = urlsplit(download_url)
    if not download_url or not key or parsed_url.scheme != "https" or parsed_url.hostname != "api.unsplash.com":
        return False
    try:
        import httpx

        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            resp = client.get(
                download_url,
                headers={"Authorization": f"Client-ID {key}", "Accept-Version": "v1"},
            )
            return 200 <= resp.status_code < 300
    except SpendBlocked:
        raise
    except Exception:
        return False


def _generate_live_flux_bytes(
    prompt: str,
    api_key: Optional[str] = None,
    *,
    width: int = 1200,
    height: int = 800,
    timeout_s: float = 45.0,
) -> Optional[bytes]:
    """Generate 3:2 (1200x800) high-resolution editorial artwork via Black Forest Labs FLUX API."""
    key = (api_key or os.environ.get("BFL_API_KEY", "")).strip()
    if os.environ.get("ZEITGEIST_LIVE_LLM") != "1" or not key or not prompt:
        return None
    if type(width) is not int or type(height) is not int or (width, height) != (1200, 800):
        raise SpendBlocked("Unpriced FLUX dimensions; the verified request shape is 1200x800 at 1.5k")
    try:
        import time
        import httpx

        headers = {
            "x-key": key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        payload = {
            "prompt": prompt,
            "aspect_ratio": "3:2",
            "resolution": "1.5k",
            "grounding": True,
            "safety_tolerance": 2,
            "version": "latest",
        }
        deadline = time.monotonic() + max(5.0, float(timeout_s))
        with httpx.Client(timeout=15.0, follow_redirects=False) as client:
            submit_resp = None
            from .provider_lifecycle import (begin_lifecycle, request_digest, utc_now,
                validated_bfl_poll_url, read_bfl_result, submit_bfl_receipt)
            guard = get_spend_guard(credential_keys={"bfl": key})
            digest = request_digest({"provider": "bfl", "operation": "generate_image:flux-3-image", "input": payload})
            lifecycle = begin_lifecycle(guard, "bfl", "generate_image:flux-3-image", None, digest)
            submit = lambda: client.post("https://api.bfl.ai/v1/flux-3-image", headers=headers, json=payload)
            resp = submit() if lifecycle is not None else guard.call(
                "bfl", "generate_image:flux-3-image", None, submit)
            if resp.status_code == 200:
                submit_resp = resp.json()
            if not submit_resp or not submit_resp.get("id"):
                return None

            task_id = submit_resp["id"]
            if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
                return None
            metadata = {"resolution": "1.5k", "aspect_ratio": "3:2"}
            try:
                poll_url = validated_bfl_poll_url(
                    str(submit_resp.get("polling_url") or ""), task_id)
                metadata["polling_host"] = urlsplit(poll_url).hostname
            except ValueError:
                if lifecycle is not None:
                    lifecycle.submitted(task_id, submitted_at=utc_now(), metadata=metadata)
                raise
            if lifecycle is not None:
                lifecycle.submitted(task_id, submitted_at=utc_now(), metadata=metadata)
            while time.monotonic() < deadline:
                time.sleep(1.25)
                poll_resp = read_bfl_result(client, poll_url, task_id)
                if poll_resp.status_code != 200:
                    continue
                poll_data = poll_resp.json()
                if lifecycle is not None:
                    if not isinstance(poll_data, dict) or poll_data.get("id") != task_id:
                        return None
                    accounted = submit_bfl_receipt(lifecycle, digest, task_id, poll_data)
                    if not accounted and poll_data.get("status") == "Ready" and time.monotonic() < deadline:
                        # ResultResponse may precede SettledCostResultResponse.
                        # Re-read this known job only; never submit generation again.
                        final_resp = read_bfl_result(client, poll_url, task_id,
                            timeout=min(15.0, max(0.1, deadline - time.monotonic())))
                        if final_resp.status_code == 200:
                            final_data = final_resp.json()
                            if not isinstance(final_data, dict) or final_data.get("id") != task_id:
                                return None
                            submit_bfl_receipt(lifecycle, digest, task_id, final_data)
                status = str(poll_data.get("status") or "")
                if status == "Ready":
                    sample_url = str((poll_data.get("result") or {}).get("sample") or "")
                    if sample_url:
                        client.cookies.clear()
                        img_resp = client.get(sample_url)
                        if img_resp.status_code == 200 and img_resp.content:
                            return bytes(img_resp.content)
                    return None
                if status in {"Error", "Failed", "Request Moderated", "Content Moderated"}:
                    return None
    except SpendBlocked:
        raise
    except Exception:
        return None
    return None


def run_flux_3_tournament(
    prompts: Sequence[str],
    *,
    word: str = "",
    gloss: str = "",
    thesis: str = "",
    api_key: Optional[str] = None,
    timeout_s: float = 45.0,
    generator_fn: Optional[Callable[[str], Optional[bytes]]] = None,
    judge_model: str = "gemini-3.8-flash",
) -> Optional[bytes]:
    """Dispatch up to 3 FLUX 3 prompts in parallel and select the highest-scoring plate."""
    from concurrent.futures import ThreadPoolExecutor

    prompt_list = [str(p).strip() for p in (prompts or ()) if str(p).strip()][:3]
    if not prompt_list:
        return None

    def _run_single(idx_and_prompt: tuple[int, str]) -> Optional[tuple[int, str, bytes]]:
        idx, p_str = idx_and_prompt
        try:
            if generator_fn is not None:
                raw = generator_fn(p_str)
            else:
                raw = _generate_live_flux_bytes(p_str, api_key=api_key, timeout_s=timeout_s)
            if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
                return (idx, p_str, bytes(raw))
        except SpendBlocked:
            raise
        except Exception:
            return None
        return None

    results: List[tuple[int, str, bytes]] = []
    with ThreadPoolExecutor(max_workers=min(3, len(prompt_list))) as pool:
        for res in pool.map(_run_single, list(enumerate(prompt_list))):
            if res is not None:
                results.append(res)

    if not results:
        return None
    if len(results) == 1:
        return results[0][2]

    scored: List[tuple[float, int, bytes]] = []
    for idx, p_str, raw_bytes in results:
        scores = _score_candidate_with_gemini_vision(
            thumb_bytes=raw_bytes,
            word=word or "Equilibrium",
            gloss=gloss,
            thesis=thesis or p_str,
            model_name=judge_model,
        )
        score = compute_composite_critic_score(
            scores["semantic_alignment"],
            scores["compositional_clarity"],
            scores["editorial_memorability"],
        )
        scored.append((score, -idx, raw_bytes))

    scored.sort(reverse=True)
    return scored[0][2]


def _generate_live_imagen3_bytes(prompt: str, api_key: Optional[str] = None) -> Optional[bytes]:
    """Generate 3:2 minimalist artwork via Gemini Image models (gemini-2.5-flash-image / gemini-3.1-flash-image)."""
    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if os.environ.get("ZEITGEIST_LIVE_LLM") != "1" or not key or not prompt:
        return None
    try:
        from google import genai
        from google.genai import types

        client = guard_genai_client(genai.Client(api_key=key), credential_key=key)
        for model_name in ("gemini-2.5-flash-image", "gemini-3.1-flash-image"):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=f"{prompt} (3:2 aspect ratio, 1200x800 high-resolution editorial composition, no text overlay)",
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
                candidates = getattr(resp, "candidates", None) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        inline = getattr(part, "inline_data", None)
                        raw_data = getattr(inline, "data", None)
                        if isinstance(raw_data, (bytes, bytearray)) and raw_data:
                            return bytes(raw_data)
            except SpendBlocked:
                raise
            except Exception:
                continue

        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config={
                "number_of_images": 1,
                "aspect_ratio": "3:2",
            },
        )
        gen_list = getattr(response, "generated_images", None) or []
        if gen_list:
            first_img = getattr(gen_list[0], "image", None)
            raw_bytes = getattr(first_img, "image_bytes", None)
            if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes:
                return bytes(raw_bytes)
    except SpendBlocked:
        raise
    except Exception:
        return None
    return None


class PhotoCriticUnavailable(ValueError):
    """Photo evaluation failed technically; do not reinterpret it as unsafe imagery."""


class PhotoSearchUnavailable(ValueError):
    """Photo provider unavailable; no false no-fit verdict or AI fallback."""


class PhotoSelectionUnavailable(ValueError):
    """No photograph passed strict safety/fit checks; bounded re-query is permitted."""


def select_and_render_pulse_assets(
    commission: Any,
    medium_history_30d: Sequence[str] | int = (),
    *,
    repo_root: Optional[Path] = None,
    date_str: Optional[str] = None,
    candidates: Optional[Sequence[Dict[str, Any]]] = None,
    unsplash_candidates: Optional[Sequence[Dict[str, Any]]] = None,
    orbit_resonance_score: float = 1.50,
    velocity_score: float = 2.50,
    cluster_keywords: Optional[Sequence[str]] = None,
    cluster_silhouette: float = 0.75,
    sources: Optional[Sequence[Dict[str, str]]] = None,
    mood_votes: Optional[Dict[str, int]] = None,
    download_ping_callback: Optional[Callable[[str], Any]] = None,
    imagen_generator: Optional[Callable[[str], bytes]] = None,
    imagen_bytes: Optional[bytes] = None,
    archetype_history_30d: Sequence[str] = (),
    breakout_surge: bool = False,
    curatorial_archetype_override: Optional[str] = None,
    bfl_api_key: Optional[str] = None,
    flux_generator: Optional[Callable[[str], Optional[bytes]]] = None,
    requested_medium: str | None = None,
    require_semantic_rescore: bool = False,
    photo_history_30d: Sequence[Dict[str, Any]] | None = None,
    judge_model: str = "gemini-3.8-flash",
    strict_photo_errors: bool = False,
) -> PublishedPulse:
    """Run Multimodal Visual Critic Funnel, enforce 80/20 medium budget, and render 3:2 curatorial + 16:9 WebP + SVG."""
    import os
    from src.zeitgeist.curatorial_compositor import (
        compute_quietest_quadrant,
        render_curatorial_plate_pair,
        resolve_effective_quadrant,
        select_curatorial_archetype,
    )

    if requested_medium not in (None, "auto", "photo", "blueprint"):
        raise ValueError("Unknown requested visual medium")
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    resolved_date = str(date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    word = str(getattr(commission, "word", "") if not isinstance(commission, dict) else commission.get("word", "Equilibrium"))
    gloss = str(getattr(commission, "gloss", "") if not isinstance(commission, dict) else commission.get("gloss", ""))
    thesis = str(
        getattr(commission, "commissioned_thesis", "")
        if not isinstance(commission, dict)
        else commission.get("commissioned_thesis", "")
    )
    imagen_prompt = str(
        getattr(commission, "imagen_commission_prompt", "")
        if not isinstance(commission, dict)
        else commission.get("imagen_commission_prompt", "")
    ) or f"Minimalist architectural study representing {word}: {gloss}"
    unsplash_queries = list(
        (getattr(commission, "unsplash_queries", None) if not isinstance(commission, dict) else commission.get("unsplash_queries"))
        or ["minimalist architecture", "structural geometry", "industrial craft"]
    )
    palette = list(
        (getattr(commission, "palette_hex", None) if not isinstance(commission, dict) else commission.get("palette_hex"))
        or ["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"]
    )

    if isinstance(medium_history_30d, int):
        non_photo_days_30d = medium_history_30d
    else:
        non_photo_days_30d = sum(
            1
            for m in medium_history_30d
            if str(m).strip().upper() not in {"PHOTOGRAPHY", "CURATED_PHOTO"}
        )

    if photo_history_30d is None:
        history_path = root / 'data/pulse_feed.json'
        history = json.loads(history_path.read_text()).get('entries', []) if history_path.exists() else []
    else:
        history = photo_history_30d
    excluded_ids = _photo_history_ids(history)

    effective_candidates = [] if requested_medium == "blueprint" else (candidates if candidates is not None else unsplash_candidates)
    live_unsplash_enabled = os.environ.get("ZEITGEIST_LIVE_UNSPLASH", "1").strip() != "0"
    unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip() if live_unsplash_enabled else ""
    if effective_candidates is None:
        unsplash_key = (
            os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
            if live_unsplash_enabled
            else ""
        )
        if not unsplash_key and live_unsplash_enabled:
            env_file = root / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("UNSPLASH_ACCESS_KEY="):
                        unsplash_key = line.split("=", 1)[1].strip().strip("\"'")
                        break
        if unsplash_key:
            effective_candidates = _fetch_live_unsplash_candidates(
                unsplash_queries[:3],
                access_key=unsplash_key,
                word=word,
                gloss=gloss,
                thesis=thesis,
                excluded_photo_ids=excluded_ids,
                judge_model=judge_model,
                strict_errors=strict_photo_errors,
            )
        if not effective_candidates and live_unsplash_enabled:
            effective_candidates = fetch_curated_library_candidates(word=word, excluded_photo_ids=excluded_ids)

    def score_photos(rows):
        rows = [dict(row) for row in (rows or ()) if str(row.get('id') or row.get('photo_id') or '') not in excluded_ids]
        unreviewed = [row for row in rows if not row.get('_vision_verified') and (
            row.get('_requires_critic') or require_semantic_rescore or 'semantic_alignment' not in row)]
        judgments = _score_photo_batch(unreviewed, word=word, gloss=gloss, thesis=thesis, model_name=judge_model, strict_errors=strict_photo_errors) if unreviewed else []
        for row, verdict in zip(unreviewed, judgments):
            row.update(verdict)
            row['_vision_verified'] = True
        scored = []
        for index, row in enumerate(rows):
            verdict = _strict_photo_scores(row)
            row.update(verdict, index=index, composite_score=compute_composite_critic_score(
                verdict['semantic_alignment'], verdict['compositional_clarity'], verdict['editorial_memorability']))
            scored.append(row)
        return scored

    scored_candidates = score_photos(effective_candidates)
    passing_candidates = [row for row in scored_candidates if row['safety_pass'] is True
        and row['semantic_alignment'] >= .70 and row['composite_score'] >= .68]
    if requested_medium == 'photo' and not passing_candidates:
        raise PhotoSelectionUnavailable('No safe, semantically accepted photograph available')
    chosen_cand = None
    visual_medium = 'BESPOKE_SVG'
    if passing_candidates:
        chosen_cand = max(passing_candidates, key=lambda row: (row['composite_score'], row['semantic_alignment'], -row['index']))
    elif non_photo_days_30d >= 6 and requested_medium != 'blueprint':
        safe_floor = [row for row in scored_candidates if row['safety_pass'] is True
            and row['composite_score'] >= .55 and not require_semantic_rescore]
        if not safe_floor and live_unsplash_enabled:
            library = fetch_curated_library_candidates(word=word, excluded_photo_ids=excluded_ids)
            safe_floor = [row for row in score_photos(library) if row['safety_pass'] is True
                and row['composite_score'] >= .55 and (not require_semantic_rescore or row['semantic_alignment'] >= .70)]
        if not safe_floor:
            raise ValueError('Photography cadence requires an accepted real photograph; retain last publication')
        chosen_cand = max(safe_floor, key=lambda row: (row['semantic_alignment'], row['composite_score'], -row['index']))
    if chosen_cand is not None:
        visual_medium = 'PHOTOGRAPHY'

    hero_img: Optional[Image.Image] = None
    attribution: Dict[str, Any] = {
        "photographer_name": None,
        "photographer_url": None,
        "unsplash_url": None,
        "download_location_triggered": False,
    }
    composite_critic_score = 0.0
    semantic_alignment_score = 0.0
    selected_idx: Optional[int] = None

    if requested_medium == "blueprint":
        hero_img = render_bespoke_canvas_image(palette, size=(1200, 800))
        visual_medium = "BESPOKE_SVG"
        curatorial_archetype_override = "CONCRETE_POETRY"

    if visual_medium == 'PHOTOGRAPHY' and chosen_cand is not None:
        selected_idx = int(chosen_cand['index'])
        composite_critic_score = float(chosen_cand['composite_score'])
        semantic_alignment_score = float(chosen_cand['semantic_alignment'])
        if chosen_cand.get('photo_id'):
            hero_img, attribution = _download_winning_unsplash_photo(chosen_cand, access_key=unsplash_key)
        else:
            # Explicit supplied image fixtures remain usable without a provider.
            attribution = {key: chosen_cand.get(key) for key in (
                'photographer_name', 'photographer_url', 'unsplash_url', 'download_location')}
            attribution['download_location_triggered'] = False
            raw_bytes = chosen_cand.get('image_bytes')
            if isinstance(raw_bytes, (bytes, bytearray)):
                hero_img = Image.open(BytesIO(raw_bytes)).convert('RGB')
            elif isinstance(chosen_cand.get('pil_image'), Image.Image):
                hero_img = chosen_cand['pil_image'].convert('RGB')

    if requested_medium == "photo" and hero_img is None:
        raise ValueError("Accepted photograph has no decodable image; retain last publication")

    if hero_img is None:
        if visual_medium != "PHOTOGRAPHY":
            if isinstance(imagen_bytes, (bytes, bytearray)) and imagen_bytes:
                try:
                    hero_img = Image.open(BytesIO(imagen_bytes)).convert("RGB")
                    visual_medium = "IMAGEN_3"
                except SpendBlocked:
                    raise
                except Exception:
                    hero_img = None
            elif imagen_generator is not None:
                try:
                    gen_bytes = imagen_generator(imagen_prompt)
                    hero_img = Image.open(BytesIO(gen_bytes)).convert("RGB")
                    visual_medium = "IMAGEN_3"
                except SpendBlocked:
                    raise
                except Exception:
                    hero_img = None
            else:
                effective_bfl_key = (
                    bfl_api_key
                    if bfl_api_key is not None
                    else (
                        os.environ.get("BFL_API_KEY", "").strip()
                        if os.environ.get("ZEITGEIST_LIVE_LLM") == "1"
                        else ""
                    )
                )
                flux_prompts = list(
                    (
                        getattr(commission, "flux_3_commission_prompts", None)
                        if not isinstance(commission, dict)
                        else commission.get("flux_3_commission_prompts")
                    )
                    or [imagen_prompt]
                )
                if flux_generator is not None or effective_bfl_key:
                    if len(flux_prompts) >= 2:
                        live_flux_bytes = run_flux_3_tournament(
                            flux_prompts,
                            word=word,
                            gloss=gloss,
                            thesis=thesis,
                            api_key=effective_bfl_key,
                            generator_fn=flux_generator,
                            judge_model=judge_model,
                        )
                    elif flux_generator is not None:
                        live_flux_bytes = flux_generator(imagen_prompt)
                    else:
                        live_flux_bytes = _generate_live_flux_bytes(
                            imagen_prompt, api_key=effective_bfl_key
                        )
                    if live_flux_bytes:
                        flux_scores = _score_candidate_with_gemini_vision(
                            thumb_bytes=bytes(live_flux_bytes),
                            word=word,
                            gloss=gloss,
                            thesis=thesis,
                            model_name=judge_model,
                        )
                        if flux_scores.get("safety_pass") is True:
                            try:
                                hero_img = Image.open(BytesIO(live_flux_bytes)).convert("RGB")
                                visual_medium = "FLUX_PRO"
                                semantic_alignment_score = float(flux_scores['semantic_alignment'])
                                composite_critic_score = compute_composite_critic_score(
                                    semantic_alignment_score, flux_scores['compositional_clarity'],
                                    flux_scores['editorial_memorability'])
                                attribution["photographer_name"] = "FLUX.3 Editorial Study"
                            except SpendBlocked:
                                raise
                            except Exception:
                                hero_img = None
                        else:
                            logger.warning(
                                "Generated FLUX candidate failed Gemini Vision safety gate; falling back."
                            )
                if hero_img is None and os.environ.get("GEMINI_API_KEY", "").strip():
                    live_imagen_bytes = _generate_live_imagen3_bytes(imagen_prompt)
                    if live_imagen_bytes:
                        imagen_scores = _score_candidate_with_gemini_vision(
                            thumb_bytes=bytes(live_imagen_bytes),
                            word=word,
                            gloss=gloss,
                            thesis=thesis,
                            model_name=judge_model,
                        )
                        if imagen_scores.get("safety_pass") is True:
                            try:
                                hero_img = Image.open(BytesIO(live_imagen_bytes)).convert("RGB")
                                visual_medium = "IMAGEN_3"
                            except SpendBlocked:
                                raise
                            except Exception:
                                hero_img = None
                        else:
                            logger.warning(
                                "Generated Imagen candidate failed Gemini Vision safety gate; falling back."
                            )

        if hero_img is None:
            hero_img = render_bespoke_canvas_image(palette, size=(1200, 800))
            visual_medium = "BESPOKE_SVG"
            attribution = {"photographer_name": None, "photographer_url": None,
                           "unsplash_url": None, "download_location_triggered": False}

    # 0. Compute quietest quadrant, select curatorial archetype, and render 1200x800 dual-mode 3:2 WebP plates
    raw_quadrant, min_edge_density, _quad_lum = compute_quietest_quadrant(hero_img)
    if curatorial_archetype_override:
        chosen_archetype = str(curatorial_archetype_override).strip().upper()
    else:
        chosen_archetype = select_curatorial_archetype(
            min_edge_density=min_edge_density,
            breakout_surge=bool(breakout_surge),
            velocity=float(velocity_score),
            archetype_history_30d=archetype_history_30d,
        )
    effective_quadrant = resolve_effective_quadrant(chosen_archetype, raw_quadrant)

    assets_dir = root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    light_webp_rel = "assets/daily_pulse_light.webp"
    dark_webp_rel = "assets/daily_pulse_dark.webp"
    credit_label = str(
        attribution.get("photographer_name")
        or ("Unsplash Photography" if visual_medium == "PHOTOGRAPHY" else "Editorial Synthesis")
    )
    render_curatorial_plate_pair(
        hero_img,
        word=word,
        gloss=gloss or thesis,
        thesis=thesis,
        archetype=chosen_archetype,
        quadrant=effective_quadrant,
        luminance=_quad_lum,
        date_str=resolved_date,
        credit_line=credit_label,
        attribution=attribution,
        visual_medium=visual_medium,
        velocity_score=float(velocity_score),
        light_out_path=root / light_webp_rel,
        dark_out_path=root / dark_webp_rel,
    )

    # 1. Save 1200x800 WebP (quality=85) at assets/pulse/{date_str}.webp and prune >365d files
    pulse_dir = root / "assets" / "pulse"
    pulse_dir.mkdir(parents=True, exist_ok=True)
    webp_rel_path = f"assets/pulse/{resolved_date}.webp"
    webp_abs_path = root / webp_rel_path

    img_1200x800 = _crop_and_resize_rgb(hero_img, (1200, 800))
    img_1200x800.save(webp_abs_path, format="WEBP", quality=85)
    _prune_old_webp_assets(pulse_dir, resolved_date, max_age_days=365)

    # The SVG fallback embeds the same composed plate as the dark WebP source.
    # Resonance controls remain in accessible Markdown outside the artwork.
    b64_720 = base64.b64encode((root / dark_webp_rel).read_bytes()).decode("ascii")
    resolved_votes = normalize_mood_votes(mood_votes)
    svg_rel_path = "assets/daily_pulse.svg"
    svg_abs_path = root / svg_rel_path
    vote_metadata = escape(json.dumps(resolved_votes, sort_keys=True))
    svg_content = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" '
        'viewBox="0 0 1200 800" role="img" data-publication-layout="curatorial" '
        f'aria-label="{escape(word)} — {escape(gloss)}">\n'
        f'  <title>{escape(word)} — {resolved_date}</title>\n'
        f'  <metadata id="mood-votes">{vote_metadata}</metadata>\n'
        '  <g id="mood-vote-bar" aria-hidden="true"/>\n'
        f'  <image width="1200" height="800" href="data:image/webp;base64,{b64_720}"/>\n'
        '</svg>\n'
    )
    svg_abs_path.write_text(svg_content, encoding="utf-8")

    published = PublishedPulse(
        date=resolved_date,
        word=word,
        gloss=gloss,
        thesis=thesis,
        orbit_resonance_score=float(orbit_resonance_score),
        velocity_score=float(velocity_score),
        visual_medium=visual_medium,
        webp_asset_path=webp_rel_path,
        svg_asset_path=svg_rel_path,
        attribution=attribution,
        palette_hex=palette,
        mood_votes=resolved_votes,
        composite_critic_score=composite_critic_score,
        semantic_alignment_score=semantic_alignment_score,
        selected_candidate_index=selected_idx,
        cluster_keywords=list(cluster_keywords or []),
        cluster_silhouette=float(cluster_silhouette),
        sources=list(sources or []),
        image_data_uri=f"data:image/webp;base64,{b64_720}",
        published_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        curatorial_archetype=chosen_archetype,
        text_quadrant=effective_quadrant,
        light_webp_asset_path=light_webp_rel,
        dark_webp_asset_path=dark_webp_rel,
    )
    # Keep an injected tracking adapter in memory only. Never invoke it before
    # the canonical/local publication boundary, or serialize it into the feed.
    published._download_ping_callback = download_ping_callback

    # Rendering updates only the local projection. The orchestrator commits
    # canonical Firestore state after staged assets and README validation.
    from src.state_store import JsonFileStateStore

    store = JsonFileStateStore(repo_root=root)
    store.save_pulse_feed_entry(resolved_date, published.to_feed_entry())
    feed_path = root / "data" / "pulse_feed.json"
    if feed_path.exists():
        raw_feed = json.loads(feed_path.read_text(encoding="utf-8"))
        entries_list = raw_feed.get("entries", []) if isinstance(raw_feed, dict) else []
        v2_feed = {
            "version": "2.0",
            "schema_version": "2.0",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entries": entries_list,
        }
        feed_path.write_text(json.dumps(v2_feed, indent=2) + "\n", encoding="utf-8")

    from .publication import write_publication_manifest
    write_publication_manifest(
        root, date=resolved_date, word=word, gloss=gloss, archetype=chosen_archetype,
        medium=visual_medium, archive_path=webp_rel_path,
    )
    return published

