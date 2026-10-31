"""Quantitative Valence-Arousal-Dominance (VAD) & Sublinear TF-IDF Sentiment Methodology.

Combines four weighted lenses into one cohesive daily sentiment score and word:
  - Personal Orbit (40%): Creators, builders, Virginia/East Coast corridor, academic/music/photo craft
  - Global & Civic Macro (25%): Multilateral development, energy, infrastructure, trade
  - Markets & FinOps (20%): Yield curve, capital discipline, real-economy liquidity
  - AI & Open-Source Builders (15%): Deterministic tooling, agent harnesses, systems craft
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FeedDocument:
    """A single public feed item sampled from one of the four orbit lenses."""

    lens_id: str
    source_id: str
    title: str
    summary: str
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LensScore:
    """Quantitative VAD + Epistemic Tension metrics for a single orbit lens."""

    lens_id: str
    label: str
    weight: float
    valence: float          # 0..100 (constructive/warm vs. contracted/bleak)
    arousal: float          # 0..100 (kinetic/urgent vs. calm/still)
    dominance: float        # 0..100 (deliberate/controlled vs. volatile/reactive)
    epistemic_tension: float  # 0..100 (structural debate/shift vs. settled consensus)
    composite_index: float  # 0..100 weighted lens index
    top_terms: Tuple[str, ...]


@dataclass(frozen=True)
class WeightedOrbitAnalysis:
    """Single unified output of the 40/25/20/15 weighted sentiment methodology."""

    date_iso: str
    composite_score: float       # Single 0..100 weighted score across all 4 lenses
    valence: float
    arousal: float
    dominance: float
    epistemic_tension: float
    word_of_the_day: str
    explanation_phrase: str
    visual_concrete_anchor: str
    is_builder_systems_day: bool
    interjection_applied: bool
    personal_note: Optional[str]
    lens_breakdown: Tuple[LensScore, ...]
    dominant_keywords: Tuple[str, ...]


# Curated NRC/Warriner-inspired Valence-Arousal-Dominance (VAD) & Tension lexicon
# Values normalized to [-1.0, +1.0] offsets from a 50.0 neutral baseline.
VAD_LEXICON: Dict[str, Tuple[float, float, float, float]] = {
    # Constructive / deliberate / architectural terms
    "architecture": (0.45, 0.15, 0.65, 0.20),
    "verification": (0.55, 0.20, 0.75, 0.15),
    "deterministic": (0.50, 0.10, 0.80, 0.10),
    "equilibrium": (0.60, -0.25, 0.70, -0.10),
    "calibrated": (0.55, -0.10, 0.75, 0.05),
    "reconciliation": (0.45, 0.05, 0.65, 0.15),
    "stewardship": (0.65, -0.15, 0.60, -0.05),
    "resilience": (0.60, 0.15, 0.65, 0.25),
    "craft": (0.70, 0.05, 0.60, -0.10),
    "minimalist": (0.50, -0.35, 0.55, -0.20),
    "autumn": (0.55, -0.20, 0.40, -0.10),
    "commonwealth": (0.45, -0.05, 0.45, 0.05),
    "williamsburg": (0.50, -0.15, 0.45, -0.05),
    "blue-ridge": (0.65, -0.30, 0.50, -0.15),
    "multilateral": (0.35, 0.10, 0.40, 0.35),
    "discipline": (0.45, 0.05, 0.70, 0.20),
    "liquidity": (0.20, 0.25, 0.35, 0.40),
    "patience": (0.60, -0.40, 0.65, -0.15),
    "convergence": (0.50, 0.25, 0.55, 0.30),
    "tempered": (0.35, -0.15, 0.65, 0.25),
    # High-tension / caution / macro friction terms
    "volatility": (-0.45, 0.65, -0.40, 0.75),
    "grid": (0.10, 0.35, 0.25, 0.55),
    "tariff": (-0.35, 0.45, -0.20, 0.65),
    "constraint": (-0.15, 0.30, 0.20, 0.55),
    "tightening": (-0.25, 0.35, 0.10, 0.50),
    "fragile": (-0.50, 0.40, -0.45, 0.60),
    "hype": (-0.30, 0.60, -0.25, 0.50),
    "slop": (-0.65, 0.35, -0.30, 0.45),
    "prune": (0.35, 0.10, 0.60, 0.15),
    "harness": (0.45, 0.20, 0.70, 0.20),
    "horizon": (0.55, -0.20, 0.50, 0.10),
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over",
    "under", "across", "while", "after", "before", "their", "about", "amid",
    "more", "less", "than", "very", "have", "been", "were", "what", "when",
    "where", "which", "would", "could", "should", "into", "onto", "upon",
}


# Structured Circumplex Mood Rubric mapping (Valence, Arousal, Dominance, Tension)
# into a single evocative Word of the Day, a short explanation phrase, and a concrete visual anchor.
MOOD_TAXONOMY = [
    {
        "word": "TEMPERED",
        "target": (62.0, 44.0, 68.0, 56.0),
        "builder_mode": True,
        "explanation_template": (
            "Deliberate engineering craft and local autumn calm counterbalance "
            "cautious capital liquidity and global grid constraints."
        ),
        "visual_anchor": "Minimalist concrete architectural walkway in crisp morning light and long geometric shadow",
    },
    {
        "word": "CONVERGENCE",
        "target": (66.0, 58.0, 65.0, 54.0),
        "builder_mode": True,
        "explanation_template": (
            "Independent threads in agentic software design, development finance, "
            "and personal creative circles are locking into shared structure."
        ),
        "visual_anchor": "Two rivers meeting in a quiet misty valley seen from above, clean high-contrast geometry",
    },
    {
        "word": "STILLNESS",
        "target": (70.0, 32.0, 62.0, 35.0),
        "builder_mode": False,
        "explanation_template": (
            "A rare pocket of low macro volatility and unhurried focus across "
            "Virginia autumn air, acoustic craft, and steady system baselines."
        ),
        "visual_anchor": "Single wooden pier extending into glass-calm morning water under pale overcast light",
    },
    {
        "word": "THRESHOLD",
        "target": (54.0, 64.0, 52.0, 68.0),
        "builder_mode": False,
        "explanation_template": (
            "Markets and multilateral institutions pause at an inflection point "
            "while builders shift from rapid prototyping to verified foundations."
        ),
        "visual_anchor": "Solitary lighthouse on a dark basalt headland as first light breaks along a sharp horizon line",
    },
    {
        "word": "BALLAST",
        "target": (58.0, 40.0, 72.0, 60.0),
        "builder_mode": True,
        "explanation_template": (
            "Grounded personal routines and strict verification gates act as ballast "
            "against noisy macro headlines and speculative software churn."
        ),
        "visual_anchor": "Heavy cut-stone bridge arch standing steady over fast-moving dark water in monochrome tones",
    },
]


def load_orbit_config(config_path: Path) -> dict:
    """Load the weighted orbit configuration JSON."""
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_curated_corpus() -> List[FeedDocument]:
    """Deterministic representative daily feed sample across the 4 weighted lenses."""
    return [
        # Lens 1: Personal Orbit (weight 0.40)
        FeedDocument(
            lens_id="personal_orbit",
            source_id="matt_pocock",
            title="Designing Deep Modules and Verification Harnesses for Autonomous Coding Agents",
            summary="Why pruning shallow wrappers and testing at module boundaries keeps AI-assisted codebases navigable and deterministic.",
            tags=("architecture", "verification", "harness", "prune", "craft"),
        ),
        FeedDocument(
            lens_id="personal_orbit",
            source_id="simon_willison",
            title="Pragmatic Local-First Tooling and Structured Cost Control with Small Multimodal Models",
            summary="Combining deterministic lexical indexing with structured LLM evaluation keeps daily pipelines fast, inspectable, and affordable.",
            tags=("deterministic", "calibrated", "minimalist", "craft"),
        ),
        FeedDocument(
            lens_id="personal_orbit",
            source_id="virginia_commonwealth_corridor",
            title="Crisp October Morning Across Williamsburg, Richmond, and the Blue Ridge Corridor",
            summary="Early autumn frost clears over historic brickwork and Piedmont ridges as collegiate Saturdays and regional logistics hit stride.",
            tags=("autumn", "williamsburg", "blue-ridge", "commonwealth", "equilibrium"),
        ),
        FeedDocument(
            lens_id="personal_orbit",
            source_id="wm_global_research_aiddata",
            title="Tracking Multilateral Development Commitments with Verified Taxonomies",
            summary="Calibrated sector classification prevents cross-category drift across infrastructure, water sanitation, and energy resilience.",
            tags=("multilateral", "verification", "reconciliation", "resilience"),
        ),
        FeedDocument(
            lens_id="personal_orbit",
            source_id="classic_melodic_craft",
            title="Enduring Studio Arrangement Craft and Unhurried Acoustic Patience",
            summary="Why acoustic space, deliberate piano voicings, and lyrical restraint outlast compressed algorithmic production.",
            tags=("craft", "patience", "minimalist", "horizon"),
        ),
        # Lens 2: Global & Civic Macro (weight 0.25)
        FeedDocument(
            lens_id="global_macro",
            source_id="oecd_economic_outlook",
            title="Global Trade Corridors Adapt to Power Grid Constraints and Industrial Realignment",
            summary="Multilateral supply chains show tempered resilience despite regional tariff friction and power capacity bottlenecks.",
            tags=("multilateral", "grid", "constraint", "tariff", "resilience"),
        ),
        # Lens 3: Markets & FinOps (weight 0.20)
        FeedDocument(
            lens_id="markets_finops",
            source_id="fred_macro_rates",
            title="Term Liquidity and Operating Margin Discipline Anchor Enterprise Allocations",
            summary="Capital allocators favor verified cash-flow durability and disciplined unit economics over high-volatility multiple expansion.",
            tags=("liquidity", "discipline", "tightening", "tempered", "reconciliation"),
        ),
        # Lens 4: AI & Open-Source Builders (weight 0.15)
        FeedDocument(
            lens_id="ai_builders",
            source_id="hackernews_systems",
            title="Builders Shift from Prompt Hype to Deterministic Verification and SQLite State Engines",
            summary="Open-source engineering communities reject generative slop in favor of reproducible benchmarks, schema guardrails, and small tools.",
            tags=("deterministic", "verification", "slop", "hype", "convergence"),
        ),
    ]


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z][a-z\-]{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _score_lens_documents(
    lens_id: str,
    label: str,
    weight: float,
    docs: List[FeedDocument],
    idf_map: Dict[str, float],
) -> LensScore:
    if not docs:
        return LensScore(
            lens_id=lens_id,
            label=label,
            weight=weight,
            valence=50.0,
            arousal=50.0,
            dominance=50.0,
            epistemic_tension=50.0,
            composite_index=50.0,
            top_terms=(),
        )

    tf_counts: Dict[str, int] = {}
    for doc in docs:
        combined = f"{doc.title} {doc.summary} {' '.join(doc.tags)}"
        for tok in _tokenize(combined):
            tf_counts[tok] = tf_counts.get(tok, 0) + 1
        for tag in doc.tags:
            tf_counts[tag] = tf_counts.get(tag, 0) + 2

    tfidf_weights: Dict[str, float] = {}
    for term, count in tf_counts.items():
        sublinear_tf = 1.0 + math.log(count)
        tfidf_weights[term] = sublinear_tf * idf_map.get(term, 1.0)

    val_sum = 0.0
    aro_sum = 0.0
    dom_sum = 0.0
    ten_sum = 0.0
    matched_weight = 0.0

    for term, w in tfidf_weights.items():
        if term in VAD_LEXICON:
            v, a, d, t = VAD_LEXICON[term]
            val_sum += v * w
            aro_sum += a * w
            dom_sum += d * w
            ten_sum += t * w
            matched_weight += w

    norm = max(matched_weight, 1.0)
    valence = max(0.0, min(100.0, 50.0 + 38.0 * (val_sum / norm)))
    arousal = max(0.0, min(100.0, 50.0 + 38.0 * (aro_sum / norm)))
    dominance = max(0.0, min(100.0, 50.0 + 38.0 * (dom_sum / norm)))
    tension = max(0.0, min(100.0, 50.0 + 38.0 * (ten_sum / norm)))

    # Composite lens index rewards constructive valence + deliberate dominance while accounting for tension
    composite = round(0.45 * valence + 0.35 * dominance + 0.20 * (100.0 - 0.35 * abs(arousal - 50.0)), 1)

    ranked_terms = sorted(tfidf_weights.items(), key=lambda kv: (-kv[1], kv[0]))
    top_terms = tuple(term for term, _ in ranked_terms[:5])

    return LensScore(
        lens_id=lens_id,
        label=label,
        weight=weight,
        valence=round(valence, 1),
        arousal=round(arousal, 1),
        dominance=round(dominance, 1),
        epistemic_tension=round(tension, 1),
        composite_index=composite,
        top_terms=top_terms,
    )


def validate_owner_interjection(raw: dict) -> dict:
    """Normalize trusted-owner controls before configuration or publication writes."""
    if not isinstance(raw, dict):
        raise ValueError("Owner interjection must be an object")
    if not raw.get("enabled"):
        return {"enabled": False}
    result = dict(raw)
    mode = result.get("override_visual_mode") or "auto"
    if mode not in {"auto", "photo", "blueprint"}:
        raise ValueError("Visual mode must be auto, photo, or blueprint")
    result["override_visual_mode"] = None if mode == "auto" else mode
    for key, maximum in (("override_word", 37), ("override_phrase", 50), ("personal_note", 1000)):
        value = result.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be plain text")
        value = str(value or "").strip()
        if len(value) > maximum or any(ord(ch) < 32 or ord(ch) == 127 or ch in '<>"' for ch in value):
            raise ValueError(f"{key} exceeds its plain-text length/markup limit")
        result[key] = value or None
    word = result.get("override_word")
    if word and not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]{2,18}(?: [A-Za-zÀ-ÖØ-öø-ÿ'-]{2,18})?", word):
        raise ValueError("Owner word must contain one or two Latin-script words")
    if word:
        result["override_word"] = word.upper()
    phrase = result.get("override_phrase")
    if phrase and len(phrase.split()) > 12:
        raise ValueError("Owner phrase must contain at most twelve words")
    if result.get("override_image_url"):
        raise ValueError("Direct image URLs require curator validation; use photo mode")
    return result


def analyze_seven_bucket_orbit(
    config: dict,
    documents: Optional[List[FeedDocument]] = None,
    date_iso: str = "2026-10-23",
) -> WeightedOrbitAnalysis:
    """Compute the single weighted sentiment score across the 7-bucket Cultural Orbit & Macro architecture."""
    docs = documents if documents is not None else default_curated_corpus()
    lenses_cfg = config.get("lenses", {})

    # Compute corpus-level smooth IDF
    doc_freq: Dict[str, int] = {}
    total_docs = max(len(docs), 1)
    for doc in docs:
        unique_terms = set(_tokenize(f"{doc.title} {doc.summary}")) | set(doc.tags)
        for term in unique_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    idf_map = {
        term: math.log((1.0 + total_docs) / (1.0 + df)) + 1.0
        for term, df in doc_freq.items()
    }

    lens_scores: List[LensScore] = []
    for lens_id in ("personal_orbit", "global_macro", "markets_finops", "ai_builders"):
        spec = lenses_cfg.get(lens_id, {})
        weight = float(spec.get("weight", 0.25))
        label = str(spec.get("label", lens_id))
        lens_docs = [d for d in docs if d.lens_id == lens_id]
        lens_scores.append(_score_lens_documents(lens_id, label, weight, lens_docs, idf_map))

    # Normalize weights so they sum to 1.00 (Personal Orbit = 0.40)
    total_weight = sum(ls.weight for ls in lens_scores) or 1.0
    valence = sum(ls.valence * (ls.weight / total_weight) for ls in lens_scores)
    arousal = sum(ls.arousal * (ls.weight / total_weight) for ls in lens_scores)
    dominance = sum(ls.dominance * (ls.weight / total_weight) for ls in lens_scores)
    tension = sum(ls.epistemic_tension * (ls.weight / total_weight) for ls in lens_scores)
    composite_score = sum(ls.composite_index * (ls.weight / total_weight) for ls in lens_scores)

    # Select closest mood entry in circumplex VAD+T space
    best_entry = MOOD_TAXONOMY[0]
    best_dist = float("inf")
    for entry in MOOD_TAXONOMY:
        tv, ta, td, tt = entry["target"]
        dist = math.sqrt(
            (valence - tv) ** 2
            + 0.8 * (arousal - ta) ** 2
            + (dominance - td) ** 2
            + 0.7 * (tension - tt) ** 2
        )
        if dist < best_dist:
            best_dist = dist
            best_entry = entry

    word = str(best_entry["word"])
    phrase = str(best_entry["explanation_template"])
    visual_anchor = str(best_entry["visual_anchor"])
    is_builder_day = bool(best_entry["builder_mode"])

    # Check if user submitted an active interjection from phone or CLI
    interjection = config.get("active_interjection", {})
    interjection_applied = False
    personal_note = None
    if interjection.get("enabled"):
        interjection_applied = True
        if interjection.get("override_word"):
            word = str(interjection["override_word"]).strip().upper()
        if interjection.get("override_phrase"):
            phrase = str(interjection["override_phrase"]).strip()
        if interjection.get("personal_note"):
            personal_note = str(interjection["personal_note"]).strip()
        if interjection.get("override_visual_mode") == "blueprint":
            is_builder_day = True
        elif interjection.get("override_visual_mode") == "photo":
            is_builder_day = False

    all_keywords: List[str] = []
    for ls in lens_scores:
        for term in ls.top_terms:
            if term not in all_keywords:
                all_keywords.append(term)

    return WeightedOrbitAnalysis(
        date_iso=date_iso,
        composite_score=round(composite_score, 1),
        valence=round(valence, 1),
        arousal=round(arousal, 1),
        dominance=round(dominance, 1),
        epistemic_tension=round(tension, 1),
        word_of_the_day=word,
        explanation_phrase=phrase,
        visual_concrete_anchor=visual_anchor,
        is_builder_systems_day=is_builder_day,
        interjection_applied=interjection_applied,
        personal_note=personal_note,
        lens_breakdown=tuple(lens_scores),
        dominant_keywords=tuple(all_keywords[:8]),
    )


# ---------------------------------------------------------------------------
# WS3 Seam 1: Stratified Asymmetric Window & Farthest-Point Diversity
# ---------------------------------------------------------------------------

import base64
from dataclasses import field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - stdlib-only visitor workflow fallback
    np = None  # type: ignore[assignment]

ORBIT_BUCKET_TOKENS = frozenset(
    {
        "ORBIT",
        "PERSONAL_ORBIT",
        "O1",
        "O2",
        "O3",
        "O4",
        "ORBIT_SCOUT",
        "O4_SCOUT",
    }
)
ORBIT_ORIGIN_TOKENS = frozenset({"curated_o1_o3", "orbit_scout_1hop"})

ORBIT_TIER_WEIGHTS: Dict[str, float] = {
    "O1": 2.0,
    "O2": 1.6,
    "O3": 1.3,
    "O4_SCOUT": 1.1,
    "O4": 1.1,
    "ORBIT_SCOUT": 1.1,
    "MACRO": 1.0,
    "M1": 1.0,
    "M2": 1.0,
    "M3": 1.0,
    "M4_SCOUT": 1.0,
}


@dataclass
class CachedItem:
    """In-memory item representation for stratified windowing and clustering."""

    url_hash: str
    source_id: str
    bucket: str = "MACRO"
    tier: str = "M1"
    origin: str = "curated_M1_M3"
    title: str = ""
    summary: str = ""
    url: str = ""
    published_at: str | datetime = ""
    embedding_f16: bytes | str | None = None
    embedding_f32: np.ndarray | Sequence[float] | None = None
    weight: Optional[float] = None
    orbit_weight: Optional[float] = None
    source_tier: Optional[str] = None
    anchored_in_14d: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)


def _get_item_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _set_item_field(item: Any, key: str, value: Any) -> None:
    if isinstance(item, dict):
        item[key] = value
    else:
        setattr(item, key, value)


def _get_item_hash(item: Any) -> str:
    for key in ("url_hash", "url_sha256", "id", "url"):
        val = _get_item_field(item, key)
        if val:
            return str(val)
    return ""


def _parse_utc_datetime(val: Any) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(float(val), tz=timezone.utc)
    if isinstance(val, str) and val.strip():
        raw = val.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _is_personal_orbit_item(item: Any) -> bool:
    bucket = str(_get_item_field(item, "bucket", "") or "").strip().upper()
    if bucket in ORBIT_BUCKET_TOKENS:
        return True
    source_tier = str(_get_item_field(item, "source_tier", "") or "").strip().upper()
    if source_tier in ORBIT_BUCKET_TOKENS:
        return True
    tier = str(_get_item_field(item, "tier", "") or "").strip().upper()
    if tier in ORBIT_BUCKET_TOKENS:
        return True
    origin = str(_get_item_field(item, "origin", "") or "").strip().lower()
    if origin in ORBIT_ORIGIN_TOKENS:
        return True
    if not bucket and not tier and not source_tier:
        source_id = str(_get_item_field(item, "source_id", "") or "").strip().lower()
        if source_id.startswith(("o1_", "o2_", "o3_", "o4_", "orbit", "ig_")):
            return True
    return False


def resolve_orbit_tier_weight(item: Any) -> float:
    """Resolve granular per-tier Orbit weight (O1=2.0, O2=1.6, O3=1.3, O4_SCOUT=1.1, MACRO=1.0)."""
    explicit_orbit_w = _get_item_field(item, "orbit_weight", None)
    if explicit_orbit_w is not None:
        return float(explicit_orbit_w)

    for field_name in ("source_tier", "tier"):
        raw_tier = str(_get_item_field(item, field_name, "") or "").strip().upper()
        if raw_tier in ORBIT_TIER_WEIGHTS:
            return float(ORBIT_TIER_WEIGHTS[raw_tier])

    source_id = str(_get_item_field(item, "source_id", "") or "").strip().lower()
    if source_id.startswith("o1_"):
        return float(ORBIT_TIER_WEIGHTS["O1"])
    if source_id.startswith("o2_"):
        return float(ORBIT_TIER_WEIGHTS["O2"])
    if source_id.startswith("o3_"):
        return float(ORBIT_TIER_WEIGHTS["O3"])
    if source_id.startswith("o4_"):
        return float(ORBIT_TIER_WEIGHTS["O4_SCOUT"])

    raw_bucket = str(_get_item_field(item, "bucket", "") or "").strip().upper()
    if raw_bucket in ORBIT_TIER_WEIGHTS:
        return float(ORBIT_TIER_WEIGHTS[raw_bucket])

    return 1.60 if _is_personal_orbit_item(item) else 1.00


def _extract_unit_embedding_f32(item: Any, default_dim: int = 768) -> np.ndarray:
    """Extract L2-normalized float32 vector from embedding_f32, embedding, or embedding_f16."""
    for arr_key in ("embedding_f32", "embedding"):
        arr_val = _get_item_field(item, arr_key)
        if arr_val is not None:
            vec = np.asarray(arr_val, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                return vec / norm
            return vec

    f16_val = _get_item_field(item, "embedding_f16")
    if f16_val is not None:
        if isinstance(f16_val, str):
            raw_bytes = base64.b64decode(f16_val)
        elif isinstance(f16_val, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(f16_val)
        else:
            raw_bytes = np.asarray(f16_val, dtype=np.float16).tobytes()
        vec = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).copy()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            return vec / norm
        return vec

    return np.zeros(default_dim, dtype=np.float32)


def farthest_point_diversity_sample(
    items: Sequence[Any],
    k: int = 4,
    near_duplicate_threshold: float = 0.85,
) -> List[Any]:
    """Collapse near-duplicates (>= 0.85) to newest item, then select k maximally distant items."""
    if not items or k <= 0:
        return []

    indexed_items = list(enumerate(items))
    # Sort newest-first by published_at; tie-break by original order
    indexed_items.sort(
        key=lambda pair: (-_parse_utc_datetime(_get_item_field(pair[1], "published_at")).timestamp(), pair[0])
    )
    sorted_items = [pair[1] for pair in indexed_items]
    vecs = [_extract_unit_embedding_f32(it) for it in sorted_items]

    # Step 1: Collapse near-duplicates with cosine similarity >= 0.85 to an already-kept newer item
    dedup_items: List[Any] = []
    dedup_vecs: List[np.ndarray] = []
    for item, vec in zip(sorted_items, vecs):
        if not dedup_vecs:
            dedup_items.append(item)
            dedup_vecs.append(vec)
            continue
        sims = [float(np.dot(vec, kept_vec)) for kept_vec in dedup_vecs]
        if max(sims) >= near_duplicate_threshold:
            continue
        dedup_items.append(item)
        dedup_vecs.append(vec)

    if len(dedup_items) <= k:
        return dedup_items

    # Step 2: Greedy farthest-point traversal starting from newest item x_1 = dedup_items[0]
    selected_indices: List[int] = [0]
    remaining_indices: List[int] = list(range(1, len(dedup_items)))
    mat = np.stack(dedup_vecs, axis=0)  # shape (M, D)

    while len(selected_indices) < k and remaining_indices:
        best_idx = remaining_indices[0]
        best_max_sim = float("inf")
        for cand_idx in remaining_indices:
            max_sim = float(np.max(mat[cand_idx] @ mat[selected_indices].T))
            if max_sim < best_max_sim - 1e-7:
                best_max_sim = max_sim
                best_idx = cand_idx
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [dedup_items[idx] for idx in selected_indices]


def load_asymmetric_window(
    cache_items: Sequence[Any],
    anchored_hashes_14d: Optional[Iterable[str]] = None,
    now_utc: Optional[datetime | str] = None,
    *,
    max_orbit_items: int = 220,
    max_macro_items: int = 180,
    macro_per_source_k: int = 4,
) -> List[Any]:
    """Build stratified active window (14d Orbit, 4d Macro) with sub-linear source damping weights."""
    if not cache_items:
        return []

    anchored_set: set[str] = set(str(h) for h in (anchored_hashes_14d or ()))

    if now_utc is not None:
        ref_now = _parse_utc_datetime(now_utc)
    else:
        parsed_times = [
            _parse_utc_datetime(_get_item_field(it, "published_at"))
            for it in cache_items
            if _get_item_field(it, "published_at")
        ]
        ref_now = max(parsed_times) if parsed_times else datetime.now(timezone.utc)

    orbit_candidates: List[Tuple[float, int, Any]] = []
    macro_by_source: Dict[str, List[Any]] = {}

    for idx, item in enumerate(cache_items):
        pub_raw = _get_item_field(item, "published_at")
        pub_dt = _parse_utc_datetime(pub_raw) if pub_raw else ref_now
        age_hours = (ref_now - pub_dt).total_seconds() / 3600.0
        is_orbit = _is_personal_orbit_item(item)

        if is_orbit:
            if pub_raw and age_hours > 336.0 + 1e-6:
                continue
            orbit_candidates.append((-pub_dt.timestamp(), idx, item))
        else:
            bucket_str = str(
                _get_item_field(item, "bucket", "") or _get_item_field(item, "tier", "") or ""
            ).strip().upper()
            max_macro_age_hours = 96.0
            if pub_raw and age_hours > max_macro_age_hours + 1e-6:
                continue
            source_id = str(_get_item_field(item, "source_id", "unknown_macro") or "unknown_macro")
            macro_by_source.setdefault(source_id, []).append(item)

    orbit_candidates.sort(key=lambda tup: (tup[0], tup[1]))
    retained_orbit = [tup[2] for tup in orbit_candidates[:max_orbit_items]]

    sampled_macro_candidates: List[Tuple[float, int, Any]] = []
    macro_counter = 0
    for source_id in sorted(macro_by_source.keys()):
        sampled_for_feed = farthest_point_diversity_sample(
            macro_by_source[source_id],
            k=macro_per_source_k,
        )
        for item in sampled_for_feed:
            pub_raw = _get_item_field(item, "published_at")
            pub_dt = _parse_utc_datetime(pub_raw) if pub_raw else ref_now
            sampled_macro_candidates.append((-pub_dt.timestamp(), macro_counter, item))
            macro_counter += 1

    sampled_macro_candidates.sort(key=lambda tup: (tup[0], tup[1]))
    retained_macro = [tup[2] for tup in sampled_macro_candidates[:max_macro_items]]

    window_items = retained_orbit + retained_macro

    # Count N_win(s) across the active window
    source_counts: Dict[str, int] = {}
    for item in window_items:
        source_id = str(_get_item_field(item, "source_id", "unknown") or "unknown")
        source_counts[source_id] = source_counts.get(source_id, 0) + 1

    # Assign sub-linear source damping weight: w_i = w_bucket(s) * c_used(i) / sqrt(N_win(s))
    for item in window_items:
        source_id = str(_get_item_field(item, "source_id", "unknown") or "unknown")
        n_win = max(1, source_counts.get(source_id, 1))
        w_bucket = resolve_orbit_tier_weight(item)

        item_hash = _get_item_hash(item)
        already_anchored = (
            (bool(item_hash) and item_hash in anchored_set)
            or bool(_get_item_field(item, "anchored_in_14d", False))
            or bool(_get_item_field(item, "used_for_anchor", False))
            or bool(_get_item_field(item, "anchored_word", None))
        )
        c_used = 0.50 if already_anchored else 1.00
        w_i = (w_bucket * c_used) / math.sqrt(float(n_win))
        _set_item_field(item, "weight", float(w_i))

    return window_items


# ---------------------------------------------------------------------------
# WS3 Seam 2: Pure-numpy Average-Linkage Clustering & Disjoint Surge Math
# ---------------------------------------------------------------------------

# Calibration Note (G-11 / G-13):
# Unlike high-density enterprise corpora in design.md (which specify cosine distance
# cutoffs of 0.28 / 0.38, i.e. similarity 0.72 / 0.62, with min_cluster_size=3),
# the daily RSS/Atom + 1-hop scout window (~40-400 items mixing 768-dim Gemini
# embeddings and deterministic lexical hash fallback vectors) operates in a sparser
# daily regime. We step down similarity thresholds tau in (0.78, 0.72, 0.68)
# (floor DEFAULT_COSINE_DISTANCE_THRESHOLD = 0.68) with DEFAULT_MIN_CLUSTER_SIZE = 2
# distinct sources so two-source cross-lens convergence (1 Orbit + 1 Macro) is
# preserved without collapsing into singletons.
DYNAMIC_CLUSTER_THRESHOLDS: Tuple[float, ...] = (0.78, 0.72, 0.68)
DEFAULT_COSINE_DISTANCE_THRESHOLD: float = 0.68
DEFAULT_MIN_CLUSTER_SIZE: int = 2


@dataclass
class ScoredCluster:
    """Scored topical cluster with c-TF-IDF keywords, silhouette, and disjoint surge metrics."""

    cluster_id: str
    items: List[Any]
    cluster_keywords: List[str]
    silhouette_score: float
    w_recent: float
    w_base_daily: float
    breakout_surge: float
    velocity: float
    orbit_resonance: float
    s_intra: float
    unique_sources: Tuple[str, ...]
    unique_sources_count: int
    commission_score: float
    threshold_used: float = 0.78

    @property
    def keywords(self) -> List[str]:
        return self.cluster_keywords

    @property
    def b_surge(self) -> float:
        return self.breakout_surge

    @property
    def m_orbit(self) -> float:
        return self.orbit_resonance

    @property
    def score(self) -> float:
        return self.commission_score

    @property
    def silhouette(self) -> float:
        return self.silhouette_score

    @property
    def size(self) -> int:
        return len(self.items)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _average_linkage_cluster_at_threshold(
    sim_matrix: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    """Greedy average-linkage hierarchical clustering using vectorized numpy operations."""
    n = int(sim_matrix.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    clusters: List[List[int]] = [[i] for i in range(n)]
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n, dtype=np.float64)
    sum_sim = sim_matrix.astype(np.float64, copy=True)

    avg_sim = sim_matrix.astype(np.float64, copy=True)
    tril_r, tril_c = np.tril_indices(n, k=0)
    avg_sim[tril_r, tril_c] = -np.inf

    while True:
        flat_idx = int(np.argmax(avg_sim))
        a, b = divmod(flat_idx, n)
        max_sim = float(avg_sim[a, b])
        if max_sim < threshold or not np.isfinite(max_sim):
            break

        # Merge cluster b into cluster a (a < b guaranteed by upper-triangular active mask)
        clusters[a].extend(clusters[b])
        clusters[b] = []
        active[b] = False
        sizes[a] += sizes[b]

        sum_sim[a, :] += sum_sim[b, :]
        sum_sim[:, a] = sum_sim[a, :]

        avg_sim[b, :] = -np.inf
        avg_sim[:, b] = -np.inf

        if a > 0:
            left_mask = active[:a]
            avg_sim[:a, a] = np.where(
                left_mask,
                sum_sim[:a, a] / (sizes[:a] * sizes[a]),
                -np.inf,
            )
        if a + 1 < n:
            right_mask = active[a + 1 :]
            avg_sim[a, a + 1 :] = np.where(
                right_mask,
                sum_sim[a, a + 1 :] / (sizes[a] * sizes[a + 1 :]),
                -np.inf,
            )

    return [c for i, c in enumerate(clusters) if active[i] and c]


def _extract_1_2_grams(text: str) -> List[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    grams = list(tokens)
    for idx in range(len(tokens) - 1):
        grams.append(f"{tokens[idx]} {tokens[idx + 1]}")
    return grams


def _extract_c_tfidf_keywords(
    cluster_indices_list: List[List[int]],
    items: Sequence[Any],
    top_k: int = 5,
) -> List[List[str]]:
    """Compute top-k 1-2 gram c-TF-IDF keywords for each cluster."""
    total_items = max(1, len(items))
    cluster_term_counts: List[Dict[str, int]] = []
    global_term_counts: Dict[str, int] = {}

    for indices in cluster_indices_list:
        counts: Dict[str, int] = {}
        for idx in indices:
            item = items[idx]
            title = str(_get_item_field(item, "title", "") or "")
            summary = str(_get_item_field(item, "summary", "") or "")
            for gram in _extract_1_2_grams(f"{title} {summary}"):
                counts[gram] = counts.get(gram, 0) + 1
                global_term_counts[gram] = global_term_counts.get(gram, 0) + 1
        cluster_term_counts.append(counts)

    results: List[List[str]] = []
    for counts in cluster_term_counts:
        total_terms_in_cluster = sum(counts.values())
        if total_terms_in_cluster == 0:
            results.append(["zeitgeist"])
            continue
        scored_terms: List[Tuple[float, str]] = []
        for term, f_tc in counts.items():
            tf = float(f_tc) / float(total_terms_in_cluster)
            global_f = max(1, global_term_counts.get(term, f_tc))
            idf = math.log(1.0 + (float(total_items) / float(global_f)))
            scored_terms.append((tf * idf, term))
        scored_terms.sort(key=lambda pair: (-pair[0], -len(pair[1]), pair[1]))
        results.append([pair[1] for pair in scored_terms[:top_k]])

    return results


def _compute_silhouette_scores(
    cluster_indices_list: List[List[int]],
    sim_matrix: np.ndarray,
) -> List[float]:
    """Compute mean cosine silhouette score s(i) = (b(i) - a(i)) / max(a(i), b(i)) per cluster."""
    num_clusters = len(cluster_indices_list)
    if num_clusters < 2:
        return [0.0 for _ in cluster_indices_list]

    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    cluster_silhouettes: List[float] = []

    for k_idx, indices in enumerate(cluster_indices_list):
        if len(indices) < 2:
            cluster_silhouettes.append(0.0)
            continue

        idx_arr = np.asarray(indices, dtype=np.int64)
        sub_dist = dist_matrix[np.ix_(idx_arr, idx_arr)]
        # Mean intra-cluster distance a(i) excluding self (diagonal is 0)
        a_vals = np.sum(sub_dist, axis=1) / float(len(indices) - 1)

        # Nearest other-cluster mean distance b(i)
        b_vals = np.full(len(indices), np.inf, dtype=np.float64)
        for other_idx, other_indices in enumerate(cluster_indices_list):
            if other_idx == k_idx or not other_indices:
                continue
            other_arr = np.asarray(other_indices, dtype=np.int64)
            cross_dist = dist_matrix[np.ix_(idx_arr, other_arr)]
            mean_cross = np.mean(cross_dist, axis=1)
            b_vals = np.minimum(b_vals, mean_cross)

        denom = np.maximum(a_vals, b_vals)
        s_vals = np.where(denom > 1e-12, (b_vals - a_vals) / denom, 0.0)
        cluster_silhouettes.append(float(np.mean(s_vals)))

    return cluster_silhouettes


def cluster_and_score(
    items: Sequence[Any],
    now_utc: Optional[datetime | str] = None,
    anchored_hashes_14d: Optional[Iterable[str]] = None,
) -> List[ScoredCluster]:
    """Cluster items via pure-numpy average linkage and rank by Disjoint Surge Velocity & Orbit Resonance."""
    if not items:
        return []

    anchored_set: set[str] = set(str(h) for h in (anchored_hashes_14d or ()))

    if now_utc is not None:
        ref_now = _parse_utc_datetime(now_utc)
    else:
        parsed_times = [
            _parse_utc_datetime(_get_item_field(it, "published_at"))
            for it in items
            if _get_item_field(it, "published_at")
        ]
        ref_now = max(parsed_times) if parsed_times else datetime.now(timezone.utc)

    # Ensure every item has a valid weight w_i (compute sub-linear source damping if not already set)
    source_counts: Dict[str, int] = {}
    for item in items:
        src = str(_get_item_field(item, "source_id", "unknown") or "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    item_weights: List[float] = []
    for item in items:
        explicit_w = _get_item_field(item, "weight", None)
        if explicit_w is not None:
            w_val = float(explicit_w)
        else:
            src = str(_get_item_field(item, "source_id", "unknown") or "unknown")
            n_win = max(1, source_counts.get(src, 1))
            w_bucket = resolve_orbit_tier_weight(item)
            item_hash = _get_item_hash(item)
            already_anchored = (
                (bool(item_hash) and item_hash in anchored_set)
                or bool(_get_item_field(item, "anchored_in_14d", False))
                or bool(_get_item_field(item, "used_for_anchor", False))
                or bool(_get_item_field(item, "anchored_word", None))
            )
            c_used = 0.50 if already_anchored else 1.00
            w_val = (w_bucket * c_used) / math.sqrt(float(n_win))
            _set_item_field(item, "weight", float(w_val))
        item_weights.append(w_val)

    # Unpack L2-normalized float32 embeddings and compute N x N cosine similarity matrix
    vecs = [_extract_unit_embedding_f32(it) for it in items]
    e_hat = np.stack(vecs, axis=0).astype(np.float32)
    sim_matrix = np.clip(e_hat @ e_hat.T, -1.0, 1.0)

    # Dynamic Similarity Thresholding: step down tau in [0.78, 0.72, 0.68] until >= 2 multi-source clusters emerge
    chosen_clusters_idx: List[List[int]] = []
    chosen_tau = DYNAMIC_CLUSTER_THRESHOLDS[0]

    for tau in DYNAMIC_CLUSTER_THRESHOLDS:
        candidate_clusters = _average_linkage_cluster_at_threshold(sim_matrix, tau)
        multi_source_count = 0
        for c_indices in candidate_clusters:
            distinct_sources = {
                str(_get_item_field(items[idx], "source_id", f"src_{idx}") or f"src_{idx}")
                for idx in c_indices
            }
            if len(distinct_sources) >= DEFAULT_MIN_CLUSTER_SIZE:
                multi_source_count += 1
        chosen_clusters_idx = candidate_clusters
        chosen_tau = tau
        if multi_source_count >= DEFAULT_MIN_CLUSTER_SIZE:
            break

    keywords_per_cluster = _extract_c_tfidf_keywords(chosen_clusters_idx, items, top_k=5)
    silhouettes_per_cluster = _compute_silhouette_scores(chosen_clusters_idx, sim_matrix)

    scored_clusters: List[ScoredCluster] = []
    for k_idx, c_indices in enumerate(chosen_clusters_idx):
        cluster_items = [items[idx] for idx in c_indices]

        w_recent = 0.0
        w_macro_base = 0.0
        w_orbit_base = 0.0

        all_sources: set[str] = set()
        orbit_sources: set[str] = set()
        macro_sources: set[str] = set()
        sources_0_24h: set[str] = set()
        macro_sources_0_24h: set[str] = set()
        sources_base: set[str] = set()
        orbit_sources_0_72h: set[str] = set()
        orbit_sources_72_336h: set[str] = set()

        for idx in c_indices:
            item = items[idx]
            w_i = item_weights[idx]
            src = str(_get_item_field(item, "source_id", f"src_{idx}") or f"src_{idx}")
            is_orbit = _is_personal_orbit_item(item)

            pub_raw = _get_item_field(item, "published_at")
            pub_dt = _parse_utc_datetime(pub_raw) if pub_raw else ref_now
            age_h = max(0.0, (ref_now - pub_dt).total_seconds() / 3600.0)

            all_sources.add(src)
            if is_orbit:
                orbit_sources.add(src)
            else:
                macro_sources.add(src)

            if age_h <= 24.0 + 1e-9:
                sources_0_24h.add(src)
                if not is_orbit:
                    macro_sources_0_24h.add(src)

            if is_orbit:
                if age_h <= 72.0 + 1e-9:
                    w_recent += w_i
                    orbit_sources_0_72h.add(src)
                elif age_h <= 336.0 + 1e-9:
                    w_orbit_base += w_i
                    sources_base.add(src)
                    orbit_sources_72_336h.add(src)
            else:
                if age_h <= 24.0 + 1e-9:
                    w_recent += w_i
                elif age_h <= 96.0 + 1e-9:
                    w_macro_base += w_i
                    sources_base.add(src)

        w_base_daily = (w_macro_base / 3.0) + (w_orbit_base / 11.0)

        # Dual-Track Breakout Surge Multiplier
        is_macro_breakout = len(sources_0_24h) >= 4 and len(sources_base) <= 1
        is_orbit_breakout = len(orbit_sources_0_72h) >= 2 and len(orbit_sources_72_336h) == 0
        b_surge = 1.25 if (is_macro_breakout or is_orbit_breakout) else 1.00

        velocity = min(
            8.0,
            b_surge * (((w_recent + 0.25) / (w_base_daily + 0.25)) ** 0.6),
        )

        # Orbit Resonance Multiplier
        if len(orbit_sources) >= 1 and len(macro_sources) >= 1:
            m_orbit = 1.50
        elif len(orbit_sources) >= 2 and len(macro_sources) == 0:
            m_orbit = 1.35
        elif len(orbit_sources) == 0 and len(macro_sources_0_24h) >= 5:
            m_orbit = 1.00
        else:
            m_orbit = 0.65

        idx_arr = np.asarray(c_indices, dtype=np.int64)
        sub_sim = sim_matrix[np.ix_(idx_arr, idx_arr)]
        s_intra = float(np.sum(sub_sim)) / float(len(c_indices) ** 2)

        unique_sources_sorted = tuple(sorted(all_sources))
        unique_count = len(unique_sources_sorted)
        commission_score = (
            velocity * m_orbit * s_intra * math.log(1.0 + float(unique_count))
        )

        scored_clusters.append(
            ScoredCluster(
                cluster_id=f"cluster_{k_idx}",
                items=cluster_items,
                cluster_keywords=keywords_per_cluster[k_idx],
                silhouette_score=silhouettes_per_cluster[k_idx],
                w_recent=float(w_recent),
                w_base_daily=float(w_base_daily),
                breakout_surge=float(b_surge),
                velocity=float(velocity),
                orbit_resonance=float(m_orbit),
                s_intra=float(s_intra),
                unique_sources=unique_sources_sorted,
                unique_sources_count=unique_count,
                commission_score=float(commission_score),
                threshold_used=float(chosen_tau),
            )
        )

    scored_clusters.sort(
        key=lambda c: (
            -c.commission_score,
            -c.w_recent,
            -c.s_intra,
            -c.unique_sources_count,
            c.cluster_id,
        )
    )
    return scored_clusters


def score_and_cluster_signals(
    cache_items: Sequence[Any],
    *,
    now_utc: Optional[datetime | str] = None,
    anchored_hashes_14d: Optional[Iterable[str]] = None,
) -> List[ScoredCluster]:
    """Run Stage 1-2 stratified windowing (14d Orbit / 4d Macro) followed by average-linkage clustering."""
    window_items = load_asymmetric_window(
        cache_items=cache_items,
        anchored_hashes_14d=anchored_hashes_14d,
        now_utc=now_utc,
    )
    return cluster_and_score(
        items=window_items,
        now_utc=now_utc,
        anchored_hashes_14d=anchored_hashes_14d,
    )



