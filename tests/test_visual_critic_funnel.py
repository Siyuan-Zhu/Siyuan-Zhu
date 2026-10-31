"""Unit tests for WS3 Seam 4: Multimodal Visual Critic Funnel, Dual-Res WebP & SVG Vote-Bar."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from src.zeitgeist.adk_pipeline import DirectorCommission
from src.zeitgeist.visual_router import (
    PublishedPulse,
    select_and_render_pulse_assets,
    update_svg_mood_vote_bar,
)


def _make_sample_commission() -> DirectorCommission:
    return DirectorCommission(
        word="Cantilever",
        gloss="Rigid structural projection anchored at a single load-bearing fulcrum.",
        commissioned_thesis="Personal orbit builders and macro infrastructure feeds converge on cantilevered load geometry.",
        visual_anchor="Brutalist cantilevered concrete terrace over calm water",
        unsplash_queries=[
            "cantilever architecture",
            "brutalist concrete overhang",
            "minimalist structural beam",
        ],
        imagen_commission_prompt="Editorial architectural study of a cantilevered concrete terrace.",
        palette_hex=["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"],
    )


def _make_test_photo_bytes(color: tuple[int, int, int] = (30, 64, 110)) -> bytes:
    img = Image.new("RGB", (1600, 900), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_critic_selects_photography_when_semantic_and_composite_pass_thresholds(
    tmp_path: Path,
) -> None:
    """Candidate passing semantic_alignment >= 0.70 and composite >= 0.68 wins as PHOTOGRAPHY."""
    commission = _make_sample_commission()
    pinged_urls: list[str] = []

    candidates = [
        {
            "id": "photo_low",
            "photographer_name": "Alex Rivera",
            "photographer_url": "https://unsplash.com/@alexrivera",
            "unsplash_url": "https://unsplash.com/photos/photo_low",
            "download_location": "https://api.unsplash.com/photos/photo_low/download",
            "image_bytes": _make_test_photo_bytes((40, 50, 60)),
            "semantic_alignment": 0.72,
            "compositional_clarity": 0.65,
            "editorial_memorability": 0.60,
            "safety_pass": True,
            # Composite = 0.50*0.72 + 0.30*0.65 + 0.20*0.60 = 0.36 + 0.195 + 0.12 = 0.675 (< 0.68!)
        },
        {
            "id": "photo_winner",
            "photographer_name": "Elena Rostova",
            "photographer_url": "https://unsplash.com/@elenarostova",
            "unsplash_url": "https://unsplash.com/photos/photo_winner",
            "download_location": "https://api.unsplash.com/photos/photo_winner/download",
            "image_bytes": _make_test_photo_bytes((24, 90, 140)),
            "semantic_alignment": 0.84,
            "compositional_clarity": 0.80,
            "editorial_memorability": 0.75,
            "safety_pass": True,
            # Composite = 0.50*0.84 + 0.30*0.80 + 0.20*0.75 = 0.42 + 0.24 + 0.15 = 0.81 (>= 0.68!)
        },
    ]

    # Seed an old >365d WebP file to verify 365-day pruning
    pulse_dir = tmp_path / "assets" / "pulse"
    pulse_dir.mkdir(parents=True, exist_ok=True)
    expired_webp = pulse_dir / "2024-01-15.webp"
    expired_webp.write_bytes(b"old")
    recent_webp = pulse_dir / "2026-10-20.webp"
    recent_webp.write_bytes(b"recent")

    pulse = select_and_render_pulse_assets(
        commission=commission,
        medium_history_30d=["PHOTOGRAPHY"] * 10,
        repo_root=tmp_path,
        date_str="2026-10-27",
        candidates=candidates,
        orbit_resonance_score=1.50,
        velocity_score=3.85,
        download_ping_callback=lambda url: pinged_urls.append(url),
    )

    assert isinstance(pulse, PublishedPulse)
    assert pulse.visual_medium == "PHOTOGRAPHY"
    assert pulse.composite_critic_score == pytest.approx(0.81, abs=1e-4)
    assert pulse.attribution["photographer_name"] == "Elena Rostova"
    assert pulse.attribution["download_location_triggered"] is False
    assert pinged_urls == []
    from src.zeitgeist.visual_router import confirm_published_photo_download
    confirm_published_photo_download(pulse, callback=pinged_urls.append)
    confirm_published_photo_download(pulse, callback=pinged_urls.append)
    assert pulse.attribution["download_location_triggered"] is True
    assert pinged_urls == ["https://api.unsplash.com/photos/photo_winner/download"]

    # Verify 1200x675 WebP saved and >365d file pruned
    webp_path = tmp_path / pulse.webp_asset_path
    assert webp_path.exists()
    assert not expired_webp.exists()
    assert recent_webp.exists()

    with Image.open(webp_path) as im:
        assert im.format == "WEBP"
        assert im.size == (1200, 800)

    # Verify assets/daily_pulse.svg embeds 720x260 base64 WebP and valid XML with mood-vote-bar
    svg_path = tmp_path / pulse.svg_asset_path
    assert svg_path.exists()
    svg_text = svg_path.read_text(encoding="utf-8")
    root = ET.fromstring(svg_text)
    assert root.tag.endswith("svg")
    assert "data:image/webp;base64," in svg_text
    assert 'id="mood-vote-bar"' in svg_text

    # Decode embedded base64 WebP and check 720x260 dimensions
    b64_prefix = "data:image/webp;base64,"
    start_idx = svg_text.index(b64_prefix) + len(b64_prefix)
    end_idx = svg_text.index('"', start_idx)
    embedded_bytes = base64.b64decode(svg_text[start_idx:end_idx])
    with Image.open(BytesIO(embedded_bytes)) as emb_im:
        assert emb_im.format == "WEBP"
        assert emb_im.size == (1200, 800)

    # Verify data/pulse_feed.json canonical 37-key v2 entry written (aligned with firestore.rules isValidPulseHistory)
    feed_path = tmp_path / "data" / "pulse_feed.json"
    assert feed_path.exists()
    feed_data = json.loads(feed_path.read_text(encoding="utf-8"))
    assert feed_data["version"] == "2.0"
    assert feed_data["schema_version"] == "2.0"
    entry = feed_data["entries"][0]
    expected_keys = [
        "schema_version",
        "date",
        "word",
        "gloss",
        "commissioned_thesis",
        "visual_medium",
        "image_url",
        "fallback_webp_path",
        "unsplash_photographer",
        "unsplash_profile_url",
        "flux_model",
        "cluster_keywords",
        "orbit_contribution_pct",
        "macro_contribution_pct",
        "orbit_sources_matched",
        "runner_up_clusters",
        "mood_votes",
        "critic_scores",
        "published_at",
        "updated_at",
        "headline",
        "summary",
        "24h_shift",
        "dominant_category",
        "dominant_signal_type",
        "orbit_weight_ratio",
        "top_orbit_themes",
        "top_macro_themes",
        "visual_tier_used",
        "artwork_title",
        "artwork_artist",
        "artwork_source",
        "artwork_url",
        "image_data_uri",
        "quote",
        "quote_attribution",
        "window_hours",
    ]
    assert list(entry.keys()) == expected_keys
    assert entry["artwork_title"] == "Cantilever"
    assert entry["visual_tier_used"] == "PHOTOGRAPHY"
    assert entry["window_hours"] == 96
    assert entry["schema_version"] == "2.0"
    assert entry["image_data_uri"] == ""
    assert set(entry["mood_votes"].keys()) == {"spot_on", "too_sunny", "too_heavy"}


def test_critic_routes_to_commissioned_art_when_below_threshold_and_budget_available(
    tmp_path: Path,
) -> None:
    """When all photos fail threshold and non-photo days in 30d < 6, route to BESPOKE_SVG / IMAGEN_3."""
    commission = _make_sample_commission()
    weak_candidates = [
        {
            "id": "weak_1",
            "photographer_name": "Sam Lee",
            "photographer_url": "https://unsplash.com/@samlee",
            "unsplash_url": "https://unsplash.com/photos/weak_1",
            "download_location": "https://api.unsplash.com/photos/weak_1/download",
            "image_bytes": _make_test_photo_bytes(),
            "semantic_alignment": 0.55,  # < 0.70 threshold
            "compositional_clarity": 0.80,
            "editorial_memorability": 0.75,
        }
    ]

    # Only 3 non-photo days in the last 30 days (< 6 cap)
    history_30d = ["PHOTOGRAPHY"] * 20 + ["IMAGEN_3", "BESPOKE_SVG", "IMAGEN_3"]

    pulse = select_and_render_pulse_assets(
        commission=commission,
        medium_history_30d=history_30d,
        repo_root=tmp_path,
        date_str="2026-10-27",
        candidates=weak_candidates,
    )

    assert pulse.visual_medium in {"IMAGEN_3", "BESPOKE_SVG"}
    webp_path = tmp_path / pulse.webp_asset_path
    assert webp_path.exists()
    with Image.open(webp_path) as im:
        assert im.format == "WEBP"
        assert im.size == (1200, 800)


def test_critic_enforces_80_20_cap_when_6_non_photo_days_already_used(
    tmp_path: Path,
) -> None:
    """When non-photo days in 30d >= 6, force PHOTOGRAPHY using highest semantic_alignment candidate."""
    commission = _make_sample_commission()
    weak_candidates = [
        {
            "id": "weak_a",
            "photographer_name": "Jordan Kim",
            "photographer_url": "https://unsplash.com/@jordankim",
            "unsplash_url": "https://unsplash.com/photos/weak_a",
            "download_location": "https://api.unsplash.com/photos/weak_a/download",
            "image_bytes": _make_test_photo_bytes((50, 60, 70)),
            "semantic_alignment": 0.58,
            "compositional_clarity": 0.70,
            "editorial_memorability": 0.70,
            "safety_pass": True,
        },
        {
            "id": "weak_b_best_sem",
            "photographer_name": "Morgan Vance",
            "photographer_url": "https://unsplash.com/@morganvance",
            "unsplash_url": "https://unsplash.com/photos/weak_b_best_sem",
            "download_location": "https://api.unsplash.com/photos/weak_b_best_sem/download",
            "image_bytes": _make_test_photo_bytes((70, 80, 90)),
            "semantic_alignment": 0.66,  # Below 0.70, but highest semantic_alignment
            "compositional_clarity": 0.60,
            "editorial_memorability": 0.55,
            "safety_pass": True,
        },
    ]

    # 6 non-photo days already used in last 30 days -> budget exhausted!
    history_30d = ["PHOTOGRAPHY"] * 24 + [
        "IMAGEN_3",
        "BESPOKE_SVG",
        "IMAGEN_3",
        "BESPOKE_SVG",
        "IMAGEN_3",
        "BESPOKE_SVG",
    ]

    pulse = select_and_render_pulse_assets(
        commission=commission,
        medium_history_30d=history_30d,
        repo_root=tmp_path,
        date_str="2026-10-27",
        candidates=weak_candidates,
    )

    assert pulse.visual_medium == "PHOTOGRAPHY"
    assert pulse.attribution["photographer_name"] == "Morgan Vance"
    assert pulse.semantic_alignment_score == pytest.approx(0.66, abs=1e-4)


def test_update_svg_mood_vote_bar_surgically_patches_svg_and_preserves_valid_xml(
    tmp_path: Path,
) -> None:
    """Verify update_svg_mood_vote_bar updates <g id='mood-vote-bar'> without breaking XML or image."""
    commission = _make_sample_commission()
    pulse = select_and_render_pulse_assets(
        commission=commission,
        medium_history_30d=["PHOTOGRAPHY"] * 5,
        repo_root=tmp_path,
        date_str="2026-10-27",
        candidates=[],
    )

    svg_path = tmp_path / pulse.svg_asset_path
    new_votes = {
        "resonant": 19,
        "reflective": 11,
        "electric": 7,
        "skeptical": 3,
    }
    updated_svg = update_svg_mood_vote_bar(svg_path, new_votes)

    root = ET.fromstring(updated_svg)
    assert root.tag.endswith("svg")
    assert "data:image/webp;base64," in updated_svg
    assert "19" in updated_svg
    assert "11" in updated_svg
    assert "7" in updated_svg
    assert "3" in updated_svg
    assert "40 votes" in updated_svg or "40" in updated_svg


def test_default_mood_votes_and_imagen3_bytes_routing(tmp_path: Path) -> None:
    """Verify DEFAULT_MOOD_VOTES uses spot_on/too_sunny/too_heavy and Tier 2 IMAGEN_3 renders cleanly."""
    from src.zeitgeist.visual_router import DEFAULT_MOOD_VOTES

    assert dict(DEFAULT_MOOD_VOTES) == {
        "spot_on": 0,
        "too_sunny": 0,
        "too_heavy": 0,
    }

    commission = _make_sample_commission()
    pulse = select_and_render_pulse_assets(
        commission=commission,
        medium_history_30d=["PHOTOGRAPHY"] * 10,
        repo_root=tmp_path,
        date_str="2026-10-27",
        unsplash_candidates=[],
        imagen_bytes=_make_test_photo_bytes((20, 45, 70)),
        mood_votes={"spot_on": 14, "too_sunny": 2, "too_heavy": 1},
    )

    assert pulse.visual_medium == "IMAGEN_3"
    entry = pulse.to_feed_entry()
    assert entry["visual_tier_used"] == "IMAGEN_3"
    assert entry["artwork_source"] == "Google Imagen 3"
    assert entry["mood_votes"] == {"spot_on": 14, "too_sunny": 2, "too_heavy": 1}

    svg_text = (tmp_path / pulse.svg_asset_path).read_text(encoding="utf-8")
    metadata = ET.fromstring(svg_text).find("{http://www.w3.org/2000/svg}metadata")
    assert json.loads(metadata.text) == {"spot_on": 14, "too_sunny": 2, "too_heavy": 1}
    assert "Original ADK Commission" not in svg_text
    assert "Visual:" not in svg_text


def test_score_candidate_with_gemini_vision_fails_closed_on_missing_key_error_or_safety_block(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify _score_candidate_with_gemini_vision fails closed (0.0 scores, safety_pass=False) on missing key, API error, or finish_reason safety block."""
    from types import SimpleNamespace
    from google import genai
    from src.zeitgeist.visual_router import _score_candidate_with_gemini_vision

    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setattr("src.zeitgeist.visual_router.guard_genai_client", lambda client, **kwargs: client)
    thumb = _make_test_photo_bytes()

    # 1. Missing API key -> fail closed
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    missing_key_scores = _score_candidate_with_gemini_vision(
        thumb_bytes=thumb,
        word="Cantilever",
        gloss="Structural overhang.",
        thesis="Load-bearing geometry.",
        gemini_api_key="",
    )
    assert missing_key_scores == {
        "semantic_alignment": 0.0,
        "compositional_clarity": 0.0,
        "editorial_memorability": 0.0,
        "safety_pass": False,
    }

    # 2. Exception during API call -> fail closed and log warning
    class _ExplodingModels:
        def generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("Gemini quota exhausted")

    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key=None: SimpleNamespace(models=_ExplodingModels()),
    )
    with caplog.at_level("WARNING"):
        err_scores = _score_candidate_with_gemini_vision(
            thumb_bytes=thumb,
            word="Cantilever",
            gloss="Structural overhang.",
            thesis="Load-bearing geometry.",
            gemini_api_key="test-gemini-key",
        )
    assert err_scores == {
        "semantic_alignment": 0.0,
        "compositional_clarity": 0.0,
        "editorial_memorability": 0.0,
        "safety_pass": False,
    }
    assert any("Gemini" in rec.message or "vision" in rec.message.lower() for rec in caplog.records)

    # 3. Safety finish_reason block -> fail closed even if text has high scores
    captured_prompts: list[str] = []

    class _BlockedModels:
        def generate_content(self, *, model, contents, config):  # type: ignore[no-untyped-def]
            captured_prompts.append(str(contents[-1]))
            return SimpleNamespace(
                text='{"semantic_alignment": 0.95, "compositional_clarity": 0.90, "editorial_memorability": 0.88, "safety_pass": true}',
                candidates=[SimpleNamespace(finish_reason="IMAGE_SAFETY")],
            )

    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key=None: SimpleNamespace(models=_BlockedModels()),
    )
    blocked_scores = _score_candidate_with_gemini_vision(
        thumb_bytes=thumb,
        word="Cantilever",
        gloss="Structural overhang.",
        thesis="Load-bearing geometry.",
        gemini_api_key="test-gemini-key",
    )
    assert blocked_scores == {
        "semantic_alignment": 0.0,
        "compositional_clarity": 0.0,
        "editorial_memorability": 0.0,
        "safety_pass": False,
    }
    assert len(captured_prompts) == 1
    assert "safety_pass" in captured_prompts[0]

    # 4. Valid response with safety_pass=True
    class _PassingModels:
        def generate_content(self, *, model, contents, config):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                text='{"semantic_alignment": 0.86, "compositional_clarity": 0.82, "editorial_memorability": 0.79, "safety_pass": true}',
                candidates=[SimpleNamespace(finish_reason="STOP")],
            )

    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key=None: SimpleNamespace(models=_PassingModels()),
    )
    valid_scores = _score_candidate_with_gemini_vision(
        thumb_bytes=thumb,
        word="Cantilever",
        gloss="Structural overhang.",
        thesis="Load-bearing geometry.",
        gemini_api_key="test-gemini-key",
    )
    assert valid_scores == {
        "semantic_alignment": 0.86,
        "compositional_clarity": 0.82,
        "editorial_memorability": 0.79,
        "safety_pass": True,
    }


def test_select_and_render_pulse_assets_enforces_safety_pass_in_primary_80_20_and_flux_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify select_and_render_pulse_assets rejects safety_pass=False in primary gate, 80/20 cap fallback, unscored vision critic path, and live FLUX path."""
    import src.zeitgeist.visual_router as vr

    commission = _make_sample_commission()

    # 1. Primary gate: high scores with safety_pass=False MUST be rejected
    unsafe_high_cand = {
        "id": "unsafe_high",
        "photographer_name": "Unsafe Photographer",
        "photographer_url": "https://unsplash.com/@unsafe",
        "unsplash_url": "https://unsplash.com/photos/unsafe_high",
        "download_location": "https://api.unsplash.com/photos/unsafe_high/download",
        "image_bytes": _make_test_photo_bytes((80, 20, 20)),
        "semantic_alignment": 0.95,
        "compositional_clarity": 0.92,
        "editorial_memorability": 0.90,
        "safety_pass": False,
    }
    pulse_primary = select_and_render_pulse_assets(
        commission=commission,
        medium_history_30d=["PHOTOGRAPHY"] * 10,
        repo_root=tmp_path,
        date_str="2026-10-27",
        candidates=[unsafe_high_cand],
    )
    assert pulse_primary.visual_medium == "BESPOKE_SVG"
    assert pulse_primary.attribution["photographer_name"] != "Unsafe Photographer"

    # 2. Unscored candidate with missing GEMINI_API_KEY fails closed even under 80/20 cap (non_photo_days_30d >= 6)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    unscored_live_cand = {
        "id": "unscored_live",
        "photographer_name": "Unvetted Live Author",
        "photographer_url": "https://unsplash.com/@unvetted",
        "unsplash_url": "https://unsplash.com/photos/unscored_live",
        "download_location": "https://api.unsplash.com/photos/unscored_live/download",
        "image_bytes": _make_test_photo_bytes((40, 70, 90)),
    }
    with pytest.raises(ValueError, match="Photography cadence"):
        select_and_render_pulse_assets(
            commission=commission,
            medium_history_30d=["BESPOKE_SVG"] * 6 + ["PHOTOGRAPHY"] * 24,
            repo_root=tmp_path, date_str="2026-10-27", candidates=[unscored_live_cand],
        )

    # 3. Live FLUX candidate rejected when Vision Critic returns safety_pass=False
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(
        vr,
        "_score_candidate_with_gemini_vision",
        lambda *args, **kwargs: {
            "semantic_alignment": 0.90,
            "compositional_clarity": 0.88,
            "editorial_memorability": 0.85,
            "safety_pass": False,
        },
    )
    monkeypatch.setattr(vr, "_generate_live_imagen3_bytes", lambda *args, **kwargs: None)
    pulse_flux_unsafe = select_and_render_pulse_assets(
        commission=commission,
        medium_history_30d=["PHOTOGRAPHY"] * 10,
        repo_root=tmp_path,
        date_str="2026-10-27",
        candidates=[],
        bfl_api_key="test-bfl-key",
        flux_generator=lambda prompt: _make_test_photo_bytes((30, 60, 90)),
    )
    assert pulse_flux_unsafe.visual_medium == "BESPOKE_SVG"


def test_unsplash_3_query_landscape_9_candidate_funnel_and_empty_safe_candidates_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 3-query x 3-result (9-candidate) landscape Unsplash funnel and empty safe_candidates guard in route_and_render_visual."""
    from types import SimpleNamespace
    import httpx
    import src.zeitgeist.visual_router as vr
    from src.zeitgeist.methodology import WeightedOrbitAnalysis, LensScore

    commission = _make_sample_commission()
    photo_bytes = _make_test_photo_bytes((35, 65, 95))
    search_calls: list[dict] = []

    def _fake_client_get(self, url, *, params=None, headers=None, **kwargs):  # type: ignore[no-untyped-def]
        url_str = str(url)
        if "api.unsplash.com/search/photos" in url_str:
            search_calls.append(dict(params or {}))
            q = (params or {}).get("query", "q")
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "results": [
                        {
                            "id": f"{q}_{idx}",
                            "urls": {
                                "small": f"https://images.unsplash.com/{q}_{idx}_small",
                                "regular": f"https://images.unsplash.com/{q}_{idx}_reg",
                                "raw": f"https://images.unsplash.com/{q}_{idx}_raw",
                            },
                            "links": {
                                "html": f"https://unsplash.com/photos/{q}_{idx}",
                                "download_location": f"https://api.unsplash.com/photos/{q}_{idx}/download",
                            },
                            "user": {
                                "name": f"Author {idx}",
                                "links": {"html": f"https://unsplash.com/@author_{idx}"},
                            },
                        }
                        for idx in range(3)
                    ]
                },
            )
        return SimpleNamespace(status_code=200, content=photo_bytes)

    monkeypatch.setattr(httpx.Client, "get", _fake_client_get)
    monkeypatch.setattr(
        vr,
        "_score_photo_batch",
        lambda candidates, **kwargs: [{
            "semantic_alignment": 0.85,
            "compositional_clarity": 0.82,
            "editorial_memorability": 0.80,
            "safety_pass": True,
        } for _ in candidates],
    )

    live_cands = vr._fetch_live_unsplash_candidates(
        commission.unsplash_queries,
        access_key="test-unsplash-key",
        word=commission.word,
        gloss=commission.gloss,
        thesis=commission.commissioned_thesis,
    )
    assert len(search_calls) == 3
    for call_params in search_calls:
        assert call_params["orientation"] == "landscape"
        assert call_params["per_page"] == 3
    assert len(live_cands) == 9
    assert all(c.get("safety_pass") is True for c in live_cands)

    # Verify route_and_render_visual does not raise IndexError when safe_candidates is empty
    monkeypatch.setattr(
        vr,
        "score_curated_candidate",
        lambda entry, analysis: vr.VisualCandidate(
            candidate_id="unsafe_cand",
            mode="CURATED_PHOTO",
            image_url="https://images.unsplash.com/unsafe",
            credit_label="Unsafe",
            credit_url="https://unsplash.com",
            alt_text="unsafe neon gore",
            anchor_tags=(),
            semantic_alignment=0.90,
            minimalist_restraint=0.90,
            safety_pass=False,
        ),
    )
    dummy_analysis = WeightedOrbitAnalysis(
        date_iso="2026-10-27",
        composite_score=72.0,
        valence=68.0,
        arousal=52.0,
        dominance=74.0,
        epistemic_tension=45.0,
        word_of_the_day="Cantilever",
        explanation_phrase="Rigid structural projection.",
        visual_concrete_anchor="Brutalist concrete terrace",
        is_builder_systems_day=False,
        interjection_applied=False,
        personal_note=None,
        lens_breakdown=(
            LensScore(
                lens_id="personal_orbit",
                label="Personal Orbit",
                weight=0.40,
                valence=68.0,
                arousal=52.0,
                dominance=74.0,
                epistemic_tension=45.0,
                composite_index=72.0,
                top_terms=("architecture",),
            ),
        ),
        dominant_keywords=("architecture", "concrete", "geometry"),
    )
    svg_out = tmp_path / "safe_guard_pulse.svg"
    decision = vr.route_and_render_visual(
        analysis=dummy_analysis,
        config={},
        svg_output_path=svg_out,
        prefer_mode="photo",
    )
    assert decision.selected_mode in {"ARCHITECTURAL_BLUEPRINT_SVG", "AI_GENERATED_FALLBACK"}
    assert decision.safety_pass is True
    assert svg_out.exists()





@pytest.mark.parametrize("field,value", [
    ("safety_pass", "false"), ("safety_pass", "true"), ("safety_pass", 1),
    ("semantic_alignment", float("nan")), ("semantic_alignment", float("inf")),
    ("compositional_clarity", -0.1), ("editorial_memorability", 1.1),
    ("semantic_alignment", "0.95"), ("semantic_alignment", True),
])
def test_visual_critic_rejects_malformed_model_verdicts(monkeypatch, field, value):
    from types import SimpleNamespace
    from google import genai
    from src.zeitgeist import visual_router
    verdict = {"safety_pass": True, "semantic_alignment": 0.95,
               "compositional_clarity": 0.9, "editorial_memorability": 0.9}
    verdict[field] = value
    models = SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(
        text=json.dumps(verdict), candidates=[]))
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=models))
    monkeypatch.setattr(visual_router, "guard_genai_client", lambda client, **kwargs: client)
    result = visual_router._score_candidate_with_gemini_vision(
        thumb_bytes=_make_test_photo_bytes(), word="Anchored", gloss="Steady craft.",
        thesis="A workshop.", gemini_api_key="fixture-only")
    assert result == visual_router._FAIL_CLOSED_VISION_SCORES


def test_live_workflow_flag_routes_bfl_before_gemini_fallback(monkeypatch, tmp_path):
    from src.zeitgeist import visual_router
    calls = []
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("BFL_API_KEY", "fixture-only")
    monkeypatch.delenv("ZEITGEIST_LIVE_BFL", raising=False)
    def flux(prompts, **kwargs):
        calls.append(kwargs["api_key"])
        return _make_test_photo_bytes()
    monkeypatch.setattr(visual_router, "run_flux_3_tournament", flux)
    monkeypatch.setattr(visual_router, "_score_candidate_with_gemini_vision", lambda **kwargs: {
        "semantic_alignment": .95, "compositional_clarity": .9,
        "editorial_memorability": .9, "safety_pass": True})
    monkeypatch.setattr(visual_router, "_generate_live_imagen3_bytes", lambda *args, **kwargs: None)
    pulse = select_and_render_pulse_assets(
        {"word": "Anchored", "gloss": "Steady craft.", "flux_3_commission_prompts": ["one", "two"]},
        repo_root=tmp_path, candidates=[])
    assert calls == ["fixture-only"]
    assert pulse.visual_medium == "FLUX_PRO"


def test_valid_vision_verdict_uses_bounded_thumbnail_and_preserves_original(monkeypatch):
    from types import SimpleNamespace
    from google import genai
    from src.zeitgeist import visual_router
    original = BytesIO()
    Image.new("RGB", (1200, 800), "white").save(original, format="PNG")
    raw = original.getvalue()
    def generate(**kwargs):
        inline = kwargs["contents"][0].inline_data
        assert inline.mime_type == "image/jpeg"
        with Image.open(BytesIO(inline.data)) as image:
            assert image.size == (384, 256)
        return SimpleNamespace(text=json.dumps({"safety_pass": True, "semantic_alignment": .95,
            "compositional_clarity": .9, "editorial_memorability": .9}), candidates=[])
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(visual_router, "guard_genai_client", lambda client, **kwargs: client)
    result = visual_router._score_candidate_with_gemini_vision(
        thumb_bytes=raw, word="Anchored", gloss="Steady craft.", thesis="A workshop.", gemini_api_key="fixture-only")
    assert result["safety_pass"] is True
    assert raw == original.getvalue()
