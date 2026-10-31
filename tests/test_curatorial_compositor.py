"""Tests for the 5-Archetype Curatorial Compositor, Real Photography Priority, and EGM Resolutions."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from scripts.process_visitor_event import parse_cli_args, process_event
from src.leaderboard.elo_ledger import BOT_INITIAL_ELO, DEFAULT_BOT_ELO
from src.state_store import InMemoryStateStore
from src.zeitgeist.adk_pipeline import DirectorCommission, ZeitgeistADKOrchestrator
from src.zeitgeist.collectors import compute_url_hash, compute_url_hash_128
from src.zeitgeist.curatorial_compositor import (
    ALL_ARCHETYPES,
    QUADRANT_NAMES,
    compute_quietest_quadrant,
    render_curatorial_plate_pair,
    render_curatorial_plates,
    resolve_effective_quadrant,
    select_curatorial_archetype,
)
from src.zeitgeist.visual_router import (
    _generate_live_flux_bytes,
    run_flux_3_tournament,
    select_and_render_pulse_assets,
)


def _make_quadrant_test_image(calm_quadrant: str = "upper_right") -> Image.Image:
    """Create a 1200x800 synthetic photograph with one calm quadrant and three high-frequency quadrants."""
    img = Image.new("RGB", (1200, 800), (235, 232, 224))
    draw = ImageDraw.Draw(img)
    quad_boxes = {
        "upper_left": (0, 0, 600, 400),
        "upper_right": (600, 0, 1200, 400),
        "lower_left": (0, 400, 600, 800),
        "lower_right": (600, 400, 1200, 800),
    }
    for q_name, (x0, y0, x1, y1) in quad_boxes.items():
        if q_name == calm_quadrant:
            draw.rectangle([x0, y0, x1, y1], fill=(242, 239, 232))
            continue
        # Draw dense high-contrast crosshatching in non-calm quadrants
        for x in range(x0, x1, 6):
            draw.line([(x, y0), (x, y1)], fill=(15, 20, 28), width=2)
        for y in range(y0, y1, 6):
            draw.line([(x0, y), (x1, y)], fill=(250, 248, 244), width=2)
    return img


def test_compute_quietest_quadrant_identifies_calm_quadrant() -> None:
    for target_quad in QUADRANT_NAMES:
        img = _make_quadrant_test_image(calm_quadrant=target_quad)
        chosen_quad, min_density, lum = compute_quietest_quadrant(img)
        assert chosen_quad == target_quad
        assert min_density < 0.05
        assert 0.5 < lum <= 1.0


def test_select_curatorial_archetype_rotation_and_anti_repetition() -> None:
    # 1. Calm image (< 0.14 edge density) defaults to POETIC_CINEMA when history is empty
    a1 = select_curatorial_archetype(
        min_edge_density=0.04,
        breakout_surge=False,
        velocity=2.1,
        archetype_history_30d=[],
    )
    assert a1 == "POETIC_CINEMA"

    # 2. Consecutive-day anti-repetition never repeats yesterday's archetype
    a2 = select_curatorial_archetype(
        min_edge_density=0.04,
        breakout_surge=False,
        velocity=2.1,
        archetype_history_30d=["POETIC_CINEMA"],
    )
    assert a2 != "POETIC_CINEMA"
    assert a2 in ALL_ARCHETYPES

    # 3. High edge density across all quadrants (>= 0.14) skips POETIC_CINEMA and selects frame-safe layout
    a_dense = select_curatorial_archetype(
        min_edge_density=0.22,
        breakout_surge=False,
        velocity=2.2,
        archetype_history_30d=[],
    )
    assert a_dense in {"LITERARY_JOURNAL", "STUDIO_MONOGRAPH", "KYOTO_TANKOBON"}

    # 4. Breakout surge (>= 4.0 sigma velocity) unlocks CONCRETE_POETRY
    a_surge = select_curatorial_archetype(
        min_edge_density=0.18,
        breakout_surge=True,
        velocity=4.2,
        archetype_history_30d=["POETIC_CINEMA", "LITERARY_JOURNAL"],
    )
    assert a_surge == "CONCRETE_POETRY"
    assert resolve_effective_quadrant(a_surge, "upper_left") == "full_canvas"


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_render_curatorial_plate_pair_all_five_archetypes(
    tmp_path: Path, archetype: str
) -> None:
    img = _make_quadrant_test_image("upper_left")
    light_out = tmp_path / f"light_{archetype.lower()}.webp"
    dark_out = tmp_path / f"dark_{archetype.lower()}.webp"
    quad = resolve_effective_quadrant(archetype, "upper_left")

    light_img, dark_img = render_curatorial_plate_pair(
        img,
        word="Still",
        gloss="Some things arrive only when we stop.",
        archetype=archetype,
        quadrant=quad,
        date_str="2026-10-28",
        credit_line="Photo by Test Photographer on Unsplash",
        light_out_path=light_out,
        dark_out_path=dark_out,
    )
    assert light_img.size == (1200, 800)
    assert dark_img.size == (1200, 800)
    assert light_out.exists() and light_out.stat().st_size > 1000
    assert dark_out.exists() and dark_out.stat().st_size > 1000


def test_real_photography_priority_and_all_archetype_compatibility(tmp_path: Path) -> None:
    """Verify real photography (~80% priority) is selected first and renders cleanly across rotations."""
    photo_img = _make_quadrant_test_image("upper_right")
    buf = BytesIO()
    photo_img.save(buf, format="JPEG", quality=90)
    photo_bytes = buf.getvalue()

    commission = DirectorCommission(
        word="Solastalgia",
        gloss="Homesickness felt while still at home as familiar landscapes shift.",
        commissioned_thesis="Quiet coastal bluff horizon under evening mist.",
        unsplash_queries=["coastal bluff mist", "minimalist horizon", "quiet sea"],
        imagen_commission_prompt="Minimalist 3:2 coastal horizon in twilight fog",
    )
    assert len(commission.flux_3_commission_prompts) == 3

    # Even with borderline candidate scores, when 6 non-photo days occurred in 30d (80% cap), PHOTOGRAPHY wins
    candidates = [
        {
            "photographer_name": "Elena Rostova",
            "photographer_url": "https://unsplash.com/@erostova",
            "unsplash_url": "https://unsplash.com/photos/bluff-1",
            "download_location": "https://api.unsplash.com/photos/bluff-1/download",
            "semantic_alignment": 0.69,
            "compositional_clarity": 0.66,
            "editorial_memorability": 0.65,
            "safety_pass": True,
            "image_bytes": photo_bytes,
        }
    ]
    pinged: list[str] = []
    published = select_and_render_pulse_assets(
        commission,
        medium_history_30d=["FLUX_PRO"] * 6 + ["PHOTOGRAPHY"] * 24,
        repo_root=tmp_path,
        date_str="2026-10-28",
        candidates=candidates,
        download_ping_callback=pinged.append,
        archetype_history_30d=["LITERARY_JOURNAL"],
    )
    assert published.visual_medium == "PHOTOGRAPHY"
    assert published.curatorial_archetype == "POETIC_CINEMA"
    assert published.text_quadrant == "upper_right"
    assert pinged == []
    from src.zeitgeist.visual_router import confirm_published_photo_download
    confirm_published_photo_download(published, callback=pinged.append)
    assert len(pinged) == 1
    assert (tmp_path / published.light_webp_asset_path).exists()
    assert (tmp_path / published.dark_webp_asset_path).exists()
    with Image.open(tmp_path / published.light_webp_asset_path) as im_light:
        assert im_light.size == (1200, 800)
    with Image.open(tmp_path / published.webp_asset_path) as im_archive:
        assert im_archive.size == (1200, 800)

    # Canonical 37-key v2 feed entry by default, 39-key when curatorial metadata is requested
    assert len(published.to_feed_entry()) == 37
    enriched_entry = published.to_feed_entry(include_curatorial_metadata=True)
    assert len(enriched_entry) == 39
    assert enriched_entry["curatorial_archetype"] == "POETIC_CINEMA"
    assert enriched_entry["text_quadrant"] == "upper_right"


def test_flux_3_tournament_concurrent_selection(tmp_path: Path) -> None:
    """Verify run_flux_3_tournament dispatches 3 prompts in parallel and selects the best plate."""
    prompts = [
        "Prompt A: architectural shadow study",
        "Prompt B: warm timber luthier workbench",
        "Prompt C: brutalist concrete courtyard",
    ]
    called_prompts: list[str] = []

    def _fake_flux_gen(prompt: str) -> bytes:
        called_prompts.append(prompt)
        img = _make_quadrant_test_image("upper_left")
        out = BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()

    winner_bytes = run_flux_3_tournament(
        prompts,
        word="Adagio",
        gloss="At ease; slow and stately craft.",
        thesis="Deliberate precision over hurried scale.",
        generator_fn=_fake_flux_gen,
    )
    assert winner_bytes is not None and len(winner_bytes) > 500
    assert len(called_prompts) == 3


def test_egm_resolutions_url_hash_128_elo_wistful_and_rate_limits(tmp_path: Path) -> None:
    # 1. URL hash 32-char (128-bit) default + explicit hex_chars support
    url = "https://arxiv.org/abs/2610.12345?utm_source=rss"
    assert len(compute_url_hash(url)) == 32
    assert len(compute_url_hash_128(url)) == 32
    assert compute_url_hash(url) == compute_url_hash_128(url)
    assert len(compute_url_hash(url, hex_chars=16)) == 16

    # 2. DEFAULT_BOT_ELO alias
    assert DEFAULT_BOT_ELO == BOT_INITIAL_ELO == 1200

    # 3. CLI --actor alias and wistful -> too_sunny vote mapping
    parsed = parse_cli_args(["--actor", "octocat", "--title", "pulse:vote:wistful", "--issue-number", "42"])
    assert parsed.login == "octocat"

    store = InMemoryStateStore()
    res = process_event(
        tmp_path,
        visitor_login="octocat",
        issue_title="pulse:vote:wistful",
        issue_number=42,
        store=store,
    )
    assert res["status"] == "recorded_vote"
    assert res["vote"] == "too_sunny"

    # 4. reconcile_to_git alias + record_completed_game
    store.set_needs_git_reconcile(True)
    assert store.reconcile_to_git() is True
    assert store.needs_git_reconcile is False

    now_epoch = int(datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc).timestamp())
    for _ in range(3):
        store.record_completed_game("octocat", now_epoch)
    ok, reason = store.check_and_increment_game_rate_limit("octocat", now_epoch + 60,is_new_round=True)
    assert ok is False
    assert reason == "daily_game_limit_exceeded"


def test_orchestrator_end_to_end_persists_curatorial_state(tmp_path: Path) -> None:
    store = InMemoryStateStore()
    orch = ZeitgeistADKOrchestrator(repo_root=tmp_path, state_store=store)
    analysis, decision, telemetry = orch.run()
    assert orch.last_published_pulse is not None
    assert orch.last_published_pulse.curatorial_archetype in ALL_ARCHETYPES
    saved_state = store.load_pulse_state()
    assert saved_state.get("curatorial_archetype") == orch.last_published_pulse.curatorial_archetype
    assert saved_state.get("text_quadrant") == orch.last_published_pulse.text_quadrant
    assert isinstance(saved_state.get("archetype_history_30d"), list)
    assert len(saved_state["archetype_history_30d"]) >= 1


def test_flux_3_strict_flagship_endpoint_no_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _generate_live_flux_bytes only dispatches to flagship /v1/flux-3-image and never legacy flux-pro-1.1 or flux-dev."""
    import httpx
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")

    called_requests: list[tuple[str, dict]] = []

    def _fake_post(self, url, *, headers=None, json=None, **kwargs):  # type: ignore[no-untyped-def]
        called_requests.append((str(url), dict(json or {})))
        raise RuntimeError("Simulated network stop")

    from types import SimpleNamespace
    monkeypatch.setattr(
        "src.zeitgeist.visual_router.get_spend_guard",
        lambda **kwargs: SimpleNamespace(call=lambda provider, operation, maximum, fn: fn()),
    )
    monkeypatch.setenv("BFL_API_KEY", "test-bfl-key")
    monkeypatch.setattr(httpx.Client, "post", _fake_post)

    res = _generate_live_flux_bytes("Architectural shadow study in 3:2")
    assert res is None
    assert len(called_requests) == 1
    # Official /v1/flux-3-image OpenAPI input observed 2026-10-30 forbids extras.
    payload = called_requests[0][1]
    assert set(payload) <= {"user", "prompt", "images", "aspect_ratio", "resolution", "width", "height", "safety_tolerance", "grounding", "version"}
    assert payload["resolution"] == "1.5k"
    assert "width" not in payload and "height" not in payload  # Exact dimensions override the priced tier.
    endpoint, payload = called_requests[0]
    assert endpoint == "https://api.bfl.ai/v1/flux-3-image"
    assert payload["aspect_ratio"] == "3:2"
    assert payload["resolution"] == "1.5k"
    assert payload["grounding"] is True


def test_firestore_rules_permits_all_eight_text_quadrant_tokens() -> None:
    """Verify firestore.rules allows all 8 curatorial placement tokens in both daily_pulse and pulse_history."""
    rules_path = Path(__file__).resolve().parent.parent / "firestore.rules"
    content = rules_path.read_text(encoding="utf-8")
    expected_tokens = [
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
        "margin_outside",
        "margin_bottom",
        "margin_right",
        "full_canvas",
    ]
    for token in expected_tokens:
        assert f"'{token}'" in content
    assert content.count("'margin_outside'") >= 2


def test_readme_omits_duplicate_markdown_caption_below_picture() -> None:
    """Verify README.md renders the <picture> tag and Stitch slash-bar without a duplicate Markdown word/phrase line."""
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "<picture>" in content
    assert "Spot On" in content and "Too Sunny" in content and "Too Heavy" in content
    assert "**Isobaric** — _Holding equal pressure across shifting fronts._" not in content


def test_render_curatorial_plates_wrapper_returns_path_dict(tmp_path: Path) -> None:
    """Verify render_curatorial_plates returns a dict with light and dark Path entries."""
    img = _make_quadrant_test_image("upper_right")
    out = render_curatorial_plates(
        img,
        word="Isobaric",
        gloss="Holding equal pressure across shifting fronts.",
        archetype="LITERARY_JOURNAL",
        quadrant="margin_outside",
        luminance=0.82,
        thesis="Equal pressure across atmospheric boundaries.",
        attribution="Photo by Test Photographer on Unsplash",
        visual_medium="PHOTOGRAPHY",
        velocity_score=2.4,
        date_str="2026-10-28",
        light_out_path=tmp_path / "plate_light.webp",
        dark_out_path=tmp_path / "plate_dark.webp",
    )
    assert set(out.keys()) == {"light", "dark"}
    assert out["light"].exists() and out["dark"].exists()

