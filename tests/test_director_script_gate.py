"""Unit tests for WS3 Seam 3: Editorial Director Agent & Latin-Script / 30-Day Deduplication Gate."""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pytest

from src.zeitgeist.adk_pipeline import (
    DirectorCommission,
    commission_daily_zeitgeist,
    validate_director_commission,
)
from src.zeitgeist.methodology import CachedItem, cluster_and_score


def test_validate_director_commission_rejects_bengali_script_and_history_duplicates() -> None:
    """Verify validate_director_commission rejects non-Latin script (e.g. Bengali) and 30d history duplicates."""
    history_30d = ["Entropy", "Isobaric Framing", "Tessellation"]

    # 1. Bengali script in word ('কিভাবে') -> rejected
    bad_bengali_word = DirectorCommission(
        word="কিভাবে",
        gloss="How systems converge under pressure.",
        commissioned_thesis="Cross-lens convergence across builder and macro dispatches.",
        visual_anchor="Brushed steel manifold",
        unsplash_queries=["steel architecture", "industrial manifold", "minimalist bridge"],
        imagen_commission_prompt="Architectural blueprint of a steel manifold.",
        palette_hex=["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"],
    )
    ok, reason = validate_director_commission(bad_bengali_word, history_30d)
    assert ok is False
    assert "format" in reason.lower() or "latin" in reason.lower()

    # 2. Bengali script leaked into commissioned_thesis -> rejected even if word is ASCII
    bad_bengali_thesis = DirectorCommission(
        word="Equilibrium",
        gloss="Balanced structural tension across independent feeds.",
        commissioned_thesis="Builder dispatches explore কিভাবে deterministic kernels stabilize latency.",
        visual_anchor="Suspension cable array",
        unsplash_queries=["suspension bridge cables", "minimalist steel", "concrete pier"],
        imagen_commission_prompt="Minimalist suspension bridge cable geometry.",
        palette_hex=["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"],
    )
    ok, reason = validate_director_commission(bad_bengali_thesis, history_30d)
    assert ok is False
    assert "bengali" in reason.lower() or "non-latin" in reason.lower()

    # 3. 30-day history duplicate (case-insensitive) -> rejected
    dup_commission = DirectorCommission(
        word="isobaric framing",
        gloss="Constant-pressure structural boundary.",
        commissioned_thesis="Macro and personal orbit sources converge on constant-pressure design.",
        visual_anchor="Barometric chamber",
        unsplash_queries=["barometer instrument", "brass gauge", "weather station"],
        imagen_commission_prompt="Technical drawing of an isobaric chamber.",
        palette_hex=["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"],
    )
    ok, reason = validate_director_commission(dup_commission, history_30d)
    assert ok is False
    assert "30 days" in reason.lower() or "duplicate" in reason.lower()

    # 4. More than 2 words -> rejected
    three_words = DirectorCommission(
        word="Three Word Phrase",
        gloss="Too many words in the zeitgeist title.",
        commissioned_thesis="Synthesis across three separate tokens.",
    )
    ok, reason = validate_director_commission(three_words, history_30d)
    assert ok is False

    # 5. Valid 1-word and 2-word Latin commissions (including accented loanwords) -> accepted
    valid_commission = DirectorCommission(
        word="Détente",
        gloss="Relaxing strained protocol boundaries.",
        commissioned_thesis="Personal orbit systems engineers and macro policy analysts converge on protocol interoperability.",
        visual_anchor="Interlocking brass gears",
        unsplash_queries=["interlocking gears", "clockwork mechanism", "archway stone"],
        imagen_commission_prompt="Editorial illustration of interlocking architectural arches.",
        palette_hex=["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"],
    )
    ok, reason = validate_director_commission(valid_commission, history_30d)
    assert ok is True
    assert reason == "valid"


def test_commission_daily_zeitgeist_retries_with_rejection_reason_and_succeeds() -> None:
    """Verify up to 2 retries (3 attempts total) at temperature=0.3 passing explicit rejection reasons."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    vec = np.zeros(768, dtype=np.float32)
    vec[0] = 1.0
    f16 = vec.astype(np.float16).tobytes()

    items = [
        CachedItem(
            url_hash="h1",
            source_id="o1_willison",
            bucket="ORBIT",
            title="Deterministic sandboxes for local agents",
            summary="Evaluating reproducible execution boundaries.",
            published_at=now_utc.isoformat(),
            embedding_f16=f16,
        ),
        CachedItem(
            url_hash="h2",
            source_id="m1_ars",
            bucket="MACRO",
            title="Sandbox isolation in cloud runtimes",
            summary="Kernel isolation primitives for multi-tenant workloads.",
            published_at=now_utc.isoformat(),
            embedding_f16=f16,
        ),
    ]
    clusters = cluster_and_score(items, now_utc=now_utc)
    history_30d = ["Containment", "Lattice"]

    prompts_seen: list[str] = []
    temps_seen: list[float] = []

    def fake_llm(prompt: str, temperature: float) -> DirectorCommission:
        prompts_seen.append(prompt)
        temps_seen.append(temperature)
        attempt = len(prompts_seen)
        if attempt == 1:
            # Attempt 1: collides with history_30d
            return DirectorCommission(
                word="Containment",
                gloss="Bounded execution.",
                commissioned_thesis="First attempt collides with 30d history.",
            )
        if attempt == 2:
            # Attempt 2: leaks Bengali script
            return DirectorCommission(
                word="Bulkhead",
                gloss="Structural partition কিভাবে isolating fault domains.",
                commissioned_thesis="Second attempt leaks non-Latin characters.",
            )
        # Attempt 3: valid commission
        return DirectorCommission(
            word="Bulkhead",
            gloss="Partitions isolating runtime fault domains.",
            commissioned_thesis="Orbit and macro engineers converge on bulkhead isolation primitives.",
            visual_anchor="Ship hull watertight bulkhead door",
            unsplash_queries=["steel bulkhead", "industrial concrete wall", "brutalist corridor"],
            imagen_commission_prompt="Architectural cutaway of a watertight steel bulkhead.",
            palette_hex=["#0F172A", "#1E293B", "#38BDF8", "#F59E0B", "#F8FAFC"],
        )

    result = commission_daily_zeitgeist(
        clusters[:3],
        history_30d,
        llm_callable=fake_llm,
    )

    assert len(prompts_seen) == 3
    assert temps_seen == [0.3, 0.3, 0.3]
    assert "Containment" in prompts_seen[1]
    assert "30 days" in prompts_seen[1] or "duplicate" in prompts_seen[1].lower()
    assert "BENGALI" in prompts_seen[2].upper() or "NON-LATIN" in prompts_seen[2].upper()
    assert result.word == "Bulkhead"
    assert result.attempts_used == 3
    assert result.fallback_used is False


def test_commission_daily_zeitgeist_deterministic_fallback_when_all_attempts_fail() -> None:
    """If all 3 LLM attempts fail validation, fallback deterministically selects a valid non-duplicate word."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    vec = np.zeros(768, dtype=np.float32)
    vec[0] = 1.0
    f16 = vec.astype(np.float16).tobytes()

    items = [
        CachedItem(
            url_hash="f1",
            source_id="o1_a",
            bucket="ORBIT",
            title="Harmonic acoustic resonance",
            summary="Acoustic resonance in timber frames.",
            published_at=now_utc.isoformat(),
            embedding_f16=f16,
        ),
        CachedItem(
            url_hash="f2",
            source_id="m1_b",
            bucket="MACRO",
            title="Structural harmonic damping",
            summary="Tuned mass dampers in tall towers.",
            published_at=now_utc.isoformat(),
            embedding_f16=f16,
        ),
    ]
    clusters = cluster_and_score(items, now_utc=now_utc)
    history_30d = ["Harmonic", "Resonance"]

    def always_invalid_llm(prompt: str, temperature: float) -> dict[str, str]:
        return {
            "word": "কিভাবে",
            "gloss": "Invalid script every time.",
            "commissioned_thesis": "Always fails.",
        }

    result = commission_daily_zeitgeist(
        clusters[:3],
        history_30d,
        llm_callable=always_invalid_llm,
    )
    ok, reason = validate_director_commission(result, history_30d)
    assert ok is True, reason
    assert result.fallback_used is True
    assert result.word.lower() not in {w.lower() for w in history_30d}


def test_adk_orchestrator_v2_end_to_end_and_run_zeitgeist_cli(tmp_path) -> None:
    """Verify ZeitgeistADKOrchestrator.run() and scripts/run_zeitgeist.py execute the full 4-stage v2 pipeline."""
    import json
    from pathlib import Path
    import shutil
    from scripts.run_zeitgeist import main as run_zeitgeist_main
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator

    repo_src = Path(__file__).resolve().parents[1]
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_src / "config" / "orbit_sources.json", tmp_path / "config" / "orbit_sources.json")

    orchestrator = ZeitgeistADKOrchestrator(tmp_path)
    analysis, decision, telemetry = orchestrator.run(date_iso="2026-10-27", offline_demo=True)

    assert analysis.date_iso == "2026-10-27"
    assert telemetry.within_budget is False
    assert telemetry.estimated_monthly_cost_index is None
    assert orchestrator.last_commission is not None
    assert orchestrator.last_published_pulse is not None

    # Verify assets/pulse/2026-10-27.webp and assets/daily_pulse.svg were rendered
    assert (tmp_path / "assets" / "pulse" / "2026-10-27.webp").exists()
    assert (tmp_path / "assets" / "daily_pulse.svg").exists()

    # Verify the legacy entry fields plus stable publication provenance.
    feed_payload = json.loads((tmp_path / "data" / "pulse_feed.json").read_text(encoding="utf-8"))
    assert len(feed_payload["entries"]) == 1
    entry = feed_payload["entries"][0]
    assert len(entry.keys()) == 38
    assert entry["publication_id"] == json.loads((tmp_path / "assets/pulse_manifest.json").read_text())["publication_id"]
    assert set(entry["mood_votes"].keys()) == {"spot_on", "too_sunny", "too_heavy"}
    assert entry["window_hours"] == 96
    assert entry["schema_version"] == "2.0"
    assert entry["image_data_uri"] == ""

    # Verify scripts/run_zeitgeist.py CLI arg parsing works cleanly inside isolated tmp_path
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {"ZEITGEIST_OFFLINE_DEMO": "1"}):
        rc = run_zeitgeist_main(["--repo-root", str(tmp_path), "--date", "2026-10-27", "--skip-readme"])
    assert rc == 0

