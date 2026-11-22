#!/usr/bin/env python3
"""A production private run: the Director and the photo path on the chosen models.

It reads the ingestion cache and the 30-day feed history, commissions one edition
exactly as the edition would (the Stage 0 packet only under the same switch), and
scores photos for it. It publishes nothing: no commit,
no README, no artifact. Its one state write is the judge calibration: when Hydrogen judged
the photos and blocked no batch for safety, Flash scores them again and Hydrogen's matched
pass marks are stored. The photo stage renders into a temporary directory that is removed
at once. Only numbers and public model names are printed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zeitgeist import adk_pipeline as pipeline  # noqa: E402
from src.zeitgeist import judge_calibration as jc  # noqa: E402
from src.zeitgeist import visual_router as vr  # noqa: E402
from src.zeitgeist.model_choice import (  # noqa: E402
    FLASH, HYDROGEN, RunModels, activate, model_choice_line, resolve_run_models,
)
from src.zeitgeist.understanding import understanding_v1_enabled  # noqa: E402

DIRECTOR_LINE = "[private_run] director_ok=%d attempts=%d packet=%d fallback=%d items=%d clusters=%d"
PHOTO_LINE = "[private_run] photo_ok=%d candidates=%d safe=%d best_meaning=%.3f best_composite=%.3f"
FAILED_LINE = "[private_run] failed_stage=%d"
CALIBRATION_LINE = ("[judge_calibration] photos=%d flash_pass_meaning=%d flash_pass_composite=%d "
                    "meaning_min=%.2f composite_min=%.2f written=%d")
# The critic's batch size, as the live photo search uses it.
CRITIC_BATCH = 9
SETUP, READ, DIRECTOR, PHOTO = 1, 2, 3, 4


@dataclass
class PrivateRun:
    """The run's printable lines and exit code, plus in-memory results that are never printed."""

    lines: List[str]
    exit_code: int
    commission: Any = None
    # Every photo the critic scored in this run, with its thumbnail bytes and scores.
    scored_candidates: List[Dict[str, Any]] = field(default_factory=list)
    # The photos of every batch the judge safety-blocked in this run.
    safety_blocked_ids: Set[str] = field(default_factory=set)


def _history_words(entries: List[Any], date_iso: str) -> List[str]:
    words = []
    for entry in entries[:30]:
        if not isinstance(entry, dict) or str(entry.get("date", ""))[:10] == date_iso:
            continue
        word = entry.get("artwork_title") or entry.get("word")
        if word:
            words.append(str(word))
    return words


def _photo_line(scored: List[Dict[str, Any]], photo_ok: bool) -> str:
    return PHOTO_LINE % (
        int(photo_ok), len(scored), sum(row.get("safety_pass") is True for row in scored),
        max((float(row["semantic_alignment"]) for row in scored), default=0.0),
        max((float(row["composite_score"]) for row in scored), default=0.0),
    )


def run_private(store: Any, *, genai_client: Any, now: datetime, run_models: RunModels) -> PrivateRun:
    """Never raises, not even KeyboardInterrupt or SystemExit: a failure ends the run with a numeric failed_stage line."""
    lines: List[str] = []
    scored: List[Dict[str, Any]] = []
    result = PrivateRun(lines, 3, scored_candidates=scored)
    date_iso = now.strftime("%Y-%m-%d")
    stage = READ
    try:
        with activate(run_models):
            cached = store.load_ingestion_cache(now)
            rows = list(cached.values()) if isinstance(cached, dict) else list(cached or ())
            items = [item for item in (pipeline._row_to_cached_item(row, index, now.isoformat())
                                       for index, row in enumerate(rows)) if item is not None]
            clusters = pipeline.score_and_cluster_signals(items, now_utc=now)
            history = store.load_pulse_feed_history(days=30)
            entries = list(history.get("entries", [])) if isinstance(history, dict) else list(history or ())

            stage = DIRECTOR
            # The same switch the edition reads decides whether the Stage 0 packet is sent.
            packet = {"evidence_packet": True, "now_utc": now} if understanding_v1_enabled() else {}
            commission = pipeline.commission_daily_zeitgeist(
                clusters, history_words_30d=_history_words(entries, date_iso), genai_client=genai_client,
                owner_override={}, model_name=run_models.current("director"), run_models=run_models, **packet,
            )
            result.commission = commission
            director_ok = not commission.fallback_used
            lines.append(model_choice_line(run_models, commission.slate_count))
            lines.append(DIRECTOR_LINE % (int(director_ok), commission.attempts_used, int(commission.packet_used),
                                          int(run_models.fallback_fired), len(items), len(clusters)))
            if not director_ok or run_models.fallback_fired:
                # The run can only end as a failure now; do not spend on the critic.
                return result

            stage = PHOTO
            photo_ok = False
            try:
                with tempfile.TemporaryDirectory() as scratch:
                    try:
                        vr.select_and_render_pulse_assets(
                            commission=commission, repo_root=Path(scratch), date_str=date_iso,
                            requested_medium="photo", photo_history_30d=entries,
                            judge_model=run_models.current("judge"), strict_photo_errors=True,
                            scored_candidates_out=scored, safety_blocked_ids=result.safety_blocked_ids,
                        )
                        photo_ok = True
                    except vr.PhotoSelectionUnavailable:
                        photo_ok = False
                lines.append(_photo_line(scored, photo_ok))
            finally:
                if run_models.fallback_fired:
                    lines.append(model_choice_line(run_models, commission.slate_count))
            result.exit_code = 0 if not run_models.fallback_fired else 3
    except BaseException:
        # Interrupts and timeouts included: their messages and chained causes are never printed.
        lines.append(FAILED_LINE % stage)
        result.exit_code = 3
    return result


def _flash_scores(scored: List[Dict[str, Any]], commission: Any) -> List[Tuple[float, float]]:
    """(meaning, composite) under Flash for each scored photo: the run's batches, the run's prompt."""
    word = str(getattr(commission, "word", ""))
    gloss = str(getattr(commission, "gloss", ""))
    thesis = str(getattr(commission, "commissioned_thesis", ""))
    scores = []
    for offset in range(0, len(scored), CRITIC_BATCH):
        verdicts = vr._score_photo_batch(scored[offset:offset + CRITIC_BATCH], word=word, gloss=gloss,
                                         thesis=thesis, model_name=FLASH, strict_errors=True)
        if all(verdict == vr._FAIL_CLOSED_VISION_SCORES for verdict in verdicts):
            # A blocked or unanswered batch is not a measurement; it would read as "Flash passed none".
            raise ValueError("Flash batch failed closed")
        for verdict in verdicts:
            scores.append((float(verdict["semantic_alignment"]), float(vr.compute_composite_critic_score(
                verdict["semantic_alignment"], verdict["compositional_clarity"], verdict["editorial_memorability"]))))
    return scores


def measure_judges(store: Any, result: PrivateRun, run_models: RunModels, date_iso: str) -> Optional[str]:
    """After a completed Hydrogen-judged run, score its photos again with Flash and store Hydrogen's matched marks.

    None when the step does not apply. Otherwise one numeric line; never raises, and a failure is written=0.
    """
    if result.exit_code != 0 or run_models.current("judge") != HYDROGEN:
        return None
    scored = result.scored_candidates
    counts: Tuple[int, int] = (0, 0)
    marks: Tuple[float, float] = jc.DEFAULT_MARKS
    written = 0
    try:
        # A safety-blocked batch is never shown to another model, so a blocked run is not measured.
        if len(scored) >= jc.MIN_PHOTOS and not result.safety_blocked_ids:
            # Flash answers on its own run models, so nothing here can move the run's judge.
            with activate(RunModels(FLASH, FLASH, run_models.embedding)):
                flash = _flash_scores(scored, result.commission)
            judge = [(float(row["semantic_alignment"]), float(row["composite_score"])) for row in scored]
            counts = tuple(int(sum(row[axis] >= jc.DEFAULT_MARKS[axis] for row in flash)) for axis in (0, 1))
            matched = jc.matched_marks(flash, judge)
            if matched is not None:
                marks = (float(matched[0]), float(matched[1]))
                document = jc.build_document(HYDROGEN, marks, {
                    "photos": len(scored), "flash_pass_meaning": counts[0], "flash_pass_composite": counts[1],
                }, date_iso)
                store.save_calibration(jc.DOCUMENT_NAME, document)
                written = 1
    except BaseException:
        # Interrupts included: the step never fails the run, and no message is printed.
        written = 0
    return CALIBRATION_LINE % (len(scored), counts[0], counts[1], marks[0], marks[1], written)


def _store() -> Any:
    from src.state_store import get_state_store

    return get_state_store(repo_root=ROOT)


def _genai_client() -> Optional[Any]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or os.environ.get("ZEITGEIST_LIVE_LLM") != "1":
        return None
    from google import genai

    from src.zeitgeist.spend_guard import guard_genai_client

    return guard_genai_client(genai.Client(api_key=key), credential_key=key)


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    # Library prints and log records can carry private text; none of it leaves this process.
    hidden = StringIO()
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(hidden), redirect_stderr(hidden):
            try:
                store = _store()
                genai_client = _genai_client()
                run_models = resolve_run_models(pipeline.load_orbit_config(ROOT / "config" / "orbit_sources.json"))
            except BaseException:
                result = PrivateRun([FAILED_LINE % SETUP], 3)
            else:
                now = datetime.now(timezone.utc)
                result = run_private(store, genai_client=genai_client, now=now, run_models=run_models)
                calibration = measure_judges(store, result, run_models, now.strftime("%Y-%m-%d"))
                if calibration is not None:
                    result.lines.append(calibration)
    finally:
        logging.disable(logging.NOTSET)
    for line in result.lines:
        print(line)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
