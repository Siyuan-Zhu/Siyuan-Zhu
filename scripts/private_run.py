#!/usr/bin/env python3
"""A production private run: the Director and the photo path on the chosen models.

It reads the ingestion cache and the 30-day feed history, commissions one edition
exactly as the edition would (the Stage 0 packet only under the same switch), and
scores photos for it. It publishes nothing: no commit,
no README, no state write, no artifact. The photo stage renders into a temporary
directory that is removed at once. Only numbers and public model names are printed.
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
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zeitgeist import adk_pipeline as pipeline  # noqa: E402
from src.zeitgeist import visual_router as vr  # noqa: E402
from src.zeitgeist.model_choice import RunModels, activate, model_choice_line, resolve_run_models  # noqa: E402
from src.zeitgeist.understanding import understanding_v1_enabled  # noqa: E402

DIRECTOR_LINE = "[private_run] director_ok=%d attempts=%d packet=%d fallback=%d items=%d clusters=%d"
PHOTO_LINE = "[private_run] photo_ok=%d candidates=%d safe=%d best_meaning=%.3f best_composite=%.3f"
FAILED_LINE = "[private_run] failed_stage=%d"
SETUP, READ, DIRECTOR, PHOTO = 1, 2, 3, 4


@dataclass
class PrivateRun:
    """The run's printable lines and exit code, plus in-memory results that are never printed."""

    lines: List[str]
    exit_code: int
    commission: Any = None
    # Every photo the critic scored in this run, with its thumbnail bytes and scores.
    scored_candidates: List[Dict[str, Any]] = field(default_factory=list)


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
                            scored_candidates_out=scored,
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
                result = run_private(store, genai_client=genai_client, now=datetime.now(timezone.utc),
                                     run_models=run_models)
    finally:
        logging.disable(logging.NOTSET)
    for line in result.lines:
        print(line)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
