#!/usr/bin/env python3
"""A production private run: the Director and the photo path on the chosen models.

It reads the ingestion cache and the 30-day feed history, commissions one edition
exactly as the edition would (the Stage 0 packet only under the same switch), and
scores photos for it. It publishes nothing: no commit,
no README, no artifact. Its one state write is the judge calibration: when Hydrogen judged
the photos and blocked no batch for safety, Flash scores them again and Hydrogen's matched
pass marks are stored. With ZEITGEIST_RUN_RECORD=1 it also writes one private run record
after the Director step. The photo stage renders into a temporary directory that is removed
at once. Only numbers and public model names are printed.

With --arms it runs comparison arms instead: each named arm (a fixed set of models and prompt
path) names the same frozen intake --repeats times, and one private run record is written per
arm and repeat. In that mode the store refuses every write but the run record, no judge
calibration is measured, and vectors from another embedding model are made again in memory only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zeitgeist import adk_pipeline as pipeline  # noqa: E402
from src.zeitgeist import embedding_migration as em  # noqa: E402
from src.zeitgeist import judge_calibration as jc  # noqa: E402
from src.zeitgeist import run_record  # noqa: E402
from src.zeitgeist import visual_router as vr  # noqa: E402
from src.zeitgeist.model_choice import (  # noqa: E402
    EMBED_001, EMBED_2, FLASH, HYDROGEN, RunModels, activate, model_choice_line, resolve_run_models,
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

# Comparison arms: one numeric line per arm and repeat, a failure line (repeat 0 is the arm's own window),
# the count of records written, and the refusal of a dispatch before any call.
ARM_LINE = "[private_run] arm=%d repeat=%d director_ok=%d attempts=%d packet=%d fallback=%d"
ARM_FAILED_LINE = "[private_run] arm=%d repeat=%d failed_stage=%d"
RECORDS_LINE = "[private_run] records_written=%d"
REFUSED_LINE = "[private_run] arms_refused=%d"
REFUSE_REPEATS, REFUSE_UNKNOWN, REFUSE_NOT_BUILT, REFUSE_OVER_CAP, REFUSE_DUPLICATE = 1, 2, 3, 4, 5
MAX_REPEATS = 3
MAX_NAMING_ATTEMPTS = 12
MAX_PHOTO_ARMS = 2
SHOWN_GROUPS = 5
INTAKE_FIELDS = ("url_hash", "source_id", "tier", "published_at", "title", "summary")


@dataclass(frozen=True)
class Arm:
    """One arm: the same code with fixed models and prompt path."""

    director: str
    judge: str
    embedding: str
    packet: bool
    runnable: bool = True


ARMS: Mapping[str, Arm] = MappingProxyType({
    "previous": Arm(FLASH, FLASH, EMBED_001, packet=False),
    "current": Arm(HYDROGEN, HYDROGEN, EMBED_2, packet=True),
    # Known, but their parts are not built yet.
    "current_links": Arm(HYDROGEN, HYDROGEN, EMBED_2, packet=True, runnable=False),
    "current_reflector": Arm(HYDROGEN, HYDROGEN, EMBED_2, packet=True, runnable=False),
})


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


@dataclass
class Grouping:
    """The night's groups, which cutoffs made them, and the model that made the grouped vectors."""

    clusters: List[Any]
    cutoffs_source: str
    embedding_model: str


def _vector_model(rows: Sequence[Any]) -> str:
    """The one embedding model named on the rows; "mixed" for several, "unknown" for none."""
    models = {row.get("embedding_model") if isinstance(row, dict) else None for row in rows}
    if len(models) > 1:
        return "mixed"
    model = next(iter(models), None)
    return model if isinstance(model, str) and model else "unknown"


def _cluster_stored(store: Any, rows: List[Any], items: List[Any], now: datetime) -> Grouping:
    """Group the stored vectors with the cutoffs of the model that made them; reads only, writes nothing.

    Rows that all carry embedding-2 vectors use the accepted calibration's cutoffs. Any other
    window (embedding-001, unmarked or mixed rows, or no accepted calibration) keeps the legacy cutoffs.
    """
    model = _vector_model(rows)
    if model == em.EMBED_2:
        plan = em.plan_embedding_run(store, em.EMBED_2)
        if plan is not None and plan.mode == "migrated":
            # Migrated mode never touches the store; it is given none so nothing can be written.
            clusters = em.cluster_for_run(items, plan, store=None, now_utc=now)
            calibrated = plan.cutoffs is not None and not plan.fallback
            return Grouping(clusters, "calibrated" if calibrated else "legacy", model)
    return Grouping(pipeline.score_and_cluster_signals(items, now_utc=now), "legacy", model)


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
            grouping = _cluster_stored(store, rows, items, now)
            clusters = grouping.clusters
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
            if run_record.run_record_enabled():
                _write_private_record(store, now, date_iso, run_models, grouping, commission)
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


def _write_private_record(store: Any, now: datetime, date_iso: str, run_models: RunModels,
                          grouping: Grouping, commission: Any) -> None:
    """The private run record after the Director step. Prints and logs nothing.

    Any failure is swallowed, except an interrupt or an exit, which end the run as any interrupt does.
    """
    try:
        evidence = pipeline.director_evidence(commission)
        record = run_record.build_run_record(
            kind="private_run", date_iso=date_iso, now_utc=now, environ=os.environ, run_models=run_models,
            embedding_model=grouping.embedding_model, clusters=grouping.clusters,
            evidence_text=evidence["evidence_text"], commission=commission,
            candidate_words=evidence["candidate_words"], chosen_cluster_id=commission.selected_cluster_id,
            prompt_sha256=evidence["prompt_sha256"], cutoffs_source=grouping.cutoffs_source,
        )
        run_record.write_run_record(store, run_record.make_run_id("private_run", date_iso), record)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        pass


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


def intake_hash(rows: Sequence[Any]) -> str:
    """sha256 over the sorted (url_hash, source_id, tier, published_at, title, summary) of every row; no vectors."""
    keyed = sorted([["" if row.get(name) is None else str(row.get(name)) for name in INTAKE_FIELDS]
                    for row in rows if isinstance(row, dict)])
    payload = json.dumps(keyed, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_order(count: int, digest: str, repeat: int) -> List[int]:
    """The order the leading groups are shown in: ranked on repeat 1, else a permutation fixed by intake and repeat."""
    ranked = list(range(count))
    if repeat <= 1 or count < 2:
        return ranked
    order = list(ranked)
    rng = random.Random("%s:%d" % (digest, repeat))
    while order == ranked:
        rng.shuffle(order)
    return order


class ArmWriteRefused(RuntimeError):
    """An arm tried a store write other than its run record."""


class ArmStore:
    """Reads (load_*) pass through and the run record is saved; every other attribute raises and is counted."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.refused = 0

    def save_run_record(self, run_id: str, document: Dict[str, Any]) -> None:
        self._store.save_run_record(run_id, document)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("load_"):
            return getattr(self._store, name)
        self.refused += 1
        raise ArmWriteRefused("arm store write refused")


def parse_arms(arms: str, repeats: Optional[str]) -> Tuple[List[str], int, int]:
    """(arm names, repeats, refusal code); a refusal code of 0 means the dispatch may run."""
    text = (repeats or "").strip() or str(MAX_REPEATS)
    if not text.isdigit() or not 1 <= int(text) <= MAX_REPEATS:
        return [], 0, REFUSE_REPEATS
    count = int(text)
    names = [part.strip() for part in arms.split(",")]
    if any(name not in ARMS for name in names):
        return [], 0, REFUSE_UNKNOWN
    if len(names) * count > MAX_NAMING_ATTEMPTS:
        return [], 0, REFUSE_OVER_CAP
    if len(set(names)) != len(names):
        return [], 0, REFUSE_DUPLICATE
    if not all(ARMS[name].runnable for name in names):
        return [], 0, REFUSE_NOT_BUILT
    return names, count, 0


class ArmNotReady(RuntimeError):
    """An arm on the second embedding model without its own vectors and an accepted calibration."""


def _items(rows: List[Any], now: datetime) -> List[Any]:
    return [item for item in (pipeline._row_to_cached_item(row, index, now.isoformat())
                              for index, row in enumerate(rows)) if item is not None]


def _calibrated_grouping(store: ArmStore, rows: List[Any], now: datetime) -> Grouping:
    """The stored embedding-2 vectors on the accepted calibration's cutoffs; no call. Raises if either is missing."""
    if _vector_model(rows) != em.EMBED_2:
        raise ArmNotReady("rows are not all embedding-2")
    grouping = _cluster_stored(store, rows, _items(rows, now), now)
    if grouping.cutoffs_source != "calibrated":
        raise ArmNotReady("no accepted calibration")
    return grouping


def _reembedded_grouping(store: ArmStore, rows: List[Any], arm: Arm, genai_client: Any, now: datetime) -> Grouping:
    """The first model's vectors (made again in memory when the rows carry another model) on the legacy cutoffs."""
    if any(not isinstance(row, dict) or row.get("embedding_model") != arm.embedding for row in rows):
        keyed = {str(index): row for index, row in enumerate(rows) if isinstance(row, dict)}
        embedded = pipeline._embed_rows(keyed, {}, genai_client, arm.embedding)
        if embedded is None:
            raise ValueError("arm embedding unavailable")
        rows = list(embedded)
    return _cluster_stored(store, rows, _items(rows, now), now)


def _arm_repeat(store: ArmStore, *, index: int, name: str, arm: Arm, repeat: int, grouping: Grouping,
                digest: str, history_entries: List[Any], genai_client: Any, now: datetime,
                lines: List[str]) -> Tuple[bool, bool]:
    """One naming attempt of one arm, then its record, then (first repeat only) its photo path.

    Returns (ok, record written). Raises on any failure; the caller prints the numeric line.
    """
    date_iso = now.strftime("%Y-%m-%d")
    run_models = RunModels(arm.director, arm.judge, arm.embedding)
    clusters = grouping.clusters
    order = candidate_order(min(SHOWN_GROUPS, len(clusters)), digest, repeat)
    shown = [clusters[position] for position in order] + list(clusters[len(order):])
    with activate(run_models):
        packet = {"evidence_packet": True, "now_utc": now} if arm.packet else {}
        commission = pipeline.commission_daily_zeitgeist(
            shown, history_words_30d=_history_words(history_entries, date_iso), genai_client=genai_client,
            owner_override={}, model_name=run_models.current("director"), run_models=run_models, **packet,
        )
    director_ok = not commission.fallback_used
    lines.append(ARM_LINE % (index, repeat, int(director_ok), commission.attempts_used,
                             int(commission.packet_used), int(run_models.fallback_fired)))
    evidence = pipeline.director_evidence(commission)
    record = run_record.build_run_record(
        kind="private_run", date_iso=date_iso, now_utc=now, environ=os.environ, run_models=run_models,
        embedding_model=grouping.embedding_model, clusters=clusters, evidence_text=evidence["evidence_text"],
        commission=commission, candidate_words=evidence["candidate_words"],
        chosen_cluster_id=commission.selected_cluster_id, prompt_sha256=evidence["prompt_sha256"],
        intake_hash=digest, arm=name, repeat=repeat, cutoffs_source=grouping.cutoffs_source,
    )
    written = run_record.write_run_record(
        store, run_record.make_run_id("private_run", date_iso, arm=name, repeat=repeat), record)
    ok = director_ok and not run_models.fallback_fired
    if ok and repeat == 1 and index <= MAX_PHOTO_ARMS:
        scored: List[Dict[str, Any]] = []
        photo_ok = False
        with activate(run_models), tempfile.TemporaryDirectory() as scratch:
            try:
                vr.select_and_render_pulse_assets(
                    commission=commission, repo_root=Path(scratch), date_str=date_iso,
                    requested_medium="photo", photo_history_30d=history_entries, judge_model=run_models.current("judge"),
                    strict_photo_errors=True, scored_candidates_out=scored, safety_blocked_ids=set(),
                )
                photo_ok = True
            except vr.PhotoSelectionUnavailable:
                photo_ok = False
        lines.append(_photo_line(scored, photo_ok))
        ok = not run_models.fallback_fired
    return ok, written


def run_arms(store: Any, *, genai_client: Any, now: datetime, names: List[str], repeats: int) -> PrivateRun:
    """Every arm and repeat on one frozen intake, writing only run records. Never raises.

    One arm failing does not stop the others; an interrupt or exit stops the dispatch.
    """
    lines: List[str] = []
    result = PrivateRun(lines, 3)
    try:
        reads = ArmStore(store)
        cached = reads.load_ingestion_cache(now)
        rows = list(cached.values()) if isinstance(cached, dict) else list(cached or ())
        digest = intake_hash(rows)
        history = reads.load_pulse_feed_history(days=30)
        entries = list(history.get("entries", [])) if isinstance(history, dict) else list(history or ())
    except BaseException:
        lines.append(FAILED_LINE % READ)
        return result
    failed = False
    written = 0
    stop = False
    stores = [ArmStore(store) for _ in names]
    # Before any paid call of the dispatch: an arm on the second embedding model needs the stored
    # embedding-2 vectors and an accepted calibration, and is grouped now; it never runs on legacy cutoffs.
    groupings: Dict[int, Optional[Grouping]] = {}
    for index, name in enumerate(names, 1):
        if ARMS[name].embedding != em.EMBED_2:
            continue
        try:
            groupings[index] = _calibrated_grouping(stores[index - 1], rows, now)
        except BaseException as error:
            groupings[index] = None
            if not isinstance(error, Exception):
                lines.append(ARM_FAILED_LINE % (index, 0, READ))
                failed = stop = True
                break
    for index, name in enumerate(names, 1):
        if stop:
            break
        arm = ARMS[name]
        arm_store = stores[index - 1]
        try:
            if index in groupings:
                grouping = groupings[index]
                if grouping is None:
                    raise ArmNotReady("arm not ready")
            else:
                with activate(RunModels(arm.director, arm.judge, arm.embedding)):
                    grouping = _reembedded_grouping(arm_store, rows, arm, genai_client, now)
            if arm_store.refused:
                raise ArmWriteRefused("arm store write refused")
        except BaseException as error:
            lines.append(ARM_FAILED_LINE % (index, 0, READ))
            failed = True
            if not isinstance(error, Exception):
                break
            continue
        for repeat in range(1, repeats + 1):
            refused_before = arm_store.refused
            try:
                ok, wrote = _arm_repeat(arm_store, index=index, name=name, arm=arm, repeat=repeat,
                                        grouping=grouping, digest=digest, history_entries=entries,
                                        genai_client=genai_client, now=now, lines=lines)
                written += int(wrote)
                if arm_store.refused != refused_before:
                    lines.append(ARM_FAILED_LINE % (index, repeat, DIRECTOR))
                    ok = False
                failed = failed or not ok or not wrote
            except BaseException as error:
                # Messages and chained causes are never printed.
                lines.append(ARM_FAILED_LINE % (index, repeat, DIRECTOR))
                failed = True
                if not isinstance(error, Exception):
                    stop = True
                    break
    lines.append(RECORDS_LINE % written)
    result.exit_code = 3 if failed else 0
    return result


def _now() -> datetime:
    return datetime.now(timezone.utc)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=None, help="comma-separated comparison arms")
    parser.add_argument("--repeats", default=None, help="naming attempts per arm, 1 to 3 (default 3)")
    args = parser.parse_args(argv)
    arms_mode = bool((args.arms or "").strip())
    if not arms_mode and (args.repeats or "").strip():
        parser.error("--repeats needs --arms")
    names: List[str] = []
    repeats = 0
    if arms_mode:
        # Checked before the store, the client or any call; the line carries only the refusal code.
        names, repeats, refusal = parse_arms(args.arms, args.repeats)
        if refusal:
            print(REFUSED_LINE % refusal)
            return 3
    # Library prints and log records can carry private text; none of it leaves this process.
    hidden = StringIO()
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(hidden), redirect_stderr(hidden):
            try:
                store = _store()
                genai_client = _genai_client()
                if not arms_mode:
                    run_models = resolve_run_models(pipeline.load_orbit_config(ROOT / "config" / "orbit_sources.json"))
            except BaseException:
                result = PrivateRun([FAILED_LINE % SETUP], 3)
            else:
                now = _now()
                if arms_mode:
                    # The judge calibration is not measured: an arm writes nothing but its records.
                    result = run_arms(store, genai_client=genai_client, now=now, names=names, repeats=repeats)
                else:
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
