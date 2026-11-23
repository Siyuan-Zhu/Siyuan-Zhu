#!/usr/bin/env python3
"""The reflector on a stored run: has the night's standing group been named by a recent gloss?

It reads the newest run record of one kind (and arm) written in the last 36 hours that holds the
reflector's inputs, and the published glosses of the last 30 days. It answers twice: by arithmetic,
on one embedding call of the kind the collection step makes, and as a pointing task on a second model
with two planted answers, three times with the glosses shuffled (six small calls at most). Its numbers,
and the standing group's item hashes when it found the group unnamed, are stored on that record under
tests["reflector"]. It prints one line of numbers and always exits 0.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zeitgeist import adk_pipeline as pipeline  # noqa: E402
from src.zeitgeist import reflector as rf  # noqa: E402
from src.zeitgeist import run_record  # noqa: E402
from src.zeitgeist.collectors import embed_missing_items  # noqa: E402
from src.zeitgeist.methodology import _extract_unit_embedding_f32  # noqa: E402
from src.zeitgeist.model_choice import FLASH, HYDROGEN  # noqa: E402
from src.zeitgeist.spend_guard import SpendBlocked, refused_before_dispatch  # noqa: E402

KEY = "reflector"
MODEL_ENV = "ZEITGEIST_REFLECTOR_MODEL"
MODELS = (FLASH, HYDROGEN)
MAX_AGE = timedelta(hours=36)
HISTORY_DAYS = 30
MAX_GLOSSES = 25
RUNS = 3
# Two questions a run: by construction never more than this many generate_content calls.
MAX_QUESTIONS = 6
TEMPERATURE = 0.3
LOAD_LIMIT = 500
STATUS_NUMBERS = {"pointer": 1, "named": 2, "void": 3, "skipped": 4, "no_group": 5, "no_glosses": 6,
                  "no_control": 7, "no_record": 8, "failed": 9, "disputed": 10}
LINE = "[reflector] status=%d plants_ok=%d agreement=%d arithmetic_named=%d calls=%d"
PLANT = "plant"


class _Stopped(Exception):
    """A SpendBlocked that was not refused before dispatch: the run ends at once."""


class _Calls:
    """The genai client as the reflector uses it, counting each paid call that was made.

    A call the spend guard refused before dispatch was not made and is not counted; any other
    outcome, a failure included, is.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.made = 0
        self.questions = 0
        self.models = self

    def _call(self, name: str, kwargs: Dict[str, Any]) -> Any:
        try:
            reply = getattr(self._client.models, name)(**kwargs)
        except SpendBlocked as error:
            if not refused_before_dispatch(error):
                self.made += 1
            raise
        except BaseException:
            self.made += 1
            raise
        self.made += 1
        return reply

    def embed_content(self, **kwargs: Any) -> Any:
        return self._call("embed_content", kwargs)

    def generate_content(self, **kwargs: Any) -> Any:
        self.questions += 1
        return self._call("generate_content", kwargs)


def _numbers(status: str, model: str, now: datetime, reason: Optional[str] = None,
             same_model: Optional[bool] = None) -> Dict[str, Any]:
    numbers: Dict[str, Any] = {"status": status}
    if reason is not None:
        numbers["reason"] = reason
    numbers.update({
        "arithmetic_named": None, "arithmetic_margin": None, "control_answers": None, "answers": None,
        "plants_ok": None, "agreement": None, "pointer_item_hashes": None, "model": model,
        "same_model": same_model, "written_at": now.isoformat(),
    })
    return numbers


def _newest_record(store: Any, kind: str, arm: Optional[str], now: datetime) -> Optional[Tuple[str, Dict[str, Any]]]:
    """The newest record of the kind (and arm) holding the reflector's inputs, written in the last 36 hours."""
    found = []
    for record in store.load_run_records(kind=kind, limit=LOAD_LIMIT) or ():
        if not isinstance(record, dict) or record.get("kind") != kind:
            continue
        if arm is not None and record.get("arm") != arm:
            continue
        run_id, tests = record.get("run_id"), record.get("tests")
        if not isinstance(run_id, str) or not isinstance(tests, dict) or not isinstance(tests.get(rf.INPUTS_KEY), dict):
            continue
        moment = run_record._written_at(record)
        if moment is None or now - moment > MAX_AGE:
            continue
        found.append((moment, run_id, record))
    if not found:
        return None
    # Newest first; records written at one moment (the repeats of one dispatch) in run id order.
    found.sort(key=lambda entry: entry[1])
    found.sort(key=lambda entry: entry[0], reverse=True)
    return found[0][1], found[0][2]


def _past_glosses(store: Any, record_date: str) -> List[Tuple[str, str]]:
    """(date, gloss) of the published entries, newest first, at most 25, the record's own date left out."""
    loader = getattr(store, "load_pulse_feed_history", None)
    if loader is None:
        return []
    history = loader(days=HISTORY_DAYS)
    entries = list(history.get("entries", [])) if isinstance(history, dict) else list(history or ())
    by_date: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        day = str(entry.get("date") or "")[:10]
        gloss = " ".join(str(entry.get("gloss") or "").split())
        if day and gloss and day != record_date and day not in by_date:
            by_date[day] = gloss
    return sorted(by_date.items(), reverse=True)[:MAX_GLOSSES]


def _arithmetic(calls: _Calls, inputs: Dict[str, Any], past: Sequence[Tuple[str, str]],
                own_gloss: str) -> Optional[Dict[str, Any]]:
    """One embedding call for the past glosses and tonight's own, then named_by_arithmetic; None when it cannot."""
    standing_centre = inputs["standing"].get("centre")
    chosen_centre = inputs["chosen"].get("centre")
    if not standing_centre or not chosen_centre:
        return None
    texts = [gloss for _, gloss in past] + [own_gloss]
    try:
        # The collection step's own request, as the own-group check makes it: the record's model, verified vectors only.
        embedded = embed_missing_items(
            [{"url_hash": "%s_%d" % (KEY, index), "title": text} for index, text in enumerate(texts)], {},
            genai_client=calls, model=str(inputs.get("embedding_model") or ""), require_live=True,
        )
    except SpendBlocked as error:
        if refused_before_dispatch(error):
            return None
        raise _Stopped() from None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None
    try:
        vectors = [_extract_unit_embedding_f32(item) for item in embedded]
        return rf.named_by_arithmetic(standing_centre, vectors[:-1], vectors[-1], chosen_centre)
    except Exception:
        return None


def _ask(calls: _Calls, model: str, question: str, count: int) -> Any:
    """One pointing question; the parsed answer, or None for a refused, failed or unreadable reply."""
    if calls.questions >= MAX_QUESTIONS:
        return None
    try:
        reply = calls.models.generate_content(model=model, contents=question,
                                              config=pipeline._director_config(model, TEMPERATURE))
        text = reply.text
    except SpendBlocked as error:
        if refused_before_dispatch(error):
            return None
        raise _Stopped() from None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None
    return rf.parse_answer(text, count)


def _reflect_on(store: Any, run_id: str, record: Dict[str, Any], calls: Optional[_Calls], model: str,
                allow_same_model: bool, now: datetime) -> Dict[str, Any]:
    inputs = record["tests"][rf.INPUTS_KEY]
    standing, chosen = inputs.get("standing"), inputs.get("chosen")
    own_gloss = " ".join(str(record.get("gloss") or "").split())
    models = record.get("models") if isinstance(record.get("models"), dict) else {}
    # Whether the reflector answers on the model that answered as Director that night.
    same_model = model == models.get("director")

    def early(status: str, reason: Optional[str] = None) -> Dict[str, Any]:
        return _numbers(status, model, now, reason=reason, same_model=same_model)

    # A night whose Director fell back to another model is not the reading the reflector checks.
    if models.get("fallback_fired") is True:
        return early("skipped", reason="fallback")
    if not isinstance(standing, dict) or not isinstance(standing.get("block"), str):
        return early("no_group")
    if not isinstance(chosen, dict) or not isinstance(chosen.get("block"), str) or not own_gloss:
        return early("no_control")
    if same_model and not allow_same_model:
        return early("skipped", reason="same_model")
    past = _past_glosses(store, str(record.get("date") or "")[:10])
    if not past:
        return early("no_glosses")
    if calls is None:
        return early("skipped", reason="no_client")

    arithmetic = _arithmetic(calls, inputs, past, own_gloss)
    past_texts = [gloss for _, gloss in past]
    plant = len(past_texts)
    control, real = [], []
    for run in range(1, RUNS + 1):
        rng = random.Random("%s:%d" % (run_id, run))
        control_order = list(range(plant + 1))
        rng.shuffle(control_order)
        real_order = list(range(plant + 1))
        rng.shuffle(real_order)
        # The control: tonight's candidate among the glosses with tonight's own, which it must find.
        answer = _ask(calls, model, rf.build_question(chosen["block"], past_texts + [own_gloss], control_order),
                      plant + 1)
        control.append(rf.answer_identity(answer, control_order))
        # The real question: the standing group among the past glosses and one any-day gloss it must not choose.
        any_day = rf.ANY_DAY_GLOSSES[(run - 1) % len(rf.ANY_DAY_GLOSSES)]
        answer = _ask(calls, model, rf.build_question(standing["block"], past_texts + [any_day], real_order),
                      plant + 1)
        real.append(rf.answer_identity(answer, real_order))
    verdict = rf.verdict(control, [plant] * len(control), real, [plant] * len(real))

    def label(identity: Any) -> Any:
        if identity == rf.NONE_ANSWER or identity is None:
            return identity
        return PLANT if identity == plant else past[identity][0]

    status = verdict["status"]
    if status == "pointer" and (arithmetic is None or arithmetic["named"] is not False):
        # The pointing task found no gloss, but the arithmetic found one or could not answer: no pointer.
        status = "disputed"
    numbers = early(status)
    numbers.update({
        "arithmetic_named": None if arithmetic is None else arithmetic["named"],
        "arithmetic_margin": None if arithmetic is None else arithmetic["margin"],
        "control_answers": [int(identity == plant) for identity in control],
        "answers": [label(identity) for identity in real],
        "plants_ok": verdict["plants_ok"],
        "agreement": verdict["agreement"],
        "pointer_item_hashes": (list(standing.get("item_hashes") or ()) if status == "pointer" else None),
    })
    return numbers


def reflect(store: Any, *, genai_client: Any, kind: str = "daily_run", arm: Optional[str] = None,
            model: Optional[str] = None, allow_same_model: bool = False,
            now: Optional[datetime] = None) -> Dict[str, Any]:
    """Run the reflector on the newest matching record; the numbers stored, plus `calls`, the paid calls made.

    Never raises but KeyboardInterrupt and SystemExit; prints and logs nothing.
    """
    now = now or datetime.now(timezone.utc)
    chosen_model = str(model or os.environ.get(MODEL_ENV) or FLASH)
    if chosen_model not in MODELS:
        return dict(_numbers("skipped", chosen_model, now, reason="model"), calls=0)
    calls = _Calls(genai_client) if genai_client is not None else None
    try:
        found = _newest_record(store, kind, arm, now)
        if found is None:
            return dict(_numbers("no_record", chosen_model, now), calls=0)
        run_id, record = found
        try:
            numbers = _reflect_on(store, run_id, record, calls, chosen_model, allow_same_model, now)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            numbers = _numbers("failed", chosen_model, now)
        run_record.add_test_numbers(store, run_id, KEY, numbers)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        numbers = _numbers("failed", chosen_model, now)
    return dict(numbers, calls=calls.made if calls is not None else 0)


def line(outcome: Dict[str, Any]) -> str:
    return LINE % (STATUS_NUMBERS.get(outcome.get("status"), STATUS_NUMBERS["failed"]),
                   int(outcome.get("plants_ok") is True), int(outcome.get("agreement") is True),
                   int(outcome.get("arithmetic_named") is True), int(outcome.get("calls") or 0))


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
    parser.add_argument("--kind", default="daily_run", help="the record kind: daily_run or private_run")
    parser.add_argument("--arm", default=None, help="only records of this comparison arm")
    parser.add_argument("--model", default=None, help="the reflector's model; only the two allowed models run")
    parser.add_argument("--allow-same-model", action="store_true",
                        help="run even when the reflector's model answered as Director")
    outcome: Dict[str, Any] = {"status": "failed"}
    # Library prints, log records and argument errors can carry private text; none of it leaves this process.
    hidden = StringIO()
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(hidden), redirect_stderr(hidden):
            try:
                args = parser.parse_args(argv)
                store = _store()
                genai_client = _genai_client()
                outcome = reflect(store, genai_client=genai_client, kind=args.kind, arm=(args.arm or None),
                                  model=(args.model or None), allow_same_model=args.allow_same_model, now=_now())
            except BaseException:
                outcome = {"status": "failed"}
    finally:
        logging.disable(logging.NOTSET)
    print(line(outcome))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
