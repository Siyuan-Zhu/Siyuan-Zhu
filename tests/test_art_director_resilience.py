"""The art director's retry and exit behaviour through the real spend guard (SIY-284 follow-ups)."""
import json
import pathlib
import subprocess
import sys
import time
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types
from PIL import Image

from src.zeitgeist import art_director as AD
from src.zeitgeist.spend_guard import guard_genai_client
from tests.test_hydrogen_admission_allowance import counter
from tests.test_neon_compatibility import NEON_MODEL, neon_guard

VERDICT = '{"verdict": "ok"}'


def _plate():
    return AD._jpeg(Image.new("RGB", (900, 600), (120, 130, 140)), AD.PLATE_PX)


def _answer():
    return types.GenerateContentResponse(
        response_id="synthetic-response", model_version="gemini-4-neon",
        candidates=[{"content": {"parts": [{"text": VERDICT}], "role": "model"}, "finish_reason": "STOP"}],
        usage_metadata={"prompt_token_count": 1400, "candidates_token_count": 40, "thoughts_token_count": 300,
                        "total_token_count": 1740})


def _deadline():
    return errors.ServerError(504, {"error": {"code": 504, "message": "Deadline expired", "status": "DEADLINE_EXCEEDED"}})


def _guarded(script, count):
    guard = neon_guard()
    dispatched = []

    def generate(**kwargs):
        dispatched.append(kwargs)
        step = script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    raw = SimpleNamespace(models=SimpleNamespace(generate_content=generate, count_tokens=count))
    return guard, guard_genai_client(raw, guard, request_timeout_seconds=AD.REQUEST_SECONDS), dispatched


def _records(guard):
    return sorted((r["status"] for r in guard.ledger.document["reservations"].values()))


def test_a_deadline_failure_is_retried_through_the_real_guard_with_its_own_hold():
    guard, api, dispatched = _guarded([_deadline(), _answer()], counter(1400))
    session = AD._Session(api, NEON_MODEL, time.monotonic)
    assert session.ask([_plate()], ["The plate:"], "Check one finished plate.") == json.loads(VERDICT)
    assert session.calls == 2 and session.retries == 1
    assert _records(guard) == ["reserved", "settled"]          # the failed attempt keeps its full hold
    timeouts = [call["config"]["http_options"]["timeout"] for call in dispatched]
    assert timeouts == [AD.REQUEST_SECONDS * 1000] * 2


def test_a_token_count_stall_is_retried_without_any_paid_call():
    counted = counter(1400)
    stalls = [httpx.ReadTimeout("timed out")]

    def count(**kwargs):
        if stalls:
            raise stalls.pop()
        return counted(**kwargs)

    guard, api, dispatched = _guarded([_answer()], count)
    session = AD._Session(api, NEON_MODEL, time.monotonic)
    assert session.ask([_plate()], ["The plate:"], "Check one finished plate.") == json.loads(VERDICT)
    assert session.calls == 2 and session.retries == 1 and len(dispatched) == 1
    assert _records(guard) == ["settled"]                        # the stalled count reserved nothing


def test_other_spend_blocks_are_still_never_retried():
    from src.zeitgeist.spend_guard import SpendBlocked

    blocked = SpendBlocked("Call would exceed approved monthly cash ceiling")
    blocked.refused_before_dispatch = True
    assert AD._retryable(blocked) is False


_EXIT = """
import sys, time
sys.path.insert(0, {root!r})
from src.zeitgeist import art_director as AD
from tests.test_art_director import CREDIT, DAY, GLOSS, THESIS, WORD, FakeClient, _happy, _photo
AD.BUDGET_SECONDS, AD.CALL_SECONDS, AD.EXIT_GRACE_SECONDS = 1.0, 0.1, {grace}
client = FakeClient(_happy()[4])
answer = client.generate_content
def late(**kw):
    time.sleep({sleep})
    print("answered", flush=True)
    return answer(**kw)
client.generate_content = late
built = AD.build_options(_photo(), word=WORD, gloss=GLOSS, date_iso=DAY, credit=CREDIT, history=[])
AD.build_options = lambda *a, **k: built
outcome = {{}}
AD.direct_night(_photo(), word=WORD, gloss=GLOSS, thesis=THESIS, date_iso=DAY, credit=CREDIT, history=[],
                model="test-model", client_factory=lambda: client, outcome=outcome)
print("returned", outcome["reason"], flush=True)
"""


def _exit_run(grace, sleep):
    root = str(pathlib.Path(__file__).resolve().parents[1])
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", _EXIT.format(root=root, grace=grace, sleep=sleep)], cwd=root,
                          capture_output=True, text=True, timeout=120)
    return done, time.monotonic() - started


def test_an_abandoned_call_that_answers_soon_settles_before_exit():
    done, elapsed = _exit_run(grace=10.0, sleep=3.0)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "returned time" in done.stdout and "answered" in done.stdout   # the process waited for the answer


def test_an_abandoned_call_that_hangs_holds_the_exit_for_the_grace_only():
    done, elapsed = _exit_run(grace=1.0, sleep=30.0)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "returned time" in done.stdout and "answered" not in done.stdout
    assert elapsed < 20, elapsed


def test_the_exit_grace_fits_the_job_limit():
    assert 380 + AD.BUDGET_SECONDS + AD.EXIT_GRACE_SECONDS <= 900 - 60
