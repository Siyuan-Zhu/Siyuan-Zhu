"""Conditional ledger writes retry known conflicts, never unknown outcomes."""

import ast
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import httpx
import pytest

from src import state_store as state
from src.zeitgeist import spend_guard as spend
from tests.test_spend_hardening import NOW, document


class LedgerWire:
    def __init__(self, monkeypatch, replies):
        self.doc = document()
        self.revision = 1
        self.replies = list(replies)
        self.methods = []
        original_client = httpx.Client
        self.transport = httpx.MockTransport(self.handle)
        monkeypatch.setattr(
            spend.httpx,
            "Client",
            lambda **kwargs: original_client(transport=self.transport, **kwargs),
        )
        monkeypatch.setattr(
            state._FirestoreRestClient,
            "_resolve_token",
            lambda _: "TEST_TOKEN_DO_NOT_LOG",
        )
        self.ledger = spend.FirestoreLedger("fixture", "2026-10")
        self.guard = spend.BudgetGuard(self.ledger, lambda: NOW)

    def handle(self, request):
        self.methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "updateTime": str(self.revision),
                    "fields": {
                        k: state._py_to_firestore_value(v) for k, v in self.doc.items()
                    },
                },
            )
        assert request.method == "PATCH"
        assert request.url.params["currentDocument.updateTime"] == str(self.revision)
        reply = self.replies.pop(0) if self.replies else None
        if isinstance(reply, tuple):
            status, code = reply
            # Simulate a concurrent writer winning, without losing its reservation.
            if code in {"FAILED_PRECONDITION", "ABORTED"} or status in {409, 412}:
                self.doc.setdefault("reservations", {})["other-writer"] = {
                    "provider": "apify",
                    "operation": "run",
                    "max_gross_usd": ".01",
                    "status": "reserved",
                }
                self.revision += 1
            return httpx.Response(
                status,
                json={
                    "error": {
                        "status": code,
                        "message": "TEST_TOKEN_DO_NOT_LOG https://private.invalid?key=secret",
                    }
                },
            )
        self.doc = {
            k: state._firestore_value_to_py(v)
            for k, v in json.loads(request.content)["fields"].items()
        }
        self.revision += 1
        if reply == "lost":
            raise httpx.ReadError(
                "TEST_TOKEN_DO_NOT_LOG https://private.invalid?key=secret",
                request=request,
            )
        if reply == "invalid-ack":
            return httpx.Response(200, content=b'{"updateTime":')
        return httpx.Response(
            200,
            json={
                "updateTime": str(self.revision),
                "fields": {
                    k: state._py_to_firestore_value(v) for k, v in self.doc.items()
                },
            },
        )


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "FAILED_PRECONDITION"),
        (400, "ABORTED"),
        (409, "CONFLICT"),
        (412, "FAILED_PRECONDITION"),
    ],
)
def test_known_conditional_conflict_rereads_and_preserves_other_reservation(
    monkeypatch, status, code
):
    wire = LedgerWire(monkeypatch, [(status, code)])
    reservation, cap = wire.guard.reserve("apify", "run", ".02")
    assert cap == spend.money(".02")
    assert wire.methods == ["GET", "PATCH", "GET", "PATCH"]
    assert set(wire.doc["reservations"]) == {"other-writer", reservation}


def test_invalid_argument_is_definitive_fatal_without_retry_or_public_body(monkeypatch):
    wire = LedgerWire(monkeypatch, [(400, "INVALID_ARGUMENT")])
    with pytest.raises(spend.SpendBlocked) as failure:
        wire.guard.reserve("apify", "run", ".02")
    assert wire.methods == ["GET", "PATCH"]
    assert not wire.doc.get("reservations")
    assert failure.value.public_diagnostics == {
        "provider": "firestore",
        "code": "shared-ledger-unavailable",
        "status": 400,
        "error_type": "INVALID_ARGUMENT",
    }
    assert "TEST_TOKEN" not in json.dumps(failure.value.public_diagnostics)


@pytest.mark.parametrize("reply", ["lost", "invalid-ack"])
def test_unknown_patch_outcome_keeps_committed_hold_and_never_retries_provider(
    monkeypatch, reply
):
    wire = LedgerWire(monkeypatch, [reply])
    calls = []
    with pytest.raises(spend.SpendBlocked) as failure:
        wire.guard.call("apify", "run", ".02", lambda: calls.append("PAID"))
    assert not calls
    assert wire.methods == ["GET", "PATCH"]
    assert len(wire.doc["reservations"]) == 1
    assert next(iter(wire.doc["reservations"].values()))["status"] == "reserved"
    assert (
        failure.value.public_diagnostics["code"]
        == "shared-ledger-write-outcome-unknown"
    )
    assert failure.value.public_diagnostics["provider"] == "firestore"
    assert "TEST_TOKEN" not in json.dumps(failure.value.public_diagnostics)


def test_bounded_conflict_exhaustion_is_diagnostic_without_provider_dispatch(
    monkeypatch,
):
    wire = LedgerWire(monkeypatch, [(400, "FAILED_PRECONDITION")] * 5)
    with pytest.raises(spend.SpendBlocked) as failure:
        wire.guard.reserve("apify", "run", ".02")
    assert wire.methods == ["GET", "PATCH"] * 5
    assert failure.value.public_diagnostics["code"] == "shared-ledger-contention"
    assert failure.value.public_diagnostics["status"] == 400
    assert failure.value.public_diagnostics["error_type"] == "FAILED_PRECONDITION"


@pytest.mark.parametrize(
    "code,expected",
    [
        ("FAILED_PRECONDITION", state.PublicationConflict),
        ("ABORTED", state.PublicationConflict),
        ("INVALID_ARGUMENT", RuntimeError),
    ],
)
def test_atomic_commit_400_classifies_only_structured_known_conflicts(
    monkeypatch, code, expected
):
    requests = []

    def urlopen(request, **kwargs):
        requests.append(request)
        raise HTTPError(
            request.full_url,
            400,
            "private",
            {},
            io.BytesIO(
                json.dumps(
                    {"error": {"status": code, "message": "TEST_TOKEN_DO_NOT_LOG"}}
                ).encode()
            ),
        )

    monkeypatch.setattr(state.urllib.request, "urlopen", urlopen)
    client = state._FirestoreRestClient("fixture", access_token="synthetic-token")
    with pytest.raises(expected) as failure:
        state._commit_documents(
            client,
            [
                (
                    "feed",
                    "v1",
                    {"entries": []},
                    state._FirestoreRestDocSnapshot("v1", False),
                )
            ],
        )
    assert type(failure.value) is expected
    assert len(requests) == 1
    assert "TEST_TOKEN" not in str(failure.value)


def test_all_literal_spend_reasons_have_explicit_public_codes():
    source = Path(spend.__file__).resolve().parents[2]
    nodes = [
        node
        for folder in ("src", "scripts")
        for path in (source / folder).rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
    ]
    reasons = {
        node.args[0].value
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SpendBlocked"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert reasons <= spend.SpendBlocked._PUBLIC_REASONS.keys()
    for reason in reasons:
        assert (
            spend.SpendBlocked(reason).public_diagnostics["code"]
            != "budget-admission-rejected"
        )
    assert spend.SpendBlocked("private DO_NOT_LOG").public_diagnostics == {
        "code": "budget-admission-rejected"
    }


@pytest.mark.parametrize(
    "reply,code,status",
    [
        ((503, "UNAVAILABLE"), "shared-ledger-write-outcome-unknown", 503),
        ((400, "SECRET_KEY_NOT_AN_ENUM"), "shared-ledger-unavailable", 400),
    ],
)
def test_unknown_server_write_status_or_arbitrary_error_enum_never_retries(
    monkeypatch, reply, code, status
):
    wire = LedgerWire(monkeypatch, [reply])
    with pytest.raises(spend.SpendBlocked) as failure:
        wire.guard.reserve("apify", "run", ".02")
    assert wire.methods == ["GET", "PATCH"]
    diagnostics = failure.value.public_diagnostics
    assert diagnostics == {
        "provider": "firestore",
        "code": code,
        "status": status,
        "error_type": "UNAVAILABLE" if status == 503 else "HTTPStatusError",
    }
    assert "SECRET_KEY" not in json.dumps(diagnostics)


def test_staging_report_keeps_safe_ledger_failure_diagnostics(tmp_path, monkeypatch):
    from scripts import rehearse_staging_application as application
    from src.zeitgeist import adk_pipeline as pipeline
    from tests.test_staging_application_rehearsal import PREFIX, setup

    root, wire, factory = setup(tmp_path, monkeypatch)

    def reject(*args, **kwargs):
        raise spend.SpendBlocked(
            "Shared spend ledger unavailable or unprovisioned",
            provider_status=400,
            provider_error_type="INVALID_ARGUMENT",
        )

    monkeypatch.setattr(pipeline, "collect_96h_signals", reject)
    with pytest.raises(application.ApplicationRehearsalError) as failure:
        application.rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
        )
    assert failure.value.report["budget_diagnostics"] == {
        "provider": "firestore",
        "status": 400,
        "error_type": "INVALID_ARGUMENT",
        "code": "shared-ledger-unavailable",
    }
