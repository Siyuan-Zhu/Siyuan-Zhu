"""Operator script that reads and clears the private embedding calibration document."""

from __future__ import annotations

import re

import pytest

from scripts import embedding_calibration as script
from src.state_store import InMemoryStateStore
from src.zeitgeist.embedding_migration import (
    CALIBRATION_DOCUMENT,
    Calibration,
    calibration_document,
)

# The plan's anchored pattern, applied with fullmatch: the repository forbids a literal dollar sign.
LINE = re.compile(r"\[embedding_calibration\]( [a-z0-9_]+=-?[0-9.,-]+)+")


def _document(status: str) -> dict:
    return calibration_document(
        Calibration(
            status=status,
            date="2026-11-22",
            item_count=240,
            percentiles=(0.9975, 0.991, 0.982, 0.99995),
            multi_source=4,
            largest_share=0.125,
            top5_survivors=4,
        )
    )


def _store(document=None) -> InMemoryStateStore:
    store = InMemoryStateStore()
    if document is not None:
        store.save_calibration(CALIBRATION_DOCUMENT, document)
    return store


def _lines(capsys) -> list:
    lines = capsys.readouterr().out.splitlines()
    for line in lines:
        assert LINE.fullmatch(line), line
    return lines


class _StickyStore(InMemoryStateStore):
    """A store whose clear reports success but leaves the document in place."""

    def clear_calibration(self, name: str) -> bool:
        return True


def test_show_on_empty_store_reports_absent(capsys):
    assert script.main(["show"], store=_store()) == 0
    assert _lines(capsys) == [
        "[embedding_calibration] present=0 state=0 date=-1 items=-1 p=-1 "
        "multi_source=-1 largest_share=-1 top5_survivors=-1"
    ]


def test_show_reports_stored_numbers(capsys):
    assert script.main(["show"], store=_store(_document("accepted"))) == 0
    assert _lines(capsys) == [
        "[embedding_calibration] present=1 state=1 date=2026-11-22 items=240 "
        "p=0.997500,0.991000,0.982000,0.999950 multi_source=4 largest_share=0.125 "
        "top5_survivors=4"
    ]


def test_show_reports_invalid_document_without_its_values(capsys):
    store = _store({"schema_version": 99, "status": "accepted", "date": "2026-11-22"})
    assert script.main(["show"], store=store) == 0
    assert _lines(capsys) == [
        "[embedding_calibration] present=1 state=3 date=-1 items=-1 p=-1 "
        "multi_source=-1 largest_share=-1 top5_survivors=-1"
    ]


def test_clear_refuses_a_status_mismatch_and_keeps_the_document(capsys):
    store = _store(_document("rejected"))
    assert script.main(["clear", "--expected-status", "accepted"], store=store) == 2
    assert store.load_calibration(CALIBRATION_DOCUMENT) == _document("rejected")
    assert not any("cleared=1" in line for line in _lines(capsys))


def test_clear_with_the_matching_status_removes_the_document(capsys):
    store = _store(_document("rejected"))
    assert script.main(["clear", "--expected-status", "rejected"], store=store) == 0
    assert store.load_calibration(CALIBRATION_DOCUMENT) is None
    assert "[embedding_calibration] cleared=1" in _lines(capsys)


def test_clear_accepted_document(capsys):
    store = _store(_document("accepted"))
    assert script.main(["clear", "--expected-status", "rejected"], store=store) == 2
    assert store.load_calibration(CALIBRATION_DOCUMENT) is not None
    assert script.main(["clear", "--expected-status", "accepted"], store=store) == 0
    assert store.load_calibration(CALIBRATION_DOCUMENT) is None
    _lines(capsys)


def test_invalid_document_is_cleared_only_with_expected_invalid(capsys):
    hand_edited = dict(_document("accepted"), largest_share=1.5)
    store = _store(hand_edited)
    for status in ("accepted", "rejected"):
        assert script.main(["clear", "--expected-status", status], store=store) == 2
        assert store.load_calibration(CALIBRATION_DOCUMENT) == hand_edited
    assert script.main(["clear", "--expected-status", "invalid"], store=store) == 0
    assert store.load_calibration(CALIBRATION_DOCUMENT) is None
    assert "[embedding_calibration] cleared=1" in _lines(capsys)


@pytest.mark.parametrize("status", ["accepted", "rejected", "invalid"])
def test_clear_on_an_absent_document_returns_2(capsys, status):
    store = _store()
    assert script.main(["clear", "--expected-status", status], store=store) == 2
    assert store.load_calibration(CALIBRATION_DOCUMENT) is None
    assert not any("cleared=1" in line for line in _lines(capsys))


def test_clear_that_leaves_the_document_returns_3(capsys):
    store = _StickyStore()
    store.save_calibration(CALIBRATION_DOCUMENT, _document("rejected"))
    assert script.main(["clear", "--expected-status", "rejected"], store=store) == 3
    assert store.load_calibration(CALIBRATION_DOCUMENT) is not None
    assert not any("cleared=1" in line for line in _lines(capsys))


def test_clear_requires_a_known_expected_status(capsys):
    store = _store(_document("rejected"))
    with pytest.raises(SystemExit) as exit_info:
        script.main(["clear", "--expected-status", "anything"], store=store)
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit):
        script.main(["clear"], store=store)
    assert store.load_calibration(CALIBRATION_DOCUMENT) == _document("rejected")


def test_script_never_writes_a_calibration(capsys):
    class _NoSave(InMemoryStateStore):
        def save_calibration(self, name, document):  # pragma: no cover - must not run
            raise AssertionError("the operator script never writes a calibration")

    store = _NoSave()
    InMemoryStateStore.save_calibration(store, CALIBRATION_DOCUMENT, _document("rejected"))
    assert script.main(["show"], store=store) == 0
    assert script.main(["clear", "--expected-status", "rejected"], store=store) == 0
    _lines(capsys)
