"""The six false public momentum fields leave the feed entry behind ZEITGEIST_DROP_FALSE_FIELDS, and the
breakout flag handed to the classic template choice becomes truthful. Synthetic data only."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from src.state_store import FirestoreStateStore, JsonFileStateStore
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import run_record
from src.zeitgeist.public_schema import (
    FALSE_MOMENTUM_FIELDS,
    drop_false_fields_enabled,
    validate_editorial_entry,
)
from tests.test_director_cluster_choice import _staged
from tests.test_firestore_store import _RecordingFirestoreClient, _make_sample_published_pulse
from tests.test_publication_atomic_commit import _publication_fixture
from tests.test_run_record_wiring import _job_lines

SWITCH = "ZEITGEIST_DROP_FALSE_FIELDS"
SIX = ("24h_shift", "dominant_category", "dominant_signal_type", "orbit_weight_ratio",
       "orbit_contribution_pct", "macro_contribution_pct")

# Fingerprints taken from main (caa0976) before the change, for the sample pulse in test_firestore_store.
MAIN_ENTRY = "586dea953be29f601975e3656e4b455c8f210e648c687c4e0ee9517b5c50e98b"
MAIN_ENTRY_CURATORIAL = "47e28cd566d5357b38fd3ef3ab70ee7f40aa46e46d7585d3c5fef9a1aa082c82"
MAIN_HISTORY_DOC = "964f38bebd06676663e3ba3b6f449aaa277bc6d7cd462b74767ffeb1eeb7a029"


@pytest.fixture(autouse=True)
def _switch_unset(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# ---- the switch ----


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False), ("true", False)])
def test_the_switch_is_on_only_for_one(value, expected):
    assert drop_false_fields_enabled({SWITCH: value}) is expected
    assert drop_false_fields_enabled({}) is False


def test_the_six_fields_are_the_ones_david_approved():
    assert set(FALSE_MOMENTUM_FIELDS) == set(SIX)


def test_the_switch_is_recorded_in_the_run_record_after_the_evidence_archive():
    names = run_record.SWITCH_NAMES
    assert names.index(SWITCH) == names.index("ZEITGEIST_EVIDENCE_ARCHIVE") + 1


# ---- switch off: byte-for-byte what main produced ----


def test_off_the_feed_entry_matches_the_main_fingerprint():
    pulse = _make_sample_published_pulse()
    assert _fingerprint(pulse.to_feed_entry()) == MAIN_ENTRY
    assert _fingerprint(pulse.to_feed_entry(include_curatorial_metadata=True)) == MAIN_ENTRY_CURATORIAL


def _legacy_history_doc(tmp_path):
    client = _RecordingFirestoreClient()
    store = FirestoreStateStore(client=client, project_id="t", fallback_store=JsonFileStateStore(repo_root=tmp_path))
    store.save_pulse_state({
        "date": "2026-10-28", "word": "Cantilever", "gloss": "g",
        "analysis": {"date_iso": "2026-10-28", "word_of_the_day": "Cantilever", "explanation_phrase": "e"},
        "published_at": "2026-10-28T06:00:00Z", "updated_at": "2026-10-28T06:00:00Z",
        "mood_votes": {"spot_on": 1, "too_sunny": 0, "too_heavy": 0},
    })
    return client.docs["pulse_history/2026-10-28"]


def test_off_the_legacy_history_row_matches_the_main_fingerprint(tmp_path):
    assert _fingerprint(_legacy_history_doc(tmp_path)) == MAIN_HISTORY_DOC


# ---- switch on: the six are gone, the rest stays ----


def test_on_the_six_fields_are_absent_and_nothing_else_changes(monkeypatch):
    pulse = _make_sample_published_pulse()
    before = pulse.to_feed_entry()
    monkeypatch.setenv(SWITCH, "1")
    after = pulse.to_feed_entry()
    assert not set(SIX) & set(after)
    assert after == {key: value for key, value in before.items() if key not in SIX}
    assert after["window_hours"] == 96 and "orbit_sources_matched" in after
    assert "top_orbit_themes" in after and "top_macro_themes" in after
    validate_editorial_entry(after)


def test_on_the_curatorial_projection_drops_the_six_too(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    entry = _make_sample_published_pulse().to_feed_entry(include_curatorial_metadata=True)
    assert not set(SIX) & set(entry) and entry["curatorial_archetype"] and entry["text_quadrant"]


def test_on_the_legacy_history_row_has_no_invented_values(monkeypatch, tmp_path):
    monkeypatch.setenv(SWITCH, "1")
    row = _legacy_history_doc(tmp_path)
    assert not set(SIX) & set(row)
    assert row["word"] == "CANTILEVER" and row["window_hours"] == 96


def test_on_the_local_feed_entry_equals_the_generated_entry(monkeypatch, tmp_path):
    monkeypatch.setenv(SWITCH, "1")
    pulse = _make_sample_published_pulse()
    generated = pulse.to_feed_entry()
    JsonFileStateStore(repo_root=tmp_path).save_pulse_feed_entry("2026-10-28", generated)
    [local] = json.loads((tmp_path / "data/pulse_feed.json").read_text())["entries"]
    assert local == generated


# ---- a same-date republication across the switch is accepted ----


def test_a_same_date_republication_after_the_switch_flips_is_accepted(tmp_path, monkeypatch):
    from src.zeitgeist.publication_commit import publish_bundle

    wire, store = _publication_fixture(tmp_path, monkeypatch)
    old = store.load_pulse_state()
    previous = old["published_pulse"]
    assert set(SIX) <= set(previous)
    monkeypatch.setenv(SWITCH, "1")
    entry = {key: value for key, value in previous.items() if key not in SIX} | {"word": "AGAIN"}
    publish_bundle(store, old, entry, expected=old, local_feed={"entries": []}, prepare=lambda *args: None)
    [fed] = [e for e in wire.docs["staging_pulse_feed/v1"]["entries"] if e["date"] == previous["date"]]
    assert fed["word"] == "AGAIN" and not set(SIX) & set(fed)
    validate_editorial_entry(fed)


# ---- the breakout flag ----


def _breakout_given_to_the_renderer(tmp_path, monkeypatch, surge):
    real = pipeline.score_and_cluster_signals
    monkeypatch.setattr(pipeline, "score_and_cluster_signals",
                        lambda rows, **kw: [dataclasses.replace(c, breakout_surge=surge) for c in real(rows, **kw)])
    seen = _staged(tmp_path, monkeypatch, choose=lambda clusters, kw: clusters[0].cluster_id)
    [kw] = seen["render"]
    return kw["breakout_surge"]


@pytest.mark.parametrize("surge, expected", [(1.0, False), (1.25, True)])
def test_on_the_flag_is_true_only_for_a_real_surge(tmp_path, monkeypatch, surge, expected):
    monkeypatch.setenv(SWITCH, "1")
    assert _breakout_given_to_the_renderer(tmp_path, monkeypatch, surge) is expected


@pytest.mark.parametrize("surge", [1.0, 1.25])
def test_off_the_flag_stays_always_true(tmp_path, monkeypatch, surge):
    assert _breakout_given_to_the_renderer(tmp_path, monkeypatch, surge) is True


# ---- the workflows pass the switch, off by default ----

ROOT = Path(__file__).resolve().parents[1]
DOLLAR = chr(36)
LINE = SWITCH + ": " + DOLLAR + "{{ vars." + SWITCH + " || '0' }}"


def test_the_daily_workflow_passes_the_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    assert LINE in _job_lines(text, "synthesize-pulse")
    assert LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_workflows_pass_the_switch(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(LINE) == 1


def test_the_provider_canary_does_not_pass_the_switch():
    assert SWITCH not in (ROOT / ".github/workflows/provider_canary.yml").read_text()
