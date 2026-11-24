"""SIY-222: one account's copies of the same text count up to 15; from the 16th on they are dropped from the window."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.zeitgeist.methodology import CachedItem, load_asymmetric_window

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
DIM = 768
# The repository forbids a literal dollar character in test files; build the expression prefix instead.
DOLLAR = chr(36)
COPY_CAP_LINE = "ZEITGEIST_COPY_CAP: " + DOLLAR + "{{ vars.ZEITGEIST_COPY_CAP || '0' }}"
CANDIDATE_LINE = "ZEITGEIST_REFLECTOR_CANDIDATE: " + DOLLAR + "{{ vars.ZEITGEIST_REFLECTOR_CANDIDATE || '0' }}"


@pytest.fixture(autouse=True)
def _switch_unset(monkeypatch):
    """A shell that exports the switch must not change these tests; those that want it set it."""
    monkeypatch.delenv("ZEITGEIST_COPY_CAP", raising=False)


def _vec(seed: int) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    vec[seed % DIM] = 1.0
    vec[(seed * 7 + 3) % DIM] = 0.5
    return vec / float(np.linalg.norm(vec))


def _item(n: int, source_id: str, title: str, summary: str = "", *, hours: float | None = None,
          bucket: str = "ORBIT") -> CachedItem:
    return CachedItem(
        url_hash=f"h{n:03d}",
        source_id=source_id,
        bucket=bucket,
        tier="O1" if bucket == "ORBIT" else "M1",
        title=title,
        summary=summary,
        published_at=(NOW - timedelta(hours=float(n) if hours is None else hours)).isoformat(),
        embedding_f32=_vec(n),
    )


def _copies(count: int, source_id: str = "o1_ig_synth_a", start: int = 0, text: str = "Synthetic caption one"):
    return [_item(start + i, source_id, text, text) for i in range(count)]


def _hashes(window) -> list[str]:
    return [item.url_hash for item in window]


def test_six_copies_from_one_account_are_all_kept(monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    window = load_asymmetric_window(_copies(6), now_utc=NOW)
    assert len(window) == 6
    assert capsys.readouterr().out == "copy_cap_dropped=0\n"


def test_twenty_copies_from_one_account_keep_the_fifteen_most_recent(monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    items = _copies(20)
    window = load_asymmetric_window(list(reversed(items)), now_utc=NOW)
    # Item n is n hours old, so the fifteen most recent are h000..h014.
    assert sorted(_hashes(window)) == [f"h{n:03d}" for n in range(15)]
    assert capsys.readouterr().out == "copy_cap_dropped=5\n"
    # The kept copies are damped as fifteen items from one source, not twenty.
    for item in window:
        assert item.weight == pytest.approx(2.0 / math.sqrt(15.0))


def test_twenty_copies_over_two_accounts_are_all_kept(monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    items = _copies(10, "o1_ig_synth_a", start=0) + _copies(10, "o1_ig_synth_b", start=10)
    window = load_asymmetric_window(items, now_utc=NOW)
    assert sorted(_hashes(window)) == sorted(_hashes(items))
    assert capsys.readouterr().out == "copy_cap_dropped=0\n"


def test_trivial_differences_are_the_same_text(monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    variants = [
        ("Synthetic caption one!", "Synthetic caption one"),
        ("SYNTHETIC CAPTION ONE", "synthetic   caption\none"),
        ("synthetic caption one", "Synthetic caption one https://example.invalid/p/abc123"),
        ("  Synthetic, caption... one ", "Synthetic caption one www.example.invalid/x"),
    ]
    items = [_item(n, "o1_ig_synth_a", *variants[n % len(variants)]) for n in range(20)]
    # A different text from the same account is its own copy group and is not capped.
    items.append(_item(20, "o1_ig_synth_a", "Synthetic caption two", "Synthetic caption two"))
    window = load_asymmetric_window(items, now_utc=NOW)
    assert sorted(_hashes(window)) == [f"h{n:03d}" for n in range(15)] + ["h020"]
    assert capsys.readouterr().out == "copy_cap_dropped=5\n"


def test_items_with_no_text_are_not_treated_as_copies(monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    items = [_item(n, "o1_ig_synth_a", "", "") for n in range(18)]
    window = load_asymmetric_window(items, now_utc=NOW)
    assert len(window) == 18
    assert capsys.readouterr().out == "copy_cap_dropped=0\n"


def test_copies_outside_the_window_do_not_count_against_the_cap(monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    # Ten copies inside the 336h orbit window, ten older than it.
    items = _copies(10) + [_item(10 + i, "o1_ig_synth_a", "Synthetic caption one", "Synthetic caption one",
                                 hours=400.0 + i) for i in range(10)]
    window = load_asymmetric_window(items, now_utc=NOW)
    assert sorted(_hashes(window)) == [f"h{n:03d}" for n in range(10)]
    assert capsys.readouterr().out == "copy_cap_dropped=0\n"


def _mixed_night():
    """Orbit spam, orbit copies over two accounts, macro copies and plain items: today's window over all of them."""
    items = _copies(20, "o1_ig_synth_a", start=0)
    items += _copies(8, "o2_ig_synth_b", start=20, text="Synthetic caption shared")
    items += _copies(8, "o3_ig_synth_c", start=28, text="Synthetic caption shared")
    items += [_item(36 + i, "m1_synth_feed", "Synthetic headline", "Synthetic headline", bucket="MACRO")
              for i in range(18)]
    items += [_item(54 + i, f"m2_synth_feed_{i % 3}", f"Synthetic story {i}", "", bucket="MACRO", hours=2.0 + i)
              for i in range(9)]
    return items


def _fingerprint(window) -> str:
    rows = [[item.url_hash, repr(float(item.weight))] for item in window]
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


# sha256 of (url_hash, weight) for the mixed night's window as main built it before this change.
GOLDEN_OFF = "1662e8a43ec315c78b6f0e802069142364dd9a88ba021f40fe398ee5a0139939"


@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_switch_off_is_todays_window(monkeypatch, capsys, value):
    if value is not None:
        monkeypatch.setenv("ZEITGEIST_COPY_CAP", value)
    window = load_asymmetric_window(_mixed_night(), now_utc=NOW)
    assert len([item for item in window if item.source_id == "o1_ig_synth_a"]) == 20
    assert _fingerprint(window) == GOLDEN_OFF
    assert capsys.readouterr().out == ""


def test_switch_on_changes_only_the_spam_account(monkeypatch, capsys):
    off = _hashes(load_asymmetric_window(_mixed_night(), now_utc=NOW))
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    on = _hashes(load_asymmetric_window(_mixed_night(), now_utc=NOW))
    assert on == [h for h in off if h not in {f"h{n:03d}" for n in range(15, 20)}]
    # Five orbit copies, and three of the eighteen macro copies that the per-feed sample of four left out anyway.
    assert capsys.readouterr().out == "copy_cap_dropped=8\n"


def test_the_public_line_carries_a_number_only(monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_COPY_CAP", "1")
    load_asymmetric_window(_mixed_night(), now_utc=NOW)
    assert re.fullmatch(r"copy_cap_dropped=\d+\n", capsys.readouterr().out)


# ---- the workflows pass the switch, off by default ----


def _job_lines(text, job):
    from tests.test_run_record_wiring import _job_lines as job_lines

    return job_lines(text, job)


def test_the_daily_workflow_passes_the_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition[edition.index(CANDIDATE_LINE) + 1] == COPY_CAP_LINE
    assert COPY_CAP_LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(COPY_CAP_LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_workflows_pass_the_switch(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(COPY_CAP_LINE) == 1
    assert lines[lines.index(CANDIDATE_LINE) + 1] == COPY_CAP_LINE


def test_the_provider_canary_does_not_pass_the_switch():
    assert "ZEITGEIST_COPY_CAP" not in (ROOT / ".github/workflows/provider_canary.yml").read_text()
