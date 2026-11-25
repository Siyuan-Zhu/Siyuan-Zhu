"""Temporary: which parts of the published-night fingerprint differ between platforms. Synthetic data only."""

import hashlib
import json
import re
import warnings

import pytest

from tests.test_failed_night_archive import _cell


def _d(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _masked(value):
    """Digits masked: what is left when every number is taken out."""
    return re.sub(r"[0-9]+", "#", json.dumps(value, sort_keys=True, default=str))


def _rounded(value, digits):
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    return value


def test_platform_probe_2(tmp_path, monkeypatch, capsys):
    value = _cell("published_old_on", tmp_path, monkeypatch, capsys)
    night = value["night"]
    parts = {}
    groups = night["groups"]
    for key in sorted(groups[0]):
        column = [group.get(key) for group in groups]
        parts["g." + key] = _d(column)
        if any(isinstance(v, float) for v in column):
            parts["g." + key + ".r6"] = _d(_rounded(column, 6))
            parts["g." + key + ".r4"] = _d(_rounded(column, 4))
    attempts = night["director_attempts"]
    parts["att.req"] = _d([a["request_text"] for a in attempts])
    parts["att.req.masked"] = _d(_masked([a["request_text"] for a in attempts]))
    parts["att.reply"] = _d([a["reply_text"] for a in attempts])
    parts["att.rest"] = _d([{k: v for k, v in a.items() if k not in ("request_text", "reply_text")} for a in attempts])
    parts["ev.text"] = _d([e["evidence_text"] for e in night["director_evidence"]])
    parts["ev.text.masked"] = _d(_masked([e["evidence_text"] for e in night["director_evidence"]]))
    svg = (tmp_path / "assets/daily_pulse.svg").read_text()
    parts["svg.len"] = str(len(svg))
    parts["svg.masked"] = _d(_masked(svg))
    parts["meta.minus_commit"] = _d({k: v for k, v in value["meta"].items() if k != "commit"})
    warnings.warn("PROBE2 " + " ".join(f"{k}={v}" for k, v in parts.items()))
    # A few raw numbers of the first two groups and the first float lines of the evidence, numbers only.
    warnings.warn("PROBE2N " + " ".join(f"{k}={groups[0][k]!r}" for k in sorted(groups[0]) if isinstance(groups[0][k], float)))
