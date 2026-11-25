"""Temporary: which parts of the published-night fingerprint differ between platforms. Synthetic data only."""

import hashlib
import json
import warnings

import pytest

from tests.test_failed_night_archive import _cell


def _d(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _rounded(value):
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


@pytest.mark.parametrize("cell", ["published_old_on", "published_off"])
def test_platform_probe(tmp_path, monkeypatch, capsys, cell):
    value = _cell(cell, tmp_path, monkeypatch, capsys)
    manifest = json.loads((tmp_path / "assets/pulse_manifest.json").read_text())
    parts = {"files." + k.rsplit("/", 1)[-1]: v[:12] for k, v in sorted(manifest["files"].items())}
    parts["publication_id"] = manifest["publication_id"][:12]
    entry = value.get("state") or value["night"]["published"][0]["entry"]
    parts["entry_minus_id"] = _d({k: v for k, v in entry.items() if k != "publication_id"})
    parts["out"] = _d(value["out"])
    if "night" in value:
        for name, part in sorted(value["night"].items()):
            if name != "published":
                parts["night." + name] = _d(part)
                parts["night." + name + ".r9"] = _d(_rounded(part))
        parts["meta"] = _d(value["meta"])
    warnings.warn("PROBE " + cell + " " + " ".join(f"{k}={v}" for k, v in parts.items()))
