"""Nights that fail are archived behind ZEITGEIST_FAILED_NIGHT_ARCHIVE, with what they reached. Synthetic data only."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import evidence_archive
from tests.test_evidence_archive import META_ID, _ArchiveStore, _edition, _lines, _untimed
from tests.test_run_record_wiring import SLATE, _edition_setup
from tests.test_understanding_pipeline import _reply

SWITCH = "ZEITGEIST_FAILED_NIGHT_ARCHIVE"
OLD_SWITCH = evidence_archive.SWITCH_ENV

# Fingerprints taken from main (b07c994) before the change: the published night with the evidence archive on and
# with it off, a failed publication and a failed Director call with the evidence archive on. The one key left out
# is this switch's own entry in the archive meta's switch map, which every listed switch adds.
MAIN_FINGERPRINTS = {
    "published_old_on": "40113ba8c4336b6b7bc57a67277018430f823c0556a024ce8c724e28fab95509",
    "published_off": "cae70e748ad745063d1ddbe1d311b03ea187e57ab00593546675610bce9f7fe3",
    "publish_fails_old_on": "6b7915b7a8fe62dfaf3206ce1809a3543649a4d469afaa28ae65ceffe55132e8",
    "director_fails_old_on": "f9fb1d92fbf3beb3153c9832877b2d6b29ee615c87f3b5e831aa675663235436",
}


@pytest.fixture(autouse=True)
def _switches_unset(monkeypatch):
    for name in (SWITCH, OLD_SWITCH, "ZEITGEIST_OWN_GROUP", "ZEITGEIST_REFLECTOR", "ZEITGEIST_RUN_STARTED_AT"):
        monkeypatch.delenv(name, raising=False)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _night_print(store, out):
    """The archived night and the output, without the clock readings and this switch's own meta entry."""
    if not store.archive:
        return {"archive": {}, "out": _lines(out.out), "err": _lines(out.err)}
    night = evidence_archive.assemble(store.archive, META_ID)
    for published in night.get("published", []):
        published["entry"] = _untimed(published["entry"])
    meta = {key: value for key, value in store.archive[META_ID].items() if key != "written_at"}
    meta["switches"] = {key: value for key, value in meta["switches"].items() if key != SWITCH}
    return {"night": night, "meta": meta,
            "out": [line for line in _lines(out.out) if not line.startswith("evidence_archive=ok")],
            "err": _lines(out.err),
            "line_shape": [line.split(" ")[0] for line in _lines(out.out) if line.startswith("evidence_archive")]}


def _cell(cell, tmp_path, monkeypatch, capsys):
    if cell == "published_old_on":
        store, _, result, _ = _edition(tmp_path, monkeypatch, archive="1", seed=True,
                                       replies=["not json", _reply(selected_cluster_id="cluster_0", candidate_words=SLATE)])
        return {"result": result, **_night_print(store, capsys.readouterr())}
    if cell == "published_off":
        store, _, result, _ = _edition(tmp_path, monkeypatch, archive="0", seed=True)
        return {"result": result, "state": _untimed(store.load_pulse_state()["published_pulse"]),
                **_night_print(store, capsys.readouterr())}
    client, events = _edition_setup(tmp_path, monkeypatch, switch="0", publish_fails=cell == "publish_fails_old_on")
    if cell == "director_fails_old_on":
        def refuse(**kwargs):
            raise RuntimeError("synthetic director transport failure")
        client.generate_content = refuse
    monkeypatch.setenv(OLD_SWITCH, "1")
    store = _ArchiveStore(tmp_path, events)
    with pytest.raises(BaseException) as info:
        pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run("2026-11-21", "blueprint", offline_demo=True)
    return {"error": type(info.value).__name__, "events": list(events), "reconcile": store.needs_git_reconcile,
            "journal": (tmp_path / "data/publication_reconcile.json").exists(),
            **_night_print(store, capsys.readouterr())}


# ---- switch off: what main produced ----


@pytest.mark.parametrize("value", [None, "0", "true"])
@pytest.mark.parametrize("cell", sorted(MAIN_FINGERPRINTS))
def test_switch_off_matches_the_main_fingerprint(tmp_path, monkeypatch, capsys, cell, value):
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    assert _fingerprint(_cell(cell, tmp_path, monkeypatch, capsys)) == MAIN_FINGERPRINTS[cell]
