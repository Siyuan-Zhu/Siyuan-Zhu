"""Nights that fail are archived behind ZEITGEIST_FAILED_NIGHT_ARCHIVE, with what they reached. Synthetic data only."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import evidence_archive, run_record
from src.zeitgeist.collectors import IngestionUnavailable
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_evidence_archive import META_ID, _ArchiveStore, _job_lines, _lines, _seed_cache, _untimed
from tests.test_run_record_wiring import SLATE, _edition_setup
from tests.test_understanding_pipeline import _reply

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_LINE = re.compile(r"evidence_archive=(ok docs=\d+ bytes=\d+|failed error_class=[A-Za-z_][A-Za-z0-9_]*"
                         r"|skipped reason=no_time_left)")

SWITCH = "ZEITGEIST_FAILED_NIGHT_ARCHIVE"
OLD_SWITCH = evidence_archive.SWITCH_ENV

# Fingerprints taken from main (b07c994) before the change: the published night with the evidence archive on and
# with it off, a failed publication and a failed Director call with the evidence archive on. The one key left out
# is this switch's own entry in the archive meta's switch map, which every listed switch adds. The published
# nights run on one-hot vectors (_seed_exact) and without GITHUB_SHA, so the digests hold on macOS and Linux CI.
MAIN_FINGERPRINTS = {
    "published_old_on": "1f41efdb3775b5b362df9da22da8b5e14b7bb5c4c86593e5deb8f9ab83897b23",
    "published_off": "cae70e748ad745063d1ddbe1d311b03ea187e57ab00593546675610bce9f7fe3",
    "publish_fails_old_on": "6b7915b7a8fe62dfaf3206ce1809a3543649a4d469afaa28ae65ceffe55132e8",
    "director_fails_old_on": "f9fb1d92fbf3beb3153c9832877b2d6b29ee615c87f3b5e831aa675663235436",
}


@pytest.fixture(autouse=True)
def _switches_unset(monkeypatch):
    # GITHUB_SHA is the archive meta's "commit": set in CI, unset locally.
    for name in (SWITCH, OLD_SWITCH, "ZEITGEIST_OWN_GROUP", "ZEITGEIST_REFLECTOR", "ZEITGEIST_RUN_STARTED_AT", "GITHUB_SHA"):
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


def _seed_exact(store):
    """Six stored rows whose vectors are one-hot, so every similarity is exactly 0 or 1 under any BLAS.

    The hashed vectors of _seed_cache give similarities whose last bits differ between Accelerate (macOS) and
    OpenBLAS (Linux CI); among tied groups that changed the order, the chosen group, the artwork and its digest.
    """
    from src.zeitgeist.collectors import embed_missing_items

    rows = [{
        "url_hash": f"seed{i}", "source_id": f"m1_seed_{i}", "bucket": "MACRO", "tier": "M1", "origin": "curated_M1_M3",
        "title": f"Seeded title {i}", "summary": f"Seeded summary {i} about looms and weaving", "caption": f"Seeded caption {i}",
        "url": f"https://example.invalid/seed/{i}", "published_at": "2026-11-21T09:00:00+00:00",
        "fetched_at": "2026-11-21T09:30:00+00:00",
    } for i in range(6)]
    embedded = embed_missing_items(rows, client=None)
    for i, row in enumerate(embedded):
        vector = np.zeros(len(row["embedding_f16"]) // 2, dtype=np.float16)
        vector[i] = 1.0
        row["embedding_f16"] = vector.tobytes()
        row["embedding_f16_b64"] = base64.b64encode(vector.tobytes()).decode("ascii")
    store.save_ingestion_cache_day("2026-11-21", {row["url_hash"]: row for row in embedded})


def _published(tmp_path, monkeypatch, *, archive, replies=None):
    client, events = _edition_setup(tmp_path, monkeypatch, switch="0")
    if replies is not None:
        client.replies = list(replies)
    monkeypatch.setenv(OLD_SWITCH, archive)
    store = _ArchiveStore(tmp_path, events)
    _seed_exact(store)
    analysis, decision, _ = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run(
        "2026-11-21", "blueprint", offline_demo=True)
    return store, (analysis.word_of_the_day, analysis.explanation_phrase, decision.selected_mode)


def _cell(cell, tmp_path, monkeypatch, capsys):
    if cell == "published_old_on":
        store, result = _published(tmp_path, monkeypatch, archive="1",
                                   replies=["not json", _reply(selected_cluster_id="cluster_0", candidate_words=SLATE)])
        return {"result": result, **_night_print(store, capsys.readouterr())}
    if cell == "published_off":
        store, result = _published(tmp_path, monkeypatch, archive="0")
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


@pytest.mark.parametrize("cell", ["published_old_on", "published_off"])
def test_switch_on_a_published_night_still_matches_the_main_fingerprint(tmp_path, monkeypatch, capsys, cell):
    """On, a published night takes the Director's attempts into the capture; its one archive is main's."""
    monkeypatch.setenv(SWITCH, "1")
    assert _fingerprint(_cell(cell, tmp_path, monkeypatch, capsys)) == MAIN_FINGERPRINTS[cell]


# ---- the switch: on only with the evidence archive on too, recorded last, passed by the three workflows ----


@pytest.mark.parametrize("old, new, on", [(None, None, False), ("1", None, False), ("1", "0", False), ("1", "true", False),
                                          (None, "1", False), ("0", "1", False), ("1", "1", True)])
def test_the_switch_is_on_only_for_one_with_the_evidence_archive_on(old, new, on):
    environ = {name: value for name, value in ((OLD_SWITCH, old), (SWITCH, new)) if value is not None}
    assert evidence_archive.FAILED_NIGHT_SWITCH_ENV == SWITCH
    assert evidence_archive.failed_night_enabled(environ) is on


def test_the_switch_is_recorded_last_in_the_run_record():
    assert run_record.SWITCH_NAMES[-1] == SWITCH


DOLLAR = chr(36)
SWITCH_LINE = SWITCH + ": " + DOLLAR + "{{ vars." + SWITCH + " || '0' }}"
DROP_LINE = "ZEITGEIST_DROP_FALSE_FIELDS: " + DOLLAR + "{{ vars.ZEITGEIST_DROP_FALSE_FIELDS || '0' }}"


def test_the_daily_workflow_passes_the_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition[edition.index(DROP_LINE) + 1] == SWITCH_LINE
    assert SWITCH_LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(SWITCH_LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_runs_pass_the_switch_too(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(SWITCH_LINE) == 1
    assert lines[lines.index(DROP_LINE) + 1] == SWITCH_LINE


@pytest.mark.parametrize("name", ["provider_canary.yml", "staging_zeitgeist_dryrun.yml"])
def test_no_other_workflow_passes_the_switch(name):
    assert SWITCH not in (ROOT / ".github/workflows" / name).read_text()


# ---- the budget: 120 s, never past the job's own deadline ----


NOW = datetime(2026, 11, 25, 8, 0, tzinfo=timezone.utc)
DEADLINE = 12 * 60


def _started(seconds_ago):
    return (NOW - timedelta(seconds=seconds_ago)).isoformat()


@pytest.mark.parametrize("raw, budget", [
    (None, 60.0), ("", 60.0), ("not-a-time", 60.0), ("2026-11-25T07:59:00", 60.0),
    (_started(60), 120.0), ("2026-11-25T07:59:00Z", 120.0), (_started(DEADLINE - 30), 30.0),
    (_started(DEADLINE), 0.0), (_started(DEADLINE + 300), 0.0),
])
def test_the_budget_is_capped_to_the_time_left_before_the_deadline(raw, budget):
    assert evidence_archive.FAILED_NIGHT_DEADLINE_SECONDS == DEADLINE
    environ = {} if raw is None else {"ZEITGEIST_RUN_STARTED_AT": raw}
    assert evidence_archive.failed_night_budget(environ, now=NOW) == pytest.approx(budget)


# ---- a failed night: archived with what it reached, the same exception raised again ----


SENTINEL = "synthetic-private-message-7f3a"
ALL_PARTS = ("intake_rows", "groups", "director_attempts", "director_evidence", "commissions", "photo_candidates",
             "published")
JOURNAL = "data/publication_reconcile.json"


class _WatchingStore(_ArchiveStore):
    """Notes the recovery marker and journal as the first archive document is saved."""

    def save_evidence_archive_doc(self, doc_id, document):
        if not hasattr(self, "seen_at_archive"):
            journal = self.repo_root / JOURNAL
            self.seen_at_archive = {"reconcile": self.needs_git_reconcile,
                                    "journal": json.loads(journal.read_text())["status"] if journal.exists() else None}
        super().save_evidence_archive_doc(doc_id, document)


def _fail(tmp_path, monkeypatch, capsys, *, switch="1", old="1", publish_fails=False, store_cls=_WatchingStore,
          setup=None, expect=BaseException, mode="blueprint", **run_kwargs):
    """One failing night; setup(client) arranges the failure and returns the exception it will raise."""
    client, events = _edition_setup(tmp_path, monkeypatch, switch="0", publish_fails=publish_fails)
    for name, value in ((OLD_SWITCH, old), (SWITCH, switch)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    store = store_cls(tmp_path, events)
    seeded = _seed_cache(store)
    raised = setup(client) if setup is not None else None
    with pytest.raises(expect) as info:
        pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run("2026-11-21", mode, offline_demo=True,
                                                                           **run_kwargs)
    out = capsys.readouterr()
    assert SENTINEL not in out.out + out.err
    assert SENTINEL not in json.dumps(store.archive)
    assert all(PUBLIC_LINE.fullmatch(line) for line in _lines(out.out) if "evidence_archive" in line)
    return {"store": store, "error": info.value, "raised": raised, "out": out, "client": client, "seeded": seeded}


def _night(run):
    store = run["store"]
    meta = store.archive[META_ID]
    return meta, evidence_archive.assemble(store.archive, META_ID)


def _raise_in_collection(monkeypatch, error):
    real = pipeline.collect_96h_signals

    def collect(*args, **kwargs):
        real(*args, **kwargs)
        raise error

    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)
    return error


def _raise_in_grouping(monkeypatch):
    error = RuntimeError(SENTINEL)

    def group(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipeline, "score_and_cluster_signals", group)
    return error


def _raise_in_director(client, error):
    replies = iter(["not json"])

    def generate_content(*, model, contents, config):
        client.prompts.append(contents)
        reply = next(replies, None)
        if reply is None:
            raise error
        return SimpleNamespace(text=reply)

    client.generate_content = generate_content
    return error


def _raise_in_photo(monkeypatch):
    from src.zeitgeist.visual_router import PhotoSelectionUnavailable

    error = PhotoSelectionUnavailable(SENTINEL)

    def render(*args, **kwargs):
        # Given only with the evidence archive on.
        scored = kwargs.get("scored_candidates_out")
        if scored is not None:
            scored.extend([
                {"index": 0, "photo_id": "p0", "composite_score": 0.41, "safety_pass": True, "thumb_bytes": b"\xff"},
                {"index": 1, "photo_id": "p1", "composite_score": 0.38, "safety_pass": True},
            ])
        raise error

    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    return error


def _raise_in_validation(monkeypatch):
    from src.zeitgeist import public_schema

    error = ValueError(SENTINEL)

    def validate(entry):
        raise error

    monkeypatch.setattr(public_schema, "validate_editorial_entry", validate)
    return error


def _raise_in_assembly(monkeypatch):
    """After rendering: the compatibility records built from the published artwork."""
    error = RuntimeError(SENTINEL)

    def analyze(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipeline, "analyze_seven_bucket_orbit", analyze)
    return error


def _raise_in_setup(monkeypatch):
    from src.zeitgeist.publication_commit import PublicationWorkspace

    error = RuntimeError(SENTINEL)

    def stage(self, store):
        raise error

    monkeypatch.setattr(PublicationWorkspace, "stage_canonical_game", stage)
    return error


STAGES = {
    "setup": (lambda mp, client: _raise_in_setup(mp), ()),
    "collection": (lambda mp, client: _raise_in_collection(mp, IngestionUnavailable(SENTINEL)), ("intake_rows",)),
    "grouping": (lambda mp, client: _raise_in_grouping(mp), ("intake_rows",)),
    "director": (lambda mp, client: _raise_in_director(client, RuntimeError(SENTINEL)),
                 ("intake_rows", "groups", "director_attempts")),
    "photo": (lambda mp, client: _raise_in_photo(mp), ALL_PARTS[:-1]),
    "validation": (lambda mp, client: _raise_in_validation(mp), ALL_PARTS),
    "assembly": (lambda mp, client: _raise_in_assembly(mp), ALL_PARTS),
}


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_a_failure_at_each_stage_archives_exactly_what_it_reached(tmp_path, monkeypatch, capsys, stage):
    arrange, reached = STAGES[stage]
    run = _fail(tmp_path, monkeypatch, capsys, setup=lambda client: arrange(monkeypatch, client))
    assert run["error"] is run["raised"]
    meta, night = _night(run)
    assert set(night) == set(reached)
    assert (meta["status"], meta["failed_stage"], meta["error_class"]) == ("failed", stage, type(run["raised"]).__name__)
    assert meta["date"] == "2026-11-21" and meta["run_id"] == "daily_run:2026-11-21:local:1"
    assert meta["switches"][SWITCH] == "1" and meta["switches"][OLD_SWITCH] == "1"
    if "intake_rows" in reached:
        assert {row["url_hash"] for row in night["intake_rows"]} == set(run["seeded"])
        assert meta["intake_source"] == ("collector_rows" if stage == "collection" else "stored_rows")
    if "groups" in reached:
        assert night["groups"] and night["groups"][0]["rank"] == 1
    if stage == "director":
        client = run["client"]
        assert [(a["pass"], a["attempt"], a["reply_text"]) for a in night["director_attempts"]] == [("main", 1, "not json")]
        assert night["director_attempts"][0]["request_text"] == client.prompts[0]
    if "commissions" in reached:
        assert [c["pass"] for c in night["commissions"]] == ["main"]
        assert night["commissions"][0]["word"] == "Selvedge"
        assert [e["pass"] for e in night["director_evidence"]] == ["main"]
    if stage == "photo":
        assert [(c["render_call"], c["photo_id"], c["chosen"]) for c in night["photo_candidates"]] == [
            (1, "p0", False), (1, "p1", False)]
        assert all("thumb_bytes" not in c for c in night["photo_candidates"])
    if stage == "validation":
        [published] = night["published"]
        assert published["entry"]["word"] == "Selvedge"
    if stage == "assembly":
        [published] = night["published"]
        assert published["entry"] is None and published["visual_medium"]
    [line] = [line for line in _lines(run["out"].out) if "evidence_archive" in line]
    assert line.startswith("evidence_archive=ok ")
    assert run["store"].seen_at_archive == {"reconcile": False, "journal": None}


def test_a_failed_publication_writes_the_archive_with_the_switch_on(tmp_path, monkeypatch, capsys):
    """The twin of test_evidence_archive.test_a_failed_publication_writes_no_archive, with this switch on."""
    run = _fail(tmp_path, monkeypatch, capsys, publish_fails=True, expect=RuntimeError)
    meta, night = _night(run)
    assert (meta["status"], meta["failed_stage"], meta["error_class"]) == ("failed", "publication", "RuntimeError")
    assert set(night) == set(ALL_PARTS)
    assert night["published"][0]["entry"]["word"] == "Selvedge"
    store = run["store"]
    assert store.events[0] == "publish_bundle:called" and set(store.events[1:]) == {"save_evidence_archive_doc"}
    assert store.seen_at_archive == {"reconcile": False, "journal": None}
    assert store.needs_git_reconcile is False and not (tmp_path / JOURNAL).exists()


@pytest.mark.parametrize("error_cls", [RuntimeError, SpendBlocked])
def test_a_director_failure_is_archived_and_raised_again_spend_blocked_included(tmp_path, monkeypatch, capsys, error_cls):
    run = _fail(tmp_path, monkeypatch, capsys, setup=lambda client: _raise_in_director(client, error_cls(SENTINEL)))
    assert run["error"] is run["raised"]
    meta, _ = _night(run)
    assert (meta["failed_stage"], meta["error_class"]) == ("director", error_cls.__name__)


def _install_fails(monkeypatch):
    from src.zeitgeist.publication_commit import PublicationWorkspace

    error = OSError(SENTINEL)

    def install(self, state):
        raise error

    monkeypatch.setattr(PublicationWorkspace, "install", install)
    return error


def _download_tracking_fails(monkeypatch):
    """A published Unsplash photo whose download acknowledgement fails, after install."""
    from src.zeitgeist import visual_router as vr
    from tests.test_private_run import _jpeg, _photo, _photo_world

    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_LIVE_UNSPLASH", "1")
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "synthetic-key")

    def candidates():
        rows = []
        for photo_id in ("q0", "q1"):
            row = vr._candidate_from_unsplash(_photo(photo_id), _jpeg((8, 8)))
            row.update(semantic_alignment=0.9, compositional_clarity=0.9, editorial_memorability=0.9,
                       safety_pass=True, _vision_verified=True)
            rows.append(row)
        return rows

    real_render = pipeline.select_and_render_pulse_assets

    def render(*args, **kwargs):
        return real_render(*args, **{**kwargs, "candidates": candidates(), "offline_demo": False})

    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    error = RuntimeError(SENTINEL)

    def confirm(published):
        raise error

    monkeypatch.setattr(vr, "confirm_published_photo_download", confirm)
    return error


def _followup_fails(monkeypatch):
    """A step after install that is not the download acknowledgement: the edition is committed and installed."""
    error = RuntimeError(SENTINEL)
    real_replace = pipeline.replace

    def replace(obj, **changes):
        if "mood_votes" in changes:
            raise error
        return real_replace(obj, **changes)

    monkeypatch.setattr(pipeline, "replace", replace)
    return error


def _outcome_unknown(monkeypatch):
    from src.state_store import PublicationOutcomeUnknown
    from src.zeitgeist import publication_commit

    error = PublicationOutcomeUnknown(SENTINEL)

    def publish(*args, **kwargs):
        raise error

    monkeypatch.setattr(publication_commit, "publish_bundle", publish)
    return error


PUBLICATION_BRANCH = {
    "committed_install_failed": (_install_fails, {"reconcile": True, "journal": "canonical_committed_local_install_failed"},
                                 {}),
    "committed_installed_followup_failed": (_followup_fails, {"reconcile": True, "journal": None}, {}),
    "committed_download_tracking_failed": (_download_tracking_fails,
                                           {"reconcile": True, "journal": "canonical_committed_download_tracking_pending"},
                                           {"mode": None}),
    "outcome_unknown": (_outcome_unknown, {"reconcile": True, "journal": None}, {}),
}


@pytest.mark.parametrize("status", sorted(PUBLICATION_BRANCH))
def test_the_publication_branch_is_labelled_by_its_outcome_and_archived_after_the_marker(tmp_path, monkeypatch, capsys,
                                                                                        status):
    from src.zeitgeist.publication_commit import PublicationMirrorError

    arrange, seen, run_kwargs = PUBLICATION_BRANCH[status]
    run = _fail(tmp_path, monkeypatch, capsys, setup=lambda client: arrange(monkeypatch), expect=PublicationMirrorError,
                **run_kwargs)
    assert run["error"].__cause__ is run["raised"]
    meta, night = _night(run)
    assert (meta["status"], meta["failed_stage"], meta["error_class"]) == (
        status, "publication", type(run["raised"]).__name__)
    assert run["store"].seen_at_archive == seen
    if status != "outcome_unknown":
        # The entry as committed, with its publication id.
        assert night["published"][0]["entry"]["publication_id"]


@pytest.mark.parametrize("arrange", [_install_fails, lambda mp: None], ids=["install_fails", "publish_fails"])
def test_the_recovery_state_and_output_are_the_same_with_the_switch_on_and_off(tmp_path, monkeypatch, capsys, arrange):
    seen = {}
    for value in ("0", "1"):
        root = tmp_path / value
        run = _fail(root, monkeypatch, capsys, switch=value, publish_fails=arrange is not _install_fails,
                    setup=lambda client: arrange(monkeypatch))
        journal = root / JOURNAL
        seen[value] = {
            "error": type(run["error"]).__name__, "cause": type(run["error"].__cause__).__name__,
            "reconcile": run["store"].needs_git_reconcile,
            "journal": json.loads(journal.read_text())["status"] if journal.exists() else None,
            "out": [line for line in _lines(run["out"].out) if "evidence_archive" not in line],
            "err": run["out"].err,
        }
        assert bool(run["store"].archive) is (value == "1")
    assert seen["0"] == seen["1"]


@pytest.mark.parametrize("error_cls", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_or_an_exit_skips_the_archive(tmp_path, monkeypatch, capsys, error_cls):
    run = _fail(tmp_path, monkeypatch, capsys, expect=error_cls,
                setup=lambda client: _raise_in_collection(monkeypatch, error_cls()))
    assert run["error"] is run["raised"]
    assert run["store"].archive == {}
    assert "evidence_archive" not in run["out"].out


@pytest.mark.parametrize("old, new", [("1", "0"), ("0", "1"), (None, "1")])
def test_with_either_switch_off_a_failed_night_writes_nothing_and_passes_no_new_argument(tmp_path, monkeypatch, capsys,
                                                                                          old, new):
    seen, collected = [], []
    real_collect = pipeline.collect_96h_signals

    def collect(*args, **kwargs):
        collected.append(set(kwargs))
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)

    def arrange(client):
        patched = pipeline.commission_daily_zeitgeist

        def commission(*args, **kwargs):
            seen.append(set(kwargs))
            return patched(*args, **kwargs)

        monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", commission)
        return _raise_in_photo(monkeypatch)

    run = _fail(tmp_path, monkeypatch, capsys, old=old, switch=new, setup=arrange)
    assert run["error"] is run["raised"]
    assert run["store"].archive == {}
    assert "evidence_archive" not in run["out"].out
    assert seen and all("attempts_out" not in kwargs for kwargs in seen)
    assert collected and all("reached_rows_out" not in kwargs for kwargs in collected)


def test_a_stalled_store_is_cut_off_within_the_capped_budget(tmp_path, monkeypatch, capsys):
    # About three seconds left once the night has run: seven documents at a second each cannot all land.
    started = datetime.now(timezone.utc) - timedelta(seconds=DEADLINE - 3)
    monkeypatch.setenv("ZEITGEIST_RUN_STARTED_AT", started.isoformat())

    class _SlowStore(_ArchiveStore):
        def save_evidence_archive_doc(self, doc_id, document):
            time.sleep(1.0)
            super().save_evidence_archive_doc(doc_id, document)

    began = time.monotonic()
    run = _fail(tmp_path, monkeypatch, capsys, publish_fails=True, store_cls=_SlowStore, expect=RuntimeError)
    assert time.monotonic() - began < 10
    assert [line for line in _lines(run["out"].out) if "evidence_archive" in line] == [
        "evidence_archive=failed error_class=TimeoutError"]
    time.sleep(1.5)
    assert META_ID not in run["store"].archive


def test_past_the_deadline_nothing_is_written(tmp_path, monkeypatch, capsys):
    started = datetime.now(timezone.utc) - timedelta(seconds=DEADLINE + 5)
    monkeypatch.setenv("ZEITGEIST_RUN_STARTED_AT", started.isoformat())
    run = _fail(tmp_path, monkeypatch, capsys, publish_fails=True, expect=RuntimeError)
    assert run["store"].archive == {}
    assert run["store"].events == ["publish_bundle:called"]
    assert [line for line in _lines(run["out"].out) if "evidence_archive" in line] == [
        "evidence_archive=skipped reason=no_time_left"]


def test_a_failing_archive_store_leaves_the_failure_as_it_was(tmp_path, monkeypatch, capsys):
    class _RaisingStore(_ArchiveStore):
        def save_evidence_archive_doc(self, doc_id, document):
            raise RuntimeError(SENTINEL)

    run = _fail(tmp_path, monkeypatch, capsys, store_cls=_RaisingStore, setup=lambda client: _raise_in_photo(monkeypatch))
    assert run["error"] is run["raised"]
    assert [line for line in _lines(run["out"].out) if "evidence_archive" in line] == [
        "evidence_archive=failed error_class=RuntimeError"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("arrange, expected", [
    (lambda mp: None, {"reconcile": False, "journal": None}),
    (_install_fails, {"reconcile": True, "journal": "canonical_committed_local_install_failed"}),
], ids=["publish_fails", "install_fails"])
def test_an_interrupt_while_the_archive_waits_leaves_the_failure_and_its_outcome(tmp_path, monkeypatch, capsys,
                                                                                 arrange, expected, interrupt):
    """A cancel during the wait is swallowed: the night's own exception goes on, with its recovery state."""
    from src.zeitgeist.publication_commit import PublicationMirrorError

    def interrupted(fn, seconds, *, name):
        raise interrupt

    monkeypatch.setattr(evidence_archive, "run_with_budget", interrupted)
    install = arrange is _install_fails
    run = _fail(tmp_path, monkeypatch, capsys, publish_fails=not install,
                expect=PublicationMirrorError if install else RuntimeError, setup=lambda client: arrange(monkeypatch))
    if install:
        assert run["error"].__cause__ is run["raised"]
    else:
        assert str(run["error"]) == "synthetic publication failure"
    journal = tmp_path / JOURNAL
    assert {"reconcile": run["store"].needs_git_reconcile,
            "journal": json.loads(journal.read_text())["status"] if journal.exists() else None} == expected
    assert run["store"].archive == {}
    assert "evidence_archive" not in run["out"].out


# ---- a collection that fails inside the collector keeps the rows it held ----


LIVE_ROW = {"url_hash": "live0", "source_id": "o1_synthetic_feed", "bucket": "ORBIT", "tier": "O1",
            "origin": "curated_o1_o3", "title": "Synthetic live title", "summary": "Synthetic live summary",
            "url": "https://example.invalid/live/0", "published_at": "2026-11-21T10:00:00+00:00",
            "fetched_at": "2026-11-21T11:00:00+00:00"}
NOON = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc).timestamp()


def _live_world(tmp_path, monkeypatch, *, apify):
    """Live collection with fake sources: one curated row, no Instagram; nothing leaves the process."""
    (tmp_path / "config/orbit_sources.json").write_text(json.dumps({"coverage_requirements": {
        "min_orbit_sources": 6, "min_macro_sources": 6, "required_orbit_tiers": ["O1", "O2", "O3"],
        "required_macro_tiers": ["M1", "M2", "M3"]}}))
    for name in ("APIFY_API_TOKEN", "APIFY_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    if apify:
        monkeypatch.setenv("APIFY_API_TOKEN", "synthetic-token")
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", lambda config: [dict(LIVE_ROW)])
    monkeypatch.setattr(pipeline, "fetch_instagram_batch", lambda handles, **kwargs: [])
    monkeypatch.setattr(pipeline, "fetch_macro_scout_signals", lambda **kwargs: [])
    monkeypatch.setattr(pipeline, "select_instagram_batch", lambda roster, handles, blocklist=None: ["synthetic_handle"]
                        if apify else [])


@pytest.mark.parametrize("apify", [False, True], ids=["instagram_unavailable", "coverage_short"])
def test_a_failure_inside_the_collector_archives_the_rows_it_held(tmp_path, monkeypatch, capsys, apify):
    run = _fail(tmp_path, monkeypatch, capsys, setup=lambda client: _live_world(tmp_path, monkeypatch, apify=apify),
                expect=IngestionUnavailable, live_collect=True, now_epoch=NOON)
    meta, night = _night(run)
    assert (meta["status"], meta["failed_stage"], meta["error_class"]) == ("failed", "collection", "IngestionUnavailable")
    assert set(night) == {"intake_rows"}
    assert {row["url_hash"] for row in night["intake_rows"]} == set(run["seeded"]) | {"live0"}
    assert meta["intake_source"] == "collector_rows"


def test_a_failure_before_the_collector_holds_rows_keeps_the_intake_part_empty(tmp_path, monkeypatch, capsys):
    from src.zeitgeist import embedding_migration

    error = RuntimeError(SENTINEL)

    def plan(store, requested):
        raise error

    run = _fail(tmp_path, monkeypatch, capsys, live_collect=True, now_epoch=NOON,
                setup=lambda client: monkeypatch.setattr(embedding_migration, "plan_embedding_run", plan) or error)
    assert run["error"] is error
    meta, night = _night(run)
    assert meta["failed_stage"] == "collection"
    assert night == {"intake_rows": []}
