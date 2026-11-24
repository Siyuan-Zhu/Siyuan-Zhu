"""The private evidence archive: every night's evidence kept in its own collection, behind its own switch."""

from __future__ import annotations

import base64
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from src.state_store import FirestoreStateStore, InMemoryStateStore, JsonFileStateStore, StateStore
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import evidence_archive
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_firestore_store import _RecordingFirestoreClient
from tests.test_run_record_wiring import SLATE, _edition_setup, _RecordingStore, _sha
from tests.test_understanding_pipeline import _reply

ROOT = Path(__file__).resolve().parents[1]
DOLLAR = chr(36)
ARCHIVE_LINE = "ZEITGEIST_EVIDENCE_ARCHIVE: " + DOLLAR + "{{ vars.ZEITGEIST_EVIDENCE_ARCHIVE || '0' }}"
CANDIDATE_LINE = "ZEITGEIST_REFLECTOR_CANDIDATE: " + DOLLAR + "{{ vars.ZEITGEIST_REFLECTOR_CANDIDATE || '0' }}"
PUBLIC_LINE = re.compile(r"evidence_archive=(ok docs=\d+ bytes=\d+|failed error_class=[A-Za-z_][A-Za-z0-9_]*)")
RUN_ID = "daily_run:2026-11-21:local:1"
META_ID = "2026-11-21:" + RUN_ID


@pytest.fixture(autouse=True)
def _archive_switch_off_unless_set(monkeypatch):
    """A shell that exports the switch must not change these tests; those that want it set it."""
    monkeypatch.delenv(evidence_archive.SWITCH_ENV, raising=False)
    monkeypatch.delenv("ZEITGEIST_OWN_GROUP", raising=False)
    monkeypatch.delenv("ZEITGEIST_REFLECTOR", raising=False)


def _size(document):
    return len(json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))


# ---- the switch ----


@pytest.mark.parametrize("value, on", [(None, False), ("0", False), ("true", False), ("1", True)])
def test_the_switch_is_on_only_for_one(value, on):
    environ = {} if value is None else {evidence_archive.SWITCH_ENV: value}
    assert evidence_archive.SWITCH_ENV == "ZEITGEIST_EVIDENCE_ARCHIVE"
    assert evidence_archive.archive_enabled(environ) is on


# ---- documents: a meta document and chunks under the size limit ----


def _vector_b64(seed):
    vector = np.random.default_rng(seed).standard_normal(3072).astype(np.float16)
    return base64.b64encode(vector.tobytes()).decode("ascii")


def _large_rows(count=700):
    return [{
        "url_hash": f"h{i:04d}", "source_id": f"m1_synthetic_{i % 40}", "bucket": "MACRO", "tier": "M1",
        "title": f"Synthetic title {i}", "summary": "Synthetic summary text. " * 40, "caption": "Synthetic caption " * 10,
        "url": f"https://example.invalid/item/{i}", "published_at": "2026-11-21T10:00:00+00:00",
        "embedding_f16": _vector_b64(i), "embedding_model": "synthetic-embedding",
    } for i in range(count)]


def test_a_large_night_is_chunked_with_every_document_under_the_limit():
    rows = _large_rows()
    parts = {"intake_rows": rows, "groups": [{"cluster_id": f"cluster_{i}", "item_hashes": [r["url_hash"] for r in rows[i::50]]}
                                             for i in range(50)]}
    documents = evidence_archive.archive_documents(date_iso="2026-11-21", run_id=RUN_ID, parts=parts)
    ids = [doc_id for doc_id, _ in documents]
    assert len(ids) == len(set(ids))
    assert ids[-1] == META_ID
    assert all(_size(document) < evidence_archive.MAX_DOC_BYTES for _, document in documents)
    assert sum(_size(document) for _, document in documents) > 5_000_000
    meta = dict(documents)[META_ID]
    assert meta["chunk_ids"] == ids[:-1]
    assert all(re.fullmatch(r"2026-11-21:" + re.escape(RUN_ID) + r":(intake_rows|groups):\d+", i) for i in ids[:-1])
    assert meta["parts"]["intake_rows"]["entries"] == 700
    assert meta["parts"]["intake_rows"]["chunks"] > 1
    assert evidence_archive.assemble(dict(documents), META_ID) == parts


def test_an_entry_larger_than_a_document_is_split_and_joins_again():
    huge = {"request_text": "xé\"\\" * 400_000, "reply_text": "y"}
    parts = {"director_attempts": [{"attempt": 1}, huge, {"attempt": 3}]}
    documents = evidence_archive.archive_documents(date_iso="2026-11-21", run_id=RUN_ID, parts=parts)
    assert len(documents) > 3
    assert all(_size(document) < evidence_archive.MAX_DOC_BYTES for _, document in documents)
    assert evidence_archive.assemble(dict(documents), META_ID) == parts


def test_the_same_run_id_writes_the_same_document_ids():
    store = InMemoryStateStore()
    parts = {"intake_rows": _large_rows(120)}
    first = evidence_archive.write_archive(store, evidence_archive.archive_documents(
        date_iso="2026-11-21", run_id=RUN_ID, parts=parts))
    ids = sorted(store._evidence_archive)
    second = evidence_archive.write_archive(store, evidence_archive.archive_documents(
        date_iso="2026-11-21", run_id=RUN_ID, parts=parts))
    assert first == second and first[0] == len(ids)
    assert sorted(store._evidence_archive) == ids


def test_image_bytes_and_data_uris_are_left_out_and_embedding_bytes_are_kept_as_base64():
    raw = np.arange(8, dtype=np.float16).tobytes()
    row = {"url_hash": "h1", "embedding_f16": raw, "embedding_f32": np.arange(8, dtype=np.float32),
           "thumb_bytes": b"\x89PNG", "image_data_uri": "data:image/webp;base64,AAAA", "display_url": "https://example.invalid/p.jpg"}
    [plain] = evidence_archive.plain_entries([row])
    assert plain == {"url_hash": "h1", "embedding_f16": base64.b64encode(raw).decode("ascii"),
                     "display_url": "https://example.invalid/p.jpg"}


# ---- the stores: its own collection, never mirrored ----


def test_every_store_has_the_archive_write():
    assert hasattr(StateStore, "save_evidence_archive_doc")
    for store in (InMemoryStateStore(), JsonFileStateStore(repo_root=Path("/nonexistent")),
                  FirestoreStateStore(client=_RecordingFirestoreClient(), project_id="test-proj", canonical_only=True)):
        assert callable(store.save_evidence_archive_doc)
        assert isinstance(store, StateStore)


@pytest.mark.parametrize("prefix", ["", "staging_"])
def test_firestore_writes_to_its_own_collection_with_the_prefix(tmp_path, prefix):
    client = _RecordingFirestoreClient()
    store = FirestoreStateStore(client=client, project_id="test-proj", canonical_only=True, collection_prefix=prefix,
                                mirror_store=JsonFileStateStore(repo_root=tmp_path))
    documents = evidence_archive.archive_documents(date_iso="2026-11-21", run_id=RUN_ID,
                                                   parts={"intake_rows": _large_rows(10)})
    evidence_archive.write_archive(store, documents)
    assert set(client.docs) == {f"{prefix}evidence_archive/{doc_id}" for doc_id, _ in documents}
    assert client.docs[f"{prefix}evidence_archive/{META_ID}"] == dict(documents)[META_ID]
    assert not (tmp_path / "data").exists() or list((tmp_path / "data").rglob("*")) == []


def test_an_unsafe_document_id_is_refused():
    with pytest.raises(ValueError):
        InMemoryStateStore().save_evidence_archive_doc("bad/id", {})


def test_the_json_archive_folder_is_gitignored(tmp_path):
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", "data/evidence_archive/x.json"], check=False)
    assert result.returncode == 0


# ---- the edition: written after publication, never able to cost it ----


class _ArchiveStore(_RecordingStore):
    """The real JSON store; archive documents are kept in memory and noted among the events."""

    def __init__(self, root, events):
        super().__init__(root, events)
        self.archive = {}

    def save_evidence_archive_doc(self, doc_id, document):
        self.events.append("save_evidence_archive_doc")
        self.archive[doc_id] = json.loads(json.dumps(document))


class _RaisingArchiveStore(_ArchiveStore):
    error = RuntimeError

    def save_evidence_archive_doc(self, doc_id, document):
        self.events.append("save_evidence_archive_doc")
        raise self.error("synthetic archive failure")


class _SpendBlockedArchiveStore(_RaisingArchiveStore):
    error = SpendBlocked


class _SlowArchiveStore(_ArchiveStore):
    def save_evidence_archive_doc(self, doc_id, document):
        time.sleep(1.0)
        super().save_evidence_archive_doc(doc_id, document)


def _seed_cache(store):
    """Two stored rows carrying fields the in-memory items drop (a caption, a fetch time, the embedding model)."""
    from src.zeitgeist.collectors import embed_missing_items

    rows = [{
        "url_hash": f"seed{i}", "source_id": f"m1_seed_{i}", "bucket": "MACRO", "tier": "M1", "origin": "curated_M1_M3",
        "title": f"Seeded title {i}", "summary": f"Seeded summary {i} about looms and weaving", "caption": f"Seeded caption {i}",
        "url": f"https://example.invalid/seed/{i}", "published_at": "2026-11-21T09:00:00+00:00",
        "fetched_at": "2026-11-21T09:30:00+00:00",
    } for i in range(6)]
    embedded = embed_missing_items(rows, client=None)
    store.save_ingestion_cache_day("2026-11-21", {row["url_hash"]: row for row in embedded})
    return {row["url_hash"]: row for row in embedded}


def _edition(tmp_path, monkeypatch, store_cls=_ArchiveStore, *, archive, record="0", seed=False, replies=None):
    client, events = _edition_setup(tmp_path, monkeypatch, switch=record)
    if replies is not None:
        client.replies = list(replies)
    if archive is None:
        monkeypatch.delenv(evidence_archive.SWITCH_ENV, raising=False)
    else:
        monkeypatch.setenv(evidence_archive.SWITCH_ENV, archive)
    store = store_cls(tmp_path, events)
    seeded = _seed_cache(store) if seed else None
    orchestrator = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    analysis, decision, _ = orchestrator.run("2026-11-21", "blueprint", offline_demo=True)
    return store, client, (analysis.word_of_the_day, analysis.explanation_phrase, decision.selected_mode), seeded


def _untimed(entry):
    """A published entry without the two clock readings that differ between any two runs."""
    return {key: value for key, value in entry.items() if key not in {"updated_at", "published_at"}}


def _lines(text):
    return [line for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_switch_off_writes_nothing_prints_nothing_and_passes_no_new_argument(tmp_path, monkeypatch, capsys, value):
    seen = {"collect": [], "render": [], "commissions": []}
    real_collect, real_render = pipeline.collect_96h_signals, pipeline.select_and_render_pulse_assets
    real_commission = pipeline.commission_daily_zeitgeist

    def collect(*args, **kwargs):
        seen["collect"].append(set(kwargs))
        return real_collect(*args, **kwargs)

    def render(*args, **kwargs):
        seen["render"].append(set(kwargs))
        return real_render(*args, **kwargs)

    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    client, events = _edition_setup(tmp_path, monkeypatch, switch="0")
    patched_commission = pipeline.commission_daily_zeitgeist

    def commission(*args, **kwargs):
        made = patched_commission(*args, **kwargs)
        seen["commissions"].append(made)
        return made

    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", commission)
    if value is not None:
        monkeypatch.setenv(evidence_archive.SWITCH_ENV, value)
    store = _ArchiveStore(tmp_path, events)
    pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run("2026-11-21", "blueprint", offline_demo=True)
    out = capsys.readouterr()
    assert store.events == ["publish_bundle:called", "publish_bundle:returned"]
    assert store.archive == {}
    assert not (tmp_path / "data" / "evidence_archive").exists()
    assert "evidence_archive" not in out.out + out.err
    assert seen["collect"] and all("loaded_rows_out" not in kw for kw in seen["collect"])
    assert seen["render"] and all("scored_candidates_out" not in kw for kw in seen["render"])
    assert seen["commissions"] and all(not hasattr(made, "_director_attempts") and not hasattr(made, "_director_evidence")
                                       for made in seen["commissions"])


def test_a_staged_run_keeps_the_archive_inputs_only_with_the_switch_on(tmp_path, monkeypatch):
    for value, kept in ((None, False), ("0", False), ("1", True)):
        root = tmp_path / str(value)
        _, events = _edition_setup(root, monkeypatch, switch="0")
        if value is None:
            monkeypatch.delenv(evidence_archive.SWITCH_ENV, raising=False)
        else:
            monkeypatch.setenv(evidence_archive.SWITCH_ENV, value)
        staged = pipeline.ZeitgeistADKOrchestrator(root, state_store=_ArchiveStore(root, events))
        staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
        assert (staged._evidence_archive_inputs is not None) is kept
        assert events == []


def test_switch_on_writes_the_meta_and_chunks_after_publication_with_the_nights_evidence(tmp_path, monkeypatch, capsys):
    store, client, result, seeded = _edition(tmp_path, monkeypatch, archive="1", seed=True,
                                             replies=["not json", _reply(selected_cluster_id="cluster_0",
                                                                         candidate_words=SLATE)])
    out = capsys.readouterr()
    assert store.events[:2] == ["publish_bundle:called", "publish_bundle:returned"]
    assert set(store.events[2:]) == {"save_evidence_archive_doc"}
    assert META_ID in store.archive
    meta = store.archive[META_ID]
    assert sorted(store.archive) == sorted(meta["chunk_ids"] + [META_ID])
    assert meta["date"] == "2026-11-21" and meta["run_id"] == RUN_ID and meta["archive_schema"] == 1
    assert all(_size(document) < evidence_archive.MAX_DOC_BYTES for document in store.archive.values())
    night = evidence_archive.assemble(store.archive, META_ID)

    # Every row the run loaded, with every stored field: caption, fetch time, embedding model and vector.
    rows = {row["url_hash"]: row for row in night["intake_rows"]}
    assert set(rows) == set(seeded)
    for url_hash, stored in seeded.items():
        for key in ("title", "summary", "caption", "url", "fetched_at", "embedding_model", "source_id"):
            assert rows[url_hash][key] == stored[key]
        vector = stored["embedding_f16"]
        assert vector and rows[url_hash]["embedding_f16"] == (
            base64.b64encode(vector).decode("ascii") if isinstance(vector, bytes) else vector)
    assert meta["intake_source"] == "stored_rows"

    # Every scored group, ranked, with its scores and its items by url_hash.
    groups = night["groups"]
    assert [group["rank"] for group in groups] == list(range(1, len(groups) + 1))
    assert sum(len(group["item_hashes"]) for group in groups) >= 1
    assert all(set(group["item_hashes"]) <= set(seeded) for group in groups)
    assert all(isinstance(group["commission_score"], float) for group in groups)

    # Every Director attempt, request and reply as sent and received.
    attempts = night["director_attempts"]
    assert [a["request_text"] for a in attempts] == client.prompts
    assert [a["reply_text"] for a in attempts] == client.replies[:len(client.prompts)]
    assert [a["pass"] for a in attempts] == ["main", "main"]
    [evidence] = [e for e in night["director_evidence"] if e["pass"] == "main"]
    assert evidence["candidate_words"] == SLATE
    assert evidence["prompt_sha256"] == _sha(client.prompts[-1])
    assert evidence["evidence_text"] in client.prompts[-1]

    commissions = {c["pass"]: c for c in night["commissions"]}
    assert commissions["published"]["word"] == result[0] == "Selvedge"
    assert commissions["main"]["word"] == "Selvedge"
    [published] = night["published"]
    assert published["entry"] == store.load_pulse_state()["published_pulse"]
    assert night["photo_candidates"] == []

    # One public line, numbers only; otherwise the output is the switch-off run's.
    [line] = [line for line in _lines(out.out + out.err) if "evidence_archive" in line]
    match = PUBLIC_LINE.fullmatch(line)
    assert match and line.startswith("evidence_archive=ok ")
    assert line == f"evidence_archive=ok docs={len(store.archive)} bytes={sum(_size(d) for d in store.archive.values())}"
    for text in ("Seeded", "example.invalid", "Synthword", "synthetic fit", "Selvedge", RUN_ID):
        assert text not in out.out + out.err


def test_the_archive_is_written_with_the_run_record_on_too_and_after_it(tmp_path, monkeypatch):
    store, _, _, _ = _edition(tmp_path, monkeypatch, archive="1", record="1")
    assert store.events.index("save_run_record") < store.events.index("save_evidence_archive_doc")
    assert len(store.records) == 1 and META_ID in store.archive


def test_photo_candidates_keep_every_judge_score_and_the_chosen_one_without_image_bytes(tmp_path, monkeypatch, capsys):
    real_render = pipeline.select_and_render_pulse_assets

    def render(*args, **kwargs):
        published = real_render(*args, **kwargs)
        out = kwargs.get("scored_candidates_out")
        if out is not None:
            out.extend([
                {"index": 0, "photo_id": "p0", "semantic_alignment": 0.91, "compositional_clarity": 0.8,
                 "editorial_memorability": 0.7, "composite_score": 0.84, "safety_pass": True, "thumb_bytes": b"\xff\xd8"},
                {"index": 1, "photo_id": "p1", "semantic_alignment": 0.41, "compositional_clarity": 0.5,
                 "editorial_memorability": 0.6, "composite_score": 0.47, "safety_pass": True, "thumb_bytes": b"\xff\xd8"},
            ])
            published.selected_candidate_index = 0
        return published

    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    store, _, _, _ = _edition(tmp_path, monkeypatch, archive="1")
    capsys.readouterr()
    night = evidence_archive.assemble(store.archive, META_ID)
    candidates = night["photo_candidates"]
    assert [(c["render_call"], c["photo_id"], c["composite_score"], c["semantic_alignment"]) for c in candidates] == [
        (1, "p0", 0.84, 0.91), (1, "p1", 0.47, 0.41)]
    assert all("thumb_bytes" not in c for c in candidates)
    assert [c["chosen"] for c in candidates] == [True, False]
    assert night["published"][0]["chosen_photo"] == {"render_call": 1, "selected_candidate_index": 0}


@pytest.mark.parametrize("store_cls, error_class", [(_RaisingArchiveStore, "RuntimeError"),
                                                    (_SpendBlockedArchiveStore, "SpendBlocked")])
def test_a_failing_archive_store_prints_the_failed_line_and_leaves_the_edition(tmp_path, monkeypatch, capsys,
                                                                             store_cls, error_class):
    off_store, _, off, _ = _edition(tmp_path / "off", monkeypatch, archive="0")
    off_out = capsys.readouterr()
    on_store, _, on, _ = _edition(tmp_path / "on", monkeypatch, store_cls, archive="1")
    on_out = capsys.readouterr()
    assert on == off
    assert on_store.events[:2] == ["publish_bundle:called", "publish_bundle:returned"]
    assert _untimed(on_store.load_pulse_state()["published_pulse"]) == _untimed(off_store.load_pulse_state()["published_pulse"])
    assert on_store.needs_git_reconcile is False
    assert not (tmp_path / "on" / "data" / "publication_reconcile.json").exists()
    on_lines, off_lines = _lines(on_out.out), _lines(off_out.out)
    assert [line for line in on_lines if "evidence_archive" in line] == [f"evidence_archive=failed error_class={error_class}"]
    assert [line for line in on_lines if "evidence_archive" not in line] == off_lines
    assert on_out.err == off_out.err


def test_a_stalled_archive_store_is_held_to_its_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evidence_archive, "BUDGET_SECONDS", 0.2)
    started = time.monotonic()
    store, _, result, _ = _edition(tmp_path, monkeypatch, _SlowArchiveStore, archive="1")
    elapsed = time.monotonic() - started
    out = capsys.readouterr()
    assert result[0] == "Selvedge"
    assert [line for line in _lines(out.out) if "evidence_archive" in line] == ["evidence_archive=failed error_class=TimeoutError"]
    assert elapsed < 30
    time.sleep(1.5)
    # The abandoned writer stops at its next document; the meta document, written last, never lands.
    assert META_ID not in store.archive


def test_a_failed_publication_writes_no_archive(tmp_path, monkeypatch, capsys):
    _, events = _edition_setup(tmp_path, monkeypatch, switch="0", publish_fails=True)
    monkeypatch.setenv(evidence_archive.SWITCH_ENV, "1")
    store = _ArchiveStore(tmp_path, events)
    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run("2026-11-21", "blueprint", offline_demo=True)
    assert store.archive == {}
    assert "evidence_archive" not in capsys.readouterr().out


# ---- the workflows pass the switch to the production edition, off by default ----


def _job_lines(text, job):
    lines = text.splitlines()
    start = lines.index("  " + job + ":")
    block = []
    for line in lines[start + 1:]:
        if re.fullmatch(r"  [A-Za-z0-9_-]+:", line):
            break
        block.append(line.strip())
    return block


def test_the_daily_workflow_passes_the_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition.count(ARCHIVE_LINE) == 1
    # After the switches it keeps evidence from, and only in the production edition's job.
    assert edition[edition.index(CANDIDATE_LINE) + 1] == ARCHIVE_LINE
    assert ARCHIVE_LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(ARCHIVE_LINE) == 1


def test_the_private_run_mirrors_the_daily_step_and_passes_the_switch_too():
    lines = [line.strip() for line in (ROOT / ".github/workflows/private_run.yml").read_text().splitlines()]
    assert lines.count(ARCHIVE_LINE) == 1
    assert lines[lines.index(CANDIDATE_LINE) + 1] == ARCHIVE_LINE


@pytest.mark.parametrize("name", ["comparison_run.yml", "provider_canary.yml", "staging_zeitgeist_dryrun.yml"])
def test_no_other_workflow_passes_the_switch(name):
    assert evidence_archive.SWITCH_ENV not in (ROOT / ".github/workflows" / name).read_text()
