"""The archive read back through the real Firestore REST client: what a night wrote is what an offline tool reads.

Fakes only: a stand-in for urlopen keeps each request body and serves it back the way Firestore would.
"""

import copy
import io
import json
import urllib.error
import urllib.parse

import pytest

from src import state_store
from src.state_store import FirestoreStateStore, JsonFileStateStore, _FirestoreRestClient
from src.zeitgeist import evidence_archive
from tests.test_evidence_archive import META_ID, RUN_ID, _vector_b64

INT64_MAX = 2**63 - 1
PREFIX = "evidence_archive/"


class _Response:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _strict_json(raw):
    """Firestore refuses NaN and Infinity in a request body; so does this fake."""
    def refuse(name):
        raise ValueError(name)
    return json.loads(raw.decode("utf-8"), parse_constant=refuse)


class _FakeFirestoreWire:
    """Stores each PATCH body by document path and serves it back on GET, as a 404 when absent."""

    def __init__(self):
        self.bodies = {}
        self.requests = []

    def urlopen(self, req, timeout=10):
        path = urllib.parse.unquote(req.full_url.split("/documents/", 1)[1])
        self.requests.append((req.get_method(), path))
        if req.get_method() == "PATCH":
            body = _strict_json(req.data)
            for value in body["fields"].values():
                _check_integers(value)
            self.bodies[path] = body
            return _Response({})
        if path not in self.bodies:
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, io.BytesIO(b""))
        return _Response({**copy.deepcopy(self.bodies[path]), "updateTime": "2026-11-21T10:00:00Z"})


def _check_integers(value):
    if "integerValue" in value:
        assert -(2**63) <= int(value["integerValue"]) <= INT64_MAX
    for nested in (value.get("arrayValue") or {}).get("values") or []:
        _check_integers(nested)
    for nested in ((value.get("mapValue") or {}).get("fields") or {}).values():
        _check_integers(nested)


@pytest.fixture
def wire(monkeypatch):
    fake = _FakeFirestoreWire()
    monkeypatch.setattr(state_store.urllib.request, "urlopen", fake.urlopen)
    return fake


@pytest.fixture
def store(wire, tmp_path):
    client = _FirestoreRestClient("test-proj", access_token="fake-token")
    return FirestoreStateStore(client=client, project_id="test-proj", canonical_only=True,
                               mirror_store=JsonFileStateStore(repo_root=tmp_path))


def _night_parts():
    return {
        "intake_rows": [
            {"url_hash": "h0001", "bucket": "MACRO", "rank_pairs": [[1, 2], [3, 4]], "windows": ((1, 2), (3,)),
             "embedding_f16": _vector_b64(1), "embedding_model": "synthetic-embedding", "views": 2**53 + 1,
             "big": INT64_MAX, "small": -INT64_MAX, "score": 0.1 + 0.2, "flag": False, "none": None, "empty": [],
             "nested_empty": {}},
            {"url_hash": "h0002", "_nested_list": [1, 2], "tags": ["a", ["b", "c"]]},
        ],
        "groups": [
            {"rank": 1, "cluster_id": "cluster_1", "item_hashes": ["h0001", "h0002"], "pair": (1.5, 2.5)},
        ],
        # Larger than one document, so it is stored as fragments and joined again.
        "scores": [{"url_hash": "h0003", "trace": [[i, i * 0.5] for i in range(60_000)]}],
    }


def _night(parts):
    return evidence_archive.archive_documents(date_iso="2026-11-21", run_id=RUN_ID, parts=parts)


def _read_back(store, documents):
    return {doc_id: store.load_evidence_archive_doc(doc_id) for doc_id, _ in documents}


def test_a_night_written_through_the_rest_client_assembles_to_the_same_parts(store, wire):
    documents = _night(_night_parts())
    assert len(documents) > 4
    evidence_archive.write_archive(store, documents)
    assert {path for method, path in wire.requests if method == "PATCH"} == {PREFIX + doc_id for doc_id, _ in documents}

    read = _read_back(store, documents)
    sent = dict(documents)
    assert set(read) == set(sent)
    assert evidence_archive.assemble(read, META_ID) == evidence_archive.assemble(sent, META_ID)
    for doc_id in sent:
        assert read[doc_id] == sent[doc_id], "document differs after the round trip"


def test_each_kind_of_value_survives_part_by_part(store):
    parts = _night_parts()
    documents = _night(parts)
    evidence_archive.write_archive(store, documents)
    got = evidence_archive.assemble(_read_back(store, documents), META_ID)
    want = evidence_archive.plain_entries
    for name in parts:
        assert got[name] == want(parts[name]), name


def test_an_absent_document_reads_as_none(store):
    assert store.load_evidence_archive_doc("2026-11-21:no-such-run") is None


def test_a_nan_is_a_null_in_the_archive_documents_and_never_reaches_the_wire_as_nan(store, wire):
    parts = {"intake_rows": [{"url_hash": "h1", "score": float("nan"), "wide": float("inf"), "scores": [1.0, float("nan")]}]}
    documents = _night(parts)
    evidence_archive.write_archive(store, documents)
    assert wire.bodies, "something was written"
    for body in wire.bodies.values():
        assert "NaN" not in json.dumps(body) and "Infinity" not in json.dumps(body)
    got = evidence_archive.assemble(_read_back(store, documents), META_ID)
    assert got["intake_rows"] == [{"url_hash": "h1", "score": None, "wide": None, "scores": [1.0, None]}]


# The two tests below pin real defects in the store path found by this read-back (reported, not fixed here):
# the nightly path never meets them (the archive documents are made plain first), but a direct store call does.
# They are strict, so the fix turns them into a failure that asks for the marker to come off.


@pytest.mark.xfail(strict=True, reason="a NaN handed straight to the Firestore store is sent, not refused")
def test_a_document_holding_a_nan_is_refused_before_any_request(store, wire):
    document = {"archive_schema": 1, "entries": [{"score": float("nan")}]}
    with pytest.raises(ValueError):
        store.save_evidence_archive_doc("2026-11-21:nan", document)
    assert wire.requests == []


@pytest.mark.xfail(strict=True, reason="a dict whose only key is _nested_list reads back as that key's list")
def test_a_dict_whose_only_key_is_nested_list_reads_back_as_a_dict(store):
    document = {"archive_schema": 1, "entries": [{"_nested_list": ["only", "key"]}], "note": {"_nested_list": [1]}}
    store.save_evidence_archive_doc("2026-11-21:wrapped", document)
    assert store.load_evidence_archive_doc("2026-11-21:wrapped") == document


# ---- the JSON store, and the same night through both ----


def test_a_night_round_trips_through_the_json_store(tmp_path):
    json_store = JsonFileStateStore(repo_root=tmp_path)
    documents = _night(_night_parts())
    evidence_archive.write_archive(json_store, documents)
    read = _read_back(json_store, documents)
    assert evidence_archive.assemble(read, META_ID) == evidence_archive.assemble(dict(documents), META_ID)
    assert json_store.load_evidence_archive_doc("2026-11-21:no-such-run") is None


def test_both_stores_read_back_the_same_night(store, tmp_path):
    documents = _night(_night_parts())
    json_store = JsonFileStateStore(repo_root=tmp_path / "json")
    evidence_archive.write_archive(store, documents)
    evidence_archive.write_archive(json_store, documents)
    assert _read_back(store, documents) == _read_back(json_store, documents)
