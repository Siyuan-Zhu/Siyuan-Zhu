"""Canonical votes remain atomic when an editorial publication races the vote."""
import copy
from types import SimpleNamespace

import pytest

import src.state_store as state


DATE = "2026-10-30"
TALLY = {"spot_on": 20, "too_sunny": 3, "too_heavy": 4}


class Client:
    def __init__(self):
        self.docs = {"staging_profile_state/daily_pulse": {
            "date": DATE, "word": "BEFORE", "gloss": "Old commission",
            "visitor_mood_tally": dict(TALLY), "mood_votes": dict(TALLY)},
            "staging_pulse_votes/" + DATE: dict(TALLY)}
        self.versions = {key: 1 for key in self.docs}
        self.race = None
        self.commit_calls = 0
        self.fail = False

    def collection(self, collection):
        client = self
        class Ref:
            def __init__(self, doc): self.key = collection + "/" + doc
            def get(self):
                data = copy.deepcopy(client.docs.get(self.key))
                snapshot = SimpleNamespace(exists=data is not None,
                    update_time=client.versions.get(self.key), to_dict=lambda: data)
                if self.key.endswith("/daily_pulse") and client.race:
                    race, client.race = client.race, None
                    race()
                return snapshot
            def set(self, data):
                client.docs[self.key] = copy.deepcopy(data)
                client.versions[self.key] = client.versions.get(self.key, 0) + 1
        return SimpleNamespace(document=Ref)

    def commit_documents(self, writes):
        self.commit_calls += 1
        if self.fail:
            raise RuntimeError("Simulated unavailable commit")
        for collection, doc, data, snapshot in writes:
            key = collection + "/" + doc
            if self.versions.get(key) != snapshot.update_time:
                raise state.PublicationConflict("simulated concurrent writer")
        for collection, doc, data, snapshot in writes:
            key = collection + "/" + doc
            self.docs[key] = dict(self.docs.get(key, {}), **copy.deepcopy(data))
            self.versions[key] = self.versions.get(key, 0) + 1


@pytest.fixture
def store():
    client = Client()
    return state.FirestoreStateStore(client=client, collection_prefix="staging_"), client


def test_publication_race_preserves_new_editorial_fields_and_accepts_once(store):
    adapter, client = store
    def publish():
        client.docs["staging_profile_state/daily_pulse"].update(word="AFTER", gloss="New commission")
        client.versions["staging_profile_state/daily_pulse"] += 1
    client.race = publish
    accepted, tally = adapter.record_mood_vote(DATE, "spot_on", "a" * 64)
    assert accepted and tally["spot_on"] == 21
    assert client.docs["staging_profile_state/daily_pulse"]["word"] == "AFTER"
    assert client.docs["staging_profile_state/daily_pulse"]["gloss"] == "New commission"
    assert client.commit_calls == 2
    duplicate, same = adapter.record_mood_vote(DATE, "too_sunny", "a" * 64)
    assert duplicate is False and same == tally
    assert client.docs["staging_profile_state/daily_pulse"]["mood_votes"] == tally


def test_commit_failure_creates_neither_dedup_nor_partial_tally(store):
    adapter, client = store
    before = copy.deepcopy(client.docs)
    client.fail = True
    with pytest.raises(state.PublicationOutcomeUnknown, match="outcome unknown"):
        adapter.record_mood_vote(DATE, "spot_on", "b" * 64)
    assert client.docs == before


def test_midnight_publication_does_not_receive_yesterdays_vote(store):
    adapter, client = store
    client.docs["staging_profile_state/daily_pulse"]["date"] = "2026-10-31"
    before = copy.deepcopy(client.docs)
    accepted, tally = adapter.record_mood_vote(DATE, "spot_on", "c" * 64)
    assert accepted is False
    assert client.docs == before


def test_concurrent_identical_vote_is_deduplicated_on_retry(store):
    adapter, client = store
    voter = "d" * 64
    def other_vote():
        accepted, _ = adapter.record_mood_vote(DATE, "spot_on", voter)
        assert accepted
    client.race = other_vote
    accepted, tally = adapter.record_mood_vote(DATE, "spot_on", voter)
    assert accepted is False and tally["spot_on"] == 21
    assert len([key for key in client.docs if "vote_rate_limits/" in key]) == 1


def test_concurrent_distinct_votes_both_count_without_lost_increment(store):
    adapter, client = store
    def other_vote():
        accepted, _ = adapter.record_mood_vote(DATE, "spot_on", "e" * 64)
        assert accepted
    client.race = other_vote
    accepted, tally = adapter.record_mood_vote(DATE, "spot_on", "f" * 64)
    assert accepted and tally["spot_on"] == 22
    assert client.docs["staging_pulse_votes/" + DATE] == tally
    assert len([key for key in client.docs if "vote_rate_limits/" in key]) == 2


def test_rest_vote_uses_one_conditional_commit_and_only_patches_tally(monkeypatch):
    import io
    import json
    requests = []
    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
    def urlopen(request, **kwargs):
        requests.append(request)
        if request.method == "POST":
            return Response(json.dumps({"commitTime": "2026-10-30T00:00:01Z",
                "writeResults": [{"updateTime": "2026-10-30T00:00:01Z"} for _ in range(3)]}).encode())
        if "/vote_rate_limits/" in request.full_url:
            from urllib.error import HTTPError
            raise HTTPError(request.full_url, 404, "absent", {}, io.BytesIO())
        data = dict(TALLY) if "/pulse_votes/" in request.full_url else {
            "date": DATE, "word": "UNCHANGED", "visitor_mood_tally": TALLY}
        return Response(json.dumps({"fields": {key: state._py_to_firestore_value(value)
            for key, value in data.items()}, "updateTime": "2026-10-30T00:00:00Z"}).encode())
    monkeypatch.setattr(state.urllib.request, "urlopen", urlopen)
    client = state._FirestoreRestClient("fixture", access_token="not-a-real-token")
    adapter = state.FirestoreStateStore(client=client)
    accepted, tally = adapter.record_mood_vote(DATE, "spot_on", "a" * 64)
    assert accepted and tally["spot_on"] == 21
    commits = [request for request in requests if request.method == "POST"]
    assert len(commits) == 1 and commits[0].full_url.endswith("/documents:commit")
    writes = json.loads(commits[0].data)["writes"]
    assert len(writes) == 3
    assert writes[0]["currentDocument"] == {"exists": False}
    assert set(writes[2]["updateMask"]["fieldPaths"]) == {"visitor_mood_tally", "mood_votes"}
    assert "word" not in writes[2]["update"]["fields"]
