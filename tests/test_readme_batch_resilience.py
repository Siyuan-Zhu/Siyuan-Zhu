"""One authoritative read per README batch; bounded transport-only GET retries."""
from io import BytesIO
from urllib.error import HTTPError, URLError
import json
import pytest
from scripts import build_readme
from src import state_store as state
from tests.test_canonical_runtime_factory import fixtures, files

def test_all_variants_share_one_canonical_snapshot_even_when_remote_advances(tmp_path, monkeypatch):
    root, local, game, client = fixtures(tmp_path)
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    original = store.canonical_projections
    calls = []

    def once(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            raise TimeoutError('second canonical fetch should not occur')
        result = original(**kwargs)
        client.docs['profile_state/daily_pulse']['word'] = 'NEXT'
        return result
    monkeypatch.setattr(store, 'canonical_projections', once)
    written = build_readme.build_all_readmes(root, repo_slug='Siyuan-Zhu/readme-live-sandbox', state_store=store)
    assert len(written) == 9 and len(calls) == 1
    for path in written.values():
        text = path.read_text()
        assert 'CANONICAL' in text and 'CHECKOUT' not in text and ('NEXT' not in text)

def test_missing_local_mirror_is_not_permission_for_editorial_inference(tmp_path, monkeypatch):
    root, local, game, client = fixtures(tmp_path)
    (root / 'data/zeitgeist_state.json').unlink()
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    from src.zeitgeist import adk_pipeline
    monkeypatch.setattr(adk_pipeline, 'run_daily_zeitgeist', lambda **kwargs: pytest.fail('No providers during canonical rendering'))
    written = build_readme.build_all_readmes(root, state_store=store)
    assert len(written) == 9 and 'CANONICAL' in (root / 'README.md').read_text()

def test_canonical_auth_failure_cannot_render_from_stale_checkout(tmp_path, monkeypatch):
    root, local, game, client = fixtures(tmp_path)
    before = files(root)
    store = state.get_state_store(root, backend='firestore', firestore_client=client)

    def denied(**kwargs):
        raise HTTPError('https://firestore.googleapis.com/', 403, 'denied', {}, BytesIO())
    monkeypatch.setattr(store, 'canonical_projections', denied)
    with pytest.raises(HTTPError):
        build_readme.build_all_readmes(root, state_store=store)
    assert files(root) == before

def test_firestore_get_retries_a_tls_timeout_without_any_write(monkeypatch):
    requests = []
    sleeps = []

    def fetch(request, **kwargs):
        requests.append(request)
        if len(requests) == 1:
            raise URLError(TimeoutError('TLS handshake timed out'))
        return BytesIO(json.dumps({'fields': {'turn': {'integerValue': '3'}}, 'updateTime': '2026-10-30T20:00:00Z'}).encode())
    monkeypatch.setattr(state.urllib.request, 'urlopen', fetch)
    monkeypatch.setattr(state.time, 'sleep', sleeps.append)
    snapshot = state._FirestoreRestClient('test-project', access_token='synthetic').get_document('profile_state', 'active_game')
    assert snapshot.to_dict() == {'turn': 3}
    assert len(requests) == 2 and all((r.get_method() == 'GET' for r in requests)) and (len(sleeps) == 1)

@pytest.mark.parametrize('status', [401, 403, 404, 500])
def test_firestore_http_status_never_becomes_a_transport_retry(monkeypatch, status):
    calls = []

    def fail(request, **kwargs):
        calls.append(request)
        raise HTTPError(request.full_url, status, 'fixed', {}, BytesIO())
    monkeypatch.setattr(state.urllib.request, 'urlopen', fail)
    monkeypatch.setattr(state.time, 'sleep', lambda seconds: pytest.fail('HTTP failures must not retry'))
    client = state._FirestoreRestClient('test-project', access_token='synthetic')
    if status == 404:
        assert client.get_document('profile_state', 'active_game').exists is False
    else:
        with pytest.raises(HTTPError):
            client.get_document('profile_state', 'active_game')
    assert len(calls) == 1

def test_persistent_firestore_transport_failure_stops_after_three_reads(monkeypatch):
    calls = []
    sleeps = []

    def fail(request, **kwargs):
        calls.append(request)
        raise URLError(TimeoutError('TLS handshake timed out'))
    monkeypatch.setattr(state.urllib.request, 'urlopen', fail)
    monkeypatch.setattr(state.time, 'sleep', sleeps.append)
    with pytest.raises(URLError):
        state._FirestoreRestClient('test-project', access_token='synthetic').get_document('profile_state', 'active_game')
    assert len(calls) == 3 and len(sleeps) == 2 and all((r.get_method() == 'GET' for r in calls))
