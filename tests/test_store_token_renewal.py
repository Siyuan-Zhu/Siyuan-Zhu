"""A token the Firestore REST client obtains itself is renewed before it expires. Fakes only: no network."""
import json

import pytest

from src import state_store

OIDC_ENV = {'ACTIONS_ID_TOKEN_REQUEST_URL': 'https://oidc.example/token',
            'ACTIONS_ID_TOKEN_REQUEST_TOKEN': 'req-tok', 'GCP_WIF_PROVIDER': '//iam/provider'}


class FakeResp:
    def __init__(self, body):
        self.body = json.dumps(body).encode()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def read(self):
        return self.body


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(state_store, '_token_clock', lambda: now[0], raising=False)
    return now


def fake_sts(monkeypatch, expires_in=None):
    """GitHub OIDC then Google STS, served from memory; each STS call issues a new token."""
    for k, v in OIDC_ENV.items():
        monkeypatch.setenv(k, v)
    issued = []
    def fake_urlopen(req, timeout=None):
        if req.full_url.startswith('https://oidc.example/'):
            return FakeResp({'value': 'gh-jwt'})
        assert req.full_url == 'https://sts.googleapis.com/v1/token'
        issued.append(f'sts-{len(issued) + 1}')
        body = {'access_token': issued[-1]}
        if expires_in is not None:
            body['expires_in'] = expires_in
        return FakeResp(body)
    monkeypatch.setattr(state_store.urllib.request, 'urlopen', fake_urlopen)
    return issued


def test_oidc_token_is_reused_while_fresh(monkeypatch, clock):
    issued = fake_sts(monkeypatch, expires_in=3600)
    client = state_store._FirestoreRestClient('proj')
    assert client._resolve_token() == 'sts-1'
    clock[0] += 100
    assert client._resolve_token() == 'sts-1'
    assert issued == ['sts-1']


def test_oidc_token_without_expires_in_assumes_300s(monkeypatch, clock):
    issued = fake_sts(monkeypatch)
    client = state_store._FirestoreRestClient('proj')
    assert client._resolve_token() == 'sts-1'
    clock[0] += 239  # 300 s lifetime renews at min(300 - 60, 0.8 * 300) = 240 s
    assert client._resolve_token() == 'sts-1'
    clock[0] += 2
    assert client._resolve_token() == 'sts-2'
    clock[0] += 239
    assert client._resolve_token() == 'sts-2'
    clock[0] += 2
    assert client._resolve_token() == 'sts-3'


def test_oidc_token_lifetime_is_capped_at_300s(monkeypatch, clock):
    # STS may report an hour, but the federated token stopped working after about five minutes in Actions.
    issued = fake_sts(monkeypatch, expires_in='3600')
    client = state_store._FirestoreRestClient('proj')
    assert client._resolve_token() == 'sts-1'
    clock[0] += 239  # capped to 300 s: min(300 - 60, 0.8 * 300) = 240 s
    assert client._resolve_token() == 'sts-1'
    clock[0] += 2
    assert client._resolve_token() == 'sts-2'
    assert issued == ['sts-1', 'sts-2']


def test_oidc_token_renews_60s_before_a_short_expiry(monkeypatch, clock):
    fake_sts(monkeypatch, expires_in=150)
    client = state_store._FirestoreRestClient('proj')
    assert client._resolve_token() == 'sts-1'
    clock[0] += 89  # min(150 - 60, 0.8 * 150) = 90 s
    assert client._resolve_token() == 'sts-1'
    clock[0] += 2
    assert client._resolve_token() == 'sts-2'


def test_short_lifetime_renews_no_sooner_than_30s(monkeypatch, clock):
    fake_sts(monkeypatch, expires_in=20)
    client = state_store._FirestoreRestClient('proj')
    assert client._resolve_token() == 'sts-1'
    clock[0] += 29
    assert client._resolve_token() == 'sts-1'
    clock[0] += 2
    assert client._resolve_token() == 'sts-2'


def test_exchange_github_oidc_token_still_returns_a_plain_str(monkeypatch):
    fake_sts(monkeypatch, expires_in=3600)
    assert state_store.exchange_github_oidc_token(OIDC_ENV) == 'sts-1'


def test_constructor_token_is_never_renewed(monkeypatch, clock):
    issued = fake_sts(monkeypatch)
    client = state_store._FirestoreRestClient('proj', access_token='given')
    assert client._resolve_token() == 'given'
    clock[0] += 100_000
    assert client._resolve_token() == 'given'
    assert issued == []


@pytest.mark.parametrize('name', ['FIRESTORE_ACCESS_TOKEN', 'GCP_ACCESS_TOKEN'])
def test_env_token_is_never_renewed(monkeypatch, clock, name):
    issued = fake_sts(monkeypatch)
    monkeypatch.setenv(name, 'from-env')
    client = state_store._FirestoreRestClient('proj')
    assert client._resolve_token() == 'from-env'
    clock[0] += 100_000
    assert client._resolve_token() == 'from-env'
    assert issued == []


def test_service_account_token_is_renewed_before_its_hour(monkeypatch, clock):
    monkeypatch.setenv('GCP_SA_KEY', '{"client_email":"sa@test.iam.gserviceaccount.com"}')
    minted = []
    def fake_mint(self, sa_json_str):
        minted.append(f'sa-{len(minted) + 1}')
        return minted[-1]
    monkeypatch.setattr(state_store._FirestoreRestClient, '_mint_sa_access_token', fake_mint)
    client = state_store._FirestoreRestClient('proj')
    assert client._resolve_token() == 'sa-1'
    clock[0] += 2879  # 3600 s lifetime renews at min(3600 - 60, 0.8 * 3600) = 2880 s
    assert client._resolve_token() == 'sa-1'
    clock[0] += 2
    assert client._resolve_token() == 'sa-2'
    clock[0] += 3540
    assert client._resolve_token() == 'sa-3'
    assert minted == ['sa-1', 'sa-2', 'sa-3']
