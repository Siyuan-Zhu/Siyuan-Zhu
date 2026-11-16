"""Tests for the private owner-alert emitter. Fakes only: no network, no gcloud."""
import io
import json
import subprocess
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src import state_store
from src.zeitgeist import owner_alert
from src.zeitgeist.owner_reasons import Reason

ET = ZoneInfo('America/New_York')
OIDC_ENV = {'ACTIONS_ID_TOKEN_REQUEST_URL': 'https://oidc.example/token?x=1',
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


def test_oidc_extraction_keeps_firestore_behaviour(monkeypatch):
    for k, v in OIDC_ENV.items():
        monkeypatch.setenv(k, v)
    calls = []
    def fake_urlopen(req, timeout=None):
        calls.append((req.full_url, req.get_method(), dict(req.header_items()), req.data, timeout))
        return FakeResp({'value': 'gh-jwt'} if 'oidc.example' in req.full_url else {'access_token': 'acc'})
    monkeypatch.setattr(state_store.urllib.request, 'urlopen', fake_urlopen)
    client = state_store._FirestoreRestClient('proj')
    assert client._exchange_github_oidc_token() == 'acc'
    assert calls[0][0] == 'https://oidc.example/token?x=1&audience=' + urllib.parse.quote('//iam/provider', safe='')
    assert calls[0][1] == 'GET' and calls[0][2]['Authorization'] == 'Bearer req-tok'
    assert calls[1][0] == 'https://sts.googleapis.com/v1/token' and calls[1][1] == 'POST'
    payload = json.loads(calls[1][3])
    assert payload['subjectToken'] == 'gh-jwt' and payload['audience'] == '//iam/provider'
    assert payload['grantType'] == 'urn:ietf:params:oauth:grant-type:token-exchange'
    assert calls[0][4] == 10 and calls[1][4] == 10
    with pytest.raises(RuntimeError):
        monkeypatch.delenv('ACTIONS_ID_TOKEN_REQUEST_URL')
        client._exchange_github_oidc_token()


NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
REASON = Reason('owner-act', 'Head', 'Tonight', 'Fix')
ENV = {'GITHUB_RUN_ID': '77', 'GITHUB_REPOSITORY': 'o/r', 'ZEITGEIST_EDITION_DATE': '2026-11-14'}
BASE_KEYS = {'schema_version', 'class', 'code', 'headline', 'tonight', 'fix', 'ref', 'run_id',
             'edition_date', 'repo', 'source', 'details'}


def entry(**kw):
    args = dict(code='c', reason=REASON, stage='generation', env=ENV, now=NOW)
    args.update(kw)
    return owner_alert.build_entry(**args)


def test_entry_schema_is_exact():
    e = entry()
    assert e['logName'] == 'projects/git-readme-features/logs/egm-owner-alert'
    assert e['resource'] == {'type': 'global', 'labels': {'project_id': 'git-readme-features'}}
    p = e['jsonPayload']
    assert set(p) == BASE_KEYS
    assert p['schema_version'] == 1 and p['class'] == 'owner-act' and p['code'] == 'c'
    assert p['ref'] == 'run 77 | edition 2026-11-14 | stage generation'
    assert p['repo'] == 'o/r' and p['source'] == 'runner' and p['details'] == {}
    assert set(entry(plan_code='ABC')['jsonPayload']) == BASE_KEYS | {'plan_code'}


def test_entry_defaults_and_edition_from_now():
    p = entry(env={})['jsonPayload']
    assert p['repo'] == 'local' and p['run_id'] == 'local'
    assert p['edition_date'] == '2026-11-15'
    assert p['ref'] == 'run local | edition 2026-11-15 | stage generation'


def test_details_allow_list_drops_everything_else():
    d = entry(details={'provider': 'apify', 'http_status': 403, 'runner_fp6': 'abc123',
                       'ledger_fp6': 'def456', 'sub_cause': 'credential_rejected',
                       'message': 'boom', 'body': 'x', 'token': 'a' * 40})['jsonPayload']['details']
    assert d == {'provider': 'apify', 'http_status': 403, 'runner_fp6': 'abc123',
                 'ledger_fp6': 'def456', 'sub_cause': 'credential_rejected'}
    bad = entry(details={'provider': 'other', 'http_status': 999, 'runner_fp6': 'XYZ',
                         'ledger_fp6': 'ABCDEF', 'sub_cause': 'nope', 'message': 'm'})
    assert bad['jsonPayload']['details'] == {}
    assert entry(details={'http_status': True})['jsonPayload']['details'] == {}


class Boom:
    def __init__(self, exc):
        self.exc = exc
    def __call__(self, *a, **k):
        raise self.exc


@pytest.mark.parametrize('exc', [TimeoutError('t'), urllib.error.URLError('u'),
                                 urllib.error.HTTPError('u', 500, 'm', {}, io.BytesIO(b'x'))])
def test_emit_swallows_timeout_and_http_error(exc, capsys):
    assert owner_alert.emit(entry(), env=ENV, token_provider=lambda e: 't', opener=Boom(exc)) is False
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''


def test_emit_swallows_token_failure(capsys):
    assert owner_alert.emit(entry(), env=ENV, token_provider=Boom(RuntimeError('x')), opener=Boom(AssertionError())) is False
    assert capsys.readouterr() == ('', '')


def test_emit_never_prints_on_success(capsys):
    seen = {}
    def opener(req, timeout=None):
        seen.update(url=req.full_url, method=req.get_method(), timeout=timeout,
                    auth=req.get_header('Authorization'), body=json.loads(req.data))
        return FakeResp({})
    assert owner_alert.emit(entry(), env=ENV, token_provider=lambda e: 'tok', opener=opener) is True
    assert capsys.readouterr() == ('', '')
    assert seen['url'] == 'https://logging.googleapis.com/v2/entries:write'
    assert seen['method'] == 'POST' and seen['timeout'] == 10 and seen['auth'] == 'Bearer tok'
    assert seen['body']['entries'][0]['jsonPayload']['code'] == 'c'


def test_promote_self_heal_uses_run_start():
    promote = owner_alert.promote_self_heal
    assert promote('self-heal', started_at=datetime(2026, 11, 15, 3, 29, tzinfo=ET), source='runner') == 'self-heal'
    assert promote('self-heal', started_at=datetime(2026, 11, 15, 3, 30, tzinfo=ET), source='runner') == 'owner-act'
    utc = datetime(2026, 11, 15, 8, 29, tzinfo=timezone.utc)  # 03:29 ET (EST)
    assert promote('self-heal', started_at=utc, source='runner') == 'self-heal'
    assert promote('self-heal', started_at=datetime(2026, 11, 15, 1, 0, tzinfo=ET), source='mac_fallback') == 'owner-act'
    assert promote('watch-on-it', started_at=datetime(2026, 11, 15, 5, 0, tzinfo=ET), source='mac_fallback') == 'watch-on-it'


def test_default_token_needs_both_oidc_vars_and_times_out_gcloud(monkeypatch):
    monkeypatch.delenv('ZEITGEIST_OFFLINE_DEMO', raising=False)
    monkeypatch.setattr(state_store, 'exchange_github_oidc_token', lambda env: 'oidc-token')
    runs = []
    def fake_run(cmd, **kw):
        runs.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout='gcloud-token\n', stderr='')
    monkeypatch.setattr(owner_alert.subprocess, 'run', fake_run)
    both = {'ACTIONS_ID_TOKEN_REQUEST_URL': 'u', 'ACTIONS_ID_TOKEN_REQUEST_TOKEN': 't'}
    assert owner_alert.default_token(both) == 'oidc-token' and not runs
    for partial in ({'ACTIONS_ID_TOKEN_REQUEST_URL': 'u'}, {'ACTIONS_ID_TOKEN_REQUEST_TOKEN': 't'}, {}):
        assert owner_alert.default_token(partial) == 'gcloud-token'
    cmd, kw = runs[0]
    assert cmd == ['gcloud', 'auth', 'application-default', 'print-access-token']
    assert kw['timeout'] == 10 and kw['capture_output'] is True and kw['check'] is True
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 10)
    monkeypatch.setattr(owner_alert.subprocess, 'run', slow)
    with pytest.raises(subprocess.TimeoutExpired):
        owner_alert.default_token({})


def test_default_token_refuses_offline_demo(monkeypatch):
    monkeypatch.setenv('ZEITGEIST_OFFLINE_DEMO', '1')
    def forbidden(*a, **k):
        raise AssertionError('must not reach OIDC or gcloud')
    monkeypatch.setattr(state_store, 'exchange_github_oidc_token', forbidden)
    monkeypatch.setattr(owner_alert.subprocess, 'run', forbidden)
    with pytest.raises(RuntimeError):
        owner_alert.default_token({'ACTIONS_ID_TOKEN_REQUEST_URL': 'u', 'ACTIONS_ID_TOKEN_REQUEST_TOKEN': 't'})
    with pytest.raises(RuntimeError):
        owner_alert.default_token({})
    assert owner_alert.emit(entry(), env=ENV, opener=forbidden) is False


def fake_emit(results, sink):
    def _emit(e, **kw):
        sink.append(e)
        return results
    return _emit


def test_marker_written_only_on_successful_emit(tmp_path, monkeypatch):
    sink = []
    kw = dict(stage='generation', failure_category='publication', env=ENV, now=NOW)
    monkeypatch.setattr(owner_alert, 'emit', fake_emit(False, sink))
    assert owner_alert.notify(root=tmp_path, **kw) is False
    assert not (tmp_path / owner_alert.MARKER).exists()
    monkeypatch.setattr(owner_alert, 'emit', fake_emit(True, sink))
    assert owner_alert.notify(**kw) is True
    assert not (tmp_path / owner_alert.MARKER).exists()
    assert owner_alert.notify(root=tmp_path, **kw) is True
    assert (tmp_path / owner_alert.MARKER).is_file()
    assert owner_alert.MARKER == '.daily-outcomes/owner_alerted'
    assert sink[0]['jsonPayload']['code'] == 'publication'


def test_notify_never_raises(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError('x')
    monkeypatch.setattr(owner_alert, 'emit', explode)
    assert owner_alert.notify(stage='generation', env=ENV, now=NOW, root=tmp_path) is False


def test_notify_cls_override_keeps_caller_code(monkeypatch):
    sink = []
    monkeypatch.setattr(owner_alert, 'emit', fake_emit(True, sink))
    started = datetime(2026, 11, 15, 5, 0, tzinfo=ET)
    owner_alert.notify(stage='generation', code='heartbeat', cls_override='heartbeat', env=ENV, now=NOW, started_at=started)
    owner_alert.notify(stage='generation', code='repair-plan-ready', cls_override='repair-plan', source='mac_fallback', env=ENV, now=NOW)
    assert [(e['jsonPayload']['class'], e['jsonPayload']['code']) for e in sink] == [
        ('heartbeat', 'heartbeat'), ('repair-plan', 'repair-plan-ready')]


def test_notify_promotes_by_start_not_emit_time(monkeypatch):
    sink = []
    monkeypatch.setattr(owner_alert, 'emit', fake_emit(True, sink))
    monkeypatch.setattr(owner_alert, 'resolve', lambda **k: ('spend_wait_code', Reason('self-heal', 'h', 't', 'f')))
    late = datetime(2026, 11, 15, 8, 40, tzinfo=timezone.utc)  # 03:40 ET
    owner_alert.notify(stage='spend_maintenance', env=ENV, now=late, started_at=datetime(2026, 11, 15, 3, 29, tzinfo=ET))
    owner_alert.notify(stage='spend_maintenance', env=ENV, now=late)
    assert [e['jsonPayload']['class'] for e in sink] == ['self-heal', 'owner-act']


def test_notify_carries_sub_cause_into_details(monkeypatch):
    sink = []
    monkeypatch.setattr(owner_alert, 'emit', fake_emit(True, sink))
    owner_alert.notify(stage='spend_maintenance', code='spend_account', sub_cause='credential_rejected',
                       details={'provider': 'apify', 'http_status': 401}, env=ENV, now=NOW)
    assert sink[0]['jsonPayload']['details'] == {'provider': 'apify', 'http_status': 401, 'sub_cause': 'credential_rejected'}


def test_captured_preflight_survives_a_raise(monkeypatch):
    from scripts import provider_preflight
    monkeypatch.setattr(provider_preflight, 'run_preflight', lambda *a, **k: {'apify': {'status': 'incomplete'}})
    with pytest.raises(ValueError):
        with owner_alert.captured_preflight() as holder:
            assert holder.report is None
            provider_preflight.run_preflight(None, {})
            raise ValueError('after')
    assert holder.report == {'apify': {'status': 'incomplete'}}
    assert provider_preflight.run_preflight.__name__ == '<lambda>'


def policy(**fps):
    return {'providers': {n: {'account': {'credential_fingerprint': fp}} for n, fp in fps.items()}}


@pytest.mark.parametrize('obs,expected', [
    ({'status': 'unavailable', 'http_status': 401}, ('credential_rejected', {'provider': 'apify', 'http_status': 401})),
    ({'status': 'unavailable', 'http_status': 403}, ('credential_rejected', {'provider': 'apify', 'http_status': 403})),
    ({'status': 'unavailable', 'http_status': 500}, ('provider_unreachable', {'provider': 'apify', 'http_status': 500})),
    ({'status': 'unavailable', 'http_status': None}, ('provider_unreachable', {'provider': 'apify'})),
    ({'status': 'incomplete'}, ('metadata_incomplete', {'provider': 'apify'})),
    ({'status': 'missing_credentials'}, ('credential_missing', {'provider': 'apify'})),
    ({'status': 'metadata_verified', 'credential_fingerprint': 'aaaaaa9999999999'}, (None, {})),
    ({'status': 'weird'}, (None, {})),
])
def test_sub_cause_mapping(obs, expected):
    assert owner_alert.sub_cause_from_observation({'apify': obs}, policy(apify='aaaaaa1111111111')) == expected


@pytest.mark.parametrize('name', ['gemini', 'bfl'])
def test_sub_cause_fingerprint_mismatch_is_not_approved_for_fixed_credentials(name):
    obs = {name: {'status': 'metadata_verified', 'credential_fingerprint': 'bbbbbb0000000000'}}
    assert owner_alert.sub_cause_from_observation(obs, policy(**{name: 'aaaaaa1111111111'})) == (
        'credential_not_approved', {'provider': name, 'runner_fp6': 'bbbbbb', 'ledger_fp6': 'aaaaaa'})


def test_sub_cause_rotated_apify_key_is_not_a_mismatch():
    obs = {'apify': {'status': 'metadata_verified', 'credential_fingerprint': 'bbbbbb0000000000'}}
    assert owner_alert.sub_cause_from_observation(obs, policy(apify='aaaaaa1111111111')) == (None, {})


def test_sub_cause_rotated_apify_does_not_mask_gemini_rejection():
    obs = {'apify': {'status': 'metadata_verified', 'credential_fingerprint': 'bbbbbb0000000000'},
           'gemini': {'status': 'unavailable', 'http_status': 401}}
    assert owner_alert.sub_cause_from_observation(obs, policy(apify='aaaaaa1111111111', gemini='cccccc')) == (
        'credential_rejected', {'provider': 'gemini', 'http_status': 401})


def test_rebindable_matches_automatic_spend():
    from src.zeitgeist import automatic_spend
    assert owner_alert.REBINDABLE == frozenset(automatic_spend.REBINDABLE_CREDENTIALS)


def test_sub_cause_ignores_unsplash_and_unpolicied_providers():
    observation = {'unsplash': {'status': 'unavailable', 'http_status': 401},
                   'gemini': {'status': 'incomplete'},
                   'apify': {'status': 'missing_credentials'}}
    assert owner_alert.sub_cause_from_observation(observation, policy(unsplash='aaaaaa', apify='aaaaaa')) == (
        'credential_missing', {'provider': 'apify'})
    assert owner_alert.sub_cause_from_observation({'gemini': {'status': 'incomplete'}}, policy(apify='aaaaaa')) == (None, {})
    assert owner_alert.sub_cause_from_observation(None, None) == (None, {})
    assert owner_alert.sub_cause_from_observation({}, {'providers': 'bad'}) == (None, {})
