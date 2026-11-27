"""Spend maintenance tells the owner why, without changing what the child prints."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts import maintain_spend_control as maintenance, provider_preflight
from src.zeitgeist import owner_alert, public_tariff_http as reads

NOW = datetime(2026, 11, 5, 7, tzinfo=timezone.utc)
POLICY = {'providers': {
    'gemini': {'account': {'operation_caps_usd': {'generate:model-a': 1},
                                'credential_fingerprint': 'bbbbbb2222222222'}},
    'apify': {'actor_build': '1.2.3', 'actor_build_id': 'build1',
              'account': {'credential_fingerprint': 'aaaaaa1111111111'}}}}
BLOCKED_KEYS = {'status', 'operation', 'month', 'code'}
TRANSIENT_KEYS = BLOCKED_KEYS | {'retryable', 'retry_not_before'}


class FakeStore:
    def commit(self, values):
        pytest.fail('No accounting write permitted')


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(maintenance, 'ROOT', tmp_path)
    monkeypatch.setattr(maintenance, 'PrivateFirestoreStore', FakeStore)
    monkeypatch.delenv('ZEITGEIST_RUN_STARTED_AT', raising=False)
    emitted, calls = [], []
    monkeypatch.setattr(owner_alert, 'emit', lambda entry, **kw: emitted.append(entry) or True)
    real = owner_alert.notify
    monkeypatch.setattr(owner_alert, 'notify', lambda **kw: calls.append(kw) or real(**kw))

    def blocked_by_apify_401(monkeypatch_=monkeypatch):
        monkeypatch.setattr(provider_preflight, 'run_preflight', lambda *a, **k: {
            'observed_at': 'x', 'gemini': {'status': 'metadata_verified', 'credential_fingerprint': 'bbbbbb2222222222'},
            'apify': {'status': 'unavailable', 'http_status': 401}})
        monkeypatch.setattr(maintenance, 'maintain_spend_control',
                            lambda store, **kw: kw['observer'](POLICY, NOW))
    return tmp_path, emitted, calls, blocked_by_apify_401


def test_apply_block_emits_once_with_sub_cause(harness, capsys):
    root, emitted, calls, block = harness
    block()
    assert maintenance.main(['--apply']) == 2
    assert len(calls) == 1
    assert calls[0]['sub_cause'] == 'credential_rejected'
    assert calls[0]['stage'] == 'spend_maintenance' and calls[0]['root'] == root
    assert calls[0]['details'] == {'provider': 'apify', 'http_status': 401}
    assert len(emitted) == 1
    assert (root / owner_alert.MARKER).exists()


def test_plan_only_maintenance_never_emits(harness):
    root, emitted, calls, block = harness
    block()
    assert maintenance.main([]) == 2
    assert calls == [] and emitted == []
    assert not (root / owner_alert.MARKER).exists()


def test_emit_failure_keeps_exit_code(harness, monkeypatch, capsys):
    root, emitted, calls, block = harness
    block()
    monkeypatch.setattr(owner_alert, 'notify', lambda **kw: (_ for _ in ()).throw(RuntimeError('boom')))
    assert maintenance.main(['--apply']) == 2
    with_failure = capsys.readouterr().out
    monkeypatch.setattr(owner_alert, 'notify', lambda **kw: False)
    assert maintenance.main(['--apply']) == 2
    assert capsys.readouterr().out == with_failure


def test_bad_started_at_never_raises(harness, monkeypatch):
    root, emitted, calls, block = harness
    block()
    monkeypatch.setenv('ZEITGEIST_RUN_STARTED_AT', 'not-a-time')
    assert maintenance.main(['--apply']) == 2
    assert calls[0]['started_at'] is None
    monkeypatch.setenv('ZEITGEIST_RUN_STARTED_AT', '2026-11-05T06:00:00Z')
    assert maintenance.main(['--apply']) == 2
    assert calls[1]['started_at'] == datetime(2026, 11, 5, 6, tzinfo=timezone.utc)


def test_stdout_summary_keys_unchanged(harness, monkeypatch, capsys):
    root, emitted, calls, block = harness
    block()
    assert maintenance.main(['--apply']) == 2
    assert set(json.loads(capsys.readouterr().out)) == BLOCKED_KEYS

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    monkeypatch.setattr(maintenance, 'datetime', Frozen)
    monkeypatch.setattr(reads, 'utc_now', lambda: NOW)

    def transient(store, **kw):
        import httpx
        def handler(request):
            return httpx.Response(503, headers={'retry-after': '120'})
        monkeypatch.setattr(reads.time, 'sleep', lambda d: None)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(maintenance, 'observe_provider_metadata',
                                lambda *a: reads.public_get(client, 'https://bfl.ai/pricing'))
            return kw['observer']({}, NOW)
    monkeypatch.setattr(maintenance, 'maintain_spend_control', transient)
    assert maintenance.main(['--apply']) == 75
    assert set(json.loads(capsys.readouterr().out)) == TRANSIENT_KEYS


def _refreshed(monkeypatch, reason):
    from src.zeitgeist.automatic_spend_runtime import MaintenanceResult

    after = {'providers': {'apify': {} if reason is None else {'resolved_reason': reason}}}
    monkeypatch.setattr(maintenance, 'maintain_spend_control', lambda store, **kw: MaintenanceResult(
        status='refreshed', operation='refresh', month='2026-11', plan={'after': after}))


@pytest.mark.parametrize('reason,code', [('apify_newer_build_input_changed', 'apify-newer-build-input-changed'),
                                         ('apify_newer_build_unavailable', 'apify-newer-build-unavailable')])
def test_a_refresh_that_fell_back_to_the_apify_pin_tells_the_owner_and_still_succeeds(harness, monkeypatch, capsys,
                                                                                      reason, code):
    root, emitted, calls, block = harness
    _refreshed(monkeypatch, reason)
    assert maintenance.main(['--apply']) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {
        'status': 'refreshed', 'operation': 'refresh', 'month': '2026-11'}
    assert [c['code'] for c in calls] == [code] and len(emitted) == 1


@pytest.mark.parametrize('reason', ['newest', 'pinned_no_newer', None])
def test_a_refresh_on_the_newest_build_or_the_pin_tells_nobody(harness, monkeypatch, reason):
    root, emitted, calls, block = harness
    _refreshed(monkeypatch, reason)
    assert maintenance.main(['--apply']) == 0
    assert calls == [] and emitted == []


def test_a_plan_only_refresh_never_tells_the_owner_about_the_build(harness, monkeypatch):
    root, emitted, calls, block = harness
    _refreshed(monkeypatch, 'apify_newer_build_input_changed')
    assert maintenance.main([]) == 0
    assert calls == []
