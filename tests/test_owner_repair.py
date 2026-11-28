"""The owner repair page: guard, plan code, retry-today plan/apply, check-credentials.

Fakes only: no network, no gcloud, no Firestore, no real git or gh. The emitter's
emit is replaced by a recorder (conftest also makes the real token refuse).
"""
import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from scripts import owner_repair as repair
from scripts import provider_preflight
from src import state_store
from src.state_store import PublicationConflict
from src.zeitgeist import automatic_spend_runtime as spend_runtime
from src.zeitgeist import owner_reasons
from src.zeitgeist import owner_alert

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/owner_repair.yml'
DAILY = ROOT / '.github/workflows/daily_zeitgeist.yml'
D = chr(36)
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)  # 07:00 ET
EDITION = '2026-11-16'
CROCKFORD = set('0123456789ABCDEFGHJKMNPQRSTVWXYZ')
POLICY_SHA = 'eeecd6d8b21b65c0ddcf4b0e66a6c8be2e1fd053f83f1fbbfb318875dfa9d483'
GOOD_ENV = {
    'GITHUB_EVENT_NAME': 'workflow_dispatch',
    'GITHUB_ACTOR': 'Siyuan-Zhu',
    'GITHUB_TRIGGERING_ACTOR': 'Siyuan-Zhu',
    'GITHUB_REPOSITORY': 'Siyuan-Zhu/Siyuan-Zhu',
    'GITHUB_REF': 'refs/heads/main',
    'GITHUB_WORKFLOW_REF': 'Siyuan-Zhu/Siyuan-Zhu/.github/workflows/owner_repair.yml@refs/heads/main',
    'GITHUB_RUN_ATTEMPT': '1',
    'GITHUB_RUN_ID': '777',
    'SPEND_POLICY_SHA256': POLICY_SHA,
    'FIRESTORE_COLLECTION_PREFIX': '',
}
PLANNED = json.dumps({'month': '2026-11', 'operation': 'refresh', 'status': 'planned'}, sort_keys=True)
POLICY = {'providers': {
    'gemini': {'account': {'operation_caps_usd': {'generate:model-a': 1},
                           'credential_fingerprint': 'bbbbbb2222222222'}},
    'apify': {'actor_build': '1.2.3', 'actor_build_id': 'build1',
              'account': {'credential_fingerprint': 'aaaaaa1111111111'}}}}


# --- fakes ----------------------------------------------------------------------------

class FakeClient:
    """Firestore-like client with exact-revision conditional commits."""

    def __init__(self):
        self.docs = {}
        self.commits = []
        self.reads = []
        self.fail_commits = False

    def collection(self, name):
        client = self

        class _Collection:
            def document(self, doc_id):
                class _Doc:
                    def get(self_inner):
                        client.reads.append((name, doc_id))
                        found = client.docs.get((name, doc_id))
                        if found is None:
                            return SimpleNamespace(exists=False, to_dict=lambda: None, update_time=None)
                        value, version = found
                        snapshot_value = copy.deepcopy(value)
                        return SimpleNamespace(exists=True, to_dict=lambda: copy.deepcopy(snapshot_value),
                                               update_time=f'v{version}')
                return _Doc()
        return _Collection()

    def commit_documents(self, writes):
        self.commits.append(copy.deepcopy([(c, d, v) for c, d, v, _ in writes]))
        if self.fail_commits:
            raise PublicationConflict('read-only')
        for name, doc_id, value, snapshot in writes:
            found = self.docs.get((name, doc_id))
            if snapshot.exists:
                if found is None or snapshot.update_time != f'v{found[1]}':
                    raise PublicationConflict('changed')
            elif found is not None:
                raise PublicationConflict('exists')
        for name, doc_id, value, snapshot in writes:
            found = self.docs.get((name, doc_id))
            version = 0 if found is None else found[1] + 1
            merged = dict(found[0]) if found is not None else {}
            merged.update(copy.deepcopy(value))
            self.docs[(name, doc_id)] = (merged, version)


class FakeStore:
    def __init__(self, client=None, prefix=''):
        self.client = client or FakeClient()
        self.prefix = prefix

    def _get_client(self):
        return self.client

    def _col(self, name):
        return self.prefix + name


class FailingStore:
    def _get_client(self):
        raise AssertionError('store must not be touched')

    def _col(self, name):
        raise AssertionError('store must not be touched')


class FakeRunner:
    def __init__(self, maintenance_stdout=PLANNED, maintenance_rc=0, fail=None):
        self.calls = []
        self.maintenance_stdout = maintenance_stdout
        self.maintenance_rc = maintenance_rc
        self.fail = fail

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        assert kwargs.get('capture_output') is True
        if any('maintain_spend_control.py' in part for part in cmd):
            assert '--apply' not in cmd
            return subprocess.CompletedProcess(cmd, self.maintenance_rc,
                                               stdout='noise GEMINI secret\n' + self.maintenance_stdout + '\n',
                                               stderr='private detail')
        if self.fail and self.fail(cmd):
            raise subprocess.CalledProcessError(1, cmd, output='x', stderr='y')
        return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')

    def git_gh(self):
        return [c for c in self.calls if c and c[0] in ('git', 'gh')]

    def commits(self):
        return [c for c in self.calls if c[0] == 'git' and 'commit' in c]

    def dispatches(self):
        return [c for c in self.calls if c[0] == 'gh']


class SpendStore:
    def __init__(self, policy=POLICY):
        self.policy = policy
        self.commits = []

    def read(self, document_id):
        assert document_id == 'policy_' + POLICY_SHA
        value = copy.deepcopy(self.policy)
        return SimpleNamespace(exists=True, to_dict=lambda: copy.deepcopy(value))

    def commit(self, writes):
        self.commits.append(writes)
        raise AssertionError('read-only')


@pytest.fixture
def sent(monkeypatch):
    entries = []

    def record(entry, *, env, **kw):
        entries.append(entry['jsonPayload'])
        return True
    monkeypatch.setattr(owner_alert, 'emit', record)
    return entries


def claim(**overrides):
    value = {'schema_version': 1, 'edition_date': EDITION, 'owner': 'github-1-1',
             'source_revision': 'abc123', 'trigger': 'schedule', 'stage': 'blocked',
             'attempt': 1, 'started_at': (NOW - timedelta(hours=2)).isoformat(),
             'updated_at': (NOW - timedelta(hours=2)).isoformat(),
             'lease_until': (NOW - timedelta(hours=1)).isoformat(),
             'publication_id': None, 'failure_category': 'spend_maintenance'}
    value.update(overrides)
    return value


def store_with(claim_value, prefix=''):
    store = FakeStore(prefix=prefix)
    if claim_value is not None:
        store.client.docs[(prefix + 'editorial_executions', EDITION)] = (claim_value, 0)
    return store


def run(args, *, env=GOOD_ENV, now=NOW, store=None, runner=None, spend_store=None):
    return repair.main(args, env=dict(env), now=now, store=store or FailingStore(),
                       runner=runner or FakeRunner(), spend_store=spend_store)


def plan_code_of(entries):
    plans = [e for e in entries if e['class'] == 'repair-plan']
    assert len(plans) == 1
    return plans[0]['plan_code']


# --- guard ----------------------------------------------------------------------------

WF = 'Siyuan-Zhu/Siyuan-Zhu/.github/workflows/owner_repair.yml@refs/heads/main'


@pytest.mark.parametrize('field,value,action', [
    ('GITHUB_EVENT_NAME', 'push', 'check-credentials'),
    ('GITHUB_EVENT_NAME', 'schedule', 'plan:retry-today'),
    ('GITHUB_ACTOR', 'someone', 'check-credentials'),
    ('GITHUB_TRIGGERING_ACTOR', 'someone', 'check-credentials'),
    ('GITHUB_TRIGGERING_ACTOR', None, 'check-credentials'),
    ('GITHUB_REPOSITORY', 'Siyuan-Zhu/readme-live-sandbox', 'check-credentials'),
    ('GITHUB_REPOSITORY', 'fork/Siyuan-Zhu', 'plan:retry-today'),
    ('GITHUB_REF', 'refs/heads/feature', 'plan:retry-today'),
    ('GITHUB_REF', 'refs/heads/main2', 'check-credentials'),
    ('GITHUB_WORKFLOW_REF', 'fork/Siyuan-Zhu/.github/workflows/owner_repair.yml@refs/heads/main', 'check-credentials'),
    ('GITHUB_WORKFLOW_REF', 'evil' + WF, 'check-credentials'),
    ('GITHUB_WORKFLOW_REF', WF + '-copy', 'check-credentials'),
    ('GITHUB_WORKFLOW_REF', WF.replace('@refs/heads/main', '@refs/heads/feature'), 'check-credentials'),
    ('GITHUB_WORKFLOW_REF', WF.replace('owner_repair.yml', 'daily_zeitgeist.yml'), 'check-credentials'),
    ('GITHUB_WORKFLOW_REF', None, 'check-credentials'),
    ('GITHUB_RUN_ATTEMPT', '2', 'apply:retry-today'),
    ('GITHUB_RUN_ATTEMPT', None, 'apply:retry-today'),
    ('GITHUB_ACTOR', None, 'none'),
    (None, None, 'apply:anything-else'),
    (None, None, 'plan:check-credentials'),
])
def test_guard_refusals(field, value, action):
    env = dict(GOOD_ENV)
    if field is not None:
        if value is None:
            env.pop(field)
        else:
            env[field] = value
    with pytest.raises(RuntimeError) as caught:
        repair.owner_guard(env, action)
    assert str(caught.value) == 'Owner repair refused'


@pytest.mark.parametrize('action', ['none', 'check-credentials', 'plan:retry-today', 'apply:retry-today'])
def test_guard_accepts_owner_dispatch_on_main(action):
    repair.owner_guard(dict(GOOD_ENV), action)


def test_guard_allows_rerun_for_plan_only():
    repair.owner_guard(dict(GOOD_ENV, GITHUB_RUN_ATTEMPT='2'), 'plan:retry-today')
    repair.owner_guard(dict(GOOD_ENV, GITHUB_RUN_ATTEMPT='2'), 'check-credentials')


def test_guard_runs_before_any_store_secret_or_subprocess(sent, capsys):
    runner = FakeRunner()
    env = dict(GOOD_ENV, GITHUB_ACTOR='someone')
    for action in ('check-credentials', 'plan:retry-today', 'apply:retry-today'):
        assert run(['--action', action, '--plan-code', 'ABCDEFGHJK'], env=env, runner=runner,
                   spend_store=SpendStore()) != 0
    assert runner.calls == [] and sent == []
    assert capsys.readouterr().out == '{"status":"refused"}\n' * 3


# --- plan code ------------------------------------------------------------------------

def test_plan_code_alphabet_and_normalisation():
    plan = {'action': 'retry-today', 'edition_date': EDITION}
    salt = bytes(range(16))
    code = repair.make_plan_code(salt, plan)
    assert len(code) == 10 and set(code) <= CROCKFORD
    # Exact construction: first 50 bits of sha256(salt + canonical json), big-endian.
    digest = hashlib.sha256(salt + json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).digest()
    bits = int.from_bytes(digest[:8], 'big') >> 14
    expected = ''.join('0123456789ABCDEFGHJKMNPQRSTVWXYZ'[(bits >> (45 - 5 * i)) & 31] for i in range(10))
    assert code == expected
    messy = '-'.join([code[:5].lower(), code[5:].lower()])
    assert repair.normalise_plan_code(messy) == code
    assert repair.normalise_plan_code(' ' + code[:3] + ' ' + code[3:].lower() + ' ') == code
    assert repair.make_plan_code(bytes(16), plan) != code
    assert repair.make_plan_code(salt, dict(plan, edition_date='2026-11-17')) != code
    for bad in ('', 'ABC', code + 'A', 'ILOU012345', 'ABCDEFGHJ!'):
        assert repair.normalise_plan_code(bad) is None


# --- retry-today plan refusals --------------------------------------------------------

@pytest.mark.parametrize('claim_value,code', [
    (None, 'no-failed-run'),
    (claim(stage='published', publication_id='pub1'), 'published'),
    (claim(stage='requires_reconciliation', failure_category='generation'), 'reconcile'),
    (claim(stage='generation'), 'reconcile'),
    (claim(stage='validation', publication_id='pub1'), 'reconcile'),
    (claim(stage='awaiting_approval'), 'reconcile'),
    (claim(stage='held_for_review'), 'held'),
    (claim(stage='preflight'), 'not-retryable'),
    (claim(stage='preflight', lease_until=(NOW + timedelta(minutes=7)).isoformat()), 'lease'),
    (claim(lease_until=(NOW + timedelta(minutes=7)).isoformat()), 'lease'),
    (claim(attempt=3, failure_category='generation'), 'attempts'),
    (claim(attempt=3, stage='preflight_retryable', failure_category='transient_anonymous_price_read',
           retry_not_before=(NOW - timedelta(hours=1, minutes=55)).isoformat()), 'attempts'),
    (claim(failure_category='freshness'), 'not-retryable'),
    (claim(publication_id='pub1'), 'reconcile'),
    (claim(stage='preflight_retryable', failure_category='pricing',
           retry_not_before=(NOW - timedelta(hours=1, minutes=55)).isoformat()), 'not-retryable'),
    (claim(stage='preflight_retryable', failure_category='transient_anonymous_price_read',
           updated_at=(NOW - timedelta(minutes=5)).isoformat(),
           retry_not_before=(NOW + timedelta(minutes=12)).isoformat()), 'wait'),
    (claim(stage='preflight_retryable', failure_category='transient_anonymous_price_read',
           retry_not_before='garbage'), 'not-retryable'),
    (claim(stage='mystery'), 'not-retryable'),
])
def test_plan_refusals_published_lease_reconcile_attempts(sent, capsys, claim_value, code):
    store = store_with(claim_value)
    runner = FakeRunner()
    with pytest.raises(repair.Refusal) as caught:
        repair.retry_today_plan(store, EDITION, dict(GOOD_ENV), NOW, runner=runner)
    assert caught.value.code == code
    assert runner.calls == []  # no maintenance child for a claim the page can't retry
    assert run(['--action', 'plan:retry-today'], store=store, runner=runner) != 0
    assert store.client.commits == [] and runner.git_gh() == []
    assert capsys.readouterr().out == '{"status":"refused"}\n'
    assert len(sent) == 1 and sent[0]['class'] == 'repair-result' and 'plan_code' not in sent[0]
    if code == 'lease':
        assert '7 minutes' in sent[0]['headline']
    if code == 'wait':
        assert '12 minutes' in sent[0]['headline']


def test_plan_refusal_words():
    assert 'already published' in repair.refusal_reason(repair.Refusal('published'), EDITION).headline
    assert 'three attempts' in repair.refusal_reason(repair.Refusal('attempts'), EDITION).headline.lower()
    assert "can't retry" in repair.refusal_reason(repair.Refusal('reconcile'), EDITION).headline
    assert 'try in 3 minutes' in repair.refusal_reason(repair.Refusal('lease', minutes=3), EDITION).headline


@pytest.mark.parametrize('stdout,rc', [
    (json.dumps({'status': 'blocked', 'code': 'x', 'operation': 'maintenance', 'month': '2026-11'}), 2),
    (json.dumps({'status': 'reconciliation_ready', 'operation': 'handoff', 'month': '2026-11'}), 0),
    (PLANNED, 2),
    ('not json', 0),
])
def test_plan_refuses_unless_maintenance_plan_is_planned(stdout, rc):
    runner = FakeRunner(maintenance_stdout=stdout, maintenance_rc=rc)
    with pytest.raises(repair.Refusal) as caught:
        repair.retry_today_plan(store_with(claim()), EDITION, dict(GOOD_ENV), NOW, runner=runner)
    assert caught.value.code == 'maintenance'
    assert len(runner.calls) == 1


def test_maintenance_runner_timeout_refuses():
    def runner(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    with pytest.raises(repair.Refusal) as caught:
        repair.retry_today_plan(store_with(claim()), EDITION, dict(GOOD_ENV), NOW, runner=runner)
    assert caught.value.code == 'maintenance'


@pytest.mark.parametrize('claim_value', [
    claim(attempt=3),  # blocked/spend_maintenance is exempt from the attempt bound, as in claim()
    claim(attempt=5),
    claim(stage='preflight_retryable', failure_category='transient_anonymous_price_read',
          retry_not_before=(NOW - timedelta(hours=1, minutes=55)).isoformat()),
])
def test_plan_allows_retryable_claims(claim_value):
    runner = FakeRunner()
    plan = repair.retry_today_plan(store_with(claim_value), EDITION, dict(GOOD_ENV), NOW, runner=runner)
    assert plan == {'action': 'retry-today', 'edition_date': EDITION, 'claim_stage': claim_value['stage'],
                    'attempt': claim_value['attempt'], 'source_revision': 'abc123',
                    'failure_category': claim_value['failure_category'], 'publication_id': 'none',
                    'maintenance': 'planned'}
    assert len(runner.calls) == 1 and runner.git_gh() == []


def test_plan_is_deterministic_and_reads_prefixed_claim():
    store = store_with(claim(), prefix='stg_')
    first = repair.retry_today_plan(store, EDITION, dict(GOOD_ENV), NOW, runner=FakeRunner())
    later = repair.retry_today_plan(store, EDITION, dict(GOOD_ENV), NOW + timedelta(hours=3), runner=FakeRunner())
    assert first == later
    assert ('stg_editorial_executions', EDITION) in store.client.reads


# --- plan then apply ------------------------------------------------------------------

def test_plan_then_apply_happy_path(sent, capsys):
    store = store_with(claim())
    runner = FakeRunner()
    assert run(['--action', 'plan:retry-today'], store=store, runner=runner) == 0
    assert capsys.readouterr().out == '{"status":"ok"}\n'
    code = plan_code_of(sent)
    assert sent[0]['class'] == 'repair-plan' and code in sent[0]['fix']
    assert runner.git_gh() == []
    records = {k: v for k, v in store.client.docs.items() if k[0] == 'owner_repairs'}
    assert list(records) == [('owner_repairs', hashlib.sha256(code.encode()).hexdigest())]
    record = next(iter(records.values()))[0]
    assert set(record) == {'action', 'plan', 'salt_b64', 'created_at', 'expires_at', 'used'}
    assert record['used'] is False and record['action'] == 'retry-today'
    assert len(base64.b64decode(record['salt_b64'])) == 16
    assert datetime.fromisoformat(record['expires_at']) - datetime.fromisoformat(record['created_at']) == timedelta(hours=12)
    assert code not in json.dumps(record)
    assert repair.make_plan_code(base64.b64decode(record['salt_b64']), record['plan']) == code

    pasted = (code[:5] + '-' + code[5:]).lower()
    later = NOW + timedelta(minutes=10)
    assert run(['--action', 'apply:retry-today', '--plan-code', pasted], store=store, runner=runner,
               now=later) == 0
    assert capsys.readouterr().out == '{"status":"ok"}\n'
    record = next(v for k, v in store.client.docs.items() if k[0] == 'owner_repairs')[0]
    assert record['used'] is True
    assert runner.git_gh() == [
        ['git', '-c', 'user.name=Siyuan-Zhu',
         '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com',
         'commit', '--allow-empty', '-m', 'chore: owner retry [skip ci]'],
        ['git', 'push', 'origin', 'HEAD:refs/heads/main'],
        ['gh', 'workflow', 'run', 'daily_zeitgeist.yml', '--ref', 'main'],
    ]
    assert len(runner.commits()) == 1 and len(runner.dispatches()) == 1
    results = [e for e in sent if e['class'] == 'repair-result']
    assert len(results) == 1 and 'plan_code' not in results[0]
    assert 'started' in results[0]['headline']
    # Firestore writes: the plan record, then one compare-and-set flip.
    assert len(store.client.commits) == 2
    assert all(c[0][0] == 'owner_repairs' for c in store.client.commits)


@pytest.mark.parametrize('case', ['used', 'expired', 'foreign', 'malformed', 'missing', 'changed',
                                  'wrong-action', 'cas-race', 'bad-salt'])
def test_apply_refuses_used_expired_or_foreign_code(sent, capsys, case):
    store = store_with(claim())
    runner = FakeRunner()
    assert run(['--action', 'plan:retry-today'], store=store, runner=runner) == 0
    code = plan_code_of(sent)
    key = ('owner_repairs', hashlib.sha256(code.encode()).hexdigest())
    now = NOW + timedelta(minutes=5)
    if case == 'used':
        assert run(['--action', 'apply:retry-today', '--plan-code', code], store=store, runner=runner, now=now) == 0
        assert len(runner.commits()) == 1
    elif case == 'expired':
        now = NOW + timedelta(hours=12, minutes=1)
    elif case == 'foreign':
        code = '0123456789' if code != '0123456789' else 'ABCDEFGHJK'
    elif case == 'malformed':
        code = 'not-a-code!'
    elif case == 'missing':
        code = ''
    elif case == 'changed':
        value, version = store.client.docs[('editorial_executions', EDITION)]
        store.client.docs[('editorial_executions', EDITION)] = (dict(value, attempt=2, source_revision='def'), version + 1)
    elif case == 'wrong-action':
        value, version = store.client.docs[key]
        store.client.docs[key] = (dict(value, action='check-credentials'), version + 1)
    elif case == 'bad-salt':
        value, version = store.client.docs[key]
        store.client.docs[key] = (dict(value, salt_b64=base64.b64encode(bytes(16)).decode()), version + 1)
    elif case == 'cas-race':
        original = store.client.commit_documents

        def racing(writes):
            value, version = store.client.docs[key]
            store.client.docs[key] = (dict(value, used=True), version + 1)
            return original(writes)
        store.client.commit_documents = racing
    capsys.readouterr()
    commits_before = len(runner.commits())
    sent.clear()
    args = ['--action', 'apply:retry-today'] + (['--plan-code', code] if case != 'missing' else [])
    assert run(args, store=store, runner=runner, now=now) != 0
    assert capsys.readouterr().out == '{"status":"refused"}\n'
    assert len(runner.commits()) == commits_before
    assert len(runner.dispatches()) == (1 if case == 'used' else 0)
    assert len(sent) == 1 and sent[0]['class'] == 'repair-result' and 'plan_code' not in sent[0]
    if case not in ('used', 'cas-race'):
        assert store.client.docs[key][0]['used'] is False


def test_apply_refuses_rerun_attempt(sent, capsys):
    store = store_with(claim())
    runner = FakeRunner()
    run(['--action', 'plan:retry-today'], store=store, runner=runner)
    code = plan_code_of(sent)
    assert run(['--action', 'apply:retry-today', '--plan-code', code], store=store, runner=runner,
               env=dict(GOOD_ENV, GITHUB_RUN_ATTEMPT='2')) != 0
    assert runner.git_gh() == []


@pytest.mark.parametrize('failing,expect', [('commit', 0), ('push', 1), ('gh', 2)])
def test_apply_step_failure_after_cas_keeps_code_used(sent, capsys, failing, expect):
    store = store_with(claim())
    runner = FakeRunner(fail=lambda cmd: failing in cmd)
    run(['--action', 'plan:retry-today'], store=store, runner=runner)
    code = plan_code_of(sent)
    sent.clear()
    capsys.readouterr()
    assert run(['--action', 'apply:retry-today', '--plan-code', code], store=store, runner=runner) != 0
    assert capsys.readouterr().out == '{"status":"refused"}\n'
    record = next(v for k, v in store.client.docs.items() if k[0] == 'owner_repairs')[0]
    assert record['used'] is True
    assert len(runner.git_gh()) == expect + 1  # stops at the failing step, never retries
    assert len(sent) == 1 and sent[0]['class'] == 'repair-result'


# --- check-credentials ----------------------------------------------------------------

def test_check_credentials_is_read_only(sent, capsys, monkeypatch):
    calls = []

    def apify_401(*args, **kwargs):
        calls.append(1)
        return {'observed_at': 'x',
                'gemini': {'status': 'metadata_verified', 'credential_fingerprint': 'bbbbbb2222222222'},
                'apify': {'status': 'unavailable', 'http_status': 401, 'error': 'Bearer leaked-text'}}
    monkeypatch.setattr(provider_preflight, 'run_preflight', apify_401)
    monkeypatch.setattr(state_store, '_commit_documents', lambda *a, **k: pytest.fail('no write'))
    spend = SpendStore()
    runner = FakeRunner()
    assert run(['--action', 'check-credentials'], store=FailingStore(), runner=runner, spend_store=spend) == 0
    assert capsys.readouterr().out == '{"status":"ok"}\n'
    assert calls == [1] and spend.commits == [] and runner.calls == []
    assert len(sent) == 1
    entry = sent[0]
    assert entry['class'] == 'repair-result'
    assert entry['details'] == {'provider': 'apify', 'http_status': 401, 'sub_cause': 'credential_rejected'}
    assert entry['headline'] == 'A provider rejected our key'
    assert 'leaked' not in json.dumps(entry) and 'plan_code' not in entry


def test_check_credentials_all_good(sent, capsys, monkeypatch):
    def fine(policy, env, now):
        from scripts import provider_preflight as pf
        pf.run_preflight(None, {}, now=now, required_models=[])
        return {}
    monkeypatch.setattr(provider_preflight, 'run_preflight', lambda *a, **k: {
        'gemini': {'status': 'metadata_verified', 'credential_fingerprint': 'bbbbbb2222222222'},
        'apify': {'status': 'metadata_verified', 'credential_fingerprint': 'aaaaaa1111111111'}})
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', fine)
    assert run(['--action', 'check-credentials'], spend_store=SpendStore()) == 0
    assert len(sent) == 1 and sent[0]['headline'] == 'All providers accept our keys'


def test_check_credentials_all_good_after_apify_rebind(sent, capsys, monkeypatch):
    def fine(policy, env, now):
        from scripts import provider_preflight as pf
        pf.run_preflight(None, {}, now=now, required_models=[])
        return {}
    monkeypatch.setattr(provider_preflight, 'run_preflight', lambda *a, **k: {
        'gemini': {'status': 'metadata_verified', 'credential_fingerprint': 'bbbbbb2222222222'},
        'apify': {'status': 'metadata_verified', 'credential_fingerprint': 'dddddd7777777777'}})
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', fine)
    assert run(['--action', 'check-credentials'], spend_store=SpendStore()) == 0
    assert len(sent) == 1 and sent[0]['headline'] == 'All providers accept our keys'
    assert 'sub_cause' not in sent[0]['details']


def test_check_credentials_emit_failure_is_refused(sent, capsys, monkeypatch):
    monkeypatch.setattr(owner_alert, 'emit', lambda entry, **kw: False)
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', lambda policy, env, now: {})
    assert run(['--action', 'check-credentials'], spend_store=SpendStore()) != 0
    assert capsys.readouterr().out == '{"status":"refused"}\n'


def test_check_credentials_unexpected_error_is_generic(sent, capsys, monkeypatch):
    def broken(policy, env, now):
        raise ValueError('provider said zq-private-text')
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', broken)
    assert run(['--action', 'check-credentials'], spend_store=SpendStore()) == 0
    assert len(sent) == 1 and 'zq-private-text' not in json.dumps(sent[0])
    assert sent[0]['class'] == 'repair-result'


# --- public surface -------------------------------------------------------------------

def test_stdout_is_only_status_json(sent, capsys, monkeypatch):
    def noisy(policy, env, now):
        print('PROVIDER TEXT to stdout')
        import sys
        print('PROVIDER TEXT to stderr', file=sys.stderr)
        import logging
        logging.getLogger().error('PROVIDER TEXT logged')
        return {}
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', noisy)
    store = store_with(claim())
    runner = FakeRunner()
    outputs = []
    for args in (['--action', 'none'], ['--action', 'check-credentials'], ['--action', 'plan:retry-today'],
                 ['--action', 'bogus'], ['--plan-code'], []):
        run(args, store=store, runner=runner, spend_store=SpendStore())
        captured = capsys.readouterr()
        outputs.append(captured.out)
        assert captured.err == ''
    code = plan_code_of(sent)
    run(['--action', 'apply:retry-today', '--plan-code', code], store=store, runner=runner)
    run(['--action', 'apply:retry-today', '--plan-code', code], store=store, runner=runner)
    captured = capsys.readouterr()
    outputs.append(captured.out)
    assert captured.err == ''
    for out in outputs:
        for line in out.splitlines():
            assert line in ('{"status":"ok"}', '{"status":"refused"}')
    assert outputs[:4] == ['{"status":"ok"}\n'] * 3 + ['{"status":"refused"}\n']
    assert outputs[-1] == '{"status":"ok"}\n{"status":"refused"}\n'


def test_none_action_does_nothing(sent, capsys):
    runner = FakeRunner()
    assert run(['--action', 'none'], runner=runner) == 0
    assert runner.calls == [] and sent == []


def test_script_has_no_dollar_and_no_print_of_codes():
    text = (ROOT / 'scripts/owner_repair.py').read_text()
    assert D not in text
    assert text.count('print(') == 1


# --- workflow text --------------------------------------------------------------------

def _digest_env_line(name):
    text = DAILY.read_text()
    block = text.split('id: digest', 1)[1].split('run: |', 1)[0]
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith(name + ':')]
    assert len(lines) == 1
    return lines[0]


def test_workflow_yaml_shape():
    text = WORKFLOW.read_text()
    on = text.split('\non:', 1)[1].split('\npermissions:', 1)[0]
    assert 'workflow_dispatch:' in on and 'schedule' not in on and 'push' not in on and 'pull_request' not in on
    assert re.search(r'\n {6}action:\n(?: {8}.*\n)*? {8}type: choice', on)
    assert 'options: [none, check-credentials, "plan:retry-today", "apply:retry-today"]' in on
    assert re.search(r'\n {8}default: none\n', on)
    assert re.search(r'\n {6}plan_code:\n(?: {8}.*\n)*? {8}type: string\n(?: {8}.*\n)*? {8}required: false', on)
    perms = text.split('\npermissions:', 1)[1].split('\n\n', 1)[0]
    assert [line.strip() for line in perms.strip().splitlines()] == ['contents: write', 'actions: write', 'id-token: write']
    assert 'concurrency:\n  group: readme-state-writer\n  cancel-in-progress: false\n  queue: max\n' in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.actor == 'Siyuan-Zhu'" in text and "github.triggering_actor == 'Siyuan-Zhu'" in text
    assert "github.repository == 'Siyuan-Zhu/Siyuan-Zhu'" in text and "github.ref == 'refs/heads/main'" in text
    names = re.findall(r'\n\s+(?:- )?name: (.*)', text)
    for name in names:
        assert not re.search(r'repair|retry|credential|plan|apify|gemini|token|key|spend', name, re.I), name
    assert text.splitlines()[0].startswith('name: ') and 'repair' not in text.splitlines()[0].lower()
    assert 'upload-artifact' not in text and 'GITHUB_STEP_SUMMARY' not in text and '::' not in text
    assert 'persist-credentials: false' not in text and 'actions/checkout@v4' in text
    for name in ('GEMINI_API_KEY', 'BFL_API_KEY', 'APIFY_TOKEN', 'APIFY_API_TOKEN', 'UNSPLASH_ACCESS_KEY',
                 'SPEND_POLICY_SHA256', 'FIRESTORE_COLLECTION_PREFIX', 'GCP_PROJECT_ID', 'GOOGLE_CLOUD_PROJECT',
                 'FIRESTORE_PROJECT_ID', 'GCP_WIF_PROVIDER', 'FIRESTORE_DATABASE_ID'):
        assert '\n          ' + _digest_env_line(name) + '\n' in text, name
    assert 'GH_TOKEN: ' + D + '{{ github.token }}' in text
    # Inputs reach the script only through env, never interpolated into the run line.
    run_lines = [line for line in text.splitlines() if line.strip().startswith('run:') or line.startswith('          python')]
    assert all('inputs.' not in line for line in run_lines)
    assert 'OWNER_ACTION: ' + D + '{{ inputs.action }}' in text
    assert 'OWNER_PLAN_CODE: ' + D + '{{ inputs.plan_code }}' in text
    assert 'shell: python\n' in text
    assert '"--action", os.environ.get("OWNER_ACTION", "none"), "--plan-code", os.environ.get("OWNER_PLAN_CODE", "")' in text
    body = text.split('run: |', 1)[1]
    assert D not in body and 'inputs.' not in body


# --- fix round 1 ----------------------------------------------------------------------

SECRET_NOTE = ('The page cannot change a GitHub secret, so a token fix is a hand step,'
               ' then check-credentials, then retry-today.')
B1 = 'automatic-provider-metadata-or-credential-binding-failed'


@pytest.mark.parametrize('sub_cause', sorted(owner_alert.SUB_CAUSES))
def test_check_credentials_uses_b1_fix_and_secret_note(sent, capsys, monkeypatch, sub_cause):
    monkeypatch.setattr(owner_alert, 'sub_cause_from_observation',
                        lambda report, policy: (sub_cause, {'provider': 'apify'}))
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', lambda policy, env, now: {})
    assert run(['--action', 'check-credentials'], spend_store=SpendStore()) == 0
    _, b1 = owner_reasons.resolve(stage='spend_maintenance', code=B1, sub_cause=sub_cause)
    assert len(sent) == 1
    entry = sent[0]
    assert entry['headline'] == b1.headline
    assert entry['fix'].startswith(b1.fix)
    assert SECRET_NOTE in entry['tonight'] + ' ' + entry['fix']
    assert entry['details']['sub_cause'] == sub_cause


def test_provider_unreachable_keeps_nothing_to_do_yet(sent, capsys, monkeypatch):
    monkeypatch.setattr(owner_alert, 'sub_cause_from_observation',
                        lambda report, policy: ('provider_unreachable', {'provider': 'gemini'}))
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', lambda policy, env, now: {})
    run(['--action', 'check-credentials'], spend_store=SpendStore())
    assert sent[0]['fix'].startswith('Nothing to do yet')


@pytest.mark.parametrize('fails', [False, True])
def test_check_credentials_other_results_carry_secret_note(sent, capsys, monkeypatch, fails):
    def observe(policy, env, now):
        if fails:
            raise ValueError('x')
        return {}
    monkeypatch.setattr(spend_runtime, 'observe_provider_metadata', observe)
    run(['--action', 'check-credentials'], spend_store=SpendStore())
    assert SECRET_NOTE in sent[0]['tonight'] + ' ' + sent[0]['fix']


def test_maintenance_refusal_carries_secret_note():
    reason = repair.refusal_reason(repair.Refusal('maintenance'), EDITION)
    assert SECRET_NOTE in reason.tonight + ' ' + reason.fix


def test_plan_emit_failure_is_refused(sent, capsys, monkeypatch):
    monkeypatch.setattr(owner_alert, 'emit', lambda entry, **kw: False)
    store = store_with(claim())
    assert run(['--action', 'plan:retry-today'], store=store, runner=FakeRunner()) != 0
    assert capsys.readouterr().out == '{"status":"refused"}\n'
    records = [v[0] for k, v in store.client.docs.items() if k[0] == 'owner_repairs']
    assert len(records) == 1 and records[0]['used'] is False  # the unused record just expires


def test_attempts_wording_is_neutral():
    headline = repair.refusal_reason(repair.Refusal('attempts'), EDITION).headline
    assert headline == 'Three attempts are used, this needs a reconcile'
    assert 'paid' not in headline


def test_script_has_no_top_level_project_imports():
    import ast
    tree = ast.parse((ROOT / 'scripts/owner_repair.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith(('src', 'scripts')), node.module
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith(('src', 'scripts')) for a in node.names)


class _NoisyModule:
    def __getattr__(self, name):
        import sys
        print('import-time warning with provider text', file=sys.stderr)
        print('import-time stdout noise')
        raise ImportError('broken import')


@pytest.mark.parametrize('module', ['src.zeitgeist.owner_alert', 'src.zeitgeist.automatic_spend_runtime',
                                    'src.state_store', 'src.zeitgeist.owner_reasons'])
@pytest.mark.parametrize('action', ['check-credentials', 'plan:retry-today'])
def test_failing_import_yields_only_refused(capsys, monkeypatch, module, action):
    import sys
    monkeypatch.setitem(sys.modules, module, _NoisyModule())
    code = run(['--action', action], store=store_with(claim()), runner=FakeRunner(), spend_store=SpendStore())
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == '{"status":"refused"}\n' and captured.err == ''
