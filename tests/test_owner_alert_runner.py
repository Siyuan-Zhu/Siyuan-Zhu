"""Runner call sites, heartbeat, recovered notice and the workflow backstop.

Nothing here touches the network, gcloud or Firestore: the emitter's emit is
replaced by a recorder and every store is a fake.
"""
import copy
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import refresh_daily_digest as daily
from scripts import owner_alert_step as step
from src.zeitgeist import owner_alert
from src.zeitgeist.owner_alert import MARKER

D = chr(36)
ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/daily_zeitgeist.yml'


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'checkout'
    root.mkdir()
    for name in ('assets', 'data'):
        shutil.copytree(ROOT / name, root / name)
    pulse = json.loads((root / 'data/zeitgeist_state.json').read_text())
    store = SimpleNamespace(canonical_only=True, load_pulse_state=lambda: copy.deepcopy(pulse))
    published = date.fromisoformat(pulse['date'])
    stale_now = datetime.combine(published + timedelta(days=1), datetime.min.time(),
                                 tzinfo=timezone.utc).replace(hour=8)
    current_now = datetime.combine(published, datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    return SimpleNamespace(root=root, store=store, stale_now=stale_now, current_now=current_now)


@pytest.fixture
def entries(monkeypatch):
    sent = []

    def record(entry, *, env, **kw):
        sent.append(entry['jsonPayload'])
        return True
    monkeypatch.setattr(owner_alert, 'emit', record)
    return sent


def _fail_on(script, error):
    def runner(cmd, **kw):
        if script in cmd[1]:
            raise error
    return runner


def _alerts(sent):
    return [e for e in sent if e['class'] != 'heartbeat']


def test_heartbeat_emitted_once_per_refresh_without_marker(bundle, entries):
    assert daily.refresh(bundle.root, store=bundle.store, now=bundle.current_now, env={},
                         runner=lambda *a, **k: pytest.fail('no command')) is False
    assert [(e['class'], e['code']) for e in entries] == [('heartbeat', 'heartbeat')]
    assert not (bundle.root / MARKER).exists()


def test_stale_marker_cleared_at_start(bundle, entries):
    marker = bundle.root / MARKER
    marker.parent.mkdir(parents=True)
    marker.touch()
    daily.refresh(bundle.root, store=bundle.store, now=bundle.current_now, env={},
                  runner=lambda *a, **k: None)
    assert not marker.exists()


def test_refresh_reraises_original_error_when_emit_fails(bundle, monkeypatch):
    monkeypatch.setattr(owner_alert, 'emit', lambda *a, **k: False)
    error = RuntimeError('boom')
    with pytest.raises(RuntimeError) as caught:
        daily.refresh(bundle.root, store=bundle.store, now=bundle.stale_now, env={},
                      runner=_fail_on('run_zeitgeist', error))
    assert caught.value is error
    assert not (bundle.root / MARKER).exists()


def test_refresh_reraises_original_error_when_notify_raises(bundle, monkeypatch):
    def explode(**kw):
        if kw.get('cls_override') != 'heartbeat':
            raise KeyboardInterrupt('observer broke')
    monkeypatch.setattr(daily, 'notify', explode)
    error = RuntimeError('boom')
    with pytest.raises(RuntimeError) as caught:
        daily.refresh(bundle.root, store=bundle.store, now=bundle.stale_now, env={},
                      runner=_fail_on('run_zeitgeist', error))
    assert caught.value is error


def test_spend_stage_not_double_emitted_when_marker_exists(bundle, entries):
    def runner(cmd, **kw):
        if 'maintain_spend_control' in cmd[1]:
            (bundle.root / MARKER).parent.mkdir(parents=True, exist_ok=True)
            (bundle.root / MARKER).touch()  # the child already told the owner
            raise subprocess.CalledProcessError(2, cmd)
    with pytest.raises(subprocess.CalledProcessError):
        daily.refresh(bundle.root, store=bundle.store, now=bundle.stale_now,
                      env={'SPEND_POLICY_SHA256': 'pinned'}, runner=runner)
    assert _alerts(entries) == []


def test_generation_unknown_is_watch_on_it(bundle, entries):
    with pytest.raises(RuntimeError):
        daily.refresh(bundle.root, store=bundle.store, now=bundle.stale_now, env={},
                      runner=_fail_on('run_zeitgeist', RuntimeError('boom')))
    [alert] = _alerts(entries)
    assert (alert['class'], alert['code']) == ('watch-on-it', 'unknown')
    assert 'boom' not in json.dumps(alert)
    assert (bundle.root / MARKER).exists()


def test_freshness_message_lookup_is_exact(bundle, entries, monkeypatch):
    def refuse(message):
        def due(*a, **k):
            raise RuntimeError(message)
        monkeypatch.setattr(daily, 'publication_due', due)
        with pytest.raises(RuntimeError):
            daily.refresh(bundle.root, store=bundle.store, now=bundle.stale_now, env={},
                          runner=lambda *a, **k: None)
    refuse('Future publication date; refuse regeneration')
    refuse('Future publication date; refuse regeneration (extra detail)')
    assert [e['code'] for e in _alerts(entries)] == ['future-publication-date', 'freshness-unclassified']
    assert 'extra detail' not in json.dumps(entries)


def test_child_gets_run_started_at(bundle, entries):
    calls = []
    daily.refresh(bundle.root, store=bundle.store, now=bundle.stale_now,
                  env={'SPEND_POLICY_SHA256': 'pinned'},
                  runner=lambda cmd, **kw: calls.append((cmd, kw)))
    child = next(kw for cmd, kw in calls if 'maintain_spend_control' in cmd[1])
    assert child['env']['ZEITGEIST_RUN_STARTED_AT'] == bundle.stale_now.isoformat()
    assert 'PATH' in child['env']
    assert all('env' not in kw for cmd, kw in calls if 'maintain_spend_control' not in cmd[1])


# Stdout of the failure paths on unmodified origin/main with the same fixtures.
EXPECTED_STDOUT = {
    'freshness': '',
    'generation': '',
}


@pytest.mark.parametrize('path', ['freshness', 'generation'])
def test_no_new_public_prints(bundle, entries, capsys, monkeypatch, path):
    if path == 'freshness':
        monkeypatch.setattr(daily, 'publication_due',
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Edition held for owner review')))
        runner = lambda *a, **k: None
    else:
        runner = _fail_on('run_zeitgeist', RuntimeError('boom'))
    with pytest.raises(RuntimeError):
        daily.refresh(bundle.root, store=bundle.store, now=bundle.stale_now, env={}, runner=runner)
    assert capsys.readouterr().out == EXPECTED_STDOUT[path]


# --- the workflow backstop script -------------------------------------------------

def _attempt(root, **fields):
    path = Path(root) / '.daily-outcomes/attempt.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))


def test_failure_step_resolves_from_attempt_json(tmp_path, entries, capsys):
    _attempt(tmp_path, stage='publication', failure_category='publication',
             started_at='2026-11-16T08:00:00+00:00', edition_date='2026-11-16')
    assert step.main(['--outcome', 'failure'], root=tmp_path, env={}) == 0
    assert (entries[-1]['class'], entries[-1]['code']) == ('owner-approve', 'publication')
    assert (tmp_path / MARKER).exists()
    # A present marker means the runner or child already told the owner.
    before = len(entries)
    assert step.main(['--outcome', 'failure'], root=tmp_path, env={}) == 0
    assert len(entries) == before
    assert capsys.readouterr().out == ''


def test_failure_step_without_attempt_is_startup(tmp_path, entries):
    assert step.main(['--outcome', 'failure'], root=tmp_path, env={}) == 0
    assert (entries[-1]['class'], entries[-1]['code']) == ('watch-on-it', 'startup')
    (tmp_path / '.daily-outcomes/attempt.json').write_text('not json')
    (tmp_path / MARKER).unlink()
    assert step.main(['--outcome', 'failure'], root=tmp_path, env={}) == 0
    assert entries[-1]['code'] == 'startup'


def test_steps_never_raise_or_print(tmp_path, capsys):
    def explode(**kw):
        raise SystemExit(3)
    assert step.main(['--outcome', 'failure'], root=tmp_path, env={}, notify=explode) == 0
    assert step.main(['--outcome', 'bogus'], root=tmp_path, env={}) == 0
    assert step.main([], root=tmp_path, env={}) == 0
    assert capsys.readouterr().out == ''


def test_import_failure_still_exits_zero_quietly(tmp_path, capsys, monkeypatch):
    import builtins
    real = builtins.__import__

    def refuse(name, *a, **k):
        if name.startswith('src.zeitgeist'):
            raise ImportError('forced')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', refuse)
    import runpy
    fresh = runpy.run_path(str(ROOT / 'scripts/owner_alert_step.py'), run_name='owner_alert_step_fresh')
    assert fresh['main'](['--outcome', 'success'], root=tmp_path, env={}) == 0
    assert fresh['main'](['--outcome', 'failure'], root=tmp_path, env={}) == 0
    assert capsys.readouterr().out == ''


class _Doc:
    def __init__(self, data):
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return self._data


class _Store:
    def __init__(self, docs, fail=False):
        self.docs, self.fail, self.paths = docs, fail, []

    def _col(self, name):
        return 'pfx_' + name

    def _get_client(self):
        store = self

        class Collection:
            def __init__(self, name):
                self.name = name

            def document(self, key):
                store.paths.append(f'{self.name}/{key}')
                if store.fail:
                    raise RuntimeError('store down')
                return SimpleNamespace(get=lambda: _Doc(store.docs.get(f'{self.name}/{key}')))
        return SimpleNamespace(collection=Collection)


def _success(tmp_path, store, **attempt):
    fields = dict(stage='publication', status='published', edition_date='2026-11-16',
                  started_at='2026-11-16T08:00:00+00:00')
    fields.update(attempt)
    _attempt(tmp_path, **fields)
    return step.main(['--outcome', 'success'], root=tmp_path,
                     env={'FIRESTORE_COLLECTION_PREFIX': 'pfx_'}, store=store)


def test_success_step_recovered_only_after_prior_attempt(tmp_path, entries):
    key = 'pfx_editorial_executions/2026-11-16'
    assert _success(tmp_path, _Store({key: {'attempt': 2}})) == 0
    assert [(e['class'], e['code']) for e in entries] == [('recovered', 'recovered')]
    assert entries[0]['edition_date'] == '2026-11-16'
    # First attempt, missing claim, store failure, unpublished: all quiet.
    for store in (_Store({key: {'attempt': 1}}), _Store({}), _Store({}, fail=True)):
        assert _success(tmp_path, store) == 0
    assert _success(tmp_path, _Store({key: {'attempt': 3}}), status='already_current') == 0
    assert len(entries) == 1
    assert _success(tmp_path, _Store({key: {'attempt': 2}}), edition_date=None) == 0
    assert _success(tmp_path, _Store({key: {'attempt': 2}}), edition_date='2026-13-99') == 0
    assert len(entries) == 1
    (tmp_path / '.daily-outcomes/attempt.json').unlink()
    assert step.main(['--outcome', 'success'], root=tmp_path, env={},
                     store=_Store({key: {'attempt': 2}})) == 0
    assert len(entries) == 1


# --- workflow text ------------------------------------------------------------------

def _job_steps():
    text = WORKFLOW.read_text()
    job = text.split('  synthesize-pulse:\n', 1)[1]
    parts = job.split('\n      - name: ')
    return {p.split('\n', 1)[0]: p for p in parts[1:]}


def test_workflow_owner_steps_are_generic_and_quiet():
    steps = _job_steps()
    failure = steps['Finalize owner notice']
    summary = steps['Finalize owner summary']
    assert 'if: failure() || cancelled()' in failure
    assert 'python3 scripts/owner_alert_step.py --outcome failure' in failure
    assert 'if: success()' in summary
    assert 'python3 scripts/owner_alert_step.py --outcome success' in summary
    names = list(steps)
    final = names.index('Finalize actual daily and Git publication outcomes')
    assert names[final + 1:final + 3] == ['Finalize owner notice', 'Finalize owner summary']
    assert names[final + 3].startswith('Preserve daily outcome')
    for body in (failure, summary):
        assert 'secrets.' not in body
        assert 'id:' not in body
        for needed in ('STATE_STORE_BACKEND: ' + D + '{{ vars.STATE_STORE_BACKEND', 'FIRESTORE_COLLECTION_PREFIX',
                       'GCP_WIF_PROVIDER: ' + D + '{{ vars.GCP_WIF_PROVIDER }}', 'FIRESTORE_DATABASE_ID',
                       'GCP_PROJECT_ID: "git-readme-features"', 'GOOGLE_CLOUD_PROJECT: "git-readme-features"',
                       'FIRESTORE_PROJECT_ID: "git-readme-features"'):
            assert needed in body
        for banned in ('API_KEY', 'APIFY', 'UNSPLASH', 'FIREBASE_SERVICE_ACCOUNT',
                       'GOOGLE_APPLICATION_CREDENTIALS_JSON'):
            assert banned not in body
    head = WORKFLOW.read_text().split('\njobs:', 1)[0]
    assert 'id-token: write' in head


def test_workflow_artifact_path_unchanged_from_origin_main():
    def paths(text):
        return text.split('          path: |\n')[-1]
    current = paths(WORKFLOW.read_text())
    base = subprocess.run(['git', 'show', 'origin/main:.github/workflows/daily_zeitgeist.yml'],
                          cwd=ROOT, capture_output=True, text=True)
    if base.returncode != 0:
        pytest.skip('origin/main unavailable')
    assert current == paths(base.stdout)
