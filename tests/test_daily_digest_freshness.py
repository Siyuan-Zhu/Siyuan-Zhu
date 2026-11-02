"""Exercise actual publication verification before any maintenance/generation."""
import copy
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from scripts import refresh_daily_digest as daily

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    for name in ('assets', 'data'):
        shutil.copytree(ROOT / name, tmp_path / name)
    pulse = json.loads((tmp_path / 'data/zeitgeist_state.json').read_text())
    store = SimpleNamespace(canonical_only=True, load_pulse_state=lambda: copy.deepcopy(pulse))
    published = date.fromisoformat(pulse['date'])
    return tmp_path, pulse, store, published


def test_same_day_retry_makes_no_provider_or_maintenance_call(bundle, tmp_path):
    root, pulse, store, published = bundle
    output = tmp_path / 'github-output'
    def forbidden(*a, **kw):
        pytest.fail('A current publication must not execute paid or mutating commands')
    now = datetime.combine(published, datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    for _ in range(3):
        assert daily.refresh(root, store=store, now=now,
            env={'GITHUB_OUTPUT': str(output), 'SPEND_POLICY_SHA256': 'pinned'}, runner=forbidden) is False
    assert output.read_text() == 'publish_required=false\n' * 3


def test_stale_dispatch_keeps_existing_funding_guard_and_explicit_date(bundle, tmp_path):
    from datetime import timedelta
    root, pulse, store, published = bundle
    now = datetime.combine(published + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    calls = []
    output = tmp_path / 'github-output'
    assert daily.refresh(root, store=store, now=now,
        env={'GITHUB_OUTPUT': str(output), 'SPEND_POLICY_SHA256': 'pinned'},
        runner=lambda cmd, **kw: calls.append((cmd[1:], kw)))
    assert [c[0] for c in calls] == [
        ['scripts/maintain_spend_control.py', '--apply'],
        ['scripts/run_zeitgeist.py', '--date', str(now.date())],
        ['scripts/build_readme.py']]
    assert all(c[1] == {'cwd': root, 'check': True} for c in calls)
    assert output.read_text() == 'publish_required=false\npublish_required=true\n'


def test_green_commands_without_observed_commit_do_not_advise_reconciliation(bundle, tmp_path):
    from datetime import timedelta
    root, pulse, store, published = bundle
    receipt = tmp_path / 'daily-receipt.json'
    now = datetime.combine(published + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    assert daily.refresh(root, store=store, now=now,
        env={'DAILY_OUTCOME_PATH':str(receipt)}, runner=lambda *a, **kw:None)
    outcome = json.loads(receipt.read_text())
    assert outcome['status'] == 'awaiting_publication'
    assert outcome['canonical_commit'] == 'unknown'
    assert outcome['recovery_action'] == 'inspect_canonical_outcome'
    assert 'Do not resubmit an uncertain mutation' in outcome['recovery_guidance']


@pytest.mark.parametrize('failure', ['identity', 'word', 'asset', 'journal', 'future', 'fallback', 'unavailable'])
def test_ambiguous_or_corrupt_state_stops_before_any_command(bundle, failure):
    from datetime import timedelta
    root, pulse, store, published = bundle
    target = published + timedelta(days=1)
    if failure == 'identity': pulse['publication_id'] = 'unknown'
    if failure == 'word': pulse['word'] = 'Different'
    if failure == 'asset': (root / 'assets/daily_pulse_light.webp').write_bytes(b'corrupt')
    if failure == 'journal': (root / 'data/publication_reconcile.json').write_text('{}')
    if failure == 'future': target = published - timedelta(days=1)
    if failure == 'fallback': store.canonical_only = False
    if failure == 'unavailable':
        def unavailable(): raise RuntimeError('Storage unavailable')
        store.load_pulse_state = unavailable
    def forbidden(*a, **kw): pytest.fail('Must fail before executing any command')
    now = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    with pytest.raises((RuntimeError, ValueError)):
        daily.refresh(root, store=store, now=now, env={}, runner=forbidden)


def test_failed_generation_does_not_request_publish_or_continue(bundle, tmp_path):
    from datetime import timedelta
    import subprocess
    root, pulse, store, published = bundle
    now = datetime.combine(published + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    output = tmp_path / 'github-output'
    calls = []
    def fail(cmd, **kw):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)
    with pytest.raises(subprocess.CalledProcessError):
        daily.refresh(root, store=store, now=now, env={'GITHUB_OUTPUT': str(output)}, runner=fail)
    assert len(calls) == 1 and 'run_zeitgeist.py' in calls[0][1]
    assert output.read_text() == 'publish_required=false\n'


def test_bounded_catchups_share_serialized_writer_and_gate_push():
    text = (ROOT / '.github/workflows/daily_zeitgeist.yml').read_text()
    schedules = text.split('  schedule:', 1)[1].split('  workflow_dispatch:', 1)[0]
    assert schedules.count('- cron:') == 2
    assert 'cron: "0 2 * * *"' in schedules
    assert 'cron: "37 2,3 * * *"' in schedules
    assert schedules.count('timezone: "America/New_York"') == 2
    assert 'group: readme-state-writer' in text and 'cancel-in-progress: false' in text
    assert 'queue: max' in text
    generate = text.split('      - name: Run Daily Zeitgeist Pipeline', 1)[1].split('      - name:', 1)[0]
    assert 'id: digest' in generate and 'refresh_daily_digest.py' in generate
    push = text.split('      - name: Commit and Push', 1)[1].split('      - name:', 1)[0]
    assert "if: steps.digest.outputs.publish_required == 'true'" in push


@pytest.mark.parametrize('utc_hour', [6, 7, 8])
def test_fall_back_day_is_one_publication_date(bundle, utc_hour):
    root, pulse, store, published = bundle
    from unittest.mock import patch
    # 01:00 EDT, 02:00 EST and 03:00 EST all belong to November 1.
    with patch.object(daily, 'publication_due', return_value=False) as due:
        daily.refresh(root, store=store,
            now=datetime(2026, 11, 1, utc_hour, tzinfo=timezone.utc), env={})
    assert due.call_args.args[2] == date(2026, 11, 1)


def test_current_attempt_leaves_sanitized_receipt_without_provider_calls(bundle, tmp_path):
    root, pulse, store, published = bundle
    receipt = tmp_path / 'daily-receipt.json'
    now = datetime.combine(published, datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    env = {'DAILY_OUTCOME_PATH':str(receipt), 'GITHUB_RUN_ID':'1234',
           'GITHUB_RUN_ATTEMPT':'2', 'GITHUB_SHA':'a'*40,
           'GITHUB_REPOSITORY':'Siyuan-Zhu/Siyuan-Zhu', 'GITHUB_EVENT_NAME':'schedule',
           'DAILY_TRIGGER_SCHEDULE':'0 2 * * *', 'GEMINI_API_KEY':'never-export-this-secret'}
    def forbidden(*a, **kw): pytest.fail('Current publication must not call providers')
    assert daily.refresh(root, store=store, now=now, env=env, runner=forbidden) is False
    outcome = json.loads(receipt.read_text())
    assert outcome['schema_version'] == 1
    assert outcome['edition_date'] == str(published)
    assert outcome['status'] == 'already_current'
    assert outcome['stage'] == 'freshness'
    assert outcome['canonical_commit'] == 'not_required'
    assert outcome['git_publication'] == 'not_required'
    assert outcome['publication_id'] == pulse['publication_id']
    assert outcome['run_id'] == '1234' and outcome['run_attempt'] == '2'
    assert outcome['source_revision'] == 'a'*40
    assert outcome['trigger_schedule'] == '0 2 * * *'
    assert 'never-export-this-secret' not in receipt.read_text()


def test_unknown_generation_commit_keeps_journal_and_requests_inspection(bundle, tmp_path):
    from datetime import timedelta
    import subprocess
    root, pulse, store, published = bundle
    receipt = tmp_path / 'daily-receipt.json'
    journal = root / 'data/publication_reconcile.json'
    new_id = 'b'*20
    def fail(cmd, **kw):
        journal.write_text(json.dumps({'publication_id':new_id,
            'status':'canonical_commit_outcome_unknown', 'recovery_path':'/private/never-export'}))
        raise subprocess.CalledProcessError(1, cmd, output='private-provider-body')
    now = datetime.combine(published + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    with pytest.raises(subprocess.CalledProcessError):
        daily.refresh(root, store=store, now=now,
            env={'DAILY_OUTCOME_PATH':str(receipt)}, runner=fail)
    outcome = json.loads(receipt.read_text())
    assert outcome['status'] == 'requires_reconciliation'
    assert outcome['stage'] == 'generation'
    assert outcome['canonical_commit'] == 'unknown'
    assert outcome['publication_id'] == new_id
    assert outcome['previous_publication_id'] == pulse['publication_id']
    assert outcome['recovery_action'] == 'inspect_canonical_outcome'
    assert outcome['git_publication'] == 'not_started'
    assert journal.exists()
    assert '/private/never-export' not in receipt.read_text()
    assert 'private-provider-body' not in receipt.read_text()


def test_actual_offline_photo_cli_records_quality_failure_and_retains_digest(bundle, tmp_path):
    from datetime import timedelta
    import os
    import subprocess
    import sys
    root, pulse, store, published = bundle
    shutil.copytree(ROOT / 'config', root / 'config')
    receipt = tmp_path / 'daily-receipt.json'
    before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    child_env = {'PATH':os.environ.get('PATH',''), 'LANG':'en_US.UTF-8'}
    child_env.update(STATE_STORE_BACKEND='json', ZEITGEIST_OFFLINE_DEMO='1',
        ZEITGEIST_LIVE_LLM='0', ZEITGEIST_LIVE_COLLECT='0', ZEITGEIST_LIVE_UNSPLASH='0',
        DAILY_OUTCOME_PATH=str(receipt))
    def actual_cli(command, **kwargs):
        assert command[1] == 'scripts/run_zeitgeist.py'
        return subprocess.run([sys.executable,str(ROOT / command[1]),*command[2:],
            '--repo-root',str(root),'--prefer-visual-mode','photo'], env=child_env,
            check=True,capture_output=True)
    now = datetime.combine(published + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).replace(hour=8)
    with pytest.raises(subprocess.CalledProcessError):
        daily.refresh(root,store=store,now=now,env={'DAILY_OUTCOME_PATH':str(receipt)},runner=actual_cli)
    outcome=json.loads(receipt.read_text())
    assert outcome['status']=='failed'
    assert outcome['failure_category']=='photo_quality'
    assert outcome['canonical_commit']=='not_committed'
    assert outcome['recovery_action']=='retain_last_digest'
    assert all((root/name).read_bytes()==raw for name,raw in before.items())


def test_daily_workflow_finalizes_after_push_and_always_preserves_the_receipt():
    text=(ROOT/'.github/workflows/daily_zeitgeist.yml').read_text()
    daily=text.split('  synthesize-pulse:',1)[1]
    push=daily.index('      - name: Commit and Push')
    final=daily.index('      - name: Finalize actual daily and Git publication outcomes')
    upload=daily.index('      - name: Preserve daily outcome and any publication recovery assets')
    assert push<final<upload
    finalizer=daily[final:upload]
    assert 'if: always()' in finalizer
    assert 'steps.digest.outcome' in finalizer and 'steps.publish.outcome' in finalizer
    assert 'complete_daily_outcome.py' in finalizer
    assert 'if: always()' in daily[upload:]
    assert '.daily-outcomes/attempt.json' in daily[upload:]
    assert 'include-hidden-files: true' in daily[upload:]


def test_cli_error_after_real_offline_publication_requires_git_reconciliation(bundle, tmp_path, monkeypatch):
    from datetime import timedelta
    import subprocess
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    root,pulse,store,published=bundle
    shutil.copytree(ROOT/'config',root/'config')
    monkeypatch.setenv('STATE_STORE_BACKEND','json')
    monkeypatch.setenv('ZEITGEIST_OFFLINE_DEMO','1')
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM','0')
    monkeypatch.setenv('ZEITGEIST_LIVE_COLLECT','0')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH','0')
    target=published+timedelta(days=1)
    receipt=tmp_path/'receipt.json'
    def committed_then_failed(command,**kwargs):
        ZeitgeistADKOrchestrator(root).run(str(target),prefer_visual_mode='blueprint',offline_demo=True)
        pulse.clear();pulse.update(json.loads((root/'data/zeitgeist_state.json').read_text()))
        raise subprocess.CalledProcessError(1,command)
    with pytest.raises(subprocess.CalledProcessError):
        daily.refresh(root,store=store,
            now=datetime.combine(target,datetime.min.time(),tzinfo=timezone.utc).replace(hour=8),
            env={'DAILY_OUTCOME_PATH':str(receipt)},runner=committed_then_failed)
    outcome=json.loads(receipt.read_text())
    assert outcome['canonical_commit']=='committed'
    assert outcome['status']=='requires_reconciliation'
    assert outcome['publication_id']==pulse['publication_id']
    assert outcome['recovery_action']=='reconcile_retained_publication'
    assert not (root/'data/publication_reconcile.json').exists()


def test_canonical_auth_failure_is_not_reported_as_a_photo_provider_failure(bundle, tmp_path):
    from urllib.error import HTTPError
    root,pulse,store,published=bundle
    receipt=tmp_path/'receipt.json'
    def denied(): raise HTTPError('https://firestore.example.invalid',401,'private-error',{},None)
    store.load_pulse_state=denied
    def forbidden(*a,**kw): pytest.fail('Authentication failure must precede provider calls')
    with pytest.raises(HTTPError):
        daily.refresh(root,store=store,
            now=datetime.combine(published,datetime.min.time(),tzinfo=timezone.utc).replace(hour=8),
            env={'DAILY_OUTCOME_PATH':str(receipt)},runner=forbidden)
    outcome=json.loads(receipt.read_text())
    assert outcome['stage']=='freshness' and outcome['failure_category']=='permission'
    assert outcome['canonical_commit']=='not_attempted'
    assert 'private-error' not in receipt.read_text()
