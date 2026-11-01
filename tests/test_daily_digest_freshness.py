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
