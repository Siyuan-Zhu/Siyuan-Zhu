#!/usr/bin/env python3
"""Generate only when the verified canonical/public digest is older than today.

The Actions workflow holds readme-state-writer for this entire command and push.
An ambiguous publication fails closed; catch-up never regenerates over it.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state_store import get_state_store
from src.zeitgeist.daily_outcomes import new_outcome, receipt_path, save_outcome, record_failure, observe_committed_publication
from src.zeitgeist.publication import verify_current_pulse, verify_publication_bundle


def publication_due(root, store, target_date):
    if not store.canonical_only:
        raise RuntimeError('Canonical-only storage required for daily freshness')
    if (root / 'data/publication_reconcile.json').exists():
        raise RuntimeError('Pending publication requires recovery, not regeneration')
    manifest = verify_publication_bundle(root, require_current_pulse=True)
    pulse = store.load_pulse_state()
    verify_current_pulse(pulse, manifest)
    if pulse.get('publication_id') != manifest['publication_id']:
        raise RuntimeError('Canonical/public identity mismatch; recovery required')
    published_date = date.fromisoformat(manifest['date'])
    if published_date > target_date:
        raise RuntimeError('Future publication date; refuse regeneration')
    return published_date < target_date


def retryable_maintenance_failure(error, *, now):
    """Trust only the local CLI's exact zero-write transient classification."""
    from src.zeitgeist.public_tariff_http import validate_retry_not_before
    if (not isinstance(error,subprocess.CalledProcessError) or error.returncode!=75
            or not isinstance(error.stdout,str) or len(error.stdout)>2048):
        return None
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate recovery field')
            result[key]=value
        return result
    try:
        value=json.loads(error.stdout,object_pairs_hook=unique)
        if (not isinstance(value,dict) or set(value)!={'status','operation','month','code','retryable','retry_not_before'}
                or value['status']!='blocked' or value['operation']!='maintenance'
                or value['code']!='transient-anonymous-price-read' or value['retryable'] is not True
                or value['month']!=now.strftime('%Y-%m')):
            return None
        deadline=validate_retry_not_before(value['retry_not_before'],now=now)
        return max(now,deadline).isoformat()
    except (ValueError,TypeError,KeyError,AttributeError,RecursionError):
        return None


def refresh(root=ROOT, *, env=None, now=None, store=None, runner=subprocess.run):
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('An aware clock is required')
    target_date = now.astimezone(ZoneInfo('America/New_York')).date()
    outcome = new_outcome(env, now, target_date)
    receipt = receipt_path(root, env)
    save_outcome(receipt, outcome)
    output = env.get('GITHUB_OUTPUT')
    # Deny publication even if readiness fails before generation starts.
    if output:
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write('publish_required=false\n')
    if env.get('ZEITGEIST_REVIEW_HOLD_DATE') == str(target_date):
        outcome.update(status='held_for_review', canonical_commit='not_attempted',
                       git_publication='not_required', recovery_action='none')
        save_outcome(receipt, outcome)
        print(json.dumps({'status': 'held_for_review', 'date': str(target_date), 'generation_calls': 0}))
        return False
    execution = None
    try:
        store = store if store is not None else get_state_store(
            repo_root=root, backend='firestore',
            collection_prefix=env.get('FIRESTORE_COLLECTION_PREFIX', ''))
        due = publication_due(root, store, target_date)
        identity = json.loads((Path(root) / 'assets/pulse_manifest.json').read_text())['publication_id']
        outcome.update(publication_id=identity, previous_publication_id=identity)
        if not due:
            outcome.update(status='already_current', canonical_commit='not_required',
                           git_publication='not_required', recovery_action='none')
            save_outcome(receipt, outcome)
            print(json.dumps({'status': 'already_current', 'date': str(target_date),
                              'generation_calls': 0}))
            return False
        from src.zeitgeist.edition_execution import for_store
        execution = for_store(store, target_date, env)
        if execution is not None:
            claim = execution.claim(now=now, source_revision=env.get('GITHUB_SHA','local'),
                                    trigger=outcome['trigger'])
            if claim is None:
                raise RuntimeError('Execution is published but public state is stale; reconcile before generation')
        if env.get('SPEND_POLICY_SHA256'):
            outcome['stage'] = 'spend_maintenance'
            save_outcome(receipt, outcome)
            runner([sys.executable, 'scripts/maintain_spend_control.py', '--apply'],
                   cwd=root, check=True, capture_output=True, text=True)
        if execution is not None:
            execution.transition('generation')
        outcome['stage'] = 'generation'
        save_outcome(receipt, outcome)
        runner([sys.executable, 'scripts/run_zeitgeist.py', '--date', str(target_date)],
               cwd=root, check=True)
        observe_committed_publication(root, store, outcome)
        if execution is not None:
            execution.transition('validation', publication_id=outcome.get('publication_id'))
        outcome['stage'] = 'readme_build'
        save_outcome(receipt, outcome)
        runner([sys.executable, 'scripts/build_readme.py'], cwd=root, check=True)
        outcome.update(stage='publication', status='awaiting_publication',
                       recovery_action='reconcile_retained_publication'
                       if outcome['canonical_commit'] == 'committed' else 'inspect_canonical_outcome')
        outcome['publication_id'] = json.loads(
            (Path(root) / 'assets/pulse_manifest.json').read_text())['publication_id']
        save_outcome(receipt, outcome)
        if execution is not None:
            execution.transition('publication', publication_id=outcome['publication_id'])
        if output:
            with open(output, 'a', encoding='utf-8') as stream:
                stream.write('publish_required=true\n')
        return True
    except BaseException as error:
        try:
            record_failure(root, receipt, outcome, error, store=store)
            if execution is not None:
                observed_now=datetime.now(timezone.utc)
                retry_at=retryable_maintenance_failure(error,now=observed_now) if outcome['stage']=='spend_maintenance' else None
                if retry_at is not None:
                    execution.transition('preflight_retryable',failure_category='transient_anonymous_price_read',
                        retry_not_before=retry_at,now=observed_now)
                    outcome.update(status='failed',failure_category='spend_control',
                        failure_code='transient-anonymous-price-read',retry_not_before=retry_at,
                        recovery_action='wait_configured_catchup',canonical_commit='not_attempted')
                    save_outcome(receipt,outcome)
                else:
                    execution.transition('blocked' if outcome['stage'] in {'freshness','spend_maintenance'}
                                         else 'requires_reconciliation', failure_category=outcome['stage'])
        except Exception:
            pass  # A failed observer must not replace the actual failure.
        raise


if __name__ == '__main__':
    refresh()
