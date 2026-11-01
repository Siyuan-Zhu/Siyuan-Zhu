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


def refresh(root=ROOT, *, env=None, now=None, store=None, runner=subprocess.run):
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('An aware clock is required')
    target_date = now.astimezone(ZoneInfo('America/New_York')).date()
    store = store if store is not None else get_state_store(
        repo_root=root, backend='firestore',
        collection_prefix=env.get('FIRESTORE_COLLECTION_PREFIX', ''))
    due = publication_due(root, store, target_date)
    output = env.get('GITHUB_OUTPUT')
    # Default deny publication, including failures before generation completes.
    if output:
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write('publish_required=false\n')
    if not due:
        print(json.dumps({'status': 'already_current', 'date': str(target_date),
                          'generation_calls': 0}))
        return False
    if env.get('SPEND_POLICY_SHA256'):
        runner([sys.executable, 'scripts/maintain_spend_control.py', '--apply'],
               cwd=root, check=True)
    runner([sys.executable, 'scripts/run_zeitgeist.py', '--date', str(target_date)],
           cwd=root, check=True)
    runner([sys.executable, 'scripts/build_readme.py'], cwd=root, check=True)
    if output:
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write('publish_required=true\n')
    return True


if __name__ == '__main__':
    refresh()
