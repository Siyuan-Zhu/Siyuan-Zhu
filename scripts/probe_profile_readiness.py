#!/usr/bin/env python3
"""Read canonical application and provider metadata without publishing or spending."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
import logging
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from scripts.check_runtime_configuration import configuration_errors
from scripts.provider_preflight import inspect_unsplash
from src.state_store import get_state_store
from src.zeitgeist.automatic_spend_runtime import (
    PrivateFirestoreStore, maintain_spend_control, observe_provider_metadata,
)
from src.zeitgeist.publication import verify_current_pulse, verify_publication_bundle


class ReadOnlySpendStore:
    """Expose only reads, even if a future controller accidentally tries to write."""
    def __init__(self, store):
        self._store = store

    def read(self, document_id):
        return self._store.read(document_id)

    def commit(self, writes):
        raise RuntimeError('Read-only readiness cannot commit')


def probe(env, *, root=ROOT, now=None):
    """Return no private data; any failed dependency fails the entire probe."""
    if (env.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'
            or env.get('GITHUB_ACTOR') != 'Siyuan-Zhu'
            or env.get('GITHUB_REPOSITORY') not in {
                'Siyuan-Zhu/readme-live-sandbox', 'Siyuan-Zhu/Siyuan-Zhu'}):
        raise RuntimeError('Owner manual dispatch required')
    if configuration_errors(env, daily=True, visitor=True):
        raise RuntimeError('Incomplete reviewed configuration')
    if not re.fullmatch('[a-f0-9]{64}', env.get('SPEND_POLICY_SHA256', '')):
        raise RuntimeError('Pinned policy required')
    for name in ('GEMINI_API_KEY', 'BFL_API_KEY', 'APIFY_TOKEN', 'UNSPLASH_ACCESS_KEY'):
        if not env.get(name):
            raise RuntimeError('Provider credential required')

    # Explicit factory mode refuses checkout fallback and performs no mirrors.
    store = get_state_store(repo_root=root, backend='firestore',
                            collection_prefix=env.get('FIRESTORE_COLLECTION_PREFIX', ''))
    if not store.canonical_only:
        raise RuntimeError('Canonical-only store required')
    projection = store.canonical_projections(write_mirrors=False)
    manifest = verify_publication_bundle(root, require_manifest=True,
                                         require_current_pulse=True)
    pulse = projection['zeitgeist_state.json']
    verify_current_pulse(pulse, manifest)
    if pulse.get('publication_id') != manifest['publication_id']:
        raise RuntimeError('Canonical publication identity mismatch')

    def observer(policy, observed_now):
        report = observe_provider_metadata(policy, env, observed_now)
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            report['unsplash'] = inspect_unsplash(client, env['UNSPLASH_ACCESS_KEY'])
        if any(report.get(provider, {}).get('status') != 'metadata_verified'
               for provider in ('apify', 'gemini', 'bfl', 'unsplash')):
            raise RuntimeError('Provider metadata unavailable')
        return report

    result = maintain_spend_control(ReadOnlySpendStore(PrivateFirestoreStore()),
        expected_policy_sha=env['SPEND_POLICY_SHA256'], observer=observer,
        now=now or datetime.now(timezone.utc), apply=False)
    # Pending handoff recovery must not be presented as readiness to dispatch.
    if result.status != 'planned':
        raise RuntimeError('Accounting requires reconciliation')


def main():
    status = 'blocked'
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        # Neither dependency prints nor raw exceptions may reach a public runner.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            probe(os.environ)
        status = 'ready'
    except BaseException:
        pass
    finally:
        logging.disable(previous_logging)
    print(json.dumps({'status': status, 'mode': 'read_only',
                      'paid_calls': 0, 'ledger_writes': 0}, sort_keys=True))
    return 0 if status == 'ready' else 2


if __name__ == '__main__':
    raise SystemExit(main())
