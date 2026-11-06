#!/usr/bin/env python3
"""One-time owner reconciliation of the November 6 source-coverage failure.

Run 42229783506 stopped at required Instagram coverage before any embedding,
text or image request: the pinned Apify build reported every profile as
not_found. Its execution record is `requires_reconciliation` because a
generation-stage failure cannot prove on its own that no paid inference began.

This script proves it from durable evidence, then records the edition as
`blocked` for that exact source revision. The existing claim rules permit a
later claim only from a different source revision, within the three-attempt
bound. It never clears the record, changes the attempt count or starts work.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.state_store import _commit_documents, PublicationOutcomeUnknown

EDITION = '2026-11-06'
EXPECTED = {'stage': 'requires_reconciliation', 'owner': 'github-42229783506-1',
            'source_revision': '31834093abea5671cc8fc7caf554de767f9e3972',
            'publication_id': None, 'failure_category': 'generation'}
RUN_ID = '42229783506'


def check_outcome(outcome):
    expected = {'edition_date': EDITION, 'run_id': RUN_ID, 'status': 'failed',
                'failure_category': 'source_collection',
                'failure_code': 'required-source-coverage-unavailable',
                'canonical_commit': 'not_committed', 'git_publication': 'not_started'}
    for key, value in expected.items():
        if outcome.get(key) != value:
            raise RuntimeError('Daily outcome evidence differs: ' + key)
    if outcome.get('publication_id') != outcome.get('previous_publication_id'):
        raise RuntimeError('A new publication identity exists; this is not a pre-inference failure')
    health = outcome.get('collection_health') or {}
    if health.get('publishable') is not False or health.get('instagram_item_count') != 0:
        raise RuntimeError('Collection health does not prove the coverage stop')


def paid_rows_in_window(ledger, start, end):
    rows = []
    for key, row in (ledger.get('reservations') or {}).items():
        created = row.get('created_at')
        if created and start <= datetime.fromisoformat(created) <= end:
            rows.append((key, row))
    return rows


def check_ledger(ledger, outcome):
    start = datetime.fromisoformat(outcome['started_at'])
    end = datetime.fromisoformat(outcome['completed_at'])
    rows = paid_rows_in_window(ledger, start, end)
    if any(row.get('provider') != 'apify' for _, row in rows):
        raise RuntimeError('A non-Apify paid request exists in the failed run window')
    if len(rows) != 1 or rows[0][1].get('status') != 'settled':
        raise RuntimeError('Expected exactly one settled Apify request in the failed run window')
    return rows[0][0]


def reconcile(client, outcome, ledger, *, apply=False, now=None):
    now = now or datetime.now(timezone.utc)
    check_outcome(outcome)
    reservation = check_ledger(ledger, outcome)
    doc = client.collection('editorial_executions').document(EDITION)
    snapshot = doc.get()
    prior = snapshot.to_dict() if snapshot.exists else {}
    if prior.get('stage') == 'blocked' and prior.get('reconciliation', {}).get('run_id') == RUN_ID:
        return prior
    for key, value in EXPECTED.items():
        if prior.get(key) != value:
            raise RuntimeError('Execution record differs from reviewed state: ' + key)
    if datetime.fromisoformat(prior['lease_until']) > now:
        raise RuntimeError('Execution lease is still active')
    after = copy.deepcopy(prior)
    after.update(stage='blocked', failure_category='source_collection', updated_at=now.isoformat(),
                 reconciliation={'run_id': RUN_ID, 'reconciled_at': now.isoformat(),
                                 'evidence': 'daily outcome required-source-coverage-unavailable; '
                                             'no canonical commit; only one settled Apify request',
                                 'apify_reservation_id': reservation,
                                 'cause': 'pinned Apify build 0.0.648 reported every profile as not_found'})
    if apply:
        try:
            _commit_documents(client, [('editorial_executions', EDITION, after, snapshot)])
        except PublicationOutcomeUnknown:
            pass
        if doc.get().to_dict() != after:
            raise RuntimeError('Reconciliation outcome unresolved; read before any further action')
    return after


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outcome', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from src.state_store import _FirestoreRestClient
    client = _FirestoreRestClient('git-readme-features')
    now = datetime.now(timezone.utc)
    ledger = client.collection('spend_control').document(now.strftime('%Y-%m')).get().to_dict()
    result = reconcile(client, json.loads(args.outcome.read_text()), ledger, apply=args.apply, now=now)
    print(json.dumps({'status': 'applied' if args.apply else 'planned', 'stage': result['stage'],
                      'attempt': result.get('attempt'), 'source_revision': result.get('source_revision')}))
