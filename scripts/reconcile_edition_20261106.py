#!/usr/bin/env python3
"""One-time owner reconciliation of the November 6 source-coverage failure.

Run 42229783506 stopped at required Instagram coverage before any embedding,
text or image request: the pinned Apify build reported every profile as
not_found. Its execution record is `requires_reconciliation` because a
generation-stage failure cannot prove on its own that no paid inference began.

Each reviewed run is proven from durable evidence, then records the edition as
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
# Reviewed failed runs for this edition, each with its exact durable state.
REVIEWED = {
    '42229783506': {
        'record': {'stage': 'requires_reconciliation', 'owner': 'github-42229783506-1',
                   'source_revision': '31834093abea5671cc8fc7caf554de767f9e3972',
                   'publication_id': None, 'failure_category': 'generation', 'attempt': 1},
        'outcome': {'failure_category': 'source_collection',
                    'failure_code': 'required-source-coverage-unavailable'},
        'providers': {'apify'}, 'reservations': 1,
        'cause': 'pinned Apify build 0.0.648 reported every profile as not_found'},
    '42240720931': {
        'record': {'stage': 'requires_reconciliation', 'owner': 'github-42240720931-1',
                   'source_revision': '5880c5aee594fff6d4b96915889d7ad28086b277',
                   'publication_id': None, 'failure_category': 'generation', 'attempt': 2},
        'outcome': {'failure_category': 'photo_quality', 'recovery_action': 'retain_last_digest'},
        'providers': {'apify', 'gemini'}, 'reservations': None,
        'cause': 'no photograph met the quality threshold after three bounded searches'},
}


def check_outcome(outcome, review):
    expected = dict({'edition_date': EDITION, 'status': 'failed',
                     'canonical_commit': 'not_committed', 'git_publication': 'not_started'},
                    **review['outcome'])
    for key, value in expected.items():
        if outcome.get(key) != value:
            raise RuntimeError('Daily outcome evidence differs: ' + key)
    if outcome.get('publication_id') != outcome.get('previous_publication_id'):
        raise RuntimeError('A new publication identity exists; nothing may have been published')
    health = outcome.get('collection_health')
    if review['outcome'].get('failure_category') == 'source_collection' and (
            not health or health.get('publishable') is not False or health.get('instagram_item_count') != 0):
        raise RuntimeError('Collection health does not prove the coverage stop')


def paid_rows_in_window(ledger, start, end):
    rows = []
    for key, row in (ledger.get('reservations') or {}).items():
        created = row.get('created_at')
        if created and start <= datetime.fromisoformat(created) <= end:
            rows.append((key, row))
    return rows


def check_ledger(ledger, outcome, review):
    start = datetime.fromisoformat(outcome['started_at'])
    end = datetime.fromisoformat(outcome['completed_at'])
    rows = paid_rows_in_window(ledger, start, end)
    if any(row.get('provider') not in review['providers'] for _, row in rows):
        raise RuntimeError('An unexpected paid provider exists in the failed run window')
    if not rows or any(row.get('status') != 'settled' for _, row in rows):
        raise RuntimeError('Every paid request in the failed run window must be settled')
    if review['reservations'] is not None and len(rows) != review['reservations']:
        raise RuntimeError('Unexpected number of paid requests in the failed run window')
    return sorted(key for key, _ in rows)


def reconcile(client, outcome, ledger, *, apply=False, now=None):
    now = now or datetime.now(timezone.utc)
    run_id = outcome.get('run_id')
    review = REVIEWED.get(run_id)
    if review is None:
        raise RuntimeError('Run is not a reviewed November 6 failure')
    check_outcome(outcome, review)
    reservations = check_ledger(ledger, outcome, review)
    doc = client.collection('editorial_executions').document(EDITION)
    snapshot = doc.get()
    prior = snapshot.to_dict() if snapshot.exists else {}
    if prior.get('stage') == 'blocked' and prior.get('reconciliation', {}).get('run_id') == run_id:
        return prior
    for key, value in review['record'].items():
        if prior.get(key) != value:
            raise RuntimeError('Execution record differs from reviewed state: ' + key)
    if datetime.fromisoformat(prior['lease_until']) > now:
        raise RuntimeError('Execution lease is still active')
    after = copy.deepcopy(prior)
    after.update(stage='blocked', failure_category=review['outcome']['failure_category'],
                 updated_at=now.isoformat(),
                 reconciliation={'run_id': run_id, 'reconciled_at': now.isoformat(),
                                 'evidence': 'daily outcome failed before publication; no canonical commit; '
                                             'every paid request in the run window settled',
                                 'paid_reservation_ids': reservations, 'cause': review['cause']})
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
