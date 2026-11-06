#!/usr/bin/env python3
"""One-time, exact November 6 Apify build re-pin; no economic policy change.

Instagram changed on 2026-11-06 and pinned build 0.0.648 now reports every
profile as not_found. Apify's 0.0.659 works, with an identical pricing record
and input schema. This moves only the three build identity pins to 0.0.659.

Requires a fresh authenticated observation that matches the reviewed values.
Applies an immutable successor policy and ledger pointer atomically with a
durable receipt. A repeated or uncertain dispatch reconciles by reading the
receipt, never by replaying.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from src.zeitgeist.automatic_spend import digest, validate_policy
from src.state_store import _commit_documents, PublicationOutcomeUnknown

RECEIPT_ID = 'apify_build_repair_20261106'
OLD = {'actor_build': '0.0.648', 'actor_build_id': 'jwxpsPozhIjbsOspW',
       'actor_image_digest': '39fefb5195cc6c5f5dd53586a3fb5616200f4e5eb65ebb77a5b06cacf456f31b'}
NEW = {'actor_build': '0.0.659', 'actor_build_id': 'Ih1cRmBcVyKVYQlOj',
       'actor_image_digest': 'bc17a0030ca7534e2d4f921fbc8248d34ccf0932b4d9be740c7844bf25b928e0'}
# Unchanged contract: the successor is only valid while these stay identical.
SCHEMA_SHA = 'c05eebabee083a1a348aebebbc2084beb417dee1b8ae534b71be58f6578981c0'
PRICING_SHA = '12cc41388ad55dda87685f711086d048eda188ea00ec6e9d699c55adc9e86d0d'
ACCOUNT = '6d57321aabee388b'
# Reviewed identities of the November 6 transition; the CLI refuses any other.
EXPECTED_OLD_SHA = '9c4cd3a01eddff725768a9ffeaf82ec7b87677e94823010a2a94fa4a28888b05'
EXPECTED_NEW_SHA = '5a5e5fd1a04688a0870c321906bc3eea00713fe8110865249d52df1bff0f2aaf'


def observe(token, now):
    """Fresh authenticated metadata for the new build; no actor is started."""
    from scripts.provider_preflight import inspect_apify
    from scripts.maintain_spend_control import _safe_value
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        raw = inspect_apify(client, token, NEW['actor_build'], now)
    return _safe_value(raw, [token])


def check_observation(observation):
    expected = {'configured_build': NEW['actor_build'], 'build_pin_verified': True,
                'build_id': NEW['actor_build_id'], 'image_digest': NEW['actor_image_digest'],
                'input_schema_sha256': SCHEMA_SHA, 'account_fingerprint': ACCOUNT}
    for key, value in expected.items():
        if observation.get(key) != value:
            raise RuntimeError('Live Apify build differs from the reviewed November 6 contract: ' + key)
    if digest(observation.get('pricing')) != PRICING_SHA:
        raise RuntimeError('Live Apify pricing differs from the approved policy')


def successor(old):
    apify = old['providers']['apify']
    for key, value in OLD.items():
        if apify.get(key) != value:
            raise RuntimeError('Installed Apify pin differs from reviewed source record: ' + key)
    if apify.get('actor_schema_sha256') != SCHEMA_SHA or apify.get('actor_pricing_sha256') != PRICING_SHA:
        raise RuntimeError('Installed Apify contract differs from reviewed source record')
    new = copy.deepcopy(old)
    new['providers']['apify'].update(NEW)
    return new


def repair(store, old_sha, *, observation, apply=False, now=None):
    now = now or datetime.now(timezone.utc)
    old = store.read('policy_' + old_sha).to_dict()
    validate_policy(old, old_sha, now)
    check_observation(observation)
    new = successor(old)
    new_sha = digest(new)
    validate_policy(new, new_sha, now)
    month = now.strftime('%Y-%m')
    journal = store.client.collection('spend_control').document(RECEIPT_ID)
    prior = journal.get()
    if prior.exists:
        receipt = prior.to_dict()
        if receipt.get('new_policy_sha256') != new_sha or receipt.get('old_policy_sha256') != old_sha:
            raise RuntimeError('Repair receipt identity mismatch')
        if (store.read('policy_' + new_sha).to_dict() != new
                or store.read(month).to_dict().get('automatic_policy_sha256') != new_sha):
            raise RuntimeError('Repair receipt requires reconciliation')
        return new_sha
    ledger_snapshot = store.read(month)
    ledger = ledger_snapshot.to_dict()
    if (ledger.get('automatic_policy_sha256') != old_sha or ledger.get('reconciliation_required')
            or ledger.get('closed_for_rollover')):
        raise RuntimeError('Ledger changed or frozen; repair must be replanned')
    after = copy.deepcopy(ledger)
    after['automatic_policy_sha256'] = new_sha
    target = store.read('policy_' + new_sha)
    if target.exists:
        raise RuntimeError('Unexpected successor without repair receipt')
    receipt = {'schema_version': 1, 'old_policy_sha256': old_sha, 'new_policy_sha256': new_sha,
               'month': month, 'observed_at': now.isoformat(), 'observation': observation,
               'before_sha256': digest(ledger), 'after_sha256': digest(after),
               'change': 'providers.apify actor_build, actor_build_id, actor_image_digest only; '
                         'pricing, schema, financial terms and liabilities unchanged'}
    if apply:
        try:
            _commit_documents(store.client, [('spend_control', 'policy_' + new_sha, new, target),
                                             ('spend_control', month, after, ledger_snapshot),
                                             ('spend_control', RECEIPT_ID, receipt, prior)])
        except PublicationOutcomeUnknown:
            pass
        if (journal.get().to_dict() != receipt or store.read(month).to_dict() != after
                or store.read('policy_' + new_sha).to_dict() != new):
            raise RuntimeError('Repair outcome unresolved; read before any further action')
    return new_sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore
    if os.environ['SPEND_POLICY_SHA256'] != EXPECTED_OLD_SHA:
        raise SystemExit('Only the reviewed November 6 source policy may be repaired')
    now = datetime.now(timezone.utc)
    observation = observe(os.environ['APIFY_TOKEN'], now)
    sha = repair(PrivateFirestoreStore(), EXPECTED_OLD_SHA, observation=observation, apply=False, now=now)
    if sha != EXPECTED_NEW_SHA:
        raise SystemExit('Computed successor differs from the reviewed workflow pin; nothing written')
    if args.apply:
        sha = repair(PrivateFirestoreStore(), EXPECTED_OLD_SHA, observation=observation, apply=True, now=now)
    print(json.dumps({'status': 'applied_or_reconciled' if args.apply else 'planned', 'policy_sha256': sha}))
