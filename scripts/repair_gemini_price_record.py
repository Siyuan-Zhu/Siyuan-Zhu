#!/usr/bin/env python3
"""Exact, reviewed Gemini price-record re-pins; no economic policy change.

November 3: source-format change. November 13: Google added 2027 rates while the
current rates stayed as approved (owner-approved re-pin, 2026-11-13). Every
transition requires each approved operation's current published rate to equal
the policy's approved rate.

Requires a fresh official observation matching the reviewed contract. Applies an
immutable successor policy and ledger pointer atomically with a durable receipt.
A repeated/uncertain dispatch reconciles by reading the receipt, never replaying.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from src.zeitgeist.automatic_spend import digest, validate_policy
from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore
from src.zeitgeist.gemini_tariff import observe_gemini_tariff
from src.state_store import _commit_documents, PublicationOutcomeUnknown

from decimal import Decimal

OLD_TARIFF='5059b91574008f8992397b046c06f12f81ea63a1fca8df14c0a5ebf2a2bd0992'
NEW_TARIFF='35df15420dc97e3064ee2a669a1b89dff49d142a79a09bd20375a41d923c63f2'
NOV13_TARIFF='dda980d35a559250b8b0f6c760ab3538268fec9ca13da6433f8b94eb7abcd6d3'
# Installed tariff -> (reviewed successor, durable receipt id).
TRANSITIONS={OLD_TARIFF:(NEW_TARIFF,'price_repair_20261103'),
             NEW_TARIFF:(NOV13_TARIFF,'price_repair_20261113')}
RATE_KEYS=('input_token_usd_per_million','output_token_usd_per_million')


def check_current_rates(policy, contract):
    """Every approved operation's current published rate must equal the approved rate."""
    published=(contract or {}).get('operations',{})
    account=policy['providers']['gemini']['account']
    for operation in sorted(set(account['request_limits'])|set(account['operation_caps_usd'])):
        limits=account['request_limits'].get(operation,{})
        observed=published.get(operation)
        if not observed:
            raise RuntimeError('No published current rate for approved operation: '+operation)
        for key in RATE_KEYS:
            published_rate=observed.get(key)
            # Google publishes input-only operations (embeddings) with no output rate: that is zero.
            if published_rate is None and key=='output_token_usd_per_million' and observed.get('billing_basis')=='input_tokens':
                published_rate='0'
            if key in limits and (published_rate is None or Decimal(str(published_rate))!=Decimal(str(limits[key]))):
                raise RuntimeError('Published current rate differs from the approved rate: '+operation+' '+key)


def repair(store, old_sha, *, apply=False, now=None):
    now=now or datetime.now(timezone.utc)
    old=store.read('policy_'+old_sha).to_dict()
    validate_policy(old,old_sha,now)
    if old['providers']['gemini']['price_snapshot_sha256'] not in TRANSITIONS:
        raise RuntimeError('Installed tariff differs from reviewed source record')
    target,receipt_id=TRANSITIONS[old['providers']['gemini']['price_snapshot_sha256']]
    with httpx.Client(follow_redirects=False) as client:
        observation=observe_gemini_tariff(client,now=now)
    check_current_rates(old,observation.get('contract'))
    if observation['sha256'] != target:
        raise RuntimeError('Live price contract differs from the reviewed contract')
    new=copy.deepcopy(old)
    new['providers']['gemini']['price_snapshot_sha256']=target
    new_sha=digest(new);validate_policy(new,new_sha,now)
    month=now.strftime('%Y-%m')
    journal=store.client.collection('spend_control').document(receipt_id)
    prior=journal.get()
    if prior.exists:
        receipt=prior.to_dict()
        if receipt.get('new_policy_sha256') != new_sha or receipt.get('old_policy_sha256') != old_sha:
            raise RuntimeError('Repair receipt identity mismatch')
        if store.read('policy_'+new_sha).to_dict()!=new or store.read(month).to_dict().get('automatic_policy_sha256')!=new_sha:
            raise RuntimeError('Repair receipt requires reconciliation')
        return new_sha
    ledger_snapshot=store.read(month);ledger=ledger_snapshot.to_dict()
    if ledger.get('automatic_policy_sha256')!=old_sha or ledger.get('reconciliation_required') or ledger.get('closed_for_rollover'):
        raise RuntimeError('Ledger changed or frozen; repair must be replanned')
    after=copy.deepcopy(ledger);after['automatic_policy_sha256']=new_sha
    successor=store.read('policy_'+new_sha)
    if successor.exists: raise RuntimeError('Unexpected successor without repair receipt')
    receipt={'schema_version':1,'old_policy_sha256':old_sha,'new_policy_sha256':new_sha,
        'month':month,'observed_at':now.isoformat(),'observation':observation,
        'before_sha256':digest(ledger),'after_sha256':digest(after),
        'change':'providers.gemini.price_snapshot_sha256 only; financial terms and liabilities unchanged'}
    if apply:
        try:
            _commit_documents(store.client,[('spend_control','policy_'+new_sha,new,successor),
                ('spend_control',month,after,ledger_snapshot),
                ('spend_control',receipt_id,receipt,prior)])
        except PublicationOutcomeUnknown:
            pass
        if journal.get().to_dict()!=receipt or store.read(month).to_dict()!=after or store.read('policy_'+new_sha).to_dict()!=new:
            raise RuntimeError('Repair outcome unresolved; read before any further action')
    return new_sha


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    sha=repair(PrivateFirestoreStore(),os.environ['SPEND_POLICY_SHA256'],apply=args.apply)
    print(json.dumps({'status':'applied_or_reconciled' if args.apply else 'planned','policy_sha256':sha}))
    if args.apply and os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as out:out.write('SPEND_POLICY_SHA256='+sha+'\n')
