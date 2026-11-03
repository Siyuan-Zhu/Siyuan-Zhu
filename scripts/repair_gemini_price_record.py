#!/usr/bin/env python3
"""One-time, exact November 3 source-format repair; no economic policy change.

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

OLD_TARIFF='5059b91574008f8992397b046c06f12f81ea63a1fca8df14c0a5ebf2a2bd0992'
NEW_TARIFF='35df15420dc97e3064ee2a669a1b89dff49d142a79a09bd20375a41d923c63f2'


def repair(store, old_sha, *, apply=False, now=None):
    now=now or datetime.now(timezone.utc)
    old=store.read('policy_'+old_sha).to_dict()
    validate_policy(old,old_sha,now)
    if old['providers']['gemini']['price_snapshot_sha256'] != OLD_TARIFF:
        raise RuntimeError('Installed tariff differs from reviewed source record')
    with httpx.Client(follow_redirects=False) as client:
        observation=observe_gemini_tariff(client,now=now)
    if observation['sha256'] != NEW_TARIFF:
        raise RuntimeError('Live price contract differs from reviewed November 3 contract')
    new=copy.deepcopy(old)
    new['providers']['gemini']['price_snapshot_sha256']=NEW_TARIFF
    new_sha=digest(new);validate_policy(new,new_sha,now)
    month=now.strftime('%Y-%m')
    journal=store.client.collection('spend_control').document('price_repair_20261103')
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
                ('spend_control','price_repair_20261103',receipt,prior)])
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
