"""Prepare an offline policy installation; explicit apply is separately approved.

Only a digest-bound approved policy can change the cash ceiling. Never erases
reservations, buys credit or enables a writer. One atomic batch upgrades control.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore
from src.zeitgeist.spend_guard import BudgetGuard,money
from scripts.promote_release_state import _durable_location,_private_write


def prepare_install(snapshot,policy,observation,*,now,journal_path):
    identity=snapshot.get('target',{})
    month=now.astimezone(timezone.utc).strftime('%Y-%m')
    if identity!=dict(auto.TARGET,document_id=month) or not snapshot.get('update_time'):
        raise ValueError('Exact current canonical accounting snapshot required')
    BudgetGuard._timestamp(snapshot['update_time'])
    before=snapshot['data'];policy_sha=auto.digest(policy)
    auto.validate_policy(policy,policy_sha,now)
    if (before.get('schema_version')!=2 or before.get('credential_binding_required') is not True
        or before.get('budget_scope')!='new_requests_after_cutoff'
        or before.get('month')!=month or before.get('closed_for_rollover')
        or before.get('reconciliation_required') or money(before['cash_ceiling_usd'])!=money('2.50')):
        raise ValueError('Only the active approved legacy epoch can be upgraded')
    after=copy.deepcopy(before)
    if set(before['providers'])!=set(policy['providers']):
        raise ValueError('Provider set cannot change during installation')
    for provider,pinned in policy['providers'].items():
        account=after['providers'][provider]
        for key in ('account_id','plan_id','credential_fingerprint','credit_kind'):
            if account.get(key)!=pinned['account'].get(key):
                raise ValueError('Provider identity or funding kind changed')
        # Existing monetary credit can only shrink, including a newly approved
        # fixed prepaid tranche. No new credit is minted by schema conversion.
        account['credit_baseline_usd']=str(min(BudgetGuard._credit_baseline(account),money(pinned['credit_limit_usd'])))
        account.update({key:copy.deepcopy(pinned['account'][key]) for key in auto.STATIC})
    after.update(schema_version=3,automatic_accounting_version=1,automatic_policy_sha256=policy_sha,
        cash_ceiling_usd=str(money(policy['cash_ceiling_usd'])),
        credit_only_providers=copy.deepcopy(policy.get('credit_only_providers', [])))
    after=auto.prepare_refresh(after,policy,observation,expected_policy_sha=policy_sha,now=now)['after']
    if (after['reservations']!=before['reservations']
        or after['cash_spent_before_ledger_usd']!=before['cash_spent_before_ledger_usd']
        or after['budget_window_start']!=before['budget_window_start']
        or any(BudgetGuard._credit_baseline(after['providers'][name]) > BudgetGuard._credit_baseline(before['providers'][name])
               for name in before['providers'])):
        raise ValueError('Installation cannot release holds, grant a new cycle or reset the epoch')
    plan={'schema_version':1,'target':identity,'before':copy.deepcopy(before),
        'before_update_time':snapshot['update_time'],'before_sha256':auto.digest(before),
        'after':after,'policy':copy.deepcopy(policy),'policy_sha256':policy_sha,
        'observation_sha256':auto.digest(observation),
        'journal_path':str(_durable_location(journal_path)),
        'activation_scope':'private_control_only_no_publication_or_writer_activation'}
    plan['plan_sha256']=auto.digest(plan)
    return plan


def validate_plan(plan,expected_sha,confirmation,journal):
    actual=auto.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    target=plan.get('target',{})
    if actual!=expected_sha or plan.get('plan_sha256')!=actual:
        raise PermissionError('Exact reviewed installation plan required')
    if str(_durable_location(journal))!=plan.get('journal_path'):
        raise PermissionError('Installation journal differs from the approved plan')
    if (set(target)!=set(auto.TARGET)|{'document_id'} or any(target.get(k)!=v for k,v in auto.TARGET.items())
        or not re.fullmatch(r'\d{4}-\d{2}',target.get('document_id',''))
        or confirmation!='git-readme-features:(default):spend_control:'+target['document_id']):
        raise PermissionError('Exact reviewed private target confirmation required')
    if (auto.digest(plan['policy'])!=plan['policy_sha256'] or auto.digest(plan['before'])!=plan['before_sha256']
        or plan['after']['reservations']!=plan['before']['reservations']
        or plan['after']['automatic_policy_sha256']!=plan['policy_sha256']):
        raise ValueError('Installation policy or retained reservation binding changed')


def _read_install_result(plan,store):
    current=store.read(plan['target']['document_id'])
    policy=store.read('policy_'+plan['policy_sha256'])
    if (not current.exists or current.to_dict()!=plan['after'] or not policy.exists
        or policy.to_dict()!=plan['policy']):
        raise ValueError('Installation outcome unknown or state advanced; preserve evidence')
    return {'status':'known_committed','policy_sha256':plan['policy_sha256']}


def reconcile_install(plan,store,*,expected_sha,confirmation,journal):
    validate_plan(plan,expected_sha,confirmation,journal)
    path=_durable_location(journal)
    record=json.loads(path.read_text())
    if (record.get('plan_sha256')!=expected_sha or record.get('policy_sha256')!=plan['policy_sha256']
        or record.get('status') not in ('partial_or_unknown','complete')):
        raise ValueError('Installation journal ownership differs from the plan')
    result=_read_install_result(plan,store)
    record.update(status='complete');_private_write(path,record)
    return result


def apply_install(plan,store,*,expected_sha,confirmation,journal,now):
    validate_plan(plan,expected_sha,confirmation,journal)
    path=_durable_location(journal)
    if path.exists():
        raise FileExistsError('Installation journal already exists; reconcile without dispatch')
    auto.validate_policy(plan['policy'],plan['policy_sha256'],now)
    if plan['target']['document_id']!=now.astimezone(timezone.utc).strftime('%Y-%m'):
        raise ValueError('Installation plan expired at the UTC month boundary')
    # A reviewed fresh observation is required at activation; it cannot become a
    # reusable bootstrap after the private provider state has changed.
    observed=min(BudgetGuard._timestamp(a['automatic_observed_at']) for a in plan['after']['providers'].values())
    if not 0 <= (now-observed).total_seconds() <= 300:
        raise ValueError('Fresh verified installation observation required')
    ledger=store.read(plan['target']['document_id']);policy=store.read('policy_'+plan['policy_sha256'])
    if (not ledger.exists or ledger.to_dict()!=plan['before']
        or str(ledger.update_time)!=plan['before_update_time'] or policy.exists):
        raise ValueError('Installation source changed or policy already exists; do not overwrite')
    record={'status':'partial_or_unknown','plan_sha256':expected_sha,'policy_sha256':plan['policy_sha256']}
    _private_write(path,record,exclusive=True) # fsync before the single dispatch.
    try:
        store.commit([('policy_'+plan['policy_sha256'],plan['policy'],policy),
                      (plan['target']['document_id'],plan['after'],ledger)])
        result=_read_install_result(plan,store)
    except Exception:
        # An uncertain write is never repeated. Exact readback may prove that
        # both atomic writes committed; all other outcomes retain the journal.
        result=_read_install_result(plan,store)
    record.update(status='complete');_private_write(path,record)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    for name in ('snapshot','policy','observation','output'):prepare.add_argument('--'+name,type=Path,required=True)
    prepare.add_argument('--journal',type=Path,required=True)
    for name in ('apply','reconcile'):
        action=sub.add_parser(name)
        action.add_argument('--plan',type=Path,required=True)
        action.add_argument('--expected-plan-sha',required=True)
        action.add_argument('--confirm-target',required=True)
        action.add_argument('--journal',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            plan=prepare_install(*[json.loads(getattr(args,k).read_text()) for k in ('snapshot','policy','observation')],now=datetime.now(timezone.utc),journal_path=args.journal)
            _private_write(_durable_location(args.output),plan,exclusive=True)
            report={'status':'prepared_no_writes','plan_sha256':plan['plan_sha256'],'policy_sha256':plan['policy_sha256']}
        else:
            plan=json.loads(args.plan.read_text());validate_plan(plan,args.expected_plan_sha,args.confirm_target,args.journal)
            if args.command=='reconcile':report=reconcile_install(plan,PrivateFirestoreStore(),
                expected_sha=args.expected_plan_sha,confirmation=args.confirm_target,journal=args.journal)
            else:report=apply_install(plan,PrivateFirestoreStore(),expected_sha=args.expected_plan_sha,
                confirmation=args.confirm_target,journal=args.journal,now=datetime.now(timezone.utc))
        print(json.dumps(report,sort_keys=True));return 0
    except BaseException as error:
        print(json.dumps({'status':'blocked','error_type':type(error).__name__}));return 2


if __name__=='__main__':raise SystemExit(main())
