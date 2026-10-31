"""One-time accounting conversion is conditional and cannot erase old holds."""
import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import install_automatic_spend as install
from src.zeitgeist import automatic_spend as auto
from src.state_store import PublicationOutcomeUnknown, PublicationConflict
from tests.test_automatic_spend import fixtures,hold,NOW


def inputs():
    ledger,policy,obs=fixtures()
    ledger.update(schema_version=2,credential_binding_required=True,budget_scope='new_requests_after_cutoff')
    ledger.pop('automatic_accounting_version');ledger.pop('automatic_policy_sha256')
    ledger['reservations']['old-unknown']=hold('gemini','.05')
    source={'target':dict(auto.TARGET,document_id='2026-10'),'data':ledger,'update_time':'2026-10-31T11:59:00Z'}
    return source,policy,obs


def plan(journal=Path('/approved/private/install-journal.json')):
    return install.prepare_install(*inputs(),now=NOW,journal_path=journal)


class Store:
    def __init__(self,prepared):
        self.docs={prepared['target']['document_id']:copy.deepcopy(prepared['before'])}
        self.versions={key:prepared['before_update_time'] for key in self.docs}
        self.calls=0;self.mode=None
    def read(self,key):
        value=copy.deepcopy(self.docs.get(key))
        return SimpleNamespace(exists=value is not None,to_dict=lambda:value,update_time=self.versions.get(key))
    def commit(self,writes):
        self.calls+=1
        if self.mode=='unknown_before':raise PublicationOutcomeUnknown('synthetic')
        for key,value,snapshot in writes:
            if (key in self.docs)!=snapshot.exists or self.versions.get(key)!=snapshot.update_time:
                raise PublicationConflict('changed')
        for key,value,snapshot in writes:
            self.docs[key]=copy.deepcopy(value);self.versions[key]='2026-10-31T12:00:01Z'
        if self.mode=='unknown_after':raise PublicationOutcomeUnknown('synthetic')


def apply(prepared,store,journal):
    return install.apply_install(prepared,store,expected_sha=prepared['plan_sha256'],
        confirmation='git-readme-features:(default):spend_control:2026-10',journal=journal,now=NOW)


def test_prepare_preserves_every_old_hold_epoch_cash_and_manual_evidence():
    source,policy,obs=inputs();before=copy.deepcopy(source)
    prepared=install.prepare_install(source,policy,obs,now=NOW,journal_path=Path('/approved/private/install-journal.json'))
    assert source==before
    assert prepared['after']['reservations']==source['data']['reservations']
    assert prepared['after']['budget_window_start']==source['data']['budget_window_start']
    assert prepared['after']['cash_ceiling_usd']=='2.50'
    assert prepared['after']['schema_version']==3
    assert prepared['after']['providers']['bfl']['policy_verified_at']==source['data']['providers']['bfl']['policy_verified_at']


def test_extra_policy_account_fields_cannot_rewrite_manual_evidence_or_mint_credit():
    source,policy,obs=inputs()
    policy['providers']['bfl']['account'].update(credit_baseline_usd='999',verified_at='2026-12-01T00:00:00Z')
    prepared=install.prepare_install(source,policy,obs,now=NOW,journal_path=Path('/approved/private/install-journal.json'))
    assert prepared['after']['providers']['bfl']['verified_at']==source['data']['providers']['bfl']['verified_at']
    assert auto.money(prepared['after']['providers']['bfl']['credit_baseline_usd'])<=auto.money(source['data']['providers']['bfl']['credit_baseline_usd'])


def test_reviewed_cash_ceiling_increase_is_bound_without_resetting_prior_holds():
    from tests.test_automatic_spend import with_approved_reload
    source,policy,obs=inputs();policy=with_approved_reload(policy)
    prepared=install.prepare_install(source,policy,obs,now=NOW,journal_path=Path('/approved/private/journal.json'))
    assert prepared['before']['cash_ceiling_usd']=='2.50'
    assert prepared['after']['cash_ceiling_usd']=='10.00'
    assert prepared['after']['reservations']==prepared['before']['reservations']
    assert prepared['after']['credit_only_providers']==['apify','bfl']


@pytest.mark.parametrize('change',['target','identity','cash','frozen','already_automatic'])
def test_prepare_rejects_changed_funding_or_unapproved_epoch(change):
    source,policy,obs=inputs()
    if change=='target':source['target']['project_id']='other'
    if change=='identity':policy['providers']['bfl']['account']['credential_fingerprint']='d'*16
    if change=='cash':source['data']['cash_ceiling_usd']='3'
    if change=='frozen':source['data']['closed_for_rollover']=True
    if change=='already_automatic':source['data']['schema_version']=3
    with pytest.raises((ValueError,auto.SpendBlocked)):
        install.prepare_install(source,policy,obs,now=NOW,journal_path=Path('/approved/private/install-journal.json'))


def test_apply_one_batch_then_exact_readback_and_no_redispatch(tmp_path,monkeypatch):
    monkeypatch.setattr(install,'_durable_location',lambda path:Path(path))
    journal=tmp_path/'journal.json';prepared=plan(journal);store=Store(prepared)
    assert apply(prepared,store,journal)['status']=='known_committed'
    assert store.calls==1
    assert store.docs['2026-10']['reservations']==prepared['before']['reservations']
    with pytest.raises((ValueError,FileExistsError)):apply(prepared,store,journal)
    assert store.calls==1


@pytest.mark.parametrize('mode',['unknown_before','unknown_after'])
def test_unknown_ack_readback_only_and_exclusive_journal_prevents_blind_repeat(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(install,'_durable_location',lambda path:Path(path))
    journal=tmp_path/'journal.json';prepared=plan(journal);store=Store(prepared);store.mode=mode
    if mode=='unknown_after':assert apply(prepared,store,journal)['status']=='known_committed'
    else:
        with pytest.raises(ValueError):apply(prepared,store,journal)
    with pytest.raises((ValueError,FileExistsError)):apply(prepared,store,journal)
    assert store.calls==1 and journal.exists()


def test_apply_rejects_changed_revision_or_digest_before_dispatch(tmp_path,monkeypatch):
    monkeypatch.setattr(install,'_durable_location',lambda path:Path(path))
    prepared=plan(tmp_path/'journal.json');store=Store(prepared);store.versions['2026-10']='changed'
    with pytest.raises(ValueError):apply(prepared,store,tmp_path/'journal.json')
    assert store.calls==0
    with pytest.raises(PermissionError):install.validate_plan(prepared,'0'*64,
        'git-readme-features:(default):spend_control:2026-10',tmp_path/'journal.json')


def test_fractional_fresh_observation_is_not_rejected_as_an_integer_range(tmp_path,monkeypatch):
    monkeypatch.setattr(install,'_durable_location',lambda path:Path(path))
    prepared=plan(tmp_path/'journal.json');store=Store(prepared)
    result=install.apply_install(prepared,store,expected_sha=prepared['plan_sha256'],
        confirmation='git-readme-features:(default):spend_control:2026-10',journal=tmp_path/'journal.json',
        now=NOW+timedelta(seconds=12,microseconds=345))
    assert result['status']=='known_committed'


def test_unknown_install_cannot_be_redispatched_with_an_alternate_journal(tmp_path,monkeypatch):
    monkeypatch.setattr(install,'_durable_location',lambda path:Path(path))
    journal=tmp_path/'approved.json';prepared=plan(journal);store=Store(prepared);store.mode='unknown_before'
    with pytest.raises(ValueError):apply(prepared,store,journal)
    with pytest.raises(PermissionError):apply(prepared,store,tmp_path/'alternate.json')
    assert store.calls==1


def test_explicit_reconciliation_updates_only_the_bound_journal_without_dispatch(tmp_path,monkeypatch):
    monkeypatch.setattr(install,'_durable_location',lambda path:Path(path))
    journal=tmp_path/'approved.json';prepared=plan(journal);store=Store(prepared);store.mode='unknown_before'
    with pytest.raises(ValueError):apply(prepared,store,journal)
    # Model a previously uncertain atomic commit becoming visible later.
    store.docs.update({'2026-10':copy.deepcopy(prepared['after']),
        'policy_'+prepared['policy_sha256']:copy.deepcopy(prepared['policy'])})
    result=install.reconcile_install(prepared,store,expected_sha=prepared['plan_sha256'],
        confirmation='git-readme-features:(default):spend_control:2026-10',journal=journal)
    import json
    assert result['status']=='known_committed' and json.loads(journal.read_text())['status']=='complete'
    assert store.calls==1
