import copy
from datetime import datetime,timezone
import pytest
from scripts import repair_gemini_price_record as repair
from tests.test_automatic_spend import fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from src.zeitgeist.automatic_spend import digest

NOW=datetime(2026,11,3,12,tzinfo=timezone.utc)
NOW13=datetime(2026,11,13,17,tzinfo=timezone.utc)
# The fixture policy approves gemini-2.5-flash at 0.30 USD in and 2.50 USD out per million tokens.
FIXTURE_RATES={'operations':{'generate_content:gemini-2.5-flash':{
    'input_token_usd_per_million':'0.30','output_token_usd_per_million':'2.5',
    'future_input_token_usd_per_million':'0.60','future_output_token_usd_per_million':'5'}}}


def setup(monkeypatch):
    ledger,policy,_=fixtures(NOW)
    policy['providers']['gemini']['price_snapshot_sha256']=repair.OLD_TARIFF
    sha=digest(policy);ledger.update(month='2026-11',automatic_policy_sha256=sha)
    wire=PrivateWire(monkeypatch,{'2026-11':ledger,'policy_'+sha:policy})
    monkeypatch.setattr(repair,'observe_gemini_tariff',lambda *a,**k:{'sha256':repair.NEW_TARIFF,'contract':FIXTURE_RATES})
    return wire,sha,ledger,policy


@pytest.mark.parametrize('unknown',[False,True])
def test_conditional_repair_preserves_all_money_and_reconciles_repeated_dispatch(monkeypatch,unknown):
    wire,sha,ledger,policy=setup(monkeypatch)
    if unknown: wire.unknown_post=1
    newsha=repair.repair(wire.store,sha,apply=True,now=NOW)
    expected=copy.deepcopy(ledger);expected['automatic_policy_sha256']=newsha
    assert wire.docs['2026-11']==expected
    successor=copy.deepcopy(policy);successor['providers']['gemini']['price_snapshot_sha256']=repair.NEW_TARIFF
    assert wire.docs['policy_'+newsha]==successor
    assert repair.repair(wire.store,sha,apply=True,now=NOW)==newsha
    assert len(wire.posts)==1


def test_unreviewed_live_price_blocks_without_any_write(monkeypatch):
    wire,sha,_,_=setup(monkeypatch)
    monkeypatch.setattr(repair,'observe_gemini_tariff',lambda *a,**k:{'sha256':'unexpected'})
    with pytest.raises(RuntimeError):repair.repair(wire.store,sha,apply=True,now=NOW)
    assert not wire.posts


def setup_nov13(monkeypatch, contract=FIXTURE_RATES):
    ledger,policy,_=fixtures(NOW13)
    policy['providers']['gemini']['price_snapshot_sha256']=repair.NEW_TARIFF
    sha=digest(policy);ledger.update(month='2026-11',automatic_policy_sha256=sha)
    wire=PrivateWire(monkeypatch,{'2026-11':ledger,'policy_'+sha:policy})
    monkeypatch.setattr(repair,'observe_gemini_tariff',lambda *a,**k:{'sha256':repair.NOV13_TARIFF,'contract':contract})
    return wire,sha,ledger,policy


def test_november_13_future_rate_announcement_repins_only_the_tariff_once(monkeypatch):
    """Google added 2027 rates; current rates equal the approved ones, so only the tariff hash moves."""
    wire,sha,ledger,policy=setup_nov13(monkeypatch)
    newsha=repair.repair(wire.store,sha,apply=True,now=NOW13)
    successor=copy.deepcopy(policy);successor['providers']['gemini']['price_snapshot_sha256']=repair.NOV13_TARIFF
    assert wire.docs['policy_'+newsha]==successor
    assert wire.docs['2026-11']['automatic_policy_sha256']==newsha
    assert wire.docs['price_repair_20261113']['old_policy_sha256']==sha
    assert repair.repair(wire.store,sha,apply=True,now=NOW13)==newsha
    assert len(wire.posts)==1


@pytest.mark.parametrize('contract',[
    {'operations':{'generate_content:gemini-2.5-flash':{'input_token_usd_per_million':'0.31','output_token_usd_per_million':'2.5'}}},
    {'operations':{'generate_content:gemini-2.5-flash':{'input_token_usd_per_million':'0.30','output_token_usd_per_million':'2.6'}}},
    {'operations':{}}, {}])
def test_any_change_to_an_approved_current_rate_blocks_the_repin_without_writes(monkeypatch,contract):
    wire,sha,_,_=setup_nov13(monkeypatch,contract)
    with pytest.raises(RuntimeError,match='rate'):repair.repair(wire.store,sha,apply=True,now=NOW13)
    assert not wire.posts


def test_a_budgeted_operation_missing_from_the_published_record_blocks_the_repin(monkeypatch):
    ledger,policy,_=fixtures(NOW13)
    policy['providers']['gemini']['price_snapshot_sha256']=repair.NEW_TARIFF
    policy['providers']['gemini']['account']['operation_caps_usd']['embed_content:unpublished-model']='.01'
    sha=digest(policy);ledger.update(month='2026-11',automatic_policy_sha256=sha)
    wire=PrivateWire(monkeypatch,{'2026-11':ledger,'policy_'+sha:policy})
    monkeypatch.setattr(repair,'observe_gemini_tariff',lambda *a,**k:{'sha256':repair.NOV13_TARIFF,'contract':FIXTURE_RATES})
    with pytest.raises(RuntimeError,match='rate'):repair.repair(wire.store,sha,apply=True,now=NOW13)
    assert not wire.posts


def embedding_policy(output_rate):
    ledger,policy,_=fixtures(NOW13)
    gemini=policy['providers']['gemini']
    gemini['price_snapshot_sha256']=repair.NEW_TARIFF
    gemini['account']['request_limits']['embed_content:gemini-embedding-001']={
        'input_token_usd_per_million':'.15','output_token_usd_per_million':output_rate}
    gemini['account']['operation_caps_usd']['embed_content:gemini-embedding-001']='.03'
    contract=copy.deepcopy(FIXTURE_RATES)
    contract['operations']['embed_content:gemini-embedding-001']={'billing_basis':'input_tokens','input_token_usd_per_million':'0.15'}
    return ledger,policy,contract


@pytest.mark.parametrize('output_rate,allowed',[('0',True),('0.10',False)])
def test_input_only_published_operation_matches_only_an_approved_zero_output_rate(monkeypatch,output_rate,allowed):
    ledger,policy,contract=embedding_policy(output_rate)
    if allowed:
        repair.check_current_rates(policy,contract)
    else:
        with pytest.raises(RuntimeError,match='rate'):repair.check_current_rates(policy,contract)
    del contract['operations']['embed_content:gemini-embedding-001']['billing_basis']
    with pytest.raises(RuntimeError,match='rate'):repair.check_current_rates(policy,contract)
