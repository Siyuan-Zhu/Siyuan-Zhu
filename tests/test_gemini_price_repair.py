import copy
from datetime import datetime,timezone
import pytest
from scripts import repair_gemini_price_record as repair
from tests.test_automatic_spend import fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from src.zeitgeist.automatic_spend import digest

NOW=datetime(2026,11,3,12,tzinfo=timezone.utc)


def setup(monkeypatch):
    ledger,policy,_=fixtures(NOW)
    policy['providers']['gemini']['price_snapshot_sha256']=repair.OLD_TARIFF
    sha=digest(policy);ledger.update(month='2026-11',automatic_policy_sha256=sha)
    wire=PrivateWire(monkeypatch,{'2026-11':ledger,'policy_'+sha:policy})
    monkeypatch.setattr(repair,'observe_gemini_tariff',lambda *a,**k:{'sha256':repair.NEW_TARIFF})
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
