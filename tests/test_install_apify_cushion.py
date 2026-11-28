"""The Apify price-cushion successor (SIY-291): only the approved price terms are added, the pointer moves, it rolls back."""
import copy
from datetime import timedelta
import json

import pytest

from scripts import install_spend_successor_20261128_cushion as install_module
from scripts import rollback_spend_successor_20261128_cushion as rollback_module
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.spend_successor import (
    APIFY_APPROVED_PRICING_TERMS, APIFY_CUSHION_BASE_POLICY_SHA256, build_apify_price_cushion_successor, field_diff)
from tests.test_automatic_spend import NOW, fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from tests.test_spend_successor import hold


def base_policy():
    policy = fixtures()[1]
    policy['providers']['apify']['actor_pricing_sha256'] = auto.digest(APIFY_APPROVED_PRICING_TERMS)
    return policy


def setup(monkeypatch, *, reservations=None):
    ledger, _ = fixtures()[:2]
    policy = base_policy()
    old_sha = auto.digest(policy)
    ledger.update(month=NOW.strftime('%Y-%m'), automatic_policy_sha256=old_sha,
                  reservations=reservations if reservations is not None else {'old': hold(NOW - timedelta(hours=2))})
    wire = PrivateWire(monkeypatch, {'policy_' + old_sha: policy, ledger['month']: ledger})
    return wire, old_sha, ledger, policy


def test_the_approved_terms_are_the_live_pin():
    assert auto.digest(APIFY_APPROVED_PRICING_TERMS) == '12cc41388ad55dda87685f711086d048eda188ea00ec6e9d699c55adc9e86d0d'
    assert APIFY_CUSHION_BASE_POLICY_SHA256 == 'eeecd6d8b21b65c0ddcf4b0e66a6c8be2e1fd053f83f1fbbfb318875dfa9d483'


def test_the_successor_adds_only_the_terms():
    policy = base_policy()
    new = build_apify_price_cushion_successor(policy)
    assert field_diff(policy, new) == [
        '+ providers.apify.actor_pricing_terms = ' + field_diff({}, {'x': APIFY_APPROVED_PRICING_TERMS})[0].split(' = ', 1)[1]]
    assert new['providers']['apify']['actor_pricing_terms'] == APIFY_APPROVED_PRICING_TERMS
    auto.validate_policy(new, auto.digest(new), NOW)


def test_the_successor_refuses_terms_off_the_pin():
    with pytest.raises(RuntimeError):
        build_apify_price_cushion_successor(fixtures()[1])


def test_the_successor_refuses_a_policy_with_terms():
    with pytest.raises(RuntimeError):
        build_apify_price_cushion_successor(build_apify_price_cushion_successor(base_policy()))


def test_the_receipts_extend_the_policy_chain():
    assert runtime.APIFY_CUSHION_RECEIPT_ID == 'spend_successor_20261128_cushion'
    assert runtime.APIFY_CUSHION_ROLLBACK_RECEIPT_ID == 'spend_rollback_20261128_cushion'
    assert runtime.POLICY_REPAIR_RECEIPTS[-4:] == ('spend_successor_20261128', 'spend_rollback_20261128',
                                                   'spend_successor_20261128_cushion', 'spend_rollback_20261128_cushion')


def test_plan_writes_nothing_and_names_the_new_policy(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    report = install_module.install(wire.store, old_sha=old_sha, now=NOW)
    assert report['mode'] == 'plan' and wire.posts == []
    assert report['new_policy_sha256'] == auto.digest(build_apify_price_cushion_successor(policy))
    assert all('automatic_policy_sha256' in line for line in report['ledger_diff'])
    assert report['reservations_unchanged'] is True


def test_apply_requires_the_planned_digest(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True, expected_new_sha='0' * 64)
    assert wire.posts == []


def applied(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    new_sha = auto.digest(build_apify_price_cushion_successor(policy))
    report = install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True, expected_new_sha=new_sha)
    return wire, old_sha, new_sha, ledger, report


def test_apply_moves_only_the_pointer_and_keeps_every_hold(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    assert report['mode'] == 'applied' and report['reservations_unchanged'] is True
    after = wire.docs[ledger['month']]
    assert after['automatic_policy_sha256'] == new_sha
    assert {k: v for k, v in after.items() if k != 'automatic_policy_sha256'} == {
        k: v for k, v in ledger.items() if k != 'automatic_policy_sha256'}
    assert wire.docs['policy_' + new_sha]['providers']['apify']['actor_pricing_terms'] == APIFY_APPROVED_PRICING_TERMS
    assert wire.docs[runtime.APIFY_CUSHION_RECEIPT_ID]['new_policy_sha256'] == new_sha


def test_a_repeated_apply_reports_already_applied_without_writing(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    posts = len(wire.posts)
    again = install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True, expected_new_sha=new_sha)
    assert again['mode'] == 'already_applied' and len(wire.posts) == posts


def test_a_hold_that_may_still_be_in_flight_blocks_the_install(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch, reservations={'new': hold(NOW - timedelta(seconds=30))})
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=NOW)


def test_rollback_moves_the_pointer_back(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    plan = rollback_module.rollback(wire.store, now=NOW, required_old_sha=old_sha)
    assert plan['mode'] == 'plan' and plan['to_policy_sha256'] == old_sha
    done = rollback_module.rollback(wire.store, now=NOW, required_old_sha=old_sha, apply=True,
                                    expected_plan_sha=plan['plan_sha256'])
    assert done['mode'] == 'applied'
    assert wire.docs[ledger['month']]['automatic_policy_sha256'] == old_sha
    assert wire.docs[runtime.APIFY_CUSHION_ROLLBACK_RECEIPT_ID]['install_receipt_id'] == runtime.APIFY_CUSHION_RECEIPT_ID


def test_rollback_without_an_install_receipt_is_refused(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        rollback_module.rollback(wire.store, now=NOW)


@pytest.mark.parametrize('module', [install_module, rollback_module])
def test_main_refuses_another_installed_policy(monkeypatch, capsys, module):
    reads = []
    monkeypatch.setattr(module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', '25f5685bfd2b53eac7af18b521765ca558e8cda421ae0d3b254c81e4518366b8')
    code = module.main([])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and reads == [] and len(out) == 1 and 'error' in json.loads(out[0])


def test_after_the_install_a_refresh_tolerates_an_in_band_rise():
    ledger, policy, observation = fixtures()
    successor = build_apify_price_cushion_successor(base_policy())
    ledger['automatic_policy_sha256'] = auto.digest(successor)
    observed = copy.deepcopy(APIFY_APPROVED_PRICING_TERMS)
    observed['event_prices_usd']['post']['FREE'] = '0.002'
    observation['apify']['pricing'] = observed
    account = auto.prepare_refresh(ledger, successor, observation, expected_policy_sha=auto.digest(successor),
                                   now=NOW)['after']['providers']['apify']
    assert account['price_band_audit']['above_approval'][0]['observed'] == '0.002'
