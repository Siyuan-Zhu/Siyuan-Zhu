"""The Apify newest-safe-build successor: one policy field changes, the ledger pointer moves, and it rolls back."""
from datetime import timedelta
import pytest

from scripts import install_spend_successor_20261128 as install_module
from scripts import rollback_spend_successor_20261128 as rollback_module
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.spend_successor import build_apify_newest_safe_successor, field_diff
from tests.test_automatic_spend import NOW, fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from tests.test_spend_successor import hold


def setup(monkeypatch, *, reservations=None):
    ledger, policy = fixtures()[:2]
    old_sha = auto.digest(policy)
    ledger.update(month=NOW.strftime('%Y-%m'), automatic_policy_sha256=old_sha,
                  reservations=reservations if reservations is not None else {'old': hold(NOW - timedelta(hours=2))})
    wire = PrivateWire(monkeypatch, {'policy_' + old_sha: policy, ledger['month']: ledger})
    return wire, old_sha, ledger, policy


def test_the_successor_changes_only_the_build_mode_and_the_digest_pin():
    policy = fixtures()[1]
    new = build_apify_newest_safe_successor(policy)
    assert new['providers']['apify']['actor_build_mode'] == 'newest_safe'
    assert new['providers']['apify']['actor_image_digest'] is None
    rest = lambda p: {**p, 'providers': {**p['providers'], 'apify': {
        k: v for k, v in p['providers']['apify'].items() if k not in ('actor_build_mode', 'actor_image_digest')}}}
    assert rest(new) == rest(policy)
    auto.validate_policy(new, auto.digest(new), NOW)


def test_a_policy_already_in_newest_safe_mode_is_refused():
    policy = build_apify_newest_safe_successor(fixtures()[1])
    with pytest.raises(RuntimeError):
        build_apify_newest_safe_successor(policy)


def test_the_receipts_extend_the_policy_chain():
    assert runtime.APIFY_SUCCESSOR_RECEIPT_ID == 'spend_successor_20261128'
    assert runtime.APIFY_ROLLBACK_RECEIPT_ID == 'spend_rollback_20261128'
    assert runtime.POLICY_REPAIR_RECEIPTS[-6:-4] == ('spend_successor_20261128', 'spend_rollback_20261128')


def test_plan_writes_nothing_and_names_the_new_policy(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    report = install_module.install(wire.store, old_sha=old_sha, now=NOW)
    new_sha = auto.digest(build_apify_newest_safe_successor(policy))
    assert report['mode'] == 'plan' and report['new_policy_sha256'] == new_sha and wire.posts == []
    assert report['ledger_diff'] and all('automatic_policy_sha256' in line for line in report['ledger_diff'])
    assert report['reservations_unchanged'] is True


def test_apply_requires_the_planned_digest(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True, expected_new_sha='0' * 64)
    assert wire.posts == []


def applied(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    new_sha = auto.digest(build_apify_newest_safe_successor(policy))
    report = install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True, expected_new_sha=new_sha)
    return wire, old_sha, new_sha, ledger, report


def test_apply_moves_only_the_policy_pointer_and_keeps_every_hold(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    assert report['mode'] == 'applied' and report['reservations_unchanged'] is True
    after = wire.docs[ledger['month']]
    assert after['automatic_policy_sha256'] == new_sha
    assert {k: v for k, v in after.items() if k != 'automatic_policy_sha256'} == {
        k: v for k, v in ledger.items() if k != 'automatic_policy_sha256'}
    assert wire.docs['policy_' + new_sha]['providers']['apify']['actor_build_mode'] == 'newest_safe'
    assert wire.docs[runtime.APIFY_SUCCESSOR_RECEIPT_ID]['new_policy_sha256'] == new_sha


def test_a_repeated_apply_reports_already_applied_without_writing(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    posts = len(wire.posts)
    again = install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True, expected_new_sha=new_sha)
    assert again['mode'] == 'already_applied' and len(wire.posts) == posts


def test_a_ledger_on_another_policy_is_refused(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    wire.docs[ledger['month']]['automatic_policy_sha256'] = 'f' * 64
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=NOW)


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
    assert wire.docs[runtime.APIFY_ROLLBACK_RECEIPT_ID]['new_policy_sha256'] == old_sha


def test_rollback_without_an_install_receipt_is_refused(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        rollback_module.rollback(wire.store, now=NOW)


@pytest.mark.parametrize('module', [install_module, rollback_module])
def test_main_refuses_another_installed_policy(monkeypatch, capsys, module):
    import json

    reads = []
    monkeypatch.setattr(module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', 'e664ccce9873b17fedaa7eda48e7c928be7540fd3db6c7d99168a125045a661f')
    code = module.main([])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and reads == [] and len(out) == 1 and 'error' in json.loads(out[0])


@pytest.mark.parametrize('change', [{'closed_for_rollover': True}, {'reconciliation_required': True}])
def test_a_frozen_ledger_blocks_the_install(monkeypatch, change):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    wire.docs[ledger['month']].update(change)
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=NOW)


def test_the_month_boundary_blocks_the_install(monkeypatch):
    wire, old_sha, ledger, policy = setup(monkeypatch)
    end_of_month = NOW.replace(day=31, hour=23, minute=58)
    wire.docs[ledger['month']]['month'] = end_of_month.strftime('%Y-%m')
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=end_of_month)


def test_after_the_install_a_refresh_follows_the_newest_safe_build():
    from tests.test_automatic_spend import NEWER, refresh

    ledger, policy, observation = fixtures()
    successor = build_apify_newest_safe_successor(policy)
    ledger['automatic_policy_sha256'] = auto.digest(successor)
    observation['apify']['newest_build'] = NEWER
    account = refresh(ledger, successor, observation)['providers']['apify']
    assert (account['resolved_build'], account['resolved_reason']) == ('0.0.700', 'newest')
