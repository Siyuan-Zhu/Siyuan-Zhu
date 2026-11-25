"""Neon successor rollback to the Hydrogen successor: plan, atomic apply, reconciliation, refusals."""
import copy
from datetime import timedelta
import json
import pytest
from scripts import rollback_spend_successor_20261125 as rollback_module
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.spend_guard import HYDROGEN_OPERATION, NEON_OPERATION
from src.zeitgeist.spend_successor import NEON_BASE_POLICY_SHA256, field_diff, rollback_ledger
from tests.test_install_neon_successor import NOW, applied, setup
from tests.test_spend_successor import hold


def undo(wire, **kw):
    kw.setdefault('apply', False)
    return rollback_module.rollback(wire.store, now=NOW, **kw)


def test_plan_writes_nothing(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    before, posts = copy.deepcopy(wire.docs), len(wire.posts)
    report = undo(wire)
    ledger = wire.docs['2026-11']
    after = rollback_ledger(ledger, wire.docs['policy_' + old_sha], old_sha)
    assert report['mode'] == 'plan'
    assert report['from_policy_sha256'] == new_sha and report['to_policy_sha256'] == old_sha
    assert report['ledger_diff'] == field_diff(ledger, after)
    assert wire.docs == before and len(wire.posts) == posts


def test_apply_returns_to_the_hydrogen_successor_and_keeps_every_hold(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    wire.docs['2026-11']['reservations']['n_done'] = hold(
        NOW - timedelta(hours=3), 'settled', operation=NEON_OPERATION, actual_gross_usd='0.01', billing_reference='r')
    wire.versions['2026-11'] += 1
    ledger = copy.deepcopy(wire.docs['2026-11'])
    install_receipt = copy.deepcopy(wire.docs[runtime.NEON_SUCCESSOR_RECEIPT_ID])
    posts = len(wire.posts)
    sha = undo(wire)['plan_sha256']
    report = undo(wire, apply=True, expected_plan_sha=sha)
    assert report['mode'] == 'applied' and report['reservations_unchanged'] is True
    assert len(wire.posts) == posts + 1 and len(wire.posts[-1]['writes']) == 2
    assert wire.docs['2026-11'] == rollback_ledger(ledger, wire.docs['policy_' + old_sha], old_sha)
    receipt = wire.docs[runtime.NEON_ROLLBACK_RECEIPT_ID]
    assert receipt['kind'] == 'rollback' and receipt['install_receipt_id'] == 'spend_successor_20261125'
    assert receipt['old_policy_sha256'] == new_sha and receipt['new_policy_sha256'] == old_sha
    assert wire.docs[runtime.NEON_SUCCESSOR_RECEIPT_ID] == install_receipt
    assert runtime.ROLLBACK_RECEIPT_ID not in wire.docs
    assert runtime._successor_policy(wire.store, old_sha, '2026-11') == old_sha
    assert undo(wire, apply=True, expected_plan_sha=sha)['mode'] == 'already_applied'
    assert len(wire.posts) == posts + 1


def test_expected_plan_sha_mismatch_blocks(monkeypatch):
    wire, _, _, _ = applied(monkeypatch)
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='planned plan digest'):
        undo(wire, apply=True, expected_plan_sha='0' * 64)
    assert len(wire.posts) == posts


def test_no_neon_install_receipt_blocks(monkeypatch):
    wire, _, _, _ = setup(monkeypatch)
    # The Hydrogen install's receipt is not the Neon install's.
    wire.docs[runtime.SUCCESSOR_RECEIPT_ID] = {'schema_version': 1, 'month': '2026-11',
                                               'old_policy_sha256': 'a' * 64, 'new_policy_sha256': 'b' * 64}
    wire.versions[runtime.SUCCESSOR_RECEIPT_ID] = 1
    with pytest.raises(RuntimeError, match='No install receipt'):
        undo(wire)
    assert wire.posts == []


@pytest.mark.parametrize('age', [timedelta(hours=2), timedelta(days=9)])
def test_unsettled_neon_hold_blocks_regardless_of_age(monkeypatch, age):
    wire, old_sha, _, _ = applied(monkeypatch)
    wire.docs['2026-11']['reservations']['n'] = hold(NOW - age, operation=NEON_OPERATION)
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='unsettled Neon hold'):
        undo(wire)
    assert len(wire.posts) == posts and wire.docs['2026-11']['automatic_policy_sha256'] != old_sha


def test_unsettled_hydrogen_hold_older_than_the_window_allows_the_rollback(monkeypatch):
    wire, _, _, _ = applied(monkeypatch)
    wire.docs['2026-11']['reservations']['h'] = hold(NOW - timedelta(hours=2), operation=HYDROGEN_OPERATION)
    assert undo(wire)['mode'] == 'plan'


def test_required_old_policy_must_be_the_hydrogen_successor(monkeypatch):
    wire, _, _, _ = applied(monkeypatch)
    with pytest.raises(RuntimeError, match='reviewed installed policy'):
        undo(wire, required_old_sha='f' * 64)


def test_main_refuses_another_installed_sha(monkeypatch, capsys):
    reads = []
    monkeypatch.setattr(rollback_module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', 'e664ccce9873b17fedaa7eda48e7c928be7540fd3db6c7d99168a125045a661f')
    code = rollback_module.main([])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and reads == [] and len(out) == 1 and 'error' in json.loads(out[0])


def test_main_pins_the_install_receipt_to_the_hydrogen_successor(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(rollback_module, 'PrivateFirestoreStore', lambda *a, **k: 'store')
    monkeypatch.setattr(rollback_module, 'rollback', lambda store, **kw: seen.append(kw) or {'mode': 'plan'})
    monkeypatch.setenv('SPEND_POLICY_SHA256', NEON_BASE_POLICY_SHA256)
    assert rollback_module.main(['--apply', '--expected-plan-sha', 'a' * 64]) == 0
    assert seen == [dict(apply=True, expected_plan_sha='a' * 64, required_old_sha=NEON_BASE_POLICY_SHA256)]
