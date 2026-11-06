import copy
from datetime import datetime, timezone
import pytest
from scripts import repair_apify_build_pin as repair
from tests.test_automatic_spend import fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.automatic_spend import digest

NOW = datetime(2026, 11, 6, 15, tzinfo=timezone.utc)
PRICING = {'model': 'PAY_PER_EVENT', 'post': '0.0017'}


def observation(**changes):
    value = {'configured_build': repair.NEW['actor_build'], 'build_pin_verified': True,
             'build_id': repair.NEW['actor_build_id'], 'image_digest': repair.NEW['actor_image_digest'],
             'input_schema_sha256': repair.SCHEMA_SHA, 'account_fingerprint': repair.ACCOUNT,
             'pricing': PRICING}
    value.update(changes)
    return value


def setup(monkeypatch):
    monkeypatch.setattr(repair, 'PRICING_SHA', digest(PRICING))
    ledger, policy, _ = fixtures(NOW)
    policy['providers']['apify'].update(repair.OLD, actor_schema_sha256=repair.SCHEMA_SHA,
                                        actor_pricing_sha256=digest(PRICING))
    sha = digest(policy)
    ledger.update(month='2026-11', automatic_policy_sha256=sha)
    wire = PrivateWire(monkeypatch, {'2026-11': ledger, 'policy_' + sha: policy})
    return wire, sha, ledger, policy


@pytest.mark.parametrize('unknown', [False, True])
def test_repin_changes_only_build_identity_and_reconciles_repeated_dispatch(monkeypatch, unknown):
    wire, sha, ledger, policy = setup(monkeypatch)
    if unknown:
        wire.unknown_post = 1
    new_sha = repair.repair(wire.store, sha, observation=observation(), apply=True, now=NOW)
    expected = copy.deepcopy(ledger)
    expected['automatic_policy_sha256'] = new_sha
    assert wire.docs['2026-11'] == expected
    successor = copy.deepcopy(policy)
    successor['providers']['apify'].update(repair.NEW)
    assert wire.docs['policy_' + new_sha] == successor
    assert wire.docs[repair.RECEIPT_ID]['old_policy_sha256'] == sha
    assert repair.repair(wire.store, sha, observation=observation(), apply=True, now=NOW) == new_sha
    assert len(wire.posts) == 1


def test_plan_without_apply_writes_nothing(monkeypatch):
    wire, sha, _, _ = setup(monkeypatch)
    repair.repair(wire.store, sha, observation=observation(), apply=False, now=NOW)
    assert not wire.posts


@pytest.mark.parametrize('changes', [
    {'build_id': 'other'}, {'image_digest': 'f' * 64}, {'input_schema_sha256': '0' * 64},
    {'account_fingerprint': 'someone-else'}, {'build_pin_verified': False},
    {'pricing': {'model': 'PAY_PER_EVENT', 'post': '0.0030'}}])
def test_unreviewed_live_build_or_price_blocks_without_any_write(monkeypatch, changes):
    wire, sha, _, _ = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        repair.repair(wire.store, sha, observation=observation(**changes), apply=True, now=NOW)
    assert not wire.posts


def test_installed_pin_other_than_reviewed_source_blocks(monkeypatch):
    wire, sha, _, policy = setup(monkeypatch)
    policy['providers']['apify']['actor_build'] = '0.0.650'
    other = digest(policy)
    wire.docs['policy_' + other] = policy
    wire.versions['policy_' + other] = 1
    wire.docs['2026-11']['automatic_policy_sha256'] = other
    with pytest.raises(RuntimeError):
        repair.repair(wire.store, other, observation=observation(), apply=True, now=NOW)
    assert not wire.posts


def receipts(monkeypatch, docs):
    return PrivateWire(monkeypatch, docs).store


def test_handoff_continuity_follows_receipts_in_reviewed_order(monkeypatch):
    store = receipts(monkeypatch, {
        'price_repair_20261103': {'month': '2026-11', 'old_policy_sha256': 'a', 'new_policy_sha256': 'b'},
        'apify_build_repair_20261106': {'month': '2026-11', 'old_policy_sha256': 'b', 'new_policy_sha256': 'c'}})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'c'


def test_handoff_continuity_without_second_receipt_stays_at_first_successor(monkeypatch):
    store = receipts(monkeypatch, {
        'price_repair_20261103': {'month': '2026-11', 'old_policy_sha256': 'a', 'new_policy_sha256': 'b'}})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'b'


@pytest.mark.parametrize('second', [
    {'month': '2026-10', 'old_policy_sha256': 'b', 'new_policy_sha256': 'c'},
    {'month': '2026-11', 'old_policy_sha256': 'x', 'new_policy_sha256': 'c'}])
def test_handoff_continuity_rejects_other_month_or_broken_chain(monkeypatch, second):
    store = receipts(monkeypatch, {
        'price_repair_20261103': {'month': '2026-11', 'old_policy_sha256': 'a', 'new_policy_sha256': 'b'},
        'apify_build_repair_20261106': second})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'b'
