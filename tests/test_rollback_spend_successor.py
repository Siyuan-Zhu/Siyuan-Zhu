"""Successor rollback: plan, atomic apply, reconciliation and every refusal."""
import copy
from datetime import datetime, timedelta, timezone
import json
import pytest
from scripts import rollback_spend_successor_20261122 as rollback_module
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.spend_guard import HYDROGEN_OPERATION
from src.zeitgeist.spend_successor import rollback_ledger, field_diff
from tests.test_automatic_spend import fixtures
from tests.test_install_spend_successor import NOW, applied
from tests.test_spend_successor import hold

REPORT_KEYS = {'mode', 'from_policy_sha256', 'to_policy_sha256', 'plan_sha256', 'ledger_diff',
               'open_reservation_count', 'reservations_unchanged'}
RECEIPT_KEYS = {'schema_version', 'kind', 'old_policy_sha256', 'new_policy_sha256', 'install_receipt_id',
                'month', 'observed_at', 'before_sha256', 'after_sha256'}


def undo(wire, **kw):
    kw.setdefault('apply', False)
    return rollback_module.rollback(wire.store, now=NOW, **kw)


def plan_sha(wire):
    return undo(wire)['plan_sha256']


def test_plan_writes_nothing(monkeypatch):
    wire, old_sha, new_sha, p1 = applied(monkeypatch)
    before, posts = copy.deepcopy(wire.docs), len(wire.posts)
    report = undo(wire)
    old = wire.docs['policy_' + old_sha]
    ledger = wire.docs['2026-11']
    after = rollback_ledger(ledger, old, old_sha)
    assert set(report) == REPORT_KEYS and report['mode'] == 'plan'
    assert report['from_policy_sha256'] == new_sha and report['to_policy_sha256'] == old_sha
    assert report['plan_sha256'] == auto.digest({'before_sha256': auto.digest(ledger),
                                                 'after_sha256': auto.digest(after)})
    assert report['ledger_diff'] == field_diff(ledger, after)
    assert report['open_reservation_count'] == 1 and report['reservations_unchanged'] is True
    assert wire.docs == before and len(wire.posts) == posts


@pytest.mark.parametrize('unknown', [False, True])
def test_apply_restores_old_pointer_and_static_keeps_every_hold(monkeypatch, unknown):
    wire, old_sha, new_sha, p1 = applied(monkeypatch)
    wire.docs['2026-11']['reservations']['h_done'] = hold(
        NOW - timedelta(hours=3), 'settled', operation=HYDROGEN_OPERATION, actual_gross_usd='0.01',
        billing_reference='r')
    wire.versions['2026-11'] += 1
    ledger = copy.deepcopy(wire.docs['2026-11'])
    install_receipt = copy.deepcopy(wire.docs[runtime.SUCCESSOR_RECEIPT_ID])
    successor_policy = copy.deepcopy(wire.docs['policy_' + new_sha])
    old = wire.docs['policy_' + old_sha]
    posts = len(wire.posts)
    sha = plan_sha(wire)
    if unknown:
        wire.unknown_post = posts + 1
    report = undo(wire, apply=True, expected_plan_sha=sha)
    assert report['mode'] == 'applied' and report['plan_sha256'] == sha
    assert report['reservations_unchanged'] is True and set(report) == REPORT_KEYS
    assert len(wire.posts) == posts + 1 and len(wire.posts[-1]['writes']) == 2
    after = rollback_ledger(ledger, old, old_sha)
    assert wire.docs['2026-11'] == after
    assert wire.docs['2026-11']['reservations'] == ledger['reservations']
    receipt = wire.docs[runtime.ROLLBACK_RECEIPT_ID]
    assert set(receipt) == RECEIPT_KEYS and receipt['kind'] == 'rollback' and receipt['schema_version'] == 1
    assert receipt['old_policy_sha256'] == new_sha and receipt['new_policy_sha256'] == old_sha
    assert receipt['install_receipt_id'] == runtime.SUCCESSOR_RECEIPT_ID and receipt['month'] == '2026-11'
    assert receipt['before_sha256'] == auto.digest(ledger) and receipt['after_sha256'] == auto.digest(after)
    assert wire.docs['policy_' + new_sha] == successor_policy
    assert wire.docs[runtime.SUCCESSOR_RECEIPT_ID] == install_receipt
    assert runtime._successor_policy(wire.store, old_sha, '2026-11') == old_sha
    _, _, observation = fixtures(NOW)
    observation['price_snapshots']['gemini'] = {
        'sha256': old['providers']['gemini']['price_snapshot_sha256'],
        'source_url': old['providers']['gemini']['price_source_url'], 'observed_at': NOW.isoformat()}
    observation['gemini']['available_required_models'] = {
        'gemini-3.8-flash': ['generateContent'], 'gemini-embedding-001': ['embedContent']}
    auto.prepare_refresh(wire.docs['2026-11'], old, observation, expected_policy_sha=old_sha, now=NOW)


def test_rerun_is_already_applied(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    sha = plan_sha(wire)
    first = undo(wire, apply=True, expected_plan_sha=sha)
    posts, docs = len(wire.posts), copy.deepcopy(wire.docs)
    again = undo(wire, apply=True, expected_plan_sha=sha)
    assert again['mode'] == 'already_applied' and set(again) == REPORT_KEYS
    assert again['plan_sha256'] == first['plan_sha256'] and again['reservations_unchanged'] is True
    assert len(wire.posts) == posts and wire.docs == docs
    assert undo(wire)['mode'] == 'already_applied'


def test_already_applied_claims_unchanged_reservations_only_for_the_after_image(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    undo(wire, apply=True, expected_plan_sha=plan_sha(wire))
    wire.docs['2026-11']['reservations']['late'] = hold(NOW)
    assert undo(wire)['reservations_unchanged'] is False


@pytest.mark.parametrize('expected', [None, '0' * 64])
def test_expected_plan_sha_mismatch_blocks(monkeypatch, expected):
    wire, old_sha, _, _ = applied(monkeypatch)
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='plan digest'):
        undo(wire, apply=True, expected_plan_sha=expected)
    assert len(wire.posts) == posts and runtime.ROLLBACK_RECEIPT_ID not in wire.docs


def test_plan_digest_binds_the_ledger_it_was_planned_against(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    sha = plan_sha(wire)
    wire.docs['2026-11']['reservations']['extra'] = hold(NOW - timedelta(hours=1), 'settled',
                                                         actual_gross_usd='0.01', billing_reference='r')
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='plan digest'):
        undo(wire, apply=True, expected_plan_sha=sha)
    assert len(wire.posts) == posts


def test_no_install_receipt_blocks(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    del wire.docs[runtime.SUCCESSOR_RECEIPT_ID]
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='install receipt'):
        undo(wire)
    assert len(wire.posts) == posts


@pytest.mark.parametrize('tamper', [lambda r: r.update(month='2026-10'),
                                    lambda r: r.update(schema_version=2),
                                    lambda r: r.update(old_policy_sha256='c' * 64)])
def test_install_receipt_that_is_not_this_months_successor_blocks(monkeypatch, tamper):
    wire, old_sha, _, _ = applied(monkeypatch)
    tamper(wire.docs[runtime.SUCCESSOR_RECEIPT_ID])
    posts = len(wire.posts)
    with pytest.raises(RuntimeError):
        undo(wire)
    assert len(wire.posts) == posts


def test_required_old_policy_must_match_the_install_receipt(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    with pytest.raises(RuntimeError, match='reviewed installed policy'):
        undo(wire, required_old_sha='f' * 64)
    assert undo(wire, required_old_sha=old_sha)['mode'] == 'plan'


@pytest.mark.parametrize('change', [{'reconciliation_required': True}, {'closed_for_rollover': True},
                                    {'automatic_policy_sha256': 'f' * 64}, {'month': '2026-10'}])
def test_reconciliation_frozen_or_moved_ledger_blocks(monkeypatch, change):
    wire, old_sha, _, _ = applied(monkeypatch)
    wire.docs['2026-11'].update(change)
    posts = len(wire.posts)
    with pytest.raises(RuntimeError):
        undo(wire)
    assert len(wire.posts) == posts


def test_ledger_gemini_limits_that_drifted_from_the_successor_block(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    del wire.docs['2026-11']['providers']['gemini']['operation_caps_usd'][HYDROGEN_OPERATION]
    with pytest.raises(RuntimeError, match='differ'):
        undo(wire)


def test_in_flight_hold_blocks(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    wire.docs['2026-11']['reservations']['fresh'] = hold(NOW - timedelta(seconds=10))
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='in flight'):
        undo(wire)
    assert len(wire.posts) == posts


def test_in_flight_window_covers_the_longest_provider_plus_apify_start(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    wire.docs['2026-11']['reservations']['edge'] = dict(hold(NOW - timedelta(seconds=320)), provider='apify')
    with pytest.raises(RuntimeError, match='in flight'):
        undo(wire)


@pytest.mark.parametrize('age', [timedelta(hours=2), timedelta(days=9)])
def test_unsettled_hydrogen_hold_blocks_regardless_of_age(monkeypatch, age):
    wire, old_sha, _, _ = applied(monkeypatch)
    wire.docs['2026-11']['reservations']['h'] = hold(NOW - age, operation=HYDROGEN_OPERATION)
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='Hydrogen'):
        undo(wire)
    assert len(wire.posts) == posts and wire.docs['2026-11']['automatic_policy_sha256'] != old_sha


def test_month_boundary_blocks(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    late = datetime(2026, 11, 30, 23, 59, tzinfo=timezone.utc)
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='month'):
        rollback_module.rollback(wire.store, now=late)
    assert len(wire.posts) == posts


def test_hold_landing_between_read_and_commit_forces_a_replan(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    sha = plan_sha(wire)
    def land(server):
        server.docs['2026-11']['reservations']['late'] = hold(NOW - timedelta(seconds=5),
                                                              operation=HYDROGEN_OPERATION)
        server.versions['2026-11'] += 1
    posts = len(wire.posts)
    wire.before_post = land
    with pytest.raises(RuntimeError, match='replan'):
        undo(wire, apply=True, expected_plan_sha=sha)
    assert len(wire.posts) == posts + 1 and runtime.ROLLBACK_RECEIPT_ID not in wire.docs
    assert 'late' in wire.docs['2026-11']['reservations']


def test_unresolved_readback_is_refused_without_a_second_post(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    sha = plan_sha(wire)
    posts = len(wire.posts)
    wire.unknown_before_post = posts + 1
    before = copy.deepcopy(wire.docs)
    with pytest.raises(RuntimeError, match='unresolved'):
        undo(wire, apply=True, expected_plan_sha=sha)
    assert len(wire.posts) == posts + 1 and wire.docs == before


@pytest.mark.parametrize('tamper', [lambda r: r.update(month='2026-10'),
                                    lambda r: r.update(kind='other'),
                                    lambda r: r.update(schema_version=2),
                                    lambda r: r.update(old_policy_sha256='c' * 64),
                                    lambda r: r.update(new_policy_sha256='c' * 64),
                                    lambda r: r.update(install_receipt_id='other')])
def test_rollback_receipt_that_does_not_match_is_refused(monkeypatch, tamper):
    wire, old_sha, _, _ = applied(monkeypatch)
    sha = plan_sha(wire)
    undo(wire, apply=True, expected_plan_sha=sha)
    tamper(wire.docs[runtime.ROLLBACK_RECEIPT_ID])
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='identity mismatch'):
        undo(wire, apply=True, expected_plan_sha=sha)
    assert len(wire.posts) == posts


def test_receipt_with_a_ledger_that_moved_off_the_old_policy_needs_reconciliation(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    undo(wire, apply=True, expected_plan_sha=plan_sha(wire))
    wire.docs['2026-11']['automatic_policy_sha256'] = new_sha
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='reconciliation'):
        undo(wire)
    assert len(wire.posts) == posts


def test_main_refuses_other_installed_sha(monkeypatch, capsys):
    reads = []
    monkeypatch.setattr(rollback_module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', 'f' * 64)
    code = rollback_module.main([])
    out = capsys.readouterr().out.strip().splitlines()
    assert code != 0 and reads == [] and len(out) == 1 and 'error' in json.loads(out[0])


def test_main_prints_one_json_line_for_a_refusal(monkeypatch, capsys):
    wire, old_sha, _, _ = applied(monkeypatch)
    monkeypatch.setattr(rollback_module, 'PrivateFirestoreStore', lambda *a, **k: wire.store)
    monkeypatch.setattr(rollback_module, 'INSTALLED_POLICY_SHA256', old_sha)
    monkeypatch.setenv('SPEND_POLICY_SHA256', old_sha)
    del wire.docs[runtime.SUCCESSOR_RECEIPT_ID]
    code = rollback_module.main(['--apply', '--expected-plan-sha', '0' * 64])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and len(out) == 1 and 'error' in json.loads(out[0])


def refuses_without_posts(wire, match, **kw):
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match=match):
        undo(wire, **kw)
    assert len(wire.posts) == posts


def test_old_policy_that_fails_validation_blocks(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    old = wire.docs['policy_' + old_sha]
    old['valid_until'] = (NOW - timedelta(days=1)).isoformat()
    fresh = auto.digest(old)
    wire.docs['policy_' + fresh], wire.versions['policy_' + fresh] = old, 1
    del wire.docs['policy_' + old_sha]
    wire.docs[runtime.SUCCESSOR_RECEIPT_ID]['old_policy_sha256'] = fresh
    refuses_without_posts(wire, 'expired')


@pytest.mark.parametrize('which', ['old', 'new'])
def test_policy_document_that_no_longer_matches_its_digest_blocks(monkeypatch, which):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    wire.docs['policy_' + (old_sha if which == 'old' else new_sha)]['unreviewed'] = 'x'
    refuses_without_posts(wire, 'do not match the install receipt')


@pytest.mark.parametrize('which', ['old', 'new'])
def test_missing_policy_document_blocks(monkeypatch, which):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    del wire.docs['policy_' + (old_sha if which == 'old' else new_sha)]
    refuses_without_posts(wire, 'missing')


def test_ledger_request_limits_that_drifted_from_the_successor_block(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    limits = wire.docs['2026-11']['providers']['gemini']['request_limits']
    key = sorted(limits)[0]
    limits[key] = limits[key] + 1 if isinstance(limits[key], int) else 'drifted'
    refuses_without_posts(wire, 'differ')


def test_already_applied_needs_the_old_policy_document(monkeypatch):
    wire, old_sha, _, _ = applied(monkeypatch)
    undo(wire, apply=True, expected_plan_sha=plan_sha(wire))
    del wire.docs['policy_' + old_sha]
    refuses_without_posts(wire, 'reconciliation')


def test_main_pins_the_install_receipt_to_the_reviewed_policy(monkeypatch, capsys):
    wire, old_sha, _, _ = applied(monkeypatch)
    pin = 'f' * 64
    monkeypatch.setattr(rollback_module, 'PrivateFirestoreStore', lambda *a, **k: wire.store)
    monkeypatch.setattr(rollback_module, 'INSTALLED_POLICY_SHA256', pin)
    monkeypatch.setenv('SPEND_POLICY_SHA256', pin)
    posts = len(wire.posts)
    code = rollback_module.main([])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and len(out) == 1 and 'reviewed installed policy' in json.loads(out[0])['error']
    assert len(wire.posts) == posts
