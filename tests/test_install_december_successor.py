"""The December successor (SIY-198): a 30 USD ceiling, a fresh 30 USD reload attestation, Flash at its 2027 rate.

Every price comes from a recorded pricing page; the Firestore wire is the offline fake. Nothing is fetched.
"""
import copy
import json
from decimal import Decimal

import pytest

from scripts import install_spend_successor_december as install_module
from scripts import rollback_spend_successor_december as rollback_module
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.automatic_guard import AutomaticBudgetGuard
from src.zeitgeist.gemini_tariff import SOURCE_URL, normalize_gemini_tariff
from src.zeitgeist.spend_guard import MemoryLedger, SpendBlocked
from src.zeitgeist.spend_successor import (
    DECEMBER_VALID_UNTIL, FLASH_OPERATION, build_december_successor, ceiling_ledger, field_diff)
from tests.test_automatic_spend import NOW, fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from tests.test_gemini_tariff_undated_flash import in_flash_standard, recorded, undated
from tests.test_spend_successor import hold, installed as reload_policy
from tests.test_successor_compatibility import call, fake_client

RELOAD = {'observed_at': '2026-10-31T11:00:00Z', 'reference': 'synthetic-owner-reload-reading',
          'monthly_auto_charge_limit_usd': '30', 'reload_amount_usd': '5', 'trigger_balance_usd': '5'}
APPROVED_AT = '2026-10-31T11:30:00Z'
# Google's December page; the same page after Google drops the dated wording; the December page with a wording
# edit; an undated page with Flash input above its approval (inside the 10% band).
PAGES = {'dated': recorded(), 'undated': undated(),
         'dated_edited': in_flash_standard(recorded(), 'Our most intelligent Flash model', 'Our smartest Flash model'),
         'above': undated(input_row='* \u00241.60\n')}


def observation(page='dated'):
    contract = normalize_gemini_tariff(PAGES[page])
    return {'source_url': SOURCE_URL, 'observed_at': NOW.isoformat(), 'sha256': auto.digest(contract),
            'contract': contract}


def installed():
    """The production shape being replaced: the 10 USD reload policy with 3.8 Flash at its 2026 rate."""
    ledger, policy = reload_policy()
    policy['cash_ceiling_usd'] = '10.00'
    policy['providers']['gemini']['bounded_auto_reload_attestation']['monthly_auto_charge_limit_usd'] = '10'
    ledger.update(cash_ceiling_usd='10.00', credit_only_providers=policy['credit_only_providers'],
                  automatic_policy_sha256=auto.digest(policy), month=NOW.strftime('%Y-%m'))
    return ledger, policy


def successor(policy, page='dated', **changes):
    inputs = dict(tariff_sha256=observation(page)['sha256'], reload_observation=dict(RELOAD), approved_at=APPROVED_AT)
    inputs.update(changes)
    return build_december_successor(policy, **inputs)


def test_the_successor_changes_exactly_the_reviewed_fields():
    _, old = installed()
    new = successor(old)
    assert [line.split(' ')[1].rstrip(':') for line in field_diff(old, new)] == [
        'approval_reference', 'approved_at', 'cash_ceiling_usd',
        'providers.gemini.account.operation_caps_usd.generate_content:gemini-3.8-flash',
        'providers.gemini.account.request_limits.generate_content:gemini-3.8-flash.input_token_usd_per_million',
        'providers.gemini.account.request_limits.generate_content:gemini-3.8-flash.output_token_usd_per_million',
        'providers.gemini.account.request_limits.generate_content:gemini-3.8-flash.pricing_valid_until',
        'providers.gemini.bounded_auto_reload_attestation.monthly_auto_charge_limit_usd',
        'providers.gemini.bounded_auto_reload_attestation.observed_at',
        'providers.gemini.bounded_auto_reload_attestation.reference',
        'providers.gemini.price_snapshot_sha256', 'providers.gemini.price_valid_until', 'valid_until']
    gemini = new['providers']['gemini']
    flash = gemini['account']['request_limits'][FLASH_OPERATION]
    assert new['cash_ceiling_usd'] == '30.00' and new['valid_until'] == '2027-12-31T23:59:59Z' == DECEMBER_VALID_UNTIL
    assert (flash['input_token_usd_per_million'], flash['output_token_usd_per_million']) == ('1.50', '7.50')
    assert gemini['account']['operation_caps_usd'][FLASH_OPERATION] == '0.109554'
    assert gemini['bounded_auto_reload_attestation']['monthly_auto_charge_limit_usd'] == '30'
    assert auto.validate_policy(new, auto.digest(new), NOW) == new


def test_this_release_refuses_the_replaced_ten_dollar_policy():
    """Why the install and the pins move in one sitting: after the merge the old pin is refused."""
    _, old = installed()
    with pytest.raises(SpendBlocked, match='bounded Gemini reload'):
        auto.validate_policy(old, auto.digest(old), NOW)


@pytest.mark.parametrize('field,value', [('monthly_auto_charge_limit_usd', '10'), ('monthly_auto_charge_limit_usd', '25'),
                                         ('reload_amount_usd', '10'), ('trigger_balance_usd', '10')])
def test_an_observed_reload_setting_other_than_five_five_thirty_is_refused(field, value):
    _, old = installed()
    new = successor(old, reload_observation=dict(RELOAD, **{field: value}))
    with pytest.raises(SpendBlocked, match='bounded Gemini reload'):
        auto.validate_policy(new, auto.digest(new), NOW)


def _stale_mode(policy):
    policy['providers']['gemini']['price_evidence_mode'] = 'published_validity'


def _flash_at_2027(policy):
    policy['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]['input_token_usd_per_million'] = '1.50'


def _grown_cap(policy):
    policy['providers']['gemini']['account']['operation_caps_usd'][FLASH_OPERATION] = '0.06'


def _no_reload(policy):
    del policy['providers']['gemini']['bounded_auto_reload_attestation']


@pytest.mark.parametrize('change,inputs', [
    (_stale_mode, {}), (_flash_at_2027, {}), (_grown_cap, {}), (_no_reload, {}),
    (None, {'tariff_sha256': 'F' * 64}),
    (None, {'approved_at': '2026-10-31T10:59:59Z'}),
    (None, {'reload_observation': {k: v for k, v in RELOAD.items() if k != 'reference'}}),
    (None, {'reload_observation': dict(RELOAD, unexpected='1')}),
])
def test_the_builder_refuses_an_unreviewed_base_or_input(change, inputs):
    _, old = installed()
    if change:
        change(old)
    with pytest.raises(RuntimeError):
        successor(old, **inputs)


def refresh_observation(page):
    obs = fixtures()[2]
    obs['price_snapshots']['gemini'] = observation(page)
    obs['gemini']['available_required_models'] = {'gemini-3.8-flash': ['generateContent'],
                                                  'gemini-embedding-001': ['embedContent']}
    return obs


@pytest.mark.parametrize('page,path', [('dated', 'hash'), ('dated_edited', 'rate'), ('undated', 'rate')])
def test_the_successor_refreshes_and_admits_flash_at_the_2027_rate(page, path):
    """Pinned to the December page: it refreshes on it, on an edit of it, and after Google drops the dated wording."""
    ledger, old = installed()
    new = successor(old)
    after = ceiling_ledger(ledger, new, auto.digest(new))
    after.update(credential_binding_required=True, budget_scope='new_requests_after_cutoff')
    refreshed = auto.prepare_refresh(after, new, refresh_observation(page), expected_policy_sha=auto.digest(new),
                                     now=NOW)['after']
    gemini = refreshed['providers']['gemini']
    assert ('price_observed_sha256' in gemini) == (path == 'rate') and 'price_band_audit' not in gemini
    for item in refreshed['providers'].values():
        item['verified_at'] = '2026-10-27T00:00:00Z'
    guard = AutomaticBudgetGuard(MemoryLedger(refreshed), new, auto.digest(new), lambda: NOW,
                                 credential_fingerprints={'gemini': 'a' * 16, 'bfl': 'b' * 16, 'apify': 'c' * 16},
                                 require_credential_binding=True)
    api, calls = fake_client(guard, 'gemini-3.8-flash')
    assert call(api, 'gemini-3.8-flash').model_version == 'gemini-3.8-flash' and len(calls) == 1
    (record,) = guard.ledger.document['reservations'].values()
    assert record['request_context']['input_token_usd_per_million'] == '1.50'
    assert Decimal(record['actual_gross_usd']) == (Decimal(10) * Decimal('1.50') + Decimal(7) * Decimal('7.50')) / 10**6


def wire_for(monkeypatch, reservations=None):
    ledger, old = installed()
    ledger['reservations'] = reservations if reservations is not None else {
        'done': hold(NOW, 'settled', actual_gross_usd='0.01', billing_reference='provider_usage_v1:r')}
    old_sha = auto.digest(old)
    wire = PrivateWire(monkeypatch, {'policy_' + old_sha: old, ledger['month']: ledger})
    return wire, old_sha, ledger, old


def install(wire, old_sha, page='dated', observed=None, **kw):
    return install_module.install(wire.store, old_sha=old_sha, tariff_sha256=observation(page)['sha256'],
                                  reload_observation=dict(RELOAD), approved_at=APPROVED_AT, now=NOW,
                                  observer=lambda now: observation(observed or page), **kw)


def test_plan_writes_nothing_and_reports_the_successor(monkeypatch):
    wire, old_sha, ledger, old = wire_for(monkeypatch)
    report = install(wire, old_sha)
    assert report['mode'] == 'plan' and wire.posts == [] and report['reservations_unchanged'] is True
    assert report['new_policy_sha256'] == auto.digest(successor(old))
    assert report['successor']['cash_ceiling_usd'] == '30.00'
    assert report['successor']['monthly_auto_charge_limit_usd'] == '30'
    assert report['successor']['flash_usd_per_million'] == ['1.50', '7.50']
    assert report['successor']['effective_until'] == {'apify': '2026-12-31T00:00:00Z', 'bfl': '2026-12-31T00:00:00Z',
                                                      'gemini': DECEMBER_VALID_UNTIL}
    assert 'synthetic-owner-reload-reading' not in json.dumps(report)


def applied(monkeypatch):
    wire, old_sha, ledger, old = wire_for(monkeypatch)
    new_sha = install(wire, old_sha)['new_policy_sha256']
    report = install(wire, old_sha, apply=True, expected_new_sha=new_sha)
    return wire, old_sha, new_sha, ledger, report


def test_apply_writes_the_policy_the_ledger_and_the_receipt_in_one_commit(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    assert report['mode'] == 'applied' and report['reservations_unchanged'] is True and len(wire.posts) == 1
    after = wire.docs[ledger['month']]
    policy = wire.docs['policy_' + new_sha]
    assert after['automatic_policy_sha256'] == new_sha and after['cash_ceiling_usd'] == '30.00'
    assert after['providers']['gemini']['request_limits'] == policy['providers']['gemini']['account']['request_limits']
    assert after['reservations'] == ledger['reservations']
    assert wire.docs[runtime.DECEMBER_SUCCESSOR_RECEIPT_ID]['new_policy_sha256'] == new_sha
    again = install(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert again['mode'] == 'already_applied' and len(wire.posts) == 1


def test_readback_shows_the_successor_active_with_the_2027_flash_rate(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    back = install_module.readback(wire.store, now=NOW)
    assert back['active'] is True and back['policy_sha256'] == new_sha
    assert back['successor']['flash_usd_per_million'] == ['1.50', '7.50']
    assert back['successor']['cash_ceiling_usd'] == '30.00'


def test_apply_requires_the_planned_digest(monkeypatch):
    wire, old_sha, ledger, old = wire_for(monkeypatch)
    with pytest.raises(RuntimeError, match='planned new policy digest'):
        install(wire, old_sha, apply=True, expected_new_sha='0' * 64)
    assert wire.posts == []


@pytest.mark.parametrize('case', ['open_hold', 'other_policy', 'frozen', 'tariff_moved', 'rate_above', 'bad_digest'])
def test_the_install_refuses_and_writes_nothing(monkeypatch, case):
    reservations = {'open': hold(NOW)} if case == 'open_hold' else None
    wire, old_sha, ledger, old = wire_for(monkeypatch, reservations)
    kw = {}
    if case == 'other_policy':
        wire.docs[ledger['month']]['automatic_policy_sha256'] = 'f' * 64
    if case == 'frozen':
        wire.docs[ledger['month']]['reconciliation_required'] = True
    if case == 'tariff_moved':
        kw['observed'] = 'undated'
    if case == 'rate_above':
        kw['page'] = 'above'
    if case == 'bad_digest':
        wire.docs['policy_' + old_sha]['cash_ceiling_usd'] = '9.00'
    with pytest.raises(RuntimeError):
        install(wire, old_sha, **kw)
    assert wire.posts == []


def test_rollback_restores_the_replaced_ledger_exactly(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    plan = rollback_module.rollback(wire.store, required_old_sha=old_sha, now=NOW)
    assert plan['mode'] == 'plan' and plan['to_policy_sha256'] == old_sha
    done = rollback_module.rollback(wire.store, required_old_sha=old_sha, now=NOW, apply=True,
                                    expected_plan_sha=plan['plan_sha256'])
    assert done['mode'] == 'applied'
    assert wire.docs[ledger['month']] == ledger
    assert wire.docs[runtime.DECEMBER_ROLLBACK_RECEIPT_ID]['new_policy_sha256'] == old_sha


def test_rollback_refuses_a_month_spent_past_the_replaced_ceiling(monkeypatch):
    wire, old_sha, new_sha, ledger, report = applied(monkeypatch)
    wire.docs[ledger['month']]['reservations']['big'] = hold(
        NOW, 'settled', max_gross_usd='10.01', actual_gross_usd='10.01', billing_reference='provider_usage_v1:big')
    with pytest.raises(RuntimeError, match='above the replaced'):
        rollback_module.rollback(wire.store, required_old_sha=old_sha, now=NOW)


@pytest.mark.parametrize('module', [install_module, rollback_module])
def test_main_refuses_another_installed_policy(monkeypatch, capsys, module):
    reads = []
    monkeypatch.setattr(module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', 'e664ccce9873b17fedaa7eda48e7c928be7540fd3db6c7d99168a125045a661f')
    code = module.main([])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and reads == [] and len(out) == 1 and 'error' in json.loads(out[0])


def test_the_store_reads_the_december_receipts():
    from tests.test_spend_successor_chain import Client
    for document_id in (runtime.DECEMBER_SUCCESSOR_RECEIPT_ID, runtime.DECEMBER_ROLLBACK_RECEIPT_ID):
        runtime.PrivateFirestoreStore(Client()).read(document_id)
