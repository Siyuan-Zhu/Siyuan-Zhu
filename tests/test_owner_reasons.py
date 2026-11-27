import pytest

from src.zeitgeist import owner_reasons as R
from src.zeitgeist.daily_outcomes import _ENUMS
from src.zeitgeist.spend_guard import SpendBlocked

CLASSES = {'owner-act', 'owner-approve', 'self-heal', 'watch-on-it', 'heartbeat', 'recovered'}
EXTRA_SPEND = ['budget-admission-rejected', 'maintenance-unavailable',
               'transient-anonymous-price-read', 'private-report-unavailable']
B1 = 'automatic-provider-metadata-or-credential-binding-failed'


@pytest.mark.parametrize('code', sorted(set(SpendBlocked._PUBLIC_REASONS.values())) + EXTRA_SPEND)
def test_every_spend_public_code_resolves(code):
    out, reason = R.resolve(stage='spend_maintenance', code=code)
    assert out == code != 'unclassified'
    assert reason.cls in CLASSES


@pytest.mark.parametrize('stage', ['generation', 'readme_build', 'publication', 'startup'])
@pytest.mark.parametrize('category', sorted(_ENUMS['failure_category']))
def test_every_failure_category_resolves(stage, category):
    out, reason = R.resolve(stage=stage, failure_category=category)
    assert out != 'unclassified'
    assert reason.cls in CLASSES


def test_category_classes():
    expect = {'permission': 'owner-act', 'source_collection': 'owner-act', 'photo_quality': 'owner-act',
              'embedding_provider': 'owner-act', 'photo_provider': 'self-heal',
              'spend_control': 'owner-approve', 'unknown': 'watch-on-it', 'startup': 'watch-on-it',
              'readiness': 'owner-act', 'publication': 'owner-approve'}
    for stage in ('generation', 'readme_build', 'publication', 'startup'):
        for cat, cls in expect.items():
            assert R.resolve(stage=stage, failure_category=cat)[1].cls == cls, (stage, cat)


def test_spend_code_bug_is_owner_approve_never_watch():
    assert R.resolve(stage='spend_maintenance', code='maintenance-unavailable',
                     error_type='TypeError')[1].cls == 'owner-approve'
    for err in ('KeyError', 'ValueError', 'UnexpectedError'):
        assert R.resolve(stage='spend_maintenance', error_type=err)[1].cls == 'owner-approve'


def test_spend_network_and_family_classes():
    for err in ('URLError', 'ReadTimeout', 'ConnectError', 'TimeoutError'):
        assert R.resolve(stage='spend_maintenance', code='maintenance-unavailable', error_type=err)[1].cls == 'self-heal'
    assert R.resolve(stage='spend_maintenance', code='transient-anonymous-price-read')[1].cls == 'self-heal'
    assert R.resolve(stage='spend_maintenance', code='monthly-cash-bound')[1].cls == 'owner-act'
    assert R.resolve(stage='spend_maintenance',
                     code='automatic-primary-price-snapshot-changed-stale-or-missing')[1].cls == 'owner-act'
    assert R.resolve(stage='spend_maintenance',
                     code='automatic-automatic-policy-digest-mismatch')[1].cls == 'owner-act'
    assert R.resolve(stage='spend_maintenance',
                     code='automatic-apify-build-differs-from-policy-pin')[1].cls == 'owner-approve'
    assert R.resolve(stage='spend_maintenance',
                     code='automatic-handoff-state-requires-reconciliation')[1].cls == 'owner-approve'


def test_b1_sub_causes():
    expect = {'credential_rejected': 'owner-act', 'provider_unreachable': 'self-heal',
              'metadata_incomplete': 'owner-act', 'credential_missing': 'owner-act',
              'credential_not_approved': 'owner-act'}
    heads = set()
    for sub, cls in expect.items():
        code, reason = R.resolve(stage='spend_maintenance', code=B1, sub_cause=sub)
        assert code == B1 and reason.cls == cls
        heads.add(reason.headline)
    assert len(heads) == 5
    assert R.resolve(stage='spend_maintenance', code=B1)[1].cls == 'owner-act'


def test_freshness_messages():
    assert set(R.FRESHNESS_MESSAGES) >= {
        'Edition requires existing-work reconciliation, not new inference',
        'Edition held for owner review', 'Another edition claim is active',
        'Deterministic failure already recorded for this source revision',
        'Edition attempt bound reached; inspect before recovery'}
    expect = {'Edition requires existing-work reconciliation, not new inference': 'owner-approve',
              'Edition held for owner review': 'owner-approve',
              'Another edition claim is active': 'self-heal',
              'Deterministic failure already recorded for this source revision': 'owner-act',
              'Edition attempt bound reached; inspect before recovery': 'owner-approve'}
    for msg, cls in expect.items():
        code, reason = R.resolve(stage='freshness', message=msg)
        assert code == R.FRESHNESS_MESSAGES[msg] and reason.cls == cls
    code, reason = R.resolve(stage='freshness', message='something new')
    assert (code, reason.cls) == ('freshness-unclassified', 'owner-act')
    assert R.resolve(stage='freshness')[0] == 'freshness-unclassified'


def test_unknown_resolves_to_owner_act():
    code, reason = R.resolve(stage='x')
    assert (code, reason.cls) == ('unclassified', 'owner-act')
    code, reason = R.resolve(stage='spend_maintenance', code='never-heard-of-it')
    assert (code, reason.cls) == ('unclassified', 'owner-act')
    code, reason = R.resolve(stage='generation', failure_category='bogus')
    assert (code, reason.cls) == ('unclassified', 'owner-act')


@pytest.mark.parametrize('stage', ['freshness', 'generation', 'spend_maintenance', 'x'])
def test_heartbeat_and_recovered(stage):
    for code in ('heartbeat', 'recovered'):
        out, reason = R.resolve(stage=stage, code=code)
        assert out == code and reason.cls == code
        assert reason.headline and reason.tonight and reason.fix


def test_text_limits_and_no_dollar():
    reasons = R.all_reasons()
    assert len(reasons) > 20
    for r in reasons:
        assert 0 < len(r.headline) <= 120
        assert 0 < len(r.tonight) <= 300
        assert 0 < len(r.fix) <= 300
        for text in (r.cls, r.headline, r.tonight, r.fix):
            assert chr(36) not in text


SPIKE_CODES = {
    # B3 price
    'automatic-primary-price-snapshot-changed-stale-or-missing': ('owner-act', 'spend_price'),
    'automatic-public-gemini-tariff-is-unavailable-ambiguous-or-changed': ('owner-act', 'spend_price'),
    'automatic-public-gemini-tariff-observation-unavailable-no-price-inferred': ('owner-act', 'spend_price'),
    'automatic-public-bfl-tariff-is-unavailable-ambiguous-or-changed': ('owner-act', 'spend_price'),
    'automatic-public-bfl-tariff-observation-unavailable-no-price-inferred': ('owner-act', 'spend_price'),
    'automatic-pinned-provider-identity-tariff-or-request-policy-changed': ('owner-act', 'spend_price'),
    # B5 / B6
    'automatic-apify-account-free-grant-usage-cycle-build-or-pricing-changed': ('owner-act', 'spend_provider_changed'),
    'automatic-apify-build-differs-from-policy-pin': ('owner-approve', 'spend_build'),
    # B7 money
    'automatic-retained-liabilities-exhaust-the-approved-monthly-cash-ceiling': ('owner-act', 'spend_money'),
    'automatic-credit-only-provider-has-an-uncollateralized-retained-liability': ('owner-act', 'spend_money'),
    'automatic-credit-only-provider-has-no-remaining-verified-collateral': ('owner-act', 'spend_money'),
    'automatic-credit-only-provider-collateral-is-outside-its-verified-allowance-period': ('owner-act', 'spend_money'),
    'monthly-cash-bound': ('owner-act', 'spend_money'),
    'automatic-apify-cycle-not-active': ('owner-act', 'spend_money'),
    'automatic-apify-cycle-grant-repeated-or-overlaps-prior-allowance': ('owner-act', 'spend_money'),
    'automatic-observed-free-credit-exceeds-the-approved-grant': ('owner-act', 'spend_money'),
    'automatic-confirmed-overrun-requires-reconciliation': ('owner-act', 'spend_money'),
    'automatic-unknown-provider-liability-cannot-be-discarded': ('owner-act', 'spend_money'),
    # B8 policy
    'automatic-automatic-policy-digest-mismatch': ('owner-act', 'spend_policy'),
    'automatic-automatic-policy-expired-or-future-dated': ('owner-act', 'spend_policy'),
    'pricing-evidence-expired-or-future-dated': ('owner-act', 'spend_policy'),
    'automatic-pinned-provider-pricing-or-policy-expires-before-request-completion': ('owner-act', 'spend_policy'),
    'automatic-approved-automatic-policy-is-not-installed': ('owner-act', 'spend_policy'),
    # B9 ledger and handoff
    'automatic-legacy-frozen-or-unapproved-ledger-requires-separate-review': ('owner-approve', 'spend_ledger'),
    'automatic-handoff-state-requires-reconciliation': ('owner-approve', 'spend_ledger'),
    'automatic-handoff-belongs-to-another-approved-policy': ('owner-approve', 'spend_ledger'),
    'automatic-month-handoff-unknown-or-divergent-do-not-redispatch': ('owner-approve', 'spend_ledger'),
    'automatic-approved-prior-ledger-is-missing-initial-provisioning-is-separate': ('owner-approve', 'spend_ledger'),
    'automatic-current-ledger-month-mismatch': ('owner-approve', 'spend_ledger'),
    'automatic-private-maintenance-outcome-unknown-reconciliation-required': ('owner-approve', 'spend_ledger'),
    'automatic-refresh-plan-lost-its-source-revision': ('owner-approve', 'spend_ledger'),
    'shared-ledger-unavailable': ('owner-approve', 'spend_ledger'),
    'shared-ledger-contention': ('owner-approve', 'spend_ledger'),
    'shared-ledger-write-outcome-unknown': ('owner-approve', 'spend_ledger'),
    'automatic-unapproved-private-spend-document': ('owner-approve', 'spend_ledger'),
    # B2, B4
    'automatic-credential-change-waits-until-open-requests-for-this-provider-settle': ('self-heal', 'spend_wait'),
    'transient-anonymous-price-read': ('self-heal', 'spend_transient'),
    # keyword-audit fixes
    'unpriced-sdk-request': ('owner-approve', 'spend_other'),
    'automatic-provider-request-would-cross-the-accounting-month-boundary': ('owner-approve', 'spend_other'),
    'confirmed-overrun-must-be-reconciled-before-rollover': ('owner-act', 'spend_money'),
    'automatic-exact-automatic-request-digest-required': ('owner-approve', 'spend_other'),
    'invalid-input-tariff': ('owner-approve', 'spend_other'),
    'invalid-timezone-aware-billing-timestamp': ('owner-approve', 'spend_other'),
    'automatic-usage-receipt-request-or-credential-binding-mismatch': ('owner-approve', 'spend_other'),
    'runtime-credential-does-not-match-verified-provider-account': ('owner-act', 'spend_account'),
}


@pytest.mark.parametrize('code', sorted(SPIKE_CODES))
def test_spike_named_spend_codes_pinned(code):
    cls, key = SPIKE_CODES[code]
    out, reason = R.resolve(stage='spend_maintenance', code=code)
    assert out == code
    assert reason is R.REASONS[key] and reason.cls == cls


FRESH_REFRESH = {
    'Canonical-only storage required for daily freshness': 'owner-act',
    'Pending publication requires recovery, not regeneration': 'owner-approve',
    'Canonical/public identity mismatch; recovery required': 'owner-approve',
    'Future publication date; refuse regeneration': 'owner-approve',
    'Execution is published but public state is stale; reconcile before generation': 'owner-approve',
}


@pytest.mark.parametrize('msg', sorted(FRESH_REFRESH))
def test_refresh_digest_freshness_messages(msg):
    code, reason = R.resolve(stage='freshness', message=msg)
    assert code == R.FRESHNESS_MESSAGES[msg] != 'freshness-unclassified'
    assert reason.cls == FRESH_REFRESH[msg]
    assert len(set(R.FRESHNESS_MESSAGES.values())) == len(R.FRESHNESS_MESSAGES)


def test_unclassified_only_owner_act():
    for err in ('TypeError', 'URLError', 'ConnectTimeout', 'ConnectionError'):
        code, reason = R.resolve(stage='spend_maintenance', error_type=err)
        assert code == 'maintenance-unavailable'
        assert reason.cls == ('owner-approve' if err == 'TypeError' else 'self-heal')
    code, reason = R.resolve(stage='spend_maintenance', code='bogus', error_type='ValueError')
    assert (code, reason.cls) == ('maintenance-unavailable', 'owner-approve')


@pytest.mark.parametrize('code,cls', [('apify-newer-build-input-changed', 'owner-approve'),
                                      ('apify-newer-build-unavailable', 'watch-on-it')])
def test_an_apify_build_fallback_resolves_to_a_plain_words_review(code, cls):
    out, reason = R.resolve(stage='spend_maintenance', code=code)
    assert out == code and reason.cls == cls
    assert 'last good' in reason.tonight
