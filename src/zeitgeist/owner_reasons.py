"""Plain-words reason table for the private owner failure email.

Every entry is fixed text. No provider text, exception text or amounts are
ever copied in. Anything unknown falls to ``unclassified`` (owner-act), so a
new failure mode tells David rather than staying quiet.
"""
from dataclasses import dataclass

from src.zeitgeist.spend_guard import SpendBlocked


@dataclass(frozen=True)
class Reason:
    cls: str
    headline: str
    tonight: str
    fix: str


def _r(cls, headline, tonight, fix):
    return Reason(cls, headline, tonight, fix)


REASONS = {
    # The run went ahead on the last good Apify build (newest_safe mode); nothing was lost.
    'spend_apify_newer_input': _r('owner-approve', 'Apify released a scraper build that changes the input we send',
        'The edition ran on the last good Apify build. Nothing was lost or spent extra.',
        'Look at what the new build changed. Once the collector sends the new input, runs move to it by themselves.'),
    'spend_apify_newer_unavailable': _r('watch-on-it', "Apify's newest scraper build could not be checked",
        'The edition ran on the last good Apify build. Nothing was lost or spent extra.',
        'Nothing to do unless this repeats for several nights; then check the build list on Apify.'),
    'spend_apify_price_in_band': _r('watch-on-it', 'Apify raised its scraper price, within the 30% cushion',
        'The edition ran at the new price. The rise is measured from the price you approved and stays in the free credit.',
        'Nothing to do unless you want to approve the new price as the base; a rise above 30% would stop the run.'),
    # Spend maintenance families (no paid call has happened).
    'spend_money': _r('owner-act', 'The approved money for this month does not cover the run',
        'No edition work started and nothing was spent. Later runs stop the same way until you decide.',
        'Look at the spending ledger and the old unsettled charge. Reconcile it, or decide whether to raise the limit.'),
    'spend_price': _r('owner-act', 'A provider price page changed or could not be confirmed',
        'No edition work started. The run can go ahead today once the price record is re-pinned.',
        'Check the new price against what you approved. If it is within your limit, re-pin the policy to it.'),
    'spend_policy': _r('owner-act', 'The approved spending policy does not match or has expired',
        'No edition work started. Runs stay blocked until the policy is installed or matched.',
        'Install the successor policy, or make the pinned policy digest on GitHub match the approved one.'),
    'spend_account': _r('owner-act', 'A provider account, plan or key is not what we approved',
        'No edition work started and nothing was spent. The edition can still publish today once this is fixed.',
        'Check the provider account and the key stored in GitHub and Secret Manager. Make both match the approved account.'),
    'spend_ledger': _r('owner-approve', 'The spending ledger is in an unexpected state',
        'No edition work started. The watch will not touch the ledger; it needs a reviewed fix first.',
        'Inspect the ledger read-only, then approve a reviewed reconcile. A month rollover is the usual trigger.'),
    'spend_build': _r('owner-approve', 'The scraper version set on GitHub is not the approved one',
        'No edition work started. The run can go ahead today once the version is corrected.',
        'Set the scraper version variable on GitHub back to the approved build, then approve a retry.'),
    'spend_transient': _r('self-heal', 'A public price page was briefly unreachable',
        'No edition work started. The next catch-up run retries on its own.',
        'Nothing to do unless three catch-up runs in a row fail the same way.'),
    'spend_network': _r('self-heal', 'The spending check could not reach a provider',
        'No edition work started and nothing was spent. The next catch-up run retries on its own.',
        'Nothing to do yet. If it keeps failing, check the provider status page.'),
    'spend_code_bug': _r('owner-approve', 'The spending check hit a bug in our own code',
        'No edition work started. The watch does not edit spending code, so this needs your review.',
        'Open the run log, read the error, and approve a reviewed fix before any retry.'),
    'spend_other': _r('owner-approve', 'The spending check stopped the run on a safety rule',
        'No edition work started and nothing was spent. The safety rule needs a look before a retry.',
        'Open the run log to see which rule fired, then approve a reviewed fix or retry.'),
    'spend_provider_changed': _r('owner-act', 'A provider account or the scraper changed',
        'No edition work started and nothing was spent. The run stays blocked until the change is checked.',
        'Check the Apify plan, usage cycle and scraper version against what you approved. Fix the plan or re-pin the build.'),
    'spend_wait': _r('self-heal', 'A key was changed while an old request may still be running',
        'No edition work started. The engine waits for the old request to settle, then continues.',
        'Wait. If the old request is really stale and it is still stuck after an hour, reconcile the hold.'),
    # B1: one public code, several causes. The sub cause picks the words.
    'b1_credential_rejected': _r('owner-act', 'A provider rejected our key',
        'No edition work started and nothing was spent. The edition can still publish today once the key is fixed.',
        'Create a new key at the provider. Save it in both GitHub secrets and Secret Manager, then run the credential check.'),
    'b1_provider_unreachable': _r('self-heal', 'A provider could not be reached to check our key',
        'No edition work started and nothing was spent. The next catch-up run retries on its own.',
        'Nothing to do yet. If it keeps failing, check the provider status page.'),
    'b1_metadata_incomplete': _r('owner-act', 'A provider answered with incomplete account details',
        'No edition work started and nothing was spent. The run stays blocked until the details are complete.',
        'Check the provider account for a missing billing or build detail, then run the credential check again.'),
    'b1_credential_missing': _r('owner-act', 'A provider key is missing on the runner',
        'No edition work started. The run stays blocked until the key is present.',
        'Add the key to GitHub secrets and Secret Manager, then run the credential check.'),
    'b1_credential_not_approved': _r('owner-act', 'The provider key is not the one the ledger was approved for',
        'No edition work started and nothing was spent. A key swap needs your approval first.',
        'Put the originally approved key back, or approve the key change through the reviewed path.'),
    'b1_unknown': _r('owner-act', 'A provider key or account check failed',
        'No edition work started and nothing was spent. The cause was not recorded.',
        'Run the credential check from the owner page to find out which provider and why.'),
    # Generation, readme build, publication, startup: by failure category.
    'cat_permission': _r('owner-act', 'The engine was refused permission to do something',
        'The edition did not finish. It can still publish today once access is fixed.',
        'Check that the tokens and permissions used by the run are still valid, then run again.'),
    'cat_source_collection': _r('owner-act', 'Collecting today\'s source material failed',
        'The edition did not finish. It can still publish today once the source works again.',
        'Check the source account and the collection step in the run log, then run again.'),
    'cat_photo_quality': _r('owner-act', 'The photo did not pass its quality check',
        'The edition did not publish. No bad image went out.',
        'Look at the run log for the rejected photo and decide whether to retry or hold today.'),
    'cat_embedding_provider': _r('owner-act', 'The text-analysis provider failed',
        'The edition did not finish. It can still publish today once the provider works again.',
        'Check the provider account and key, then run again.'),
    'cat_photo_provider': _r('self-heal', 'The photo provider failed or was briefly unavailable',
        'The edition did not finish. A later catch-up run usually succeeds.',
        'Nothing to do yet. If later runs fail the same way, check the provider status page.'),
    'cat_spend_control': _r('owner-approve', 'The spending guard stopped the run mid-way',
        'The edition did not finish. Money may have been reserved, so a retry needs your review first.',
        'Inspect the spending outcome in the run log, keep limits as they are, then approve a retry.'),
    'cat_readiness': _r('owner-act', 'The run was not ready: setup or sign-in checks failed',
        'The edition did not start. It can still publish today once the setup is fixed.',
        'Open the run log for the failed readiness check and fix that setting, then run again.'),
    'cat_publication': _r('owner-approve', 'The edition is saved privately but did not reach the public profile',
        'Today\'s edition is kept. Do not generate another one; the saved edition needs a reviewed replay.',
        'Inspect the retained edition, then approve a replay of the publish step.'),
    'cat_unknown': _r('watch-on-it', 'Something unexpected stopped the engine',
        'The edition did not finish. The morning watch is looking into it.',
        'Nothing to do yet. You will hear from the watch if it needs you.'),
    'cat_startup': _r('watch-on-it', 'The run was cut off or could not start',
        'The edition did not finish. The morning watch is looking into it.',
        'Nothing to do yet. You will hear from the watch if it needs you.'),
    # Freshness (edition claim) refusals.
    'fresh_reconcile': _r('owner-approve', 'An earlier run got part way through paid work',
        'The engine will not start over. Today\'s edition waits until the earlier run is reconciled.',
        'Reconcile the earlier claim with the reviewed tool, then approve a retry.'),
    'fresh_held': _r('owner-approve', 'The edition is on hold waiting for your approval',
        'Nothing runs until you approve or release the hold.',
        'Approve or release the edition on the owner page.'),
    'fresh_active': _r('self-heal', 'Another run started this edition a few minutes ago',
        'The second run stepped aside. The first run is still in charge.',
        'Wait about 20 minutes. Nothing to do unless nothing published after that.'),
    'fresh_deterministic': _r('owner-act', 'This code version already failed this edition',
        'The engine will not repeat the same failure. A new commit lets it try again today.',
        'Fix the cause shown in the earlier failure, then make a new commit and retry.'),
    'fresh_bound': _r('owner-approve', 'Three attempts are used; the engine has stopped',
        'No more attempts today until someone looks. The edition has not published.',
        'Inspect the three attempts, then raise the bound through the reviewed tool.'),
    'fresh_unclassified': _r('owner-act', 'The run was refused at the start for a reason we do not recognise',
        'The edition did not start. The reason is not in the table, so it needs your eyes.',
        'Open the run log for the refusal message, then decide whether to retry.'),
    'fresh_config': _r('owner-act', 'The run is not set up for private daily storage',
        'The edition did not start. A setting on the runner is wrong.',
        'Check that the run uses the canonical private storage setting, then run again.'),
    'fresh_pending_recovery': _r('owner-approve', 'A publication is waiting for recovery',
        'The engine will not generate again while a saved publication is waiting to be finished.',
        'Inspect the pending publication, then approve a reviewed recovery.'),
    'fresh_identity': _r('owner-approve', 'The private and public editions do not match',
        'The engine stopped rather than guess which edition is right. Nothing new was generated.',
        'Compare the private and public editions, then approve a reviewed recovery.'),
    'fresh_future': _r('owner-approve', 'The published edition is dated in the future',
        'The engine will not generate another edition on top of it.',
        'Inspect the date on the published edition and approve a reviewed fix.'),
    'fresh_stale': _r('owner-approve', 'The edition is saved but the public page is out of date',
        'The engine will not generate again. The saved edition needs to reach the public page.',
        'Inspect the saved edition, then approve a reviewed replay of the publish step.'),
    # Fixed, stage-agnostic.
    'heartbeat': _r('heartbeat', 'The failure alert is working',
        'This is a routine test that the alert path reaches you. No edition is affected.',
        'Nothing to do.'),
    'recovered': _r('recovered', 'The edition recovered and published',
        'Today\'s edition is out. The earlier problem no longer needs action.',
        'Nothing to do.'),
    # Default.
    'unclassified': _r('owner-act', 'The run stopped for a reason we do not recognise',
        'The edition may not have published. The cause is not in the table, so it needs your eyes.',
        'Open the run log to read the cause, then decide whether to retry.'),
}

# Exact fixed messages raised by src/zeitgeist/edition_execution.py (claim);
# spike rows A6-A10. Other messages from refresh_daily_digest.py are not in the
# spike table and fall to freshness-unclassified.
FRESHNESS_MESSAGES = {
    'Edition requires existing-work reconciliation, not new inference': 'claim-needs-reconciliation',
    'Edition held for owner review': 'held-for-owner-review',
    'Another edition claim is active': 'claim-active',
    'Deterministic failure already recorded for this source revision': 'deterministic-failure-recorded',
    'Edition attempt bound reached; inspect before recovery': 'attempt-bound-reached',
    # scripts/refresh_daily_digest.py
    'Canonical-only storage required for daily freshness': 'canonical-storage-required',
    'Pending publication requires recovery, not regeneration': 'publication-recovery-pending',
    'Canonical/public identity mismatch; recovery required': 'identity-mismatch',
    'Future publication date; refuse regeneration': 'future-publication-date',
    'Execution is published but public state is stale; reconcile before generation': 'published-state-stale',
}
_FRESH_KEY = {
    'claim-needs-reconciliation': 'fresh_reconcile',
    'held-for-owner-review': 'fresh_held',
    'claim-active': 'fresh_active',
    'deterministic-failure-recorded': 'fresh_deterministic',
    'attempt-bound-reached': 'fresh_bound',
    'canonical-storage-required': 'fresh_config',
    'publication-recovery-pending': 'fresh_pending_recovery',
    'identity-mismatch': 'fresh_identity',
    'future-publication-date': 'fresh_future',
    'published-state-stale': 'fresh_stale',
}

_CATEGORY_KEY = {c: 'cat_' + c for c in (
    'permission', 'source_collection', 'photo_quality', 'embedding_provider', 'photo_provider',
    'spend_control', 'readiness', 'publication', 'unknown', 'startup')}
_CATEGORY_STAGES = frozenset({'generation', 'readme_build', 'publication', 'startup'})

_EXTRA_SPEND_CODES = frozenset({'budget-admission-rejected', 'maintenance-unavailable',
                                'transient-anonymous-price-read', 'private-report-unavailable',
                                'apify-newer-build-input-changed', 'apify-newer-build-unavailable',
                                'apify-price-in-band'})
_SPEND_CODES = frozenset(SpendBlocked._PUBLIC_REASONS.values()) | _EXTRA_SPEND_CODES
_B1_CODE = 'automatic-provider-metadata-or-credential-binding-failed'
_B1_KEY = {'credential_rejected': 'b1_credential_rejected', 'provider_unreachable': 'b1_provider_unreachable',
           'metadata_incomplete': 'b1_metadata_incomplete', 'credential_missing': 'b1_credential_missing',
           'credential_not_approved': 'b1_credential_not_approved'}
_NETWORK_ERRORS = frozenset({'URLError', 'ReadTimeout', 'ConnectError', 'TimeoutError',
                             'ConnectTimeout', 'ConnectionError'})
_CODE_BUG_ERRORS = frozenset({'TypeError', 'KeyError', 'ValueError', 'UnexpectedError'})
_P = 'automatic-'
# Codes named in spike rows B2-B9 are pinned here, ahead of the keyword rules.
_EXPLICIT = {
    'transient-anonymous-price-read': 'spend_transient',
    'apify-newer-build-input-changed': 'spend_apify_newer_input',
    'apify-newer-build-unavailable': 'spend_apify_newer_unavailable',
    'apify-price-in-band': 'spend_apify_price_in_band',
    _P + 'credential-change-waits-until-open-requests-for-this-provider-settle': 'spend_wait',
    _P + 'apify-build-differs-from-policy-pin': 'spend_build',
    _P + 'apify-account-free-grant-usage-cycle-build-or-pricing-changed': 'spend_provider_changed',
}
for _c in ('primary-price-snapshot-changed-stale-or-missing',
           'public-gemini-tariff-is-unavailable-ambiguous-or-changed',
           'public-gemini-tariff-observation-unavailable-no-price-inferred',
           'public-bfl-tariff-is-unavailable-ambiguous-or-changed',
           'public-bfl-tariff-observation-unavailable-no-price-inferred',
           'pinned-provider-identity-tariff-or-request-policy-changed'):
    _EXPLICIT[_P + _c] = 'spend_price'
for _c in (_P + 'retained-liabilities-exhaust-the-approved-monthly-cash-ceiling',
           _P + 'credit-only-provider-has-an-uncollateralized-retained-liability',
           _P + 'credit-only-provider-has-no-remaining-verified-collateral',
           _P + 'credit-only-provider-collateral-is-outside-its-verified-allowance-period',
           'monthly-cash-bound', _P + 'apify-cycle-not-active',
           _P + 'apify-cycle-grant-repeated-or-overlaps-prior-allowance',
           _P + 'observed-free-credit-exceeds-the-approved-grant',
           _P + 'confirmed-overrun-requires-reconciliation',
           'confirmed-overrun-must-be-reconciled-before-rollover',
           _P + 'unknown-provider-liability-cannot-be-discarded'):
    _EXPLICIT[_c] = 'spend_money'
for _c in (_P + 'automatic-policy-digest-mismatch', _P + 'automatic-policy-expired-or-future-dated',
           'pricing-evidence-expired-or-future-dated',
           _P + 'pinned-provider-pricing-or-policy-expires-before-request-completion',
           _P + 'approved-automatic-policy-is-not-installed'):
    _EXPLICIT[_c] = 'spend_policy'
for _c in (_P + 'legacy-frozen-or-unapproved-ledger-requires-separate-review',
           _P + 'handoff-state-requires-reconciliation', _P + 'handoff-belongs-to-another-approved-policy',
           _P + 'month-handoff-unknown-or-divergent-do-not-redispatch',
           _P + 'approved-prior-ledger-is-missing-initial-provisioning-is-separate',
           _P + 'current-ledger-month-mismatch',
           _P + 'private-maintenance-outcome-unknown-reconciliation-required',
           _P + 'refresh-plan-lost-its-source-revision', 'shared-ledger-unavailable',
           'shared-ledger-contention', 'shared-ledger-write-outcome-unknown',
           _P + 'unapproved-private-spend-document'):
    _EXPLICIT[_c] = 'spend_ledger'
# Request-shape and usage-receipt rules that merely mention a policy are code
# side, not a policy decision: owner-approve, never the watch's.
for _c in (_P + 'authenticated-gemini-usage-exceeds-the-admitted-request-policy',
           _P + 'embedding-completion-policy-covers-only-embedding-001',
           _P + 'explicit-provider-usage-accounting-policy-required',
           _P + 'legacy-holds-require-explicit-usage-accounting-policy-activation',
           _P + 'unknown-usage-receipt-policy', _P + 'usage-receipt-policy-changed-after-reservation',
           _P + 'incomplete-pinned-provider-policy', 'bounded-gemini-request-policy-required',
           'declared-operation-cap-does-not-cover-full-token-policy',
           'inline-image-count-exceeds-verified-policy', 'output-policy-changed',
           'unverified-gemini38-policy', 'unverified-model-policy',
           _P + 'provider-request-would-cross-the-accounting-month-boundary',
           'unsupported-source-accounting-epoch',
           # keyword audit: required-field and request-shape checks are code side
           _P + 'exact-automatic-request-digest-required',
           _P + 'usage-receipt-request-or-credential-binding-mismatch',
           'runtime-requires-an-approved-credential-bound-budget-epoch',
           'invalid-timezone-aware-billing-timestamp',
           _P + 'credit-only-funding-requires-a-timezone-aware-current-time',
           'explicit-approved-billing-policy-reference-required',
           'explicit-immutable-credit-baseline-usd-required',
           'explicit-none-free-prepaid-credit-kind-required',
           'a-positive-balance-requires-an-explicit-verified-credit-kind',
           'invalid-budget-liability-cutoff',
           'invalid-input-tariff', 'operation-tariff-bound', 'unverified-token-tariff',
           'only-text-output-is-covered-by-the-verified-token-tariff',
           'inline-vision-tariff-is-not-verified-for-this-model',
           'primary-provider-pricing-url-required',
           _P + 'explicit-primary-tariff-evidence-mode-required',
           'token-counting-and-tariff-evidence-required',
           'verified-single-image-generation-pricing-required',
           'unverified-model-tariff-operation-remains-blocked',
           'verified-pricing-evidence-reference-required',
           'every-admitted-operation-requires-matching-primary-price-evidence'):
    _EXPLICIT[_c] = 'spend_other'
for _c in ('public-pricing-retry-interval-invalid', 'public-pricing-retry-interval-outside-window',
           'public-pricing-response-origin-changed', 'public-pricing-anonymous-client-required'):
    _EXPLICIT[_c] = 'spend_other'

# Keyword rules match whole words between hyphens, so 'unpriced' is not 'price'.
_LEDGER_WORDS = frozenset({'ledger', 'handoff', 'rollover'})
_MONEY_WORDS = frozenset({'cash', 'liabilities', 'liability', 'collateral', 'grant', 'balance', 'credit',
                          'billing', 'overrun', 'allowance', 'prepaid', 'tranche'})
_POLICY_WORDS = frozenset({'policy', 'digest', 'expired'})
_PRICE_WORDS = frozenset({'tariff', 'price', 'pricing'})
_ACCOUNT_WORDS = frozenset({'account', 'credential', 'fingerprint'})


def _spend_key(code, error_type, sub_cause):
    if code == _B1_CODE:
        return _B1_KEY.get(sub_cause, 'b1_unknown')
    if error_type in _CODE_BUG_ERRORS:
        return 'spend_code_bug'
    if error_type in _NETWORK_ERRORS:
        return 'spend_network'
    if code in _EXPLICIT:
        return _EXPLICIT[code]
    words = set(code.split('-'))
    for family, key in ((_LEDGER_WORDS, 'spend_ledger'), (_MONEY_WORDS, 'spend_money'),
                        (_POLICY_WORDS, 'spend_policy'), (_PRICE_WORDS, 'spend_price'),
                        (_ACCOUNT_WORDS, 'spend_account')):
        if words & family:
            return key
    return 'spend_other'


def all_reasons():
    return list(REASONS.values())


def resolve(*, stage, code=None, failure_category=None, error_type=None, message=None, sub_cause=None):
    """Return (public_code, Reason). Unknown input fails toward telling David."""
    if code in ('heartbeat', 'recovered'):
        return code, REASONS[code]
    if stage == 'spend_maintenance':
        known = isinstance(code, str) and code in _SPEND_CODES
        if known or error_type in _CODE_BUG_ERRORS or error_type in _NETWORK_ERRORS:
            return (code if known else 'maintenance-unavailable'), REASONS[_spend_key(code or '', error_type, sub_cause)]
    elif stage == 'freshness':
        slug = FRESHNESS_MESSAGES.get(message) if isinstance(message, str) else None
        if slug is None and isinstance(code, str) and code in _FRESH_KEY:
            slug = code
        if slug is not None:
            return slug, REASONS[_FRESH_KEY[slug]]
        return 'freshness-unclassified', REASONS['fresh_unclassified']
    elif stage in _CATEGORY_STAGES:
        if failure_category is None:
            failure_category = 'publication' if stage == 'publication' else 'unknown'
        key = _CATEGORY_KEY.get(failure_category)
        if key is not None:
            return failure_category, REASONS[key]
    return 'unclassified', REASONS['unclassified']
