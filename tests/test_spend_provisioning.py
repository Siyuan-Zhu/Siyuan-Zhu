"""Offline initial provisioning, prepaid accounting and exact-token admission."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import copy
import pytest
from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked, guard_genai_client
from scripts.prepare_spend_ledger import prepare_candidate

NOW = datetime(2026, 10, 30, 4, 14, tzinfo=timezone.utc)
CUTOFF = '2026-10-30T04:09:08Z'

def metadata():
    return {'schema_version': 1, 'budget_window_start': CUTOFF,
      'uncontrolled_post_cutoff_usage': False, 'pre_cutoff_requests_in_flight': False,
      'providers': {'bfl': {'account_id': 'redacted-account', 'plan_id': 'prepaid',
        'credential_fingerprint': 'a'*16, 'verified_at': NOW.isoformat(),
        'verification_reference': 'sha256:'+'b'*64, 'remaining_credit_usd': '9.81',
        'credit_kind': 'prepaid', 'prepaid_credits_do_not_expire': True,
        'automatic_top_up_enabled': False,
        'credit_terms_reference': 'https://help.bfl.ai/articles/8031234243-what-are-credits',
        'max_request_duration_seconds': 300,
        'operation_caps_usd': {'generate_image:flux-3-image': '.07'},
        'price_evidence': {'generate_image:flux-3-image': {
          'source_url': 'https://bfl.ai/pricing', 'verified_at': NOW.isoformat(),
          'reference': 'sha256:'+'c'*64, 'maximum_gross_usd': '.07',
          'request_shape': {'resolution': '1.5k', 'images': 1}}}}}}

def candidate():
    return prepare_candidate(metadata(), now=NOW)

def test_prepaid_candidate_preserves_sunk_balance_without_fictional_expiry():
    doc = candidate()
    assert doc['budget_window_start'] == CUTOFF
    assert doc['cash_spent_before_ledger_usd'] == '0'
    assert doc['providers']['bfl']['credit_baseline_usd'] == '9.81'
    assert 'allowance_expires_at' not in doc['providers']['bfl']
    guard = BudgetGuard(MemoryLedger(doc), lambda: NOW, credential_fingerprints={'bfl': 'a'*16})
    guard.reserve('bfl', 'generate_image:flux-3-image')
    assert guard.snapshot()['cash_exposure_usd'] == '0'

@pytest.mark.parametrize('mutation', [
    lambda m: m.update(uncontrolled_post_cutoff_usage=True),
    lambda m: m.update(pre_cutoff_requests_in_flight=True),
    lambda m: m['providers']['bfl'].update(automatic_top_up_enabled=True),
    lambda m: m['providers']['bfl'].pop('credential_fingerprint'),
    lambda m: m['providers']['bfl']['price_evidence']['generate_image:flux-3-image'].update(source_url='https://example.com/pricing'),
])
def test_initial_provisioning_rejects_unverified_or_unbounded_account(mutation):
    data = metadata(); mutation(data)
    with pytest.raises(SpendBlocked):
        prepare_candidate(data, now=NOW)

def test_mismatched_actual_credential_cannot_consume_verified_account_credit():
    guard = BudgetGuard(MemoryLedger(candidate()), lambda: NOW, credential_fingerprints={'bfl': 'd'*16})
    with pytest.raises(SpendBlocked, match='credential'):
        guard.reserve('bfl', 'generate_image:flux-3-image')

def token_guard():
    from tests.test_spend_hardening import document
    doc = document(); doc['month'] = '2026-10'
    acct = doc['providers']['gemini']; acct['verified_at'] = NOW.isoformat()
    limits = acct['request_limits']['generate_content:gemini-2.5-flash']
    limits.update(input_token_usd_per_million='.30', output_token_usd_per_million='2.50', max_input_tokens=10000, count_tokens_reference='official-no-charge')
    acct['token_pricing_required'] = True
    return BudgetGuard(MemoryLedger(doc), lambda: NOW)

def test_exact_counted_input_and_bounded_output_determine_reservation():
    guard = token_guard(); client = Mock()
    client.models.count_tokens.return_value = SimpleNamespace(total_tokens=100)
    wrapped = guard_genai_client(client, guard)
    wrapped.models.generate_content(model='gemini-2.5-flash', contents='hello', config={'max_output_tokens': 200})
    assert guard.snapshot()['cash_exposure_usd'] == '0.00053'
    counted = client.models.count_tokens.call_args.kwargs
    assert counted['contents'] == client.models.generate_content.call_args.kwargs['contents']
    assert counted['config']['http_options']['retry_options']['attempts'] == 1

@pytest.mark.parametrize('count', [None, -1, 1.5, True, float('inf'), '12'])
def test_invalid_count_blocks_before_paid_generation(count):
    guard = token_guard(); client = Mock()
    client.models.count_tokens.return_value = SimpleNamespace(total_tokens=count)
    with pytest.raises(SpendBlocked):
        guard_genai_client(client, guard).models.generate_content(model='gemini-2.5-flash', contents='hi')
    client.models.generate_content.assert_not_called()
    assert guard.snapshot()['reservation_count'] == 0

def test_policy_change_while_counting_blocks_stale_price_reservation():
    guard = token_guard(); client = Mock()
    def count(**kwargs):
        guard.ledger.document['providers']['gemini']['verification_reference'] = 'replacement-policy'
        return SimpleNamespace(total_tokens=10)
    client.models.count_tokens.side_effect = count
    with pytest.raises(SpendBlocked, match='policy changed'):
        guard_genai_client(client, guard).models.generate_content(model='gemini-2.5-flash', contents='hi')
    client.models.generate_content.assert_not_called()
    assert guard.snapshot()['reservation_count'] == 0


def test_embeddings_count_each_distinct_string_before_paid_batch():
    guard = token_guard()
    acct = guard.ledger.document['providers']['gemini']
    op = 'embed_content:gemini-embedding-001'
    acct['operation_caps_usd'][op] = '.1'
    acct['request_limits'][op] = {'pricing_reference': 'p', 'max_input_bytes': 1000,
      'max_input_tokens': 10000, 'count_tokens_reference': 'official-no-charge',
      'input_token_usd_per_million': '.15', 'output_token_usd_per_million': '0'}
    client = Mock(); client.models.count_tokens.side_effect = [SimpleNamespace(total_tokens=10), SimpleNamespace(total_tokens=20)]
    guard_genai_client(client, guard).models.embed_content(model='gemini-embedding-001', contents=['a','b'])
    assert [c.kwargs['contents'] for c in client.models.count_tokens.call_args_list] == ['a', 'b']
    assert guard.snapshot()['cash_exposure_usd'] == '0.0000045'


@pytest.mark.parametrize('unicode_tokens, admitted', [(2040, True), (2041, False)])
def test_embedding_per_text_token_limit_blocks_silent_unicode_truncation(unicode_tokens, admitted):
    guard = token_guard()
    account = guard.ledger.document['providers']['gemini']
    operation = 'embed_content:gemini-embedding-001'
    account['operation_caps_usd'][operation] = '.1'
    account['request_limits'][operation] = {'pricing_reference': 'p', 'max_input_bytes': 1000,
      'max_input_tokens': 10000, 'count_tokens_reference': 'official-no-charge',
      'input_token_usd_per_million': '.15', 'output_token_usd_per_million': '0'}
    # Character/byte limits cannot establish the model's token count for Unicode.
    texts = ['ordinary', 'λ漢字🌿']
    client = Mock()
    client.models.count_tokens.side_effect = [SimpleNamespace(total_tokens=10), SimpleNamespace(total_tokens=unicode_tokens)]
    wrapped = guard_genai_client(client, guard)
    if admitted:
        wrapped.models.embed_content(model='gemini-embedding-001', contents=texts)
        assert client.models.embed_content.call_args.kwargs['contents'] == texts
        assert guard.snapshot()['cash_exposure_usd'] == '0.0003075'
    else:
        with pytest.raises(SpendBlocked, match='2040.*per.*text'):
            wrapped.models.embed_content(model='gemini-embedding-001', contents=texts)
        client.models.embed_content.assert_not_called()
        assert guard.snapshot()['reservation_count'] == 0
    assert [call.kwargs['contents'] for call in client.models.count_tokens.call_args_list] == texts


def test_inline_raster_has_documented_image_floor_and_maximum_dimensions():
    from io import BytesIO
    from PIL import Image
    from google.genai import types
    guard = token_guard(); client = Mock()
    guard.ledger.document['providers']['gemini']['request_limits']['generate_content:gemini-2.5-flash']['max_input_bytes'] = 10000
    buf = BytesIO(); Image.new('RGB', (20,20)).save(buf, format='PNG')
    contents = ['describe', types.Part.from_bytes(data=buf.getvalue(), mime_type='image/png')]
    # Count API underreports the image: full-input=2, independent text=5.
    client.models.count_tokens.side_effect = [SimpleNamespace(total_tokens=2), SimpleNamespace(total_tokens=5)]
    guard_genai_client(client, guard).models.generate_content(model='gemini-2.5-flash', contents=contents)
    assert guard.snapshot()['cash_exposure_usd'] == '0.0013589'
    buf = BytesIO(); Image.new('RGB', (385,20)).save(buf, format='PNG')
    client.reset_mock()
    with pytest.raises(SpendBlocked, match='384'):
        guard_genai_client(client, guard).models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type='image/png')])
    client.models.generate_content.assert_not_called()


def gemini_metadata():
    data = metadata(); account = data['providers'].pop('bfl')
    op = 'generate_content:gemini-2.5-flash'
    account.update(credit_kind='none', remaining_credit_usd='0', plan_id='tier1')
    account['operation_caps_usd'] = {op: '.01924'}
    account['price_evidence'] = {op: {'source_url': 'https://ai.google.dev/gemini-api/docs/pricing',
      'verified_at': NOW.isoformat(), 'reference': 'verified-tariff', 'maximum_gross_usd': '.01924'}}
    account['request_limits'] = {op: {'pricing_reference': 'verified-tariff',
      'count_tokens_reference': 'https://ai.google.dev/gemini-api/docs/billing#is-gettokens-billed',
      'max_input_bytes': 200000, 'max_input_tokens': 30000, 'max_output_tokens': 4096,
      'input_token_usd_per_million': '.30', 'output_token_usd_per_million': '2.50'}}
    data['providers'] = {'gemini': account}
    return data


def test_candidate_derives_maximum_from_exact_verified_model_tariff():
    doc = prepare_candidate(gemini_metadata(), now=NOW)
    assert doc['providers']['gemini']['token_pricing_required'] is True


@pytest.mark.parametrize('change', ['too_small_cap', 'unpriced_model', 'invented_rate'])
def test_candidate_refuses_understated_or_invented_generation_pricing(change):
    data = gemini_metadata(); account = data['providers']['gemini']; op = next(iter(account['request_limits']))
    if change == 'too_small_cap':
        account['operation_caps_usd'][op] = '.001'; account['price_evidence'][op]['maximum_gross_usd'] = '.001'
    elif change == 'unpriced_model':
        unknown = 'generate_content:invented-model'
        for field in ('request_limits', 'operation_caps_usd', 'price_evidence'):
            account[field][unknown] = account[field].pop(op)
    else:
        account['request_limits'][op]['input_token_usd_per_million'] = '.001'
    with pytest.raises(SpendBlocked):
        prepare_candidate(data, now=NOW)


def test_actual_client_key_is_bound_instead_of_unrelated_environment_key(monkeypatch):
    import hashlib
    from src.zeitgeist import spend_guard
    doc = prepare_candidate(gemini_metadata(), now=NOW)
    doc['providers']['gemini']['credential_fingerprint'] = hashlib.sha256(b'approved').hexdigest()[:16]
    ledger = MemoryLedger(doc)
    monkeypatch.setattr(spend_guard, 'FirestoreLedger', lambda *a: ledger)
    # Fixed runtime clock via injected factory guard for this offline test.
    monkeypatch.setenv('GEMINI_API_KEY', 'approved')
    guard = spend_guard.get_spend_guard(credential_keys={'gemini': 'different-client-key'})
    guard.now = lambda: NOW
    with pytest.raises(SpendBlocked, match='credential'):
        guard.reserve('gemini', 'generate_content:gemini-2.5-flash')
    assert guard.snapshot()['reservation_count'] == 0


def test_prepaid_refresh_retains_unknown_reservations_and_cannot_change_credential():
    from datetime import timedelta
    ledger = MemoryLedger(candidate())
    guard = BudgetGuard(ledger, lambda: NOW, credential_fingerprints={'bfl': 'a'*16})
    reservation, _ = guard.reserve('bfl', 'generate_image:flux-3-image')
    next_day = NOW + timedelta(days=1)
    fresh = copy.deepcopy(ledger.document['providers']['bfl'])
    fresh.update(verified_at=next_day.isoformat(), verification_reference='new-evidence', remaining_credit_usd='9.74')
    guard.now = lambda: next_day
    guard.refresh_verification('bfl', fresh, expected_reference='sha256:'+'b'*64)
    assert ledger.document['reservations'][reservation]['status'] == 'reserved'
    assert ledger.document['providers']['bfl']['credit_baseline_usd'] == '9.74'
    fresh.update(verification_reference='different-key-evidence', credential_fingerprint='d'*16)
    with pytest.raises(SpendBlocked, match='credential_fingerprint changed'):
        guard.refresh_verification('bfl', fresh, expected_reference='new-evidence')


def test_explicit_maximum_cannot_bypass_verified_operation_cap():
    guard = BudgetGuard(MemoryLedger(candidate()), lambda: NOW, credential_fingerprints={'bfl': 'a'*16})
    with pytest.raises(SpendBlocked, match='tariff'):
        guard.reserve('bfl', 'generate_image:flux-3-image', '.08')
    with pytest.raises(SpendBlocked, match='tariff'):
        guard.reserve('bfl', 'unknown-operation', '.01')
    assert guard.snapshot()['reservation_count'] == 0
