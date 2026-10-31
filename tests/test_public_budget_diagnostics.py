"""Budget failures expose fixed admission codes, never raw provider text."""
import json
from unittest.mock import Mock

from google.genai.errors import ClientError
import pytest

from scripts import rehearse_staging_application as application
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist.spend_guard import SpendBlocked, guard_genai_client
from tests.test_spend_provisioning import token_guard
from tests.test_staging_application_rehearsal import PREFIX, setup


def test_token_metadata_rejection_retains_status_without_paid_dispatch_or_secret():
    guard = token_guard()
    client = Mock()
    client.models.count_tokens.side_effect = ClientError(403, {
        'error': {'message': 'DO_NOT_LOG https://private.invalid?key=DO_NOT_LOG',
                  'status': 'PERMISSION_DENIED'}})
    with pytest.raises(SpendBlocked) as failure:
        guard_genai_client(client, guard).models.generate_content(
            model='gemini-2.5-flash', contents='synthetic input')
    assert failure.value.public_diagnostics == {
        'provider': 'gemini', 'code': 'token-metadata-unavailable',
        'error_type': 'ClientError', 'status': 403}
    client.models.generate_content.assert_not_called()
    assert guard.snapshot()['reservation_count'] == 0
    assert 'DO_NOT_LOG' not in json.dumps(failure.value.public_diagnostics)


@pytest.mark.parametrize('message,code', [
    ('Unpriced SDK configuration option', 'unpriced-sdk-configuration'),
    ('Token metadata unavailable; no paid call made', 'token-metadata-unavailable'),
    ('DO_NOT_LOG https://private.invalid?key=DO_NOT_LOG', 'budget-admission-rejected'),
])
def test_staging_budget_report_uses_fixed_codes_only(tmp_path, monkeypatch, message, code):
    root, wire, factory = setup(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise SpendBlocked(message)
    monkeypatch.setattr(pipeline, 'collect_96h_signals', fail)
    with pytest.raises(application.ApplicationRehearsalError) as failure:
        application.rehearse(root, PREFIX, store_factory=factory,
                             private_root=tmp_path/'private')
    report = failure.value.report
    assert report['failure_domain'] == 'budget_control'
    expected = {'code': code}
    if code == 'token-metadata-unavailable':
        expected['provider'] = 'gemini'
    assert report['budget_diagnostics'] == expected
    encoded = json.dumps(report)
    assert 'DO_NOT_LOG' not in encoded and 'private.invalid' not in encoded
