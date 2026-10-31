"""Failure-domain reporting uses typed metadata, never messages or payloads."""
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from scripts import rehearse_staging_application as app
from src.state_store import PublicationConflict, PublicationOutcomeUnknown
from src.zeitgeist.collectors import IngestionUnavailable
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_staging_application_rehearsal import PREFIX, setup

SENTINEL = 'synthetic-secret-must-not-enter-report'


def http_error(host, status=403):
    return HTTPError('https://' + host + '/private?token=' + SENTINEL, status,
                     SENTINEL, {'private-header': SENTINEL}, BytesIO(SENTINEL.encode()))


@pytest.mark.parametrize('exception,domain,status', [
    (http_error('firestore.googleapis.com'), 'storage_authentication', 403),
    (http_error('firestore.googleapis.com', 401), 'storage_authentication', 401),
    (http_error('firestore.googleapis.com', 500), 'storage_api', 500),
    (http_error('sts.googleapis.com', 403), 'authentication', 403),
    (http_error('iamcredentials.googleapis.com', 401), 'authentication', 401),
    (http_error('firestore.googleapis.com.example.test'), 'unknown', None),
    (RuntimeError('firestore.googleapis.com 403 ' + SENTINEL), 'unknown', None),
    (IngestionUnavailable(SENTINEL), 'ingestion', None),
    (SpendBlocked(SENTINEL), 'budget_control', None),
    (PublicationConflict(SENTINEL), 'storage_conflict', None),
    (PublicationOutcomeUnknown(SENTINEL), 'storage_outcome_unknown', None),
])
def test_preflight_reports_bounded_domain_without_secret_data(tmp_path, monkeypatch, capsys, exception, domain, status):
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(app, '_snapshot', fail)
    output = tmp_path/'application.json'
    assert app.main(['--repo-root', str(tmp_path), '--collection-prefix', PREFIX, '--output-json', str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['phase'] == 'preflight'
    assert report['failure_domain'] == domain
    assert report.get('http_status') == status
    serialized = output.read_text() + capsys.readouterr().out
    assert SENTINEL not in serialized
    assert 'googleapis.com' not in serialized
    assert 'private-header' not in serialized


def test_inner_storage_failure_is_not_reported_as_a_provider_failure(tmp_path, monkeypatch):
    root, wire, factory = setup(tmp_path, monkeypatch)
    def fail(*args):
        raise http_error('firestore.googleapis.com')
    monkeypatch.setattr(app, '_seed', fail)
    with pytest.raises(app.ApplicationRehearsalError) as failure:
        app.rehearse(root, PREFIX, store_factory=factory, private_root=tmp_path/'private')
    report = failure.value.report
    assert report['failure_domain'] == 'storage_authentication'
    assert report['http_status'] == 403
    assert 'provider_diagnostics' not in report
    assert SENTINEL not in json.dumps(report)
    assert 'googleapis.com' not in json.dumps(report)


def test_unknown_exception_attributes_are_not_consulted(tmp_path, monkeypatch, capsys):
    class UnknownFailure(RuntimeError):
        @property
        def url(self):
            pytest.fail('Unknown exception attributes must not be read')
        @property
        def code(self):
            pytest.fail('Unknown exception attributes must not be read')
    def fail(*args, **kwargs):
        raise UnknownFailure(SENTINEL)
    monkeypatch.setattr(app, '_snapshot', fail)
    assert app.main(['--repo-root', str(tmp_path), '--collection-prefix', PREFIX]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['failure_domain'] == 'unknown'
    assert 'http_status' not in report
    assert SENTINEL not in json.dumps(report)


def test_inner_ingestion_keeps_only_existing_public_provider_diagnostics(tmp_path, monkeypatch):
    root, wire, factory = setup(tmp_path, monkeypatch)
    def fail(*args):
        raise IngestionUnavailable(SENTINEL, public_diagnostics={
            'provider': 'apify', 'status': 400, 'code': 'invalid-input', 'url': SENTINEL,
        })
    monkeypatch.setattr(app, '_seed', fail)
    with pytest.raises(app.ApplicationRehearsalError) as failure:
        app.rehearse(root, PREFIX, store_factory=factory, private_root=tmp_path/'private')
    report = failure.value.report
    assert report['failure_domain'] == 'ingestion'
    assert report['provider_diagnostics'] == {'provider': 'apify', 'status': 400, 'code': 'invalid-input'}
    assert 'http_status' not in report
    assert SENTINEL not in json.dumps(report)


@pytest.mark.parametrize('status', ['403', True, None, 999])
def test_http_status_is_never_coerced_from_untyped_or_invalid_values(status):
    assert app._failure_classification(http_error('firestore.googleapis.com', status)) == {'failure_domain': 'unknown'}


def test_http_error_body_is_not_read_or_serialized():
    error = http_error('firestore.googleapis.com')
    assert app._failure_classification(error) == {'failure_domain': 'storage_authentication', 'http_status': 403}
    assert error.fp.tell() == 0
