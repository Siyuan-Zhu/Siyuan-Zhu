import copy
from datetime import datetime, timezone
import pytest
from scripts import reconcile_edition_20261106 as rec

NOW = datetime(2026, 11, 6, 16, tzinfo=timezone.utc)
RECORD = {'schema_version': 1, 'edition_date': '2026-11-06', 'owner': 'github-42229783506-1',
          'source_revision': '31834093abea5671cc8fc7caf554de767f9e3972', 'trigger': 'workflow_dispatch',
          'stage': 'requires_reconciliation', 'attempt': 1, 'started_at': '2026-11-06T13:58:27.163332+00:00',
          'updated_at': '2026-11-06T13:59:24.178140+00:00', 'lease_until': '2026-11-06T14:18:27.163332+00:00',
          'publication_id': None, 'failure_category': 'generation'}
OUTCOME = {'edition_date': '2026-11-06', 'run_id': '42229783506', 'status': 'failed',
           'failure_category': 'source_collection', 'failure_code': 'required-source-coverage-unavailable',
           'canonical_commit': 'not_committed', 'git_publication': 'not_started',
           'publication_id': '434ade2b29427efd9f55', 'previous_publication_id': '434ade2b29427efd9f55',
           'started_at': '2026-11-06T13:58:27.163332+00:00', 'completed_at': '2026-11-06T13:59:24.687649+00:00',
           'collection_health': {'publishable': False, 'instagram_item_count': 0}}
LEDGER = {'reservations': {
    'apify-today': {'provider': 'apify', 'status': 'settled', 'created_at': '2026-11-06T13:58:52.924613+00:00'},
    'apify-yesterday': {'provider': 'apify', 'status': 'settled', 'created_at': '2026-11-05T16:25:14.562637+00:00'}}}


class Snapshot:
    def __init__(self, value):
        self.value = copy.deepcopy(value)
        self.exists = value is not None

    def to_dict(self):
        return copy.deepcopy(self.value)


class Client:
    def __init__(self, record):
        self.record = copy.deepcopy(record)
        self.commits = []

    def collection(self, name):
        assert name == 'editorial_executions'
        return self

    def document(self, name):
        assert name == '2026-11-06'
        return self

    def get(self):
        return Snapshot(self.record)


@pytest.fixture
def client(monkeypatch):
    c = Client(RECORD)

    def commit(target, writes):
        assert target is c
        (collection, doc, value, snapshot), = writes
        assert snapshot.to_dict() == c.record
        c.commits.append(value)
        c.record = copy.deepcopy(value)
    monkeypatch.setattr(rec, '_commit_documents', commit)
    return c


def test_reconciles_to_blocked_keeping_attempt_and_revision(client):
    after = rec.reconcile(client, OUTCOME, LEDGER, apply=True, now=NOW)
    assert after['stage'] == 'blocked' and after['attempt'] == 1
    assert after['source_revision'] == RECORD['source_revision']
    assert after['reconciliation']['apify_reservation_id'] == 'apify-today'
    assert rec.reconcile(client, OUTCOME, LEDGER, apply=True, now=NOW)['stage'] == 'blocked'
    assert len(client.commits) == 1


def test_plan_writes_nothing(client):
    rec.reconcile(client, OUTCOME, LEDGER, apply=False, now=NOW)
    assert not client.commits


@pytest.mark.parametrize('change', [{'canonical_commit': 'committed'}, {'failure_code': 'other'},
                                    {'publication_id': 'new-publication'}, {'run_id': '1'}])
def test_outcome_evidence_mismatch_blocks(client, change):
    with pytest.raises(RuntimeError):
        rec.reconcile(client, dict(OUTCOME, **change), LEDGER, apply=True, now=NOW)
    assert not client.commits


def test_any_inference_spend_in_window_blocks(client):
    ledger = copy.deepcopy(LEDGER)
    ledger['reservations']['gemini'] = {'provider': 'gemini', 'status': 'settled',
                                        'created_at': '2026-11-06T13:59:10+00:00'}
    with pytest.raises(RuntimeError):
        rec.reconcile(client, OUTCOME, ledger, apply=True, now=NOW)
    assert not client.commits


@pytest.mark.parametrize('change', [{'stage': 'generation'}, {'publication_id': 'x'}, {'owner': 'other'}])
def test_unexpected_record_blocks(monkeypatch, change):
    c = Client(dict(RECORD, **change))
    monkeypatch.setattr(rec, '_commit_documents', lambda *a: pytest.fail('must not write'))
    with pytest.raises(RuntimeError):
        rec.reconcile(c, OUTCOME, LEDGER, apply=True, now=NOW)


def test_blocked_record_is_claimable_only_from_another_revision():
    from src.zeitgeist.edition_execution import EditionExecution
    blocked = dict(RECORD, stage='blocked', failure_category='source_collection')

    class Doc:
        def __init__(self):
            self.value = blocked

        def get(self):
            return Snapshot(self.value)
    execution = EditionExecution.__new__(EditionExecution)
    execution.doc, execution.edition, execution.owner = Doc(), '2026-11-06', 'github-next-1'
    execution.collection = 'editorial_executions'
    with pytest.raises(RuntimeError):
        execution.claim(now=NOW, source_revision=RECORD['source_revision'], trigger='workflow_dispatch')
    execution._write = lambda snapshot, value: value
    claimed = execution.claim(now=NOW, source_revision='new-revision', trigger='workflow_dispatch')
    assert claimed['attempt'] == 2 and claimed['stage'] == 'preflight'
