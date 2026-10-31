"""Canonical snapshot/plan recovery acceptance tests (no live Firestore)."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from scripts.migrate_release_state import create_release_backup, migrate_release_state, rollback_release_state
from src.state_store import FirestoreStateStore
from tests.test_firestore_store import _RecordingFirestoreClient


def setup_state(tmp_path):
    root = tmp_path / 'repo'
    (root / 'data').mkdir(parents=True)
    (root / 'README.md').write_text('before\n')
    (root / 'data/zeitgeist_state.json').write_text(json.dumps({'word': 'STALE', 'date': '2026-10-28', 'mood_votes': {'spot_on': 20, 'too_sunny': 3, 'too_heavy': 4}}))
    client = _RecordingFirestoreClient()
    client.docs['staging_profile_state/daily_pulse'] = {'word': 'CANONICAL', 'date': '2026-10-29', 'state_version': 9, 'mood_votes': {'spot_on': 999, 'too_sunny': 10, 'too_heavy': 10}}
    client.docs['staging_profile_state/active_game'] = {'turn': 17, 'round_id': 9, 'state_version': 17}
    client.docs['staging_pulse_votes/2026-10-29'] = {'spot_on': 999, 'too_sunny': 10, 'too_heavy': 10}
    client.docs['staging_processed_events/issue-7'] = {'status': 'accepted'}
    store = FirestoreStateStore(client=client, project_id='test-project', collection_prefix='staging_')
    backup = tmp_path / 'backup'
    create_release_backup(root, backup, firestore_store=store)
    return root, client, store, backup


def test_firestore_migration_requires_verified_backup_before_any_write(tmp_path):
    root, client, store, _ = setup_state(tmp_path)
    before = copy.deepcopy(client.docs)
    with pytest.raises(ValueError, match='backup'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, apply=True)
    assert client.docs == before


def test_canonical_snapshot_migration_preserves_newer_votes_and_game(tmp_path):
    root, client, store, backup = setup_state(tmp_path)
    plan = tmp_path / 'plan.json'
    report = migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan)
    assert not report['applied']
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, apply=True)
    pulse = client.docs['staging_profile_state/daily_pulse']
    assert pulse['word'] == 'CANONICAL'
    assert pulse['mood_votes']['spot_on'] == 999
    assert client.docs['staging_profile_state/active_game']['turn'] == 17
    assert client.docs['staging_processed_events/issue-7'] == {'status': 'accepted'}


def test_changed_live_state_refuses_apply_without_touching_files_or_database(tmp_path):
    root, client, store, backup = setup_state(tmp_path)
    plan = tmp_path / 'plan.json'
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan)
    client.docs['staging_pulse_votes/2026-10-29']['spot_on'] += 1
    before = copy.deepcopy(client.docs)
    with pytest.raises(ValueError, match='changed'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, apply=True)
    assert client.docs == before
    assert (root / 'README.md').read_text() == 'before\n'


def test_target_binding_and_corrupt_export_fail_before_mutation(tmp_path):
    root, client, store, backup = setup_state(tmp_path)
    wrong = FirestoreStateStore(client=client, project_id='other-project', collection_prefix='staging_')
    with pytest.raises(ValueError, match='target'):
        migrate_release_state(root, target_backend='firestore', firestore_store=wrong, backup_dir=backup)
    (backup / 'firestore_export.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup)


def test_apply_requires_exact_saved_plan(tmp_path):
    root, client, store, backup = setup_state(tmp_path)
    with pytest.raises(ValueError, match='plan'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, apply=True)
    plan = tmp_path / 'plan.json'
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan)
    payload = json.loads(plan.read_text())
    payload['plan']['desired']['profile_state']['daily_pulse']['word'] = 'TAMPERED'
    plan.write_text(json.dumps(payload))
    before = copy.deepcopy(client.docs)
    with pytest.raises(ValueError, match='plan'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, apply=True)
    assert client.docs == before


def test_rollback_refuses_post_cutover_events_before_restoring_local_files(tmp_path):
    root, client, store, backup = setup_state(tmp_path)
    plan = tmp_path / 'plan.json'
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan)
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, apply=True)
    client.docs['staging_processed_events/issue-8'] = {'status': 'accepted'}
    (root / 'README.md').write_text('after\n')
    before = copy.deepcopy(client.docs)
    with pytest.raises(ValueError, match='unexpected'):
        rollback_release_state(root, backup, firestore_store=store)
    assert client.docs == before
    assert (root / 'README.md').read_text() == 'after\n'


def test_staging_seed_rollback_deletes_only_migration_created_documents(tmp_path):
    root, client, store, _ = setup_state(tmp_path)
    client.docs.clear()
    backup = tmp_path / 'empty-backup'
    create_release_backup(root, backup, firestore_store=store)
    plan = tmp_path / 'seed-plan.json'
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, seed_from_json=True)
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, seed_from_json=True, apply=True)
    assert client.docs
    report = rollback_release_state(root, backup, firestore_store=store, dry_run=True)
    assert report['deleted_firestore_docs'] > 0 and client.docs
    rollback_release_state(root, backup, firestore_store=store)
    assert client.docs == {}


def test_production_cannot_be_seeded_from_local_fixture(tmp_path):
    root, client, _, _ = setup_state(tmp_path)
    store = FirestoreStateStore(client=client, project_id='test-project', collection_prefix='')
    backup = tmp_path / 'prod-backup'
    create_release_backup(root, backup, firestore_store=store)
    with pytest.raises(PermissionError, match='seed'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, seed_from_json=True, confirm_production=True)


def test_intervening_write_at_mutation_boundary_is_rejected(tmp_path, monkeypatch):
    root, client, store, backup = setup_state(tmp_path)
    plan = tmp_path / 'plan.json'
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan)
    original = client.compare_and_set_document
    def race(collection, doc_id, expected, desired):
        client.docs[f'{collection}/{doc_id}'] = {'state_version': 100, 'turn': 100}
        return original(collection, doc_id, expected, desired)
    monkeypatch.setattr(client, 'compare_and_set_document', race)
    with pytest.raises(ValueError, match='changed'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, apply=True)
    assert json.loads((backup / 'migration_journal.json').read_text())['status'] == 'partial_or_unknown'


def test_rest_recovery_uses_server_update_time_precondition(monkeypatch):
    """The Firestore wire contract protects against changes after the preflight read."""
    import urllib.error
    from scripts.migrate_release_state import _conditional_document_write
    from src.state_store import _FirestoreRestClient
    client = _FirestoreRestClient('offline-project', access_token='dummy-test-only')
    calls = []
    class Response:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.payload).encode()
    def transport(request, timeout):
        calls.append(request)
        if request.get_method() == 'GET':
            return Response({'fields': {'state_version': {'integerValue': '9'}},
                             'updateTime': '2026-10-29T00:00:00.123456Z'})
        return Response({'writeResults': []})
    monkeypatch.setattr('urllib.request.urlopen', transport)
    _conditional_document_write(client, 'staging_profile_state', 'daily_pulse',
                                {'state_version': 9}, {'state_version': 10})
    write = json.loads(calls[-1].data)['writes'][0]
    assert write['currentDocument'] == {'updateTime': '2026-10-29T00:00:00.123456Z'}
    assert write['update']['fields'] == {'state_version': {'integerValue': '10'}}
    assert calls[-1].full_url.endswith('/documents:commit')
    def conflicting_transport(request, timeout):
        if request.get_method() == 'GET': return transport(request, timeout)
        raise urllib.error.HTTPError(request.full_url, 412, 'FAILED_PRECONDITION', None, None)
    monkeypatch.setattr('urllib.request.urlopen', conflicting_transport)
    with pytest.raises(ValueError, match='changed'):
        _conditional_document_write(client, 'staging_profile_state', 'daily_pulse',
                                    {'state_version': 9}, {'state_version': 10})


def test_partial_migration_is_journaled_and_can_be_recovered(tmp_path, monkeypatch):
    root, client, store, backup = setup_state(tmp_path)
    before = copy.deepcopy(client.docs)
    plan = tmp_path / 'plan.json'
    migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan)
    original = client.compare_and_set_document
    calls = 0
    def fail_second_write(collection, doc_id, expected, desired):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interrupted transport')
        return original(collection, doc_id, expected, desired)
    monkeypatch.setattr(client, 'compare_and_set_document', fail_second_write)
    with pytest.raises(RuntimeError, match='interrupted'):
        migrate_release_state(root, target_backend='firestore', firestore_store=store, backup_dir=backup, plan_file=plan, apply=True)
    receipt = json.loads((backup / 'migration_journal.json').read_text())
    assert receipt['status'] == 'partial_or_unknown'
    assert len(receipt['completed']) == 1
    monkeypatch.setattr(client, 'compare_and_set_document', original)
    rollback_release_state(root, backup, firestore_store=store)
    assert client.docs == before
