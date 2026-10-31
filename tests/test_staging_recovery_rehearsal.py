"""Offline rehearsal through actual migration CLI, injecting only storage/transport."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import migrate_release_state as migration
from scripts.rehearse_staging_recovery import rehearse
from src.state_store import FirestoreStateStore
from tests.test_firestore_store import _RecordingFirestoreClient


def setup(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'scripts').mkdir(parents=True)
    (root / 'data').mkdir()
    (root / 'scripts/migrate_release_state.py').write_text('# offline CLI boundary fixture\n')
    (root / 'README.md').write_text('before\n')
    (root / 'data/zeitgeist_state.json').write_text(json.dumps({'word':'ANCHOR', 'date':'2026-10-30','mood_votes':{'spot_on':20,'too_sunny':3,'too_heavy':4}}))
    (root / 'data/game_state.json').write_text(json.dumps({'turn':4,'round_id':2}))
    client = _RecordingFirestoreClient()
    def factory(**kwargs): return FirestoreStateStore(client=client, **kwargs)
    monkeypatch.setattr(migration, 'FirestoreStateStore', factory)
    phases = []
    def runner(command, **kwargs):
        phases.append(command[2:])
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = migration.main(command[2:])
        return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())
    return root, client, factory, runner, phases


def test_full_staging_rehearsal_restores_empty_namespace_through_cli(tmp_path, monkeypatch):
    root, client, factory, runner, phases = setup(tmp_path, monkeypatch)
    result = rehearse(root, tmp_path/'backup', 'staging_123456_1_', runner=runner, store_factory=factory)
    assert result['status'] == 'ok'
    assert result['seeded_document_count'] > 0
    assert result['remaining_document_count'] == 0
    assert client.docs == {}
    assert [p[0] for p in phases] == ['backup','migrate','migrate','rollback','rollback']
    assert '--dry-run' in phases[1] and '--apply' in phases[2]
    assert '--confirm-production' not in [a for p in phases for a in p]


@pytest.mark.parametrize('prefix', ['', 'staging_', 'prod_', 'staging_123_0_', 'staging_../1_'])
def test_invalid_namespace_cannot_reach_storage_or_commands(tmp_path, prefix):
    runner, factory = Mock(), Mock()
    with pytest.raises(ValueError, match='namespace'):
        rehearse(tmp_path/'repo', tmp_path/'backup', prefix, runner=runner, store_factory=factory)
    factory.assert_not_called()
    runner.assert_not_called()


def test_occupied_namespace_is_never_backed_up_or_modified(tmp_path, monkeypatch):
    root, client, factory, runner, phases = setup(tmp_path, monkeypatch)
    client.docs['staging_123456_1_processed_events/keep'] = {'status':'accepted'}
    with pytest.raises(ValueError, match='occupied'):
        rehearse(root, tmp_path/'backup', 'staging_123456_1_', runner=runner, store_factory=factory)
    assert phases == []
    assert client.docs == {'staging_123456_1_processed_events/keep': {'status':'accepted'}}


def test_failed_readback_still_runs_journaled_recovery(tmp_path, monkeypatch):
    root, client, factory, runner, phases = setup(tmp_path, monkeypatch)
    def lost_response(command, **kwargs):
        result = runner(command, **kwargs)
        if command[2] == 'migrate' and '--apply' in command:
            # The server committed; the caller lost its response.
            return SimpleNamespace(returncode=1, stdout='', stderr='simulated lost response')
        return result
    with pytest.raises(RuntimeError, match='apply failed'):
        rehearse(root, tmp_path/'backup', 'staging_123456_1_', runner=lost_response, store_factory=factory)
    assert client.docs == {}
    assert [p[0] for p in phases][-2:] == ['rollback','rollback']


def test_resolved_database_mismatch_blocks_before_remote_read(tmp_path, monkeypatch):
    root, client, factory, runner, phases = setup(tmp_path, monkeypatch)
    def wrong_target(**kwargs):
        value = factory(**kwargs)
        value._database_id = 'other-database'
        return value
    with pytest.raises(ValueError, match='target differs'):
        rehearse(root, tmp_path/'backup', 'staging_123456_1_', runner=runner, store_factory=wrong_target)
    assert client.docs == {} and phases == []


def test_private_backup_cannot_be_published_inside_source_checkout(tmp_path, monkeypatch):
    root, client, factory, runner, phases = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='outside'):
        rehearse(root, root/'backup', 'staging_123456_1_', runner=runner, store_factory=factory)
    assert phases == []
