"""Actual Firestore factory and visitor handler commit each Issue exactly once."""
import copy

import pytest

from scripts import process_visitor_event as visitor
from src import state_store as stores
from tests.test_canonical_runtime_factory import fixtures
from tests.test_canonical_game_continuity import near_complete, BASE


def setup_event(tmp_path):
    root, _, _, client = fixtures(tmp_path)
    near_complete(root, client)
    store = stores.get_state_store(root, backend='firestore', firestore_client=client)
    return root, client, store


def event(root, store, *, issue=101, title='game:move:connect4:col:4:turn=4', actor='alice', at=BASE, repo='owner/sandbox'):
    return visitor.process_event(root, actor, title, issue_number=issue, timestamp_epoch=at,
                                 repo_slug=repo, store=store)


def test_actual_factory_commits_game_learning_elo_limits_and_issue_receipt_together(tmp_path, monkeypatch):
    root, client, store = setup_event(tmp_path)
    batches = []
    commit = client.commit_documents
    def capture(writes):
        batches.append(writes)
        return commit(writes)
    monkeypatch.setattr(client, 'commit_documents', capture)
    result = event(root, store)
    assert result['accepted'] is True
    assert len(batches) == 1 and len(batches[0]) == 5
    paths = {f'{col}/{doc}' for col, doc, _, _ in batches[0]}
    assert {f'profile_state/{doc}' for doc in ('active_game','bot_memory','elo_ledger','game_rate_limits')} <= paths
    receipt_paths = [path for path in paths if path.startswith('processed_events/game_issue_')]
    assert len(receipt_paths) == 1
    assert client.docs[receipt_paths[0]]['repository'] == 'owner/sandbox'
    assert client.docs[receipt_paths[0]]['issue_number'] == 101
    assert client.docs['profile_state/bot_memory']['games_learned'] == 1
    assert client.docs['profile_state/elo_ledger']['players']['alice']['elo'] == 837


def test_denied_batch_leaves_board_rewards_learning_and_quota_unchanged_then_retries_once(tmp_path, monkeypatch):
    from tests.test_firestore_store import _RecordingDocRef
    root, client, store = setup_event(tmp_path)
    before = copy.deepcopy(client.docs)
    closed = []
    monkeypatch.setattr(visitor, '_auto_close_and_lock_github_issue', lambda *a, **k: closed.append(a))
    # The old scalar-write path must never run, including the quota pre-write.
    monkeypatch.setattr(_RecordingDocRef, 'set', lambda *a, **k: pytest.fail('Scalar canonical write'))
    commit = client.commit_documents
    monkeypatch.setattr(client, 'commit_documents', lambda writes: (_ for _ in ()).throw(stores.PublicationConflict('denied')))
    result = event(root, store)
    assert result['accepted'] is False and result['status'] == 'state_conflict'
    assert client.docs == before and len(closed) == 1
    assert "state_conflict" in closed[0][1]
    assert not (root/'assets/game_board.svg').exists()
    monkeypatch.setattr(client, 'commit_documents', commit)
    assert event(root, store, at=BASE+60)['accepted'] is True
    after = copy.deepcopy(client.docs)
    assert event(root, store, at=BASE+120)['status'] == 'already_processed'
    assert client.docs == after
    assert client.docs['profile_state/bot_memory']['games_learned'] == 1
    assert client.docs['profile_state/elo_ledger']['players']['alice']['elo'] == 837
    assert len(client.docs['profile_state/game_rate_limits']['users']['alice']['hourly_timestamps']) == 1


@pytest.mark.parametrize('committed', [False, True])
def test_unknown_dispatch_stops_and_fresh_receipt_read_prevents_reward_replay(tmp_path, monkeypatch, committed):
    root, client, store = setup_event(tmp_path)
    commit = client.commit_documents
    pending = []
    def unknown(writes):
        pending.extend(writes)
        if committed:
            commit(writes)
        raise stores.PublicationOutcomeUnknown('lost acknowledgement')
    monkeypatch.setattr(client, 'commit_documents', unknown)
    closed = []
    monkeypatch.setattr(visitor, '_auto_close_and_lock_github_issue', lambda *a, **k: closed.append(a))
    with pytest.raises(stores.PublicationOutcomeUnknown, match='processed_events/game_issue_'):
        event(root, store)
    assert closed == [] and not (root/'assets/game_board.svg').exists()
    monkeypatch.setattr(client, 'commit_documents', commit)
    fresh = stores.get_state_store(root, backend='firestore', firestore_client=client)
    result = event(root, fresh, at=BASE+60)
    assert result['accepted'] is (not committed)
    assert client.docs['profile_state/bot_memory']['games_learned'] == 1
    assert client.docs['profile_state/elo_ledger']['players']['alice']['elo'] == 837
    if not committed:
        # If the first dispatched commit finishes late, the immutable receipt's
        # create precondition fences it, as do the four source revisions.
        with pytest.raises(stores.PublicationConflict):
            commit(pending)


def test_known_receipt_rebuilds_svg_after_local_failure_without_another_move(tmp_path, monkeypatch):
    root, client, store = setup_event(tmp_path)
    assert event(root, store)['accepted']
    before = copy.deepcopy(client.docs)
    (root/'assets/game_board.svg').unlink()
    duplicate = event(root, store, title='connect4: drop 1', actor='bob', at=BASE+60)
    assert duplicate['accepted'] is False and duplicate['previously_accepted'] is True
    assert (root/'assets/game_board.svg').exists() and client.docs == before


def test_receipts_bind_repository_identity_not_only_issue_number(tmp_path):
    root, client, store = setup_event(tmp_path)
    assert event(root, store)['accepted']
    other = event(root, store, repo='owner/another-repository', actor='bob', title='connect4: drop 1', at=BASE+60)
    assert other['accepted'] is True
    receipts = [doc for path, doc in client.docs.items() if path.startswith('processed_events/game_issue_')]
    assert len(receipts) == 2 and {r['repository'] for r in receipts} == {'owner/sandbox','owner/another-repository'}


def test_cross_round_turn_collision_and_legacy_titles_never_reapply_same_issue(tmp_path):
    from src.game_engine import create_initial_game_state
    root, client, store = setup_event(tmp_path)
    initial = create_initial_game_state()
    initial['mode'] = 'PVP_DUEL'
    client.docs['profile_state/active_game'] = initial
    players = [('alice',1),('bob',7),('carol',2),('frank',7),('dave',3),('grace',7),('erin',4)]
    for turn, (actor, col) in enumerate(players, 1):
        assert event(root, store, issue=200+turn, actor=actor,
            title=f'game:move:connect4:col:{col}:turn={turn}', at=BASE+turn*60)['accepted']
    learned = client.docs['profile_state/bot_memory']['games_learned']
    assert event(root, store, issue=201, title='game:move:connect4:col:1:turn=1', at=BASE+600)['status'] == 'already_processed'
    assert event(root, store, issue=999, actor='new-player', title='game:move:connect4:col:1:turn=8', at=BASE+660)['accepted']
    assert client.docs['profile_state/active_game']['round_id'] == 2
    assert client.docs['profile_state/active_game']['turn'] == 2
    before = copy.deepcopy(client.docs)
    assert event(root, store, issue=202, actor='bob', title='game:move:connect4:col:7:turn=2', at=BASE+720)['status'] == 'already_processed'
    assert client.docs == before
    assert event(root, store, issue=1000, actor='other-player', title='connect4: drop 7', at=BASE+780)['accepted']
    before = copy.deepcopy(client.docs)
    assert event(root, store, issue=1000, actor='other-player', title='connect4: drop 7', at=BASE+840)['status'] == 'already_processed'
    assert client.docs == before and client.docs['profile_state/bot_memory']['games_learned'] == learned


@pytest.mark.parametrize('issue,title,actor', [(0,'connect4: drop 1','alice'),(42,'game:move:connect4:col:4:turn=3','alice'),(43,'connect4: drop 99','alice'),(44,'connect4: drop 1','fake-bot')])
def test_unaccepted_game_events_have_no_canonical_or_reward_writes(tmp_path, issue, title, actor):
    root, client, store = setup_event(tmp_path)
    before = copy.deepcopy(client.docs)
    result = event(root, store, issue=issue, title=title, actor=actor)
    assert result['accepted'] is False and client.docs == before


def test_real_rest_request_is_one_five_document_conditional_commit(tmp_path, monkeypatch):
    from tests.test_publication_atomic_commit import WireFirestore
    root, client, _ = setup_event(tmp_path)
    wire = WireFirestore(monkeypatch, client.docs)
    store = stores.get_state_store(root, backend='firestore', firestore_client=wire.client)
    assert event(root, store)['accepted'] is True
    assert len(wire.commits) == 1
    writes = wire.commits[0]['writes']
    assert len(writes) == 5
    for write in writes:
        path = write['update']['name'].split('/documents/', 1)[1]
        expected = {'updateTime': '1'} if path in client.docs else {'exists': False}
        assert write['currentDocument'] == expected


def test_local_mirror_failure_after_known_commit_recovers_from_receipt(tmp_path, monkeypatch):
    root, client, store = setup_event(tmp_path)
    committed = []
    commit, mirror = client.commit_documents, store._mirror_projection
    def capture(writes):
        commit(writes)
        committed.append(True)
    def fail(filename, payload):
        if committed:
            raise OSError('local filesystem unavailable')
        return mirror(filename, payload)
    monkeypatch.setattr(client, 'commit_documents', capture)
    monkeypatch.setattr(store, '_mirror_projection', fail)
    with pytest.raises(OSError):
        event(root, store)
    before = copy.deepcopy(client.docs)
    fresh = stores.get_state_store(root, backend='firestore', firestore_client=client)
    assert event(root, fresh, at=BASE+60)['status'] == 'already_processed'
    assert client.docs == before and len(committed) == 1
    assert (root/'assets/game_board.svg').exists()
