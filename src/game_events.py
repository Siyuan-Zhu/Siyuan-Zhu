"""One conditional canonical game mutation, bound permanently to its GitHub Issue.

Unknown acknowledgements stop the caller. A fresh invocation reads the receipt
before computing a move; the create-if-missing receipt and source revisions also
fence a still-in-flight commit. No automatic reward replay or compensation runs.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import re

from src.game_engine import apply_visitor_move, check_issue_round_lock, normalize_game_state, parse_move_issue_title
from src.state_store import (
    PublicationConflict, PublicationOutcomeUnknown, _apply_game_rate_limit,
    _commit_documents, _strip_transient_keys,
)


def issue_receipt_id(repository: str, issue_number: int) -> str:
    identity = f'{repository.strip().lower()}\0{issue_number}'
    return 'game_issue_' + hashlib.sha256(identity.encode()).hexdigest()


def _response(result):
    return {
        'status': 'move_applied' if result.accepted else result.status_code,
        'status_code': result.status_code, 'accepted': bool(result.accepted),
        'game_key': result.new_state.get('active_game'), 'turn': result.new_state.get('turn'),
        'round_id': result.new_state.get('round_id'),
        'player_move': result.player_move_summary, 'bot_move': result.bot_move_summary,
        'round_over': result.round_over, 'winner': result.winner, 'message': result.message,
    }


def _mirror(store, documents):
    filenames = ('game_state.json', 'bot_brain.json', 'hall_of_records.json', 'game_rate_limits.json')
    for filename, document in zip(filenames, documents):
        store._mirror_projection(filename, document)
    store.set_needs_git_reconcile(True)


def apply_issue(store, *, repository, issue_number, actor, title, now_epoch):
    """Return (response, known canonical game, ledger); rejected events return None.

    Only positive Issue identities can change Firestore. JSON/in-memory callers
    continue using the existing local event path in process_visitor_event.
    """
    repository = str(repository).strip().lower()
    if not re.fullmatch(r'[a-z0-9_.-]+/[a-z0-9_.-]+', repository) or type(issue_number) is not int or issue_number <= 0:
        return {'status': 'missing_event_identity', 'accepted': False}, None, None
    client = store._get_client()
    if client is None:
        raise RuntimeError('Canonical game event requires Firestore; no fallback write allowed')
    receipt_id = issue_receipt_id(repository, issue_number)
    receipt_collection = store._col('processed_events')
    receipt_path = f'{receipt_collection}/{receipt_id}'
    receipt = client.collection(receipt_collection).document(receipt_id).get()
    if receipt.exists:
        data = receipt.to_dict() or {}
        if data.get('repository') != repository or data.get('issue_number') != issue_number or data.get('status') != 'accepted':
            raise RuntimeError(f'Unresolved or invalid game receipt; inspect {receipt_path}')
    collection = store._col('profile_state')
    names = ('active_game', 'bot_memory', 'elo_ledger', 'game_rate_limits')
    snapshots = [client.collection(collection).document(name).get() for name in names]
    if not snapshots[0].exists:
        raise RuntimeError('Canonical Firestore active_game is missing; explicit bootstrap required')
    if any(not snapshot.exists for snapshot in snapshots[1:]):
        raise RuntimeError('Canonical game domain is incomplete; recover its reviewed backup before processing')
    raw_game = snapshots[0].to_dict() or {}
    board = raw_game.get('board')
    if not isinstance(board, list) or len(board) != 6 or any(not isinstance(row, list) or len(row) != 7 for row in board) or type(raw_game.get('turn')) is not int:
        raise RuntimeError('Canonical Firestore active_game is invalid; explicit recovery required')
    documents = [copy.deepcopy(snapshot.to_dict() or {}) for snapshot in snapshots]
    game = normalize_game_state(documents[0])
    memory, ledger, limits = documents[1:]
    if (not isinstance(memory.get('states'),dict) or type(memory.get('games_learned')) is not int
        or not isinstance(ledger.get('players'),dict) or type(ledger.get('bot_elo')) is not int
        or not isinstance(ledger.get('recent_rounds'),list) or not isinstance(limits.get('users'),dict)
        or any(not isinstance(row,dict) or type(row.get('elo')) is not int
               for row in ledger['players'].values())):
        raise RuntimeError('Canonical game domain is incomplete; do not initialize over existing records')
    if receipt.exists:
        _mirror(store, [game, memory, ledger, limits])
        return {**data.get('result', {}), 'status': 'already_processed', 'accepted': False, 'previously_accepted': True,
                'receipt_path': receipt_path, 'turn': game['turn']}, game, ledger
    round_matches, round_reason = check_issue_round_lock(game, title)
    if not round_matches:
        return {'status': round_reason, 'accepted': False, 'turn': game['turn']}, None, None
    parsed = parse_move_issue_title(title, active_game_key=game['active_game'])
    if parsed is None or parsed[0] != 'connect4':
        return {'status': 'invalid_move', 'accepted': False,
                'message': 'Invalid or unsupported game command; use the current board controls.'}, None, None
    _, action, expected_turn = parsed
    if expected_turn is not None and expected_turn != game['turn']:
        return {'status':'stale_turn','accepted':False,'turn':game['turn'],
                'message':'The shared board has advanced. Refresh it before choosing another move.'},None,None
    is_new_round = not action.startswith('mode:') and (
        int(game.get('turn', 1)) <= 1 or int(game.get('moves_this_round', 0)) == 0
        or game.get('status') == 'COMPLETED' or bool(game.get('round_over')))
    allowed, reason, proposed_limits = _apply_game_rate_limit(
        limits, actor, int(now_epoch), mode=game.get('mode', 'VS_BOT'), is_new_round=is_new_round)
    if not allowed:
        return {'status': reason, 'accepted': False, 'turn': game['turn']}, None, None
    result = apply_visitor_move(game, actor_handle=actor, action=action, now_epoch=now_epoch,
                                bot_memory=memory, elo_ledger=ledger, expected_turn=expected_turn,
                                repo_slug=repository)
    response = _response(result)
    response['player'] = actor
    if not result.accepted:
        return response, None, None
    archive_write = None
    if result.new_state['round_id'] != raw_game['round_id']:
        from src.game_records import completed_record
        if result.new_state['round_id'] != raw_game['round_id'] + 1:
            raise RuntimeError('Game round must advance exactly once')
        archive_id = 'game_round_' + hashlib.sha256(
            f'{repository}\0{raw_game["round_id"]}'.encode()).hexdigest()
        archive_snapshot = client.collection(receipt_collection).document(archive_id).get()
        if archive_snapshot.exists:
            return {'status':'state_conflict','accepted':False},None,None
        archive_write = (receipt_collection,archive_id,
            dict(completed_record(normalize_game_state(raw_game)),
                 event_type='completed_game_round',repository=repository),archive_snapshot)
    proposals = [result.new_state, memory, ledger, proposed_limits]
    for proposal, snapshot in zip(proposals, snapshots):
        prior = snapshot.to_dict() or {}
        changed = {key: value for key, value in proposal.items() if key != 'state_version'} != {
            key: value for key, value in prior.items() if key != 'state_version'}
        proposal['state_version'] = int(prior.get('state_version', 0)) + int(changed)
    proposals[0] = store.game_state_document(proposals[0])
    proposals = [_strip_transient_keys(proposal) for proposal in proposals]
    accepted_receipt = {
        'schema_version': 1, 'status': 'accepted', 'event_type': 'github_issue_game',
        'repository': repository, 'issue_number': issue_number,
        'command_sha256': hashlib.sha256(f'{actor}\0{title}'.encode()).hexdigest(),
        'accepted_at': datetime.fromtimestamp(now_epoch, timezone.utc).isoformat(),
        'round_id': proposals[0]['round_id'], 'turn': proposals[0]['turn'],
        'state_version': proposals[0]['state_version'],
        'result': copy.deepcopy(response),
    }
    writes = [(collection, name, proposal, snapshot) for name, proposal, snapshot in zip(names, proposals, snapshots)]
    writes.append((receipt_collection, receipt_id, accepted_receipt, receipt))
    if archive_write:
        writes.append(archive_write)
    try:
        _commit_documents(client, writes)
    except PublicationConflict:
        return {'status': 'state_conflict', 'accepted': False, 'receipt_path': receipt_path}, None, None
    except PublicationOutcomeUnknown as exc:
        raise PublicationOutcomeUnknown(
            f'Game event outcome unknown; inspect {receipt_path} before retry. No rewards replayed.'
        ) from exc
    _mirror(store, proposals)
    return response, proposals[0], proposals[2]
