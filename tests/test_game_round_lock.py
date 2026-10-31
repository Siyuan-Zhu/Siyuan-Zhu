"""Fresh Issues created from stale round buttons cannot act on a later round."""
import copy
import html
import json
from urllib.parse import parse_qs

import pytest

from scripts import process_visitor_event as visitor, validate_release_links as release
from src.game_engine import render_readme_game_section
from src.state_store import get_state_store
from tests.test_canonical_runtime_factory import fixtures
from tests.test_canonical_game_continuity import BASE


def titles(markdown):
    return [parse_qs(html.unescape(query))['title'][0]
            for _, query in release.ISSUE_NEW_URL_RE.findall(markdown)]


def test_new_issue_from_previous_round_button_is_rejected_despite_matching_turn(tmp_path):
    root, _, current, client = fixtures(tmp_path)
    previous = dict(current, round_id=current['round_id']-1)
    old_title = titles(render_readme_game_section(previous))[0]
    store = get_state_store(root, backend='firestore', firestore_client=client)
    before = copy.deepcopy(client.docs)
    response = visitor.process_event(root, 'alice', old_title, issue_number=901, timestamp_epoch=BASE,
                                      repo_slug='Siyuan-Zhu/Siyuan-Zhu', store=store)
    assert response['accepted'] is False and response['status'] == 'stale_round'
    assert client.docs == before


@pytest.mark.parametrize('kind', ['move','mode'])
@pytest.mark.parametrize('issue,turn,round_id,expected', [(902,1,8,'move_applied'), (903,2,8,'stale_turn'), (904,1,7,'stale_round')])
def test_current_round_and_negative_locks_have_distinct_issue_identities(tmp_path, kind, issue, turn, round_id, expected):
    root, _, _, client = fixtures(tmp_path)
    store = get_state_store(root, backend='firestore', firestore_client=client)
    action = 'game:move:connect4:col:1' if kind == 'move' else 'game:mode:connect4:new_pvp'
    before = copy.deepcopy(client.docs)
    result = visitor.process_event(root, 'alice', f'{action}:turn={turn}:round={round_id}',
        issue_number=issue, timestamp_epoch=BASE, repo_slug='owner/sandbox', store=store)
    assert result['status'] == expected
    assert result['accepted'] is (expected == 'move_applied')
    if not result['accepted']:
        assert client.docs == before and not any(p.startswith('processed_events/') for p in client.docs)


def test_three_tuple_parser_and_manual_unlocked_commands_stay_compatible():
    from src.game_engine import parse_move_issue_title
    assert parse_move_issue_title('game:move:connect4:col:1:turn=2:round=8') == ('connect4','col:1',2)
    assert parse_move_issue_title('game:mode:connect4:new_pvp:turn=1:round=8') == ('connect4','mode:new_pvp',1)
    assert parse_move_issue_title('game:move:connect4:col:1:turn=2') == ('connect4','col:1',2)
    assert parse_move_issue_title('connect4: drop 1') == ('connect4','col1',None)


def test_current_completed_board_button_starts_next_round_once(tmp_path):
    root, _, current, client = fixtures(tmp_path)
    current.update(status='COMPLETED', round_over=True, turn=9)
    client.docs['profile_state/active_game'] = current
    title = titles(render_readme_game_section(current))[0]
    store = get_state_store(root, backend='firestore', firestore_client=client)
    result = visitor.process_event(root, 'alice', title, issue_number=906,
        timestamp_epoch=BASE, repo_slug='owner/sandbox', store=store)
    assert result['accepted'] is True
    assert client.docs['profile_state/active_game']['round_id'] == current['round_id'] + 1
    before = copy.deepcopy(client.docs)
    stale = visitor.process_event(root, 'bob', title, issue_number=907,
        timestamp_epoch=BASE+60, repo_slug='owner/sandbox', store=store)
    assert stale['status'] == 'stale_round' and stale['accepted'] is False
    assert client.docs == before


@pytest.mark.parametrize('suffix', ['round=0','round=-1','round=bad','round=8:round=8'])
def test_malformed_round_locks_cannot_write_or_create_receipts(tmp_path, suffix):
    root, _, _, client = fixtures(tmp_path)
    store = get_state_store(root, backend='firestore', firestore_client=client)
    before = copy.deepcopy(client.docs)
    result = visitor.process_event(root, 'alice', f'game:move:connect4:col:1:turn=1:{suffix}',
        issue_number=905, timestamp_epoch=BASE, repo_slug='owner/sandbox', store=store)
    assert result['status'] == 'invalid_round_lock' and result['accepted'] is False
    assert client.docs == before


def test_public_readme_links_bind_round_and_turn_and_pass_strict_release_gate(tmp_path, monkeypatch):
    from scripts.build_readme import check_all_readmes
    from tests.test_publication_release_gate import bundle
    monkeypatch.setenv('STATE_STORE_BACKEND', 'json')
    bundle(tmp_path)
    state = json.loads((tmp_path/'data/game_state.json').read_text())
    state.update(turn=3, mode='VS_BOT', status='ACTIVE')
    (tmp_path/'data/game_state.json').write_text(json.dumps(state))
    legacy=tmp_path/'data/connect4_state.json'
    if legacy.exists():legacy.write_text(json.dumps(state))
    from scripts.build_readme import build_all_readmes
    build_all_readmes(tmp_path)
    path = tmp_path/'README.md'
    md = path.read_text()
    game_titles = [t for t in titles(md) if t.startswith(('game:move:', 'game:mode:'))]
    assert len(game_titles) == 7
    lock = f":turn={state['turn']}:round={state['round_id']}"
    assert all(t.endswith(lock) for t in game_titles)
    assert check_all_readmes(tmp_path)
    report = release.validate_release_bundle(tmp_path, require_publication_manifest=True)
    assert report['ok'], report['errors']
    path.write_text(md.replace(f":round={state['round_id']}", ':round=9999', 1))
    rejected = release.validate_release_bundle(tmp_path, require_publication_manifest=True)
    assert rejected['ok'] is False and any('round' in e for e in rejected['errors'])
