"""Actual workflow gating and visitor controls at the reported regression boundaries."""
from pathlib import Path
from types import SimpleNamespace
import html
from urllib.parse import parse_qs, urlsplit
import pytest
from src.game_engine import create_initial_game_state, render_readme_game_section, apply_visitor_move
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('game,editorial,title,expected', [(True, False, 'game:move:connect4:col:1:turn=1:round=1003', True), (True, False, 'game:mode:connect4:new_pvp:turn=1:round=1003', True), (False, True, 'game:move:connect4:col:1:turn=1:round=1003', False), (False, False, 'game:move:connect4:col:1:turn=1:round=1003', False), (True, False, 'pulse:vote:spot_on', False), (True, False, 'orbit:word:Test', False), (False, True, 'pulse:vote:spot_on', True), (True, False, 'ordinary issue', False)])
def test_game_switch_is_independent_and_does_not_enable_editorial_events(game, editorial, title, expected):
    text = (ROOT / '.github/workflows/community_game_move.yml').read_text()
    condition = text.split('    if: >-\n', 1)[1].split('    runs-on:', 1)[0]
    expression = ' '.join(condition.split()).replace('&&', ' and ').replace('||', ' or ')
    value = eval(expression, {'__builtins__': {}}, {'vars': SimpleNamespace(COMMUNITY_GAME_ENABLED=str(game).lower(), ZEITGEIST_WRITER_ENABLED=str(editorial).lower()), 'github': SimpleNamespace(event_name='issues', event=SimpleNamespace(issue=SimpleNamespace(title=title))), 'startsWith': lambda value, prefix: value.startswith(prefix)})
    assert value is expected

@pytest.mark.parametrize('turn,offered', [(1, True), (2, True), (3, False)])
def test_duel_button_tracks_actual_engine_switch_boundary(turn, offered):
    state = create_initial_game_state(round_id=1003)
    state['turn'] = turn
    text = render_readme_game_section(state, repo_slug='Siyuan-Zhu/readme-live-sandbox')
    assert ('>Open a duel<' in text) is offered
    result = apply_visitor_move(state, actor='alice', action='mode:new_pvp', now_epoch=1900000000, expected_turn=turn)
    assert result.accepted is offered
    if not offered:
        assert result.status_code == 'invalid_mode_switch' and 'shared' not in result.message.lower()

def test_pvp_instructions_use_visible_charcoal_and_explain_shared_board():
    state = create_initial_game_state()
    state.update(mode='PVP_DUEL', turn=2, current_turn_color='YELLOW')
    text = render_readme_game_section(state, repo_slug='Siyuan-Zhu/readme-live-sandbox')
    from urllib.parse import unquote_plus
    assert 'shared board' in unquote_plus(text).casefold() and 'private room' in unquote_plus(text).casefold()
    from scripts.validate_release_links import ISSUE_NEW_URL_RE
    bodies = [parse_qs(html.unescape(query)).get('body', [''])[0] for _, query in ISSUE_NEW_URL_RE.findall(text) if 'game:move:' in query]
    assert len(bodies) == 7 and all(('Charcoal disc' in body for body in bodies))
    assert all(('Red vs. Yellow' not in body for body in bodies))

def test_release_link_audit_accepts_turn_three_without_duel_and_rejects_injected_action(tmp_path, monkeypatch):
    from tests.test_publication_release_gate import bundle
    from scripts.build_readme import build_all_readmes
    from scripts.validate_release_links import validate_release_bundle
    import json
    monkeypatch.setenv('STATE_STORE_BACKEND', 'json')
    bundle(tmp_path)
    state = create_initial_game_state(round_id=1003)
    state['turn'] = 3
    for index, row in enumerate(state['board']):
        row[0] = index % 2 + 1
    (tmp_path / 'data/game_state.json').write_text(json.dumps(state))
    legacy = tmp_path / 'data/connect4_state.json'
    if legacy.exists():
        legacy.write_text(json.dumps(state))
    build_all_readmes(tmp_path, repo_slug='Siyuan-Zhu/Siyuan-Zhu')
    report = validate_release_bundle(tmp_path, require_publication_manifest=True)
    assert report['ok'], report['errors']
    assert report['issue_links_count'] == 9
    p = tmp_path / 'README.md'
    p.write_text(p.read_text() + '\n[Invalid duel](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=game:mode:connect4:new_pvp:turn=3:round=1003)')
    assert not validate_release_bundle(tmp_path, require_publication_manifest=True)['ok']

def test_completed_even_pvp_restart_promises_the_terracotta_disc_the_engine_places():
    state = create_initial_game_state(round_id=1003)
    state.update(mode='PVP_DUEL', turn=8, status='COMPLETED', round_over=True, current_turn_color='YELLOW')
    text = render_readme_game_section(state, repo_slug='Siyuan-Zhu/readme-live-sandbox')
    from scripts.validate_release_links import ISSUE_NEW_URL_RE
    bodies = [parse_qs(html.unescape(query)).get('body', [''])[0] for _, query in ISSUE_NEW_URL_RE.findall(text) if 'game:move:' in query]
    assert len(bodies) == 7 and all(('Round 1004' in body and 'Terracotta disc' in body for body in bodies))
    result = apply_visitor_move(state, actor='alice', action='col:1', now_epoch=1900000000, expected_turn=8)
    assert result.accepted and result.new_state['round_id'] == 1004 and (result.new_state['board'][5][0] == 1)


@pytest.mark.parametrize('turn,status,delay', [(1,'ACTIVE',False),(2,'ACTIVE',True),(8,'COMPLETED',False)])
def test_takeover_delay_copy_matches_enforced_round_state(turn,status,delay):
    state=create_initial_game_state()
    state.update(mode='PVP_DUEL',turn=turn,status=status)
    text=render_readme_game_section(state,repo_slug='Siyuan-Zhu/readme-live-sandbox')
    assert ('after 15 minutes idle' in text) is delay
