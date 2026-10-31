"""Release regressions exercise the actual canonical visitor transaction."""
import copy

import pytest

from src import game_engine as game
from src.state_store import _apply_game_rate_limit
from tests.test_atomic_game_events import setup_event, event
from tests.test_canonical_game_continuity import BASE


def test_next_round_archives_final_position_without_resettling(tmp_path):
    root, client, store = setup_event(tmp_path)
    finished = event(root, store, title='game:move:connect4:col:4:turn=4:round=7')
    assert finished['accepted'] and finished['winner'] == 'red'
    previous = copy.deepcopy(client.docs['profile_state/active_game'])
    ledger = copy.deepcopy(client.docs['profile_state/elo_ledger'])
    assert event(root, store, issue=102, actor='bob',
                 title='game:move:connect4:col:1:turn=5:round=7', at=BASE+120)['accepted']
    current = client.docs['profile_state/active_game']
    archived = next(row for row in current['past_rounds'] if row['round_id']==7)
    assert archived['board'] == previous['board']
    assert archived['winner']=='red' and archived['finisher_handle']=='alice'
    assert archived['moves_count']==7 and archived['winning_cells']==previous['winning_cells']
    assert current['round_id']==8 and current['moves_this_round']==2
    assert client.docs['profile_state/elo_ledger']==ledger


def test_failed_settlement_aborts_every_canonical_document(tmp_path,monkeypatch):
    root, client, store = setup_event(tmp_path)
    before = copy.deepcopy(client.docs)
    def fail(*args,**kwargs):
        raise RuntimeError('synthetic settlement failure')
    monkeypatch.setattr('src.leaderboard.elo_ledger.settle_round',fail)
    with pytest.raises(RuntimeError,match='settlement'):
        event(root,store)
    assert client.docs==before


def test_third_admitted_round_continues_but_fourth_start_is_limited():
    rate={}
    for index in range(3):
        allowed,_,rate=_apply_game_rate_limit(rate,'alice',BASE+index*120,is_new_round=True)
        assert allowed
    allowed,_,_=_apply_game_rate_limit(rate,'alice',BASE+400,is_new_round=False)
    assert allowed,'a game-start cap must not strand the admitted third game'
    allowed,reason,_=_apply_game_rate_limit(rate,'alice',BASE+500,is_new_round=True)
    assert not allowed and reason=='daily_game_limit_exceeded'


def test_one_heading_real_history_and_simple_rating_table():
    state=game.create_initial_game_state(round_id=1003)
    state.update(status='COMPLETED',round_over=True,winner='bot',finisher_handle='House-Bot',
                 moves_this_round=22,winning_cells=[[5,c] for c in range(4)])
    state['board'][5][:4]=[2]*4
    md=game.render_readme_game_section(state)
    assert 'connect_four_title_light.webp' in md
    assert 'Next round: choose a column.' in md
    assert 'Completed rounds' in md and '| 1003' in md
    svg=game.render_board_svg(state)
    assert 'THE DAILY COLUMN' not in svg and '+0 ELO' not in svg
    assert 'CONNECTS 4 IN' not in svg


def test_winning_ack_identifies_outcome_and_stands_down_bot(tmp_path):
    root,client,store=setup_event(tmp_path)
    response=event(root,store)
    assert response['round_over'] and response['winner']=='red'
    assert 'Congratulations' in response['message']
    assert client.docs['profile_state/active_game']['moves_this_round']==7


def test_missing_legacy_outcome_never_fabricates_a_winner():
    from src.game_records import history_markdown
    state=game.create_initial_game_state()
    state['past_rounds']=[{'round_id':42,'winner':None}]
    text=history_markdown(state)
    assert 'Outcome unavailable' in text and 'won' not in text


def test_leaderboard_preserves_existing_rating_tie_rules():
    from src.game_records import leaderboard_markdown
    text=leaderboard_markdown({'players':{'alice':{'elo':800,'wins':0},'bob':{'elo':800,'wins':2}},'bot_elo':800})
    assert text.index('@bob')<text.index('@alice')


@pytest.mark.parametrize('missing',['elo_ledger','bot_memory','game_rate_limits'])
def test_missing_canonical_domain_never_resets_rewards_or_limits(tmp_path,missing):
    root,client,store=setup_event(tmp_path)
    client.docs.pop('profile_state/'+missing)
    before=copy.deepcopy(client.docs)
    with pytest.raises(RuntimeError,match='incomplete'):
        event(root,store)
    assert client.docs==before


@pytest.mark.parametrize('document',['elo_ledger','bot_memory','game_rate_limits'])
def test_incomplete_existing_domain_cannot_mint_ratings_or_reset_limits(tmp_path,document):
    root,client,store=setup_event(tmp_path)
    client.docs['profile_state/'+document]={'state_version':99}
    before=copy.deepcopy(client.docs)
    with pytest.raises(RuntimeError,match='incomplete'):
        event(root,store)
    assert client.docs==before


@pytest.mark.parametrize('mode',['VS_BOT','PVP_DUEL'])
def test_full_board_draw_has_truthful_feedback_and_no_extra_turn(tmp_path,mode):
    root,client,store=setup_event(tmp_path)
    state=game.create_initial_game_state(round_id=9)
    state['board']=[[1,1,2,2,1,1,2] if r%2==0 else [2,2,1,1,2,2,1] for r in range(6)]
    assert not any(game.has_won_bitboard(bits) for bits in game.encode_bitboard(state['board']))
    state.update(mode=mode,turn=21 if mode=='VS_BOT' else 42,
                 moves_this_round=40 if mode=='VS_BOT' else 41,
                 red_contributors={'alice':20 if mode=='VS_BOT' else 21},
                 yellow_contributors={} if mode=='VS_BOT' else {'bob':20})
    state['board'][0][2]=0
    if mode=='VS_BOT':state['board'][0][0]=0
    client.docs['profile_state/active_game']=state
    result=event(root,store,actor='alice' if mode=='VS_BOT' else 'bob',
                 title=f'game:move:connect4:col:{1 if mode=="VS_BOT" else 3}:turn={state["turn"]}:round=9')
    assert result['accepted'] and result['winner']=='draw' and 'Round drawn.' in result['message']
    assert client.docs['profile_state/active_game']['moves_this_round']==42
    assert client.docs['profile_state/bot_memory']['games_learned']==1


def test_house_win_records_its_actual_delta_and_preserves_human_loss(tmp_path):
    root,client,store=setup_event(tmp_path)
    state=game.create_initial_game_state(round_id=9)
    state['board'][5]=[2,2,2,0,1,1,1]
    state.update(turn=4,moves_this_round=6,red_contributors={'alice':3})
    client.docs['profile_state/active_game']=state
    result=event(root,store,title='game:move:connect4:col:5:turn=4:round=9')
    ledger=client.docs['profile_state/elo_ledger']
    assert result['winner']=='bot' and 'House - Bot wins.' in result['message']
    assert ledger['bot_elo']==1203 and ledger['players']['alice']['elo']==797
    assert ledger['players']['alice']['losses']==1 and ledger['recent_rounds'][0]['house_delta']==3


def test_draw_on_house_takeover_is_settled_once(tmp_path):
    root,client,store=setup_event(tmp_path)
    state=game.create_initial_game_state(round_id=9)
    state['board']=[[1,1,2,2,1,1,2] if r%2==0 else [2,2,1,1,2,2,1] for r in range(6)]
    state['board'][0][2]=0
    state.update(mode='PVP_DUEL',turn=42,current_turn_color='YELLOW',moves_this_round=41,
                 red_contributors={'alice':21},yellow_contributors={'bob':20})
    client.docs['profile_state/active_game']=state
    result=event(root,store,actor='carol',title='game:mode:connect4:new_vs_bot:turn=42:round=9')
    assert result['accepted'] and result['winner']=='draw' and 'Round drawn.' in result['message']
    assert client.docs['profile_state/bot_memory']['games_learned']==1
    assert len(client.docs['profile_state/elo_ledger']['recent_rounds'])==1


def test_bot_display_projection_preserves_record_identity():
    import copy
    from src.game_records import display_bot_copy, leaderboard_markdown, history_markdown
    from src.game_engine import create_initial_game_state, render_readme_game_section
    ledger = {'players': {'House': {'elo': 1201}}, 'bot_elo': 1200}
    before = copy.deepcopy(ledger)
    md = leaderboard_markdown(ledger)
    assert '| @House | 1201 |' in md and '| House - Bot | 1200 |' in md
    assert ledger == before
    assert display_bot_copy('House took over; @House; House-Bot; House - Bot') == 'House - Bot took over; @House; House - Bot; House - Bot'
    state = create_initial_game_state()
    state['last_move_summary'] = 'House took over Charcoal'
    state['move_log'] = [{'turn': 1, 'player': 'House-Bot', 'summary': 'House took over'}]
    before = copy.deepcopy(state)
    md = render_readme_game_section(state, elo_ledger=ledger)
    assert 'House - Bot took over' in md and '`@House-Bot`' not in md
    assert state == before
