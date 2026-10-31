"""Public game outcomes and bounded completed-round records; no persistence."""
from __future__ import annotations

import copy
import re
from html import escape


BOT_DISPLAY_NAME = "House - Bot"


def display_bot_copy(text):
    """Project legacy prose without changing stored events or account handles."""
    return re.sub(r"(?<![@\w-])(?:House-Bot|House)(?![\w-]| - Bot)", BOT_DISPLAY_NAME, str(text))


def is_complete(state):
    return state.get('status') == 'COMPLETED' or bool(state.get('round_over'))


def outcome_text(state):
    if not is_complete(state):
        charcoal = state.get('mode') == 'PVP_DUEL' and int(state.get('turn',1)) % 2 == 0
        return 'Charcoal to play.' if charcoal else 'Terracotta to play.'
    winner = state.get('winner')
    finisher = str(state.get('finisher_handle') or '').lstrip('@')
    if winner == 'draw':
        return 'Round drawn.'
    if winner in ('bot','house_bot') and state.get('mode') != 'PVP_DUEL':
        return f'{BOT_DISPLAY_NAME} wins.'
    if winner not in ('red','community','challengers','yellow','bot','house_bot'):
        return 'Outcome unavailable; awaiting recovery.'
    side = 'Charcoal' if winner in ('yellow','bot','house_bot') else 'Terracotta'
    return f'{side} wins. Congratulations, @{finisher}!' if finisher else f'{side} wins.'


def completed_record(state):
    """Archive only the known active round, never infer legacy round identities."""
    if not is_complete(state) or type(state.get('round_id')) is not int:
        raise ValueError('Completed round identity required')
    from src.game_engine import _winning_coordinates
    return {key: copy.deepcopy(value) for key,value in {
        'round_id': state['round_id'], 'game_key': state.get('active_game','connect4'),
        'mode': state.get('mode','VS_BOT'), 'winner': state.get('winner'),
        'finisher_handle': state.get('finisher_handle'),
        'moves_count': state.get('moves_this_round',0), 'board': state['board'],
        'winning_cells': [list(pair) for pair in _winning_coordinates(state.get('winning_cells',[]))],
        'ended_at_epoch': state.get('last_move_epoch') or state.get('last_move_timestamp'),
        'moves': state.get('move_history') or state.get('move_log') or [],
    }.items()}


def archive_completed_round(state):
    record = completed_record(state)
    history = copy.deepcopy(state.get('past_rounds') or [])
    existing = next((row for row in history if row.get('round_id')==record['round_id']),None)
    if existing is not None and existing != record:
        raise ValueError('Completed round conflicts with existing archive')
    if existing is None:
        history.append(record)
    state['past_rounds'] = history[-20:]


def history_markdown(state):
    history = copy.deepcopy(state.get('past_rounds') or [])
    if is_complete(state) and not any(row.get('round_id')==state['round_id'] for row in history):
        history.append(completed_record(state))
    rows = []
    for record in reversed(history[-5:]):
        rid = record.get('round_id')
        if type(rid) is not int:
            continue
        winner = record.get('winner')
        result = 'Draw' if winner=='draw' else (
            f'{BOT_DISPLAY_NAME} won' if winner in ('bot','house_bot') and record.get('mode')!='PVP_DUEL'
            else 'Charcoal won' if winner=='yellow'
            else 'Terracotta won' if winner in ('red','community','challengers')
            else 'Outcome unavailable')
        finisher = record.get('finisher_handle')
        if finisher:
            result += (' · ' + BOT_DISPLAY_NAME) if winner in ('bot', 'house_bot') and record.get('mode') != 'PVP_DUEL' and finisher == 'House-Bot' else f' · @{escape(str(finisher).lstrip("@"))}'
        count = record.get('moves_count')
        rows.append(f'| {rid} | {result} | {count if type(count) is int else "—"} |')
    content = '\n'.join(['| Round | Result | Discs |','| :---: | :--- | ---: |',*rows]) if rows else '_No completed rounds yet._'
    return '<details>\n<summary><sub>Completed rounds</sub></summary>\n\n'+content+'\n\n</details>\n\n'


def leaderboard_markdown(ledger):
    from src.leaderboard.elo_ledger import get_sorted_leaderboard
    ledger = ledger or {}
    players = ledger.get('players') or {}
    recorded = {key:row for key,row in players.items()
                if isinstance(row,dict) and type(row.get('elo')) is int} if isinstance(players,dict) else {}
    leaders = get_sorted_leaderboard({'players':recorded})
    bot_rating = ledger.get('bot_elo')
    if type(bot_rating) is int:
        # Preserve the established wins/moves/login ordering of tied humans.
        position = next((i for i,row in enumerate(leaders) if row['elo']<bot_rating),len(leaders))
        leaders.insert(position,{'github_login':'House','elo':bot_rating,'_bot':True})
    rows = [f'| {BOT_DISPLAY_NAME if row.get("_bot") else "@"+escape(row["github_login"])} | {row["elo"]} |' for row in leaders[:5]]
    content = '\n'.join(['| Player | Rating |','| :--- | ---: |',*rows]) if rows else '_No recorded ratings yet._'
    return '<details>\n<summary><sub>Leaderboard</sub></summary>\n\n'+content+'\n\n</details>\n\n'
