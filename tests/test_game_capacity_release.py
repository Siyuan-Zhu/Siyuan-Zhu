"""Bounded search and simultaneous canonical command attempts, without providers."""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from src import game_engine as game
from src.game_events import apply_issue
from src.state_store import FirestoreStateStore
from tests.test_atomic_game_events import setup_event
from tests.test_canonical_game_continuity import BASE


def test_search_budget_keeps_legal_fallback_and_immediate_win_and_block():
    board=[[0]*7 for _ in range(6)]
    stats={}
    assert game.choose_bot_move(board,max_nodes=0,stats=stats)==3
    assert stats['budget_exhausted'] and stats['nodes']==0
    board[5][:3]=[2]*3
    assert game.choose_bot_move(board,max_nodes=0)==3
    board[5][:3]=[1]*3
    assert game.choose_bot_move(board,max_nodes=0)==3
    board[5][3]=2
    stats={};before=copy.deepcopy(board)
    col=game.choose_bot_move(board,max_nodes=50,stats=stats)
    assert board[0][col]==0 and board==before and stats['nodes']<=50


@pytest.mark.parametrize('burst',[1,10,50])
@pytest.mark.parametrize('next_round',[False,True])
def test_burst_accepts_one_round_locked_event_and_one_settlement(tmp_path,burst,next_round):
    root,client,store=setup_event(tmp_path)
    if next_round:
        assert apply_issue(store,repository='owner/sandbox',issue_number=1,actor='alice',
            title='game:move:connect4:col:4:turn=4:round=7',now_epoch=BASE)[0]['accepted']
    initial=copy.deepcopy(client.docs)
    barrier=threading.Barrier(burst);lock=threading.Lock();original=client.commit_documents
    def atomic(writes):
        barrier.wait(timeout=30)
        with lock:
            original(writes)
    client.commit_documents=atomic
    def attempt(index):
        local=FirestoreStateStore(client=client,repo_root=tmp_path/str(index),canonical_only=True)
        title=('game:move:connect4:col:1:turn=5:round=7' if next_round else
               'game:move:connect4:col:4:turn=4:round=7')
        started=time.perf_counter()
        response=apply_issue(local,repository='owner/sandbox',issue_number=100+index,
                             actor=f'player-{index}',title=title,now_epoch=BASE+120)[0]
        return response,time.perf_counter()-started
    with ThreadPoolExecutor(max_workers=burst) as pool:
        results=list(pool.map(attempt,range(burst)))
    assert sum(result['accepted'] for result,_ in results)==1
    assert all(result['status'] in ('move_applied','state_conflict') for result,_ in results)
    current=client.docs['profile_state/active_game']
    assert current['round_id']==(8 if next_round else 7)
    assert client.docs['profile_state/bot_memory']['games_learned']==1
    assert len([key for key in client.docs if key.startswith('processed_events/game_issue_')])==(2 if next_round else 1)
    if next_round:
        assert client.docs['profile_state/elo_ledger']==initial['profile_state/elo_ledger']
        assert len([row for row in current['past_rounds'] if row['round_id']==7])==1

    # Optional private measurement capture uses the same actual transaction loop.
    import os, json, statistics, platform
    report=os.environ.get('EGM_CAPACITY_REPORT')
    if report:
        elapsed=sorted(seconds for _,seconds in results)
        with open(report,'a') as output:
            output.write(json.dumps(dict(burst=burst,next_round=next_round,
                environment=platform.system()+' '+platform.machine(),
                accepted=sum(result['accepted'] for result,_ in results),
                rejected=burst-1,median_seconds=statistics.median(elapsed),
                p95_seconds=elapsed[max(0,int(len(elapsed)*.95+.999)-1)],maximum_seconds=max(elapsed)))+'\n')
