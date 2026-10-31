"""Measure bounded bot work on reproducible legal fixtures, without services."""
import json
from pathlib import Path
import platform
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.game_engine import choose_bot_move, encode_bitboard, has_won_bitboard


def positions():
    rng=random.Random(20261031)
    result=[]
    for attempt in range(250):
        count=(1,7,13,19,25)[attempt%5]
        board=[[0]*7 for _ in range(6)]
        legal=True
        for move in range(count):
            choices=[c for c in range(7) if board[0][c]==0]
            if not choices:
                legal=False;break
            col=rng.choice(choices)
            row=max(r for r in range(6) if board[r][col]==0)
            board[row][col]=move%2+1
            if any(has_won_bitboard(bits) for bits in encode_bitboard(board)):
                legal=False;break
        if legal:
            result.append(board)
        if len(result)==15:
            return result
    raise RuntimeError('Fixture generation did not reach its bounded sample')


def benchmark():
    rows=[]
    for board in positions():
        stats={}
        column=choose_bot_move(board,stats=stats)
        assert board[0][column]==0 and stats['nodes']<=50000
        rows.append(dict(stats,discs=sum(bool(v) for row in board for v in row)))
    times=[row['elapsed_seconds'] for row in rows]
    return dict(environment=platform.system()+' '+platform.machine(),python=platform.python_version(),
        kind='in_process_search_fixture_only',samples=len(rows),median_seconds=statistics.median(times),
        maximum_seconds=max(times),node_limit=50000,nominal_time_budget_seconds=1.5,
        exhausted=sum(row['budget_exhausted'] for row in rows),positions=rows)


if __name__=='__main__':
    print(json.dumps(benchmark(),sort_keys=True))
