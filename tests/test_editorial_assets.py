"""Stable build artifacts and native action/record boundaries, without pixel snapshots."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit
from src.editorial_typography import ASSETS, ATTRIBUTION, write_typography
from src.game_engine import create_initial_game_state, render_readme_game_section
from tests.test_integrated_profile_layout import DisclosureParser


def test_static_typography_is_stable_and_not_rewritten(tmp_path, monkeypatch):
    write_typography(tmp_path)
    first = {p:((tmp_path/p).read_bytes(),(tmp_path/p).stat().st_mtime_ns) for p in ASSETS}
    import src.editorial_typography as typography
    def forbidden(*args):
        raise AssertionError("unchanged assets must not be rasterized")
    monkeypatch.setattr(typography, "_render", forbidden)
    write_typography(tmp_path)
    assert first == {p:((tmp_path/p).read_bytes(),(tmp_path/p).stat().st_mtime_ns) for p in ASSETS}
    assert first[ASSETS[0]][0] != first[ASSETS[1]][0]
    assert (Path(__file__).resolve().parents[1]/'docs/resonance.md').read_text()==ATTRIBUTION


def test_records_are_closed_and_outside_centered_controls_and_payloads_survive():
    state=create_initial_game_state(round_id=82)
    md=render_readme_game_section(state)
    p=DisclosureParser();p.feed(md)
    assert [r['summary'] for r in p.details]==['Recent moves','Completed rounds','Leaderboard']
    assert all('open' not in r['attrs'] for r in p.details)
    assert md.index('</div>')<md.index('last move:')<md.index('<details>')
    assert 'id="connect-four"' in md and '<h4>' in md
    assert 'How to play' not in md
    import re
    commands=re.findall(r'https://github.com/[^\s)]+',md)
    moves=[parse_qs(urlsplit(url).query) for url in commands if 'game:move:' in url]
    assert len(moves)==7
    for col,q in enumerate(moves,1):
        assert q['title']==[f'game:move:connect4:col:{col}:turn=1:round=82']
        assert 'Processing is asynchronous.' in q['body'][0] and 'House - Bot is automated' in q['body'][0]
