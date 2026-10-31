"""Presentation changes keep vote actions and state-derived game controls intact."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
import pytest
from scripts.build_readme import render_profile_markdown
from src.state_store import InMemoryStateStore
from src.game_engine import create_initial_game_state,render_readme_game_section,render_board_svg

ROOT=Path(__file__).resolve().parents[1]
class DisclosureParser(HTMLParser):
    def __init__(self):
        super().__init__();self.details=[];self.stack=[];self.summary=False;self.anchors=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='details':
            row={'summary':'','text':'','attrs':attrs};self.details.append(row);self.stack.append(row)
        if tag=='summary':self.summary=True
        if tag=='a':self.anchors.append((attrs.get('href',''),len(self.stack),self.summary))
    def handle_endtag(self,tag):
        if tag=='summary':self.summary=False
        if tag=='details':self.stack.pop()
    def handle_data(self,data):
        if self.stack:self.stack[-1]['text']+=data
        if self.summary:self.stack[-1]['summary']+=data


def test_resonance_only_toggles_attribution_and_vote_payloads_are_unchanged():
    store=InMemoryStateStore();store.save_game_state(create_initial_game_state())
    md=render_profile_markdown(ROOT,state_store=store,write_assets=False,repo_slug='owner/private')
    p=DisclosureParser();p.feed(md)
    assert not any(d['summary']=='RESONANCE' for d in p.details)
    assert 'docs/resonance.md' in md
    assert md.count('RESONANCE')==1
    votes=[(url,depth,summary) for url,depth,summary in p.anchors if 'title=pulse:vote:' in url]
    assert len(votes)==3
    for (url,depth,summary),choice in zip(votes,['spot_on','too_sunny','too_heavy']):
        assert depth==0 and not summary
        assert urlsplit(url).path=='/owner/private/issues/new'
        assert parse_qs(urlsplit(url).query)=={'title':['pulse:vote:'+choice], 'body':["Click 'Submit new issue' to cast your mood vote."]}



@pytest.mark.parametrize('mode,turn,side',[('VS_BOT',1,'Terracotta'),('VS_BOT',3,'Terracotta'),('PVP_DUEL',1,'Terracotta'),('PVP_DUEL',2,'Charcoal')])
def test_roles_and_turn_are_below_board_above_links_with_true_help(mode,turn,side):
    state=create_initial_game_state();state.update(mode=mode,turn=turn)
    md=render_readme_game_section(state);svg=render_board_svg(state)
    assert md.index('game_board.svg')<md.index(side+' to play.')<md.index('[I](')
    assert 'Mode:' not in md
    assert 'Terracotta · Visitors' in svg
    assert ('Charcoal · Visitors' if mode=='PVP_DUEL' else 'Charcoal · House') in svg
    p=DisclosureParser();p.feed(md)
    assert not any(d['summary']=='How to play' for d in p.details)
    if mode=='PVP_DUEL':
        assert '>Open a duel<' not in md
    else:
        assert ('>Open a duel<' in md)==(turn<=2)
        if turn<=2:
            url=next(url for url,_,_ in p.anchors if 'new_pvp' in url)
            assert parse_qs(urlsplit(url).query)['body']==["Just click 'Submit new issue' to switch this shared board to PvP Duel (Terracotta vs. Charcoal). This is not a private room."]


def test_completed_and_paused_guidance_stays_with_controls_without_enabling_false_actions():
    state=create_initial_game_state();state.update(status='COMPLETED',round_over=True,winner='bot')
    md=render_readme_game_section(state)
    assert md.index('game_board.svg')<md.index('House - Bot wins.')<md.index('Next round: choose a column.')<md.index('[I](')
    paused=render_readme_game_section(state,enabled=False)
    assert 'Game paused.' in paused and 'Next round: choose a column.' not in paused
    assert 'issues/new' not in paused and 'to play.' not in paused
