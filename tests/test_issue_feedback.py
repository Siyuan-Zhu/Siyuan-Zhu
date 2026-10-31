"""Rejected commands get human feedback; API errors cannot look successful."""
from types import SimpleNamespace
import copy
import pytest
from scripts import process_visitor_event as visitor
from tests.test_atomic_game_events import setup_event, event


def test_stale_firestore_move_is_commented_closed_and_locked_without_canonical_mutation(tmp_path,monkeypatch):
    root,client,store=setup_event(tmp_path); baseline=copy.deepcopy(client.docs); commands=[]
    monkeypatch.setenv('GITHUB_ACTIONS','true'); monkeypatch.setenv('GITHUB_REPOSITORY','owner/sandbox')
    monkeypatch.setenv('GH_TOKEN','synthetic')
    monkeypatch.setattr(visitor.subprocess,'run',lambda command,**kw:commands.append(command) or SimpleNamespace(returncode=0))
    result=event(root,store,title='game:move:connect4:col:4:turn=1')
    assert result['status']=='stale_turn' and client.docs==baseline
    assert [command[1:3] for command in commands]==[['issue','comment'],['issue','close'],['api','-X']]
    assert 'stale_turn' in commands[0][-1]


@pytest.mark.parametrize('failed_step',[0,1,2])
def test_feedback_failure_propagates_and_stops_following_steps(tmp_path,monkeypatch,failed_step):
    monkeypatch.setenv('GITHUB_ACTIONS','true'); monkeypatch.setenv('GITHUB_REPOSITORY','owner/sandbox')
    monkeypatch.setenv('GH_TOKEN','synthetic'); commands=[]
    def run(command,**kw):
        commands.append(command)
        return SimpleNamespace(returncode=1 if len(commands)-1==failed_step else 0)
    monkeypatch.setattr(visitor.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='Issue feedback'):
        visitor._auto_close_and_lock_github_issue(12,'Rejected stale move',repo_root=tmp_path)
    assert len(commands)==failed_step+1


def test_missing_gh_cli_fails_explicitly(tmp_path,monkeypatch):
    monkeypatch.setenv('GITHUB_ACTIONS','true'); monkeypatch.setenv('GH_TOKEN','synthetic')
    def unavailable(*a,**k): raise OSError('cannot launch')
    monkeypatch.setattr(visitor.subprocess,'run',unavailable)
    with pytest.raises(RuntimeError,match='Issue feedback'): visitor._auto_close_and_lock_github_issue(1,'feedback',repo_root=tmp_path)


def test_replayed_firestore_issue_gets_feedback_without_reward_replay(tmp_path,monkeypatch):
    root,client,store=setup_event(tmp_path); replies=[]
    monkeypatch.setattr(visitor,'_auto_close_and_lock_github_issue',lambda *a,**kw:replies.append(a))
    assert event(root,store)['accepted']
    baseline=copy.deepcopy(client.docs)
    assert event(root,store)['status']=='already_processed'
    assert client.docs==baseline and len(replies)==2
