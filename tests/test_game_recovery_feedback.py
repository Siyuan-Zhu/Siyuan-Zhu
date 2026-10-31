"""Owner recovery cannot substitute actor/title or claim an unpublished board."""
import json
from types import SimpleNamespace
import pytest
from scripts import complete_visitor_feedback as feedback
from scripts import recover_visitor_issue as recovery


@pytest.mark.parametrize('published',[False,True])
def test_feedback_distinguishes_saved_move_from_published_board(monkeypatch,published):
    calls=[]
    monkeypatch.setattr(feedback,'_auto_close_and_lock_github_issue',lambda *a,**k:calls.append(a))
    feedback.finalize({'accepted':True,'message':'House wins.'},published=published,issue_number=44,repo='Siyuan-Zhu/readme-live-sandbox')
    text=calls[0][1]
    assert 'House wins.' in text and 'Saved to the canonical game' in text
    assert ('board and history were published' in text) is published
    assert ('Do not submit it again' in text) is (not published)
    calls.clear()
    feedback.finalize({'accepted':False,'status':'state_conflict'},published=True,issue_number=44,repo='x/y')
    assert not calls


def test_recovery_uses_one_fetched_issue_actor_title_and_stable_identity(tmp_path,monkeypatch):
    env=dict(GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_ACTOR='Siyuan-Zhu',
             GITHUB_REPOSITORY='Siyuan-Zhu/readme-live-sandbox',COMMUNITY_GAME_ENABLED='true',RUNNER_TEMP=str(tmp_path))
    issue={'number':44,'title':'game:move:connect4:col:6:turn=12:round=1003','user':{'login':'actual-owner'}}
    calls=[]
    monkeypatch.setattr(recovery.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=json.dumps(issue)))
    monkeypatch.setattr(recovery,'process',lambda args:calls.append(args))
    recovery.recover('44',env=env)
    assert calls[0][1]=='actual-owner' and calls[0][3]==issue['title'] and calls[0][5]=='44'
    for key,value in [('GITHUB_ACTOR','someone-else'),('GITHUB_EVENT_NAME','issues'),('COMMUNITY_GAME_ENABLED','false')]:
        with pytest.raises(RuntimeError):recovery.recover('44',env=dict(env,**{key:value}))
    for identity in ('0','-1','44;echo secret','99999999999999999999'):
        with pytest.raises(RuntimeError):recovery.recover(identity,env=env)
