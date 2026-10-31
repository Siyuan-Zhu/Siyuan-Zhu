"""Incomplete-run reporter never executes upstream code or guesses acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('previous,preserved',[
    ('',False),
    ('Saved to the canonical game. House wins. README publication is pending.',True),
])
def test_incomplete_workflow_reports_bounded_issue_identity_without_replay(tmp_path,monkeypatch,previous,preserved):
    text=(ROOT/'.github/workflows/visitor_workflow_feedback.yml').read_text()
    script=text.split('        run: |\n',1)[1]
    script='\n'.join(line[10:] for line in script.splitlines())
    event=tmp_path/'event.json'
    run=dict(id=123,event='issues',head_repository={'full_name':'Siyuan-Zhu/readme-live-sandbox'},repository={'full_name':'Siyuan-Zhu/readme-live-sandbox'},
             display_title='Game issue #44',html_url='https://github.com/Siyuan-Zhu/readme-live-sandbox/actions/runs/123')
    event.write_text(json.dumps({'workflow_run':run}))
    monkeypatch.setenv('GITHUB_EVENT_PATH',str(event));monkeypatch.setenv('GITHUB_REPOSITORY',run['repository']['full_name'])
    import subprocess
    calls=[]
    def command(args,**kwargs):
        calls.append(args)
        if args[:2]==['gh','api']:
            content=([{'user':{'login':'github-actions[bot]'},'body':previous}] if 'comments?' in args[2] else {'number':44,'title':'game:move:connect4:col:1:turn=1:round=1'})
            return SimpleNamespace(stdout=json.dumps(content))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',command)
    exec(compile(script,'visitor-cancel-reporter','exec'),{})
    body=calls[-1][-1]
    assert ('Preserve the saved move' in body) is preserved
    assert ('Acceptance is unconfirmed' in body) is (not preserved)
    assert all(args[0]=='gh' for args in calls)
    assert '--edit-last' not in calls[-1]
    assert 'actions/checkout' not in text and 'id-token: write' not in text
