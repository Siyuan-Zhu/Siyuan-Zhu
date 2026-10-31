"""Recovery dispatch publishes canonical projections without replaying visitors."""
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('event,rebuild,actor,repo,expected', [
    ('workflow_dispatch', True, 'Siyuan-Zhu', 'Siyuan-Zhu/readme-live-sandbox', True),
    ('workflow_dispatch', True, 'Siyuan-Zhu', 'Siyuan-Zhu/Siyuan-Zhu', True),
    ('issues', True, 'Siyuan-Zhu', 'Siyuan-Zhu/readme-live-sandbox', False),
    ('workflow_dispatch', False, 'Siyuan-Zhu', 'Siyuan-Zhu/readme-live-sandbox', False),
    ('workflow_dispatch', True, 'someone-else', 'Siyuan-Zhu/readme-live-sandbox', False),
    ('workflow_dispatch', True, 'Siyuan-Zhu', 'Siyuan-Zhu/other', False),
])
def test_recovery_is_owner_manual_dispatch_without_game_replay(event,rebuild,actor,repo,expected):
    text = (ROOT/'.github/workflows/community_game_move.yml').read_text()
    job = text.split('  rebuild-recorded-profile:\n',1)[1].split('  recover-one-game-issue:',1)[0]
    condition = job.split('    if: >-\n',1)[1].split('    permissions:',1)[0]
    expression = ' '.join(condition.split()).replace('&&',' and ').replace('||',' or ')
    expression = expression.replace('true','True')
    assert eval(expression, {'__builtins__':{}}, {
        'github': SimpleNamespace(event_name=event,actor=actor,repository=repo),
        'inputs': SimpleNamespace(rebuild_only=rebuild)}) is expected
    assert 'issues: write' not in job
    assert 'process_visitor_event' not in job
    assert 'run_zeitgeist' not in job
    for secret in ('APIFY','GEMINI','BFL','UNSPLASH'):
        assert secret not in job
    assert 'scripts/check_runtime_configuration.py --visitor' in job
    assert 'scripts/build_readme.py' in job and 'scripts/publish_readme.py' in job
    assert "github.event_name == 'issues' &&" in text.split('  rebuild-recorded-profile:',1)[0]
    assert 'group: readme-state-writer' in text
