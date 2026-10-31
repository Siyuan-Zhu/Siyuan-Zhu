"""Owner-only bounded recovery of one real Issue, using its actual trusted actor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.process_visitor_event import main as process


def recover(issue_number, *, env=os.environ):
    repo=env.get('GITHUB_REPOSITORY')
    if (env.get('GITHUB_EVENT_NAME')!='workflow_dispatch' or env.get('GITHUB_ACTOR')!='Siyuan-Zhu'
        or repo not in ('Siyuan-Zhu/readme-live-sandbox','Siyuan-Zhu/Siyuan-Zhu')
        or env.get('COMMUNITY_GAME_ENABLED')!='true'
        or not str(issue_number).isdigit() or not 0<int(issue_number)<2147483648):
        raise RuntimeError('Owner recovery of one enabled game Issue required')
    response=subprocess.run(['gh','api',f'repos/{repo}/issues/{int(issue_number)}'],
                            capture_output=True,text=True,check=True,timeout=30)
    issue=json.loads(response.stdout)
    if ('pull_request' in issue or issue.get('number')!=int(issue_number)
        or not str(issue.get('title','')).startswith(('game:move:','game:mode:','connect4: drop '))):
        raise RuntimeError('Recovery requires an actual game Issue')
    # process/apply_issue checks the permanent event receipt before any move or
    # settlement. A stale original command is rejected, never refreshed/reworded.
    return process(['--login',issue['user']['login'],'--title',issue['title'],
                    '--issue-number',str(issue_number),'--repo-slug',repo,
                    '--result-file',str(Path(env['RUNNER_TEMP'])/'visitor-result.json')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--issue-number',required=True)
    raise SystemExit(recover(parser.parse_args().issue_number))
