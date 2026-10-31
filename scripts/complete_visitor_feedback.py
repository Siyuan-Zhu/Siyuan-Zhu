"""Distinguish accepted canonical moves from their later Git publication."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.process_visitor_event import _auto_close_and_lock_github_issue


def finalize(result, *, published, issue_number, repo):
    if not (result.get('accepted') or result.get('previously_accepted')):
        return
    outcome = result.get('message') or 'This Issue was previously accepted.'
    status = ('The board and history were published.' if published else
              'README publication failed after the move was saved. Do not submit it again; an owner can rebuild recorded state without replaying the move.')
    _auto_close_and_lock_github_issue(issue_number,
        f'Saved to the canonical game. {outcome}\n\n{status}\n\n[Current board](https://github.com/{repo}#connect-four).',repo_slug=repo)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-file',type=Path,required=True)
    parser.add_argument('--published',choices=('true','false'),required=True)
    args=parser.parse_args()
    if args.result_file.exists():
        finalize(json.loads(args.result_file.read_text()),published=args.published=='true',
                 issue_number=int(os.environ.get('ISSUE_NUMBER','0')),repo=os.environ['GITHUB_REPOSITORY'])


if __name__=='__main__':
    main()
