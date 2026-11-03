"""Publish generated profile changes, propagating failure to GitHub Actions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state_store import git_commit_and_push_with_retry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    from src.zeitgeist.edition_approval import require_edition_approval
    manifest = args.repo_root / 'assets/pulse_manifest.json'
    if manifest.exists():
        require_edition_approval(json.loads(manifest.read_text()))
    if not git_commit_and_push_with_retry(args.message, repo_root=args.repo_root):
        print("Profile publication failed; reconciliation is required.", file=sys.stderr)
        return 1
    if os.environ.get('GITHUB_WORKFLOW') == 'Daily Zeitgeist' or os.environ.get('ZEITGEIST_TRACK_EXECUTION') == '1':
        from src.state_store import get_state_store
        from src.zeitgeist.edition_execution import for_store
        store = get_state_store(repo_root=args.repo_root, backend='firestore',
            collection_prefix=os.environ.get('FIRESTORE_COLLECTION_PREFIX',''))
        data = json.loads(manifest.read_text())
        execution = for_store(store, data['date'], os.environ)
        execution.transition('published', publication_id=data['publication_id'])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
