#!/usr/bin/env python3
"""Finalize a sanitized daily receipt; never dispatch providers or recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.zeitgeist.daily_outcomes import complete_outcome


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    args=parser.parse_args(argv)
    outcome=complete_outcome(args.repo_root,os.environ)
    print(json.dumps({key:outcome[key] for key in ('status','stage','canonical_commit','git_publication','recovery_action')},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
