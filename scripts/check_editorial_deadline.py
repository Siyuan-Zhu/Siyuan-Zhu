#!/usr/bin/env python3
"""Observe public completion, not trigger acceptance; never generates or publishes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.zeitgeist.edition_approval import HELD_EDITIONS


def assess(manifest, now):
    local=now.astimezone(ZoneInfo('America/New_York'));edition=local.date().isoformat()
    if local.hour<4:return {'status':'before_deadline','edition_date':edition}
    if edition in HELD_EDITIONS and not HELD_EDITIONS[edition]:
        return {'status':'approval_pending','edition_date':edition,'publication_expected':False}
    return {'status':'complete' if manifest.get('date')==edition else 'publication_missing',
        'edition_date':edition,'publication_id':manifest.get('publication_id'),
        'published_date':manifest.get('date')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args()
    result=assess(json.loads(args.manifest.read_text()),datetime.now(timezone.utc))
    print(json.dumps(result));sys.exit(2 if result['status']=='publication_missing' else 0)
