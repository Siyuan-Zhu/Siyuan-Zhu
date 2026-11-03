#!/usr/bin/env python3
"""Real isolated Firestore claim/readback; no providers or publication."""
from datetime import datetime,timedelta,timezone
import json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.state_store import _FirestoreRestClient
from src.zeitgeist.edition_execution import EditionExecution

now=datetime.now(timezone.utc);run=os.environ['GITHUB_RUN_ID']
client=_FirestoreRestClient('git-readme-features')
prefix='staging_reliability_'+run+'_'
a=EditionExecution(client,prefix=prefix,edition='2026-11-03',owner='probe1')
b=EditionExecution(client,prefix=prefix,edition='2026-11-03',owner='probe2')
a.claim(now=now,source_revision=os.environ['GITHUB_SHA'],trigger='private_probe')
try:b.claim(now=now,source_revision='other',trigger='duplicate')
except RuntimeError:pass
else:raise AssertionError('Duplicate claim accepted')
a.transition('generation',now=now)
try:b.claim(now=now+timedelta(hours=1),source_revision='other',trigger='expired')
except RuntimeError:pass
else:raise AssertionError('Ambiguous paid work reclaimed')
a.transition('awaiting_approval',now=now,publication_id='probe-only-no-artifact')
assert a.doc.get().to_dict()['stage']=='awaiting_approval'
print(json.dumps({'status':'verified_live','namespace':prefix,'duplicate_rejected':True,'expired_paid_work_rejected':True,'publication_calls':0,'provider_calls':0}))
