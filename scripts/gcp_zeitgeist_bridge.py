#!/usr/bin/env python3
"""Temporary Cloud Run Job bridge; observes execution and public completion.

One dispatch only. Unknown acknowledgements are reconciled by run correlation,
never redispatched. GitHub remains the executor and sole profile writer queue.
"""
from datetime import datetime,timezone
import json,os,re,time
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from scripts.check_editorial_deadline import assess
from src.zeitgeist.edition_approval import HELD_EDITIONS

REPO='Siyuan-Zhu/Siyuan-Zhu'
API='https://api.github.com/repos/'+REPO
WORKFLOW='daily_zeitgeist.yml'


def run(client, *, correlation, now, clock=time.monotonic, sleep=time.sleep, monitor=False):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',correlation):raise ValueError('Invalid execution identity')
    manifest_url=API+'/contents/assets/pulse_manifest.json?ref=main'
    def observe():
        response=client.get(manifest_url,headers={'Accept':'application/vnd.github.raw+json'})
        response.raise_for_status();return response.json()
    before=observe();completion=assess(before,now)
    if monitor:
        print(json.dumps(dict(completion,kind='deadline_observation',correlation=correlation)))
        if completion['status']=='publication_missing':raise RuntimeError('Daily publication deadline missed')
        return completion
    edition=now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    if edition in HELD_EDITIONS and not HELD_EDITIONS[edition]:
        return {'status':'approval_pending','edition_date':edition,'kind':'dispatch_not_needed'}
    if before.get('date')==edition or completion['status']=='approval_pending':
        return dict(completion,kind='dispatch_not_needed')
    started=clock();dispatch_unknown=False
    try:
        response=client.post(API+'/actions/workflows/'+WORKFLOW+'/dispatches',json={
            'ref':'main','inputs':{'schedule_event_id':correlation}})
        if response.status_code>=500:dispatch_unknown=True
        else:response.raise_for_status()
    except httpx.TransportError:
        dispatch_unknown=True
    print(json.dumps({'kind':'dispatch_observation','edition_date':edition,'correlation':correlation,
                      'status':'unknown' if dispatch_unknown else 'accepted_not_completed'}))
    while clock()-started<1200:
        response=client.get(API+'/actions/workflows/'+WORKFLOW+'/runs',params={'event':'workflow_dispatch','per_page':100})
        response.raise_for_status()
        matches=[r for r in response.json().get('workflow_runs',[]) if r.get('display_title')=='Zeitgeist '+correlation]
        if len(matches)>1:raise RuntimeError('Duplicate correlated runs require inspection')
        if matches and matches[0]['status']=='completed':
            run=matches[0]
            if run['conclusion']!='success':raise RuntimeError('Generation execution failed; inspect run '+str(run['id']))
            after=observe()
            if after.get('date')!=edition:raise RuntimeError('Execution green but target edition not published')
            result={'kind':'publication_complete','edition_date':edition,'correlation':correlation,
                    'run_id':run['id'],'publication_id':after['publication_id']}
            print(json.dumps(result));return result
        sleep(15)
    raise RuntimeError('No verified completed publication; do not redispatch unknown work')


if __name__=='__main__':
    with httpx.Client(headers={'Authorization':'Bearer '+os.environ['GITHUB_DISPATCH_TOKEN'],
                              'Accept':'application/vnd.github+json'},timeout=30,follow_redirects=False) as client:
        run(client,correlation=os.environ['CLOUD_RUN_EXECUTION'],now=datetime.now(timezone.utc),
            monitor=os.environ.get('ZEITGEIST_MONITOR_ONLY')=='1')
