#!/usr/bin/env python3
"""Read-only inventory through the existing identity; never enable APIs or IAM."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from src.state_store import _FirestoreRestClient

project='git-readme-features'
client=_FirestoreRestClient(project)
token=client._resolve_token()
endpoints={
 'enabled_services':f'https://serviceusage.googleapis.com/v1/projects/277213321965/services?filter=state:ENABLED',
 'service_accounts':f'https://iam.googleapis.com/v1/projects/{project}/serviceAccounts',
 'secrets_metadata':f'https://secretmanager.googleapis.com/v1/projects/{project}/secrets',
 'cloud_run_jobs':f'https://run.googleapis.com/v2/projects/{project}/locations/-/jobs',
 'scheduler_locations':f'https://cloudscheduler.googleapis.com/v1/projects/{project}/locations',
 'firestore_database':f'https://firestore.googleapis.com/v1/projects/{project}/databases/(default)',
}
report={}
with httpx.Client(headers={'Authorization':'Bearer '+token},timeout=30,follow_redirects=False) as http:
    for name,url in endpoints.items():
        try:
            r=http.get(url)
            report[name]={'http_status':r.status_code}
            if r.status_code==200:report[name]['inventory']=r.json()
            else:report[name]['status']='read_permission_or_service_unavailable'
        except httpx.HTTPError:report[name]={'status':'transport_unavailable'}
Path('.private-runtime').mkdir(exist_ok=True)
Path('.private-runtime/gcp-inventory.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:{f:v for f,v in value.items() if f!='inventory'} for k,value in report.items()}))
