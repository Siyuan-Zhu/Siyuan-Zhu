"""Atomic, durable edition claims. A lease expiry never authorizes repeat inference.

Generation ambiguity requires reconciliation; only pre-inference abandoned claims
may be reclaimed. Existing per-request spending reservations remain authoritative.
"""
from datetime import datetime, timedelta, timezone
import re
import json
from src.state_store import _commit_documents, PublicationOutcomeUnknown

ACTIVE_UNCERTAIN={'generation','validation','publication','awaiting_approval','requires_reconciliation'}
FAILED_TERMINAL={'blocked','held_for_review','requires_reconciliation','preflight_retryable'}


class EditionExecution:
    def __init__(self, client, *, prefix='', edition, owner):
        from datetime import date
        if date.fromisoformat(edition).isoformat()!=edition or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,150}',owner):
            raise ValueError('Exact edition and bounded execution identity required')
        self.client=client;self.collection=prefix+'editorial_executions';self.edition=edition;self.owner=owner
        self.doc=client.collection(self.collection).document(edition)

    def _write(self, snapshot, value):
        try:
            _commit_documents(self.client,[(self.collection,self.edition,value,snapshot)])
        except PublicationOutcomeUnknown:
            if self.doc.get().to_dict()!=value:
                raise RuntimeError('Execution claim outcome unknown; do not start work') from None
        print(json.dumps({key:value.get(key) for key in ('edition_date','owner','stage',
            'publication_id','failure_category','source_revision','updated_at')},sort_keys=True))
        return value

    def claim(self, *, now, source_revision, trigger):
        if now.tzinfo is None:raise ValueError('Aware clock required')
        snap=self.doc.get();prior=snap.to_dict() if snap.exists else {}
        if prior:
            if prior['stage']=='published':return None
            if prior['stage'] in ACTIVE_UNCERTAIN:
                raise RuntimeError('Edition requires existing-work reconciliation, not new inference')
            if prior['stage']=='held_for_review':raise PermissionError('Edition held for owner review')
            if prior['stage']=='preflight_retryable':
                from .public_tariff_http import validate_retry_not_before
                if prior.get('failure_category')!='transient_anonymous_price_read' or prior.get('publication_id'):
                    raise RuntimeError('Retry requires proven pre-inference observation failure')
                deferred=validate_retry_not_before(prior.get('retry_not_before'),
                    now=datetime.fromisoformat(prior['updated_at']))
                if now<deferred:raise RuntimeError('Anonymous pricing Retry-After has not elapsed')
            if datetime.fromisoformat(prior['lease_until'])>now:
                raise RuntimeError('Another edition claim is active')
            if prior['stage']=='blocked' and prior.get('source_revision')==source_revision:
                raise RuntimeError('Deterministic failure already recorded for this source revision')
            # The bound caps attempts that may reach paid inference. Spend maintenance runs
            # before any provider call, so its block (with a new source revision) never counts.
            pre_inference=prior['stage']=='blocked' and prior.get('failure_category')=='spend_maintenance'
            if prior.get('attempt',0)>=3 and not pre_inference:
                raise RuntimeError('Edition attempt bound reached; inspect before recovery')
        value={'schema_version':1,'edition_date':self.edition,'owner':self.owner,
            'source_revision':source_revision,'trigger':trigger,'stage':'preflight',
            'attempt':prior.get('attempt',0)+1,'started_at':now.isoformat(),
            'updated_at':now.isoformat(),'lease_until':(now+timedelta(minutes=20)).isoformat(),
            'publication_id':None,'failure_category':None}
        return self._write(snap,value)

    def transition(self, stage, *, publication_id=None, failure_category=None, retry_not_before=None, now=None):
        allowed={'preflight','generation','validation','publication','awaiting_approval',
            'published','blocked','held_for_review','requires_reconciliation','preflight_retryable'}
        if stage not in allowed:raise ValueError('Unknown execution stage')
        now=now or datetime.now(timezone.utc)
        snap=self.doc.get();value=snap.to_dict() if snap.exists else {}
        if value.get('owner')!=self.owner:raise RuntimeError('Execution ownership changed')
        if value.get('stage') in FAILED_TERMINAL and (stage!=value['stage'] or
                (publication_id is not None and publication_id!=value.get('publication_id'))):
            raise RuntimeError('Failed execution is terminal; inspect or claim eligible preflight recovery')
        if value.get('stage')=='published' and stage!='published':raise RuntimeError('Published edition is terminal')
        # The lease fences the generation start; after paid work, the same owner
        # may record results after expiry, but nobody may steal uncertain work.
        if stage=='generation' and datetime.fromisoformat(value['lease_until'])<=now:
            raise RuntimeError('Lease expired before generation')
        if stage=='preflight_retryable':
            from .public_tariff_http import validate_retry_not_before
            if value.get('stage')!='preflight' or failure_category!='transient_anonymous_price_read' or value.get('publication_id'):
                raise RuntimeError('Only proven pre-inference failure may become retryable')
            validate_retry_not_before(retry_not_before,now=now)
            value['retry_not_before']=retry_not_before
        elif retry_not_before is not None:
            raise ValueError('Retry timestamp belongs only to preflight observation failure')
        value.update(stage=stage,updated_at=now.isoformat(),publication_id=publication_id or value.get('publication_id'),
                     failure_category=failure_category)
        return self._write(snap,value)


def for_store(store, edition, env):
    from src.state_store import FirestoreStateStore
    if not isinstance(store,FirestoreStateStore):return None
    owner=env.get('CLOUD_RUN_EXECUTION') or ('github-'+env.get('GITHUB_RUN_ID','local')+'-'+env.get('GITHUB_RUN_ATTEMPT','1'))
    return EditionExecution(store._get_client(),prefix=store._col('') ,edition=str(edition),owner=owner)
