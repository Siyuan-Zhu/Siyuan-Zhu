import copy
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
import pytest
from src.zeitgeist import edition_execution as execution
from src.state_store import PublicationConflict,PublicationOutcomeUnknown

NOW=datetime(2026,11,3,23,59,tzinfo=timezone.utc)


class Memory:
    def __init__(self):self.value=None;self.version=0;self.ambiguous=False
    def collection(self,*a):return self
    def document(self,*a):return self
    def get(self):
        value=copy.deepcopy(self.value)
        return SimpleNamespace(exists=value is not None,to_dict=lambda:value,version=self.version)
    def commit(self,client,writes):
        _,_,value,snap=writes[0]
        if snap.version!=self.version:raise PublicationConflict('changed')
        self.value=copy.deepcopy(value);self.version+=1
        if self.ambiguous:raise PublicationOutcomeUnknown('lost ack')


def claim(monkeypatch):
    memory=Memory();monkeypatch.setattr(execution,'_commit_documents',memory.commit)
    a=execution.EditionExecution(memory,edition='2026-11-03',owner='run1')
    b=execution.EditionExecution(memory,edition='2026-11-03',owner='run2')
    a.claim(now=NOW,source_revision='abc',trigger='schedule')
    return memory,a,b


def test_duplicate_active_claim_and_midnight_do_not_start_second_worker(monkeypatch):
    memory,a,b=claim(monkeypatch)
    with pytest.raises(RuntimeError):b.claim(now=NOW+timedelta(minutes=2),source_revision='abc',trigger='schedule')
    assert memory.value['edition_date']=='2026-11-03'


def test_expired_preflight_reclaim_fences_old_owner(monkeypatch):
    memory,a,b=claim(monkeypatch)
    b.claim(now=NOW+timedelta(minutes=21),source_revision='abc',trigger='schedule')
    with pytest.raises(RuntimeError):a.transition('generation',now=NOW+timedelta(minutes=21))


@pytest.mark.parametrize('stage',['generation','validation','publication','awaiting_approval','requires_reconciliation'])
def test_expired_paid_or_publication_work_requires_reconciliation(monkeypatch,stage):
    memory,a,b=claim(monkeypatch);a.transition(stage,now=NOW)
    with pytest.raises(RuntimeError):b.claim(now=NOW+timedelta(days=1),source_revision='new',trigger='schedule')


def test_ambiguous_claim_readback_recovers_without_duplicate_write(monkeypatch):
    memory,a,b=claim(monkeypatch);memory.ambiguous=True
    a.transition('generation',now=NOW)
    assert memory.value['stage']=='generation'


def test_pricing_failure_not_retried_until_repaired_source(monkeypatch):
    memory,a,b=claim(monkeypatch);a.transition('blocked',failure_category='pricing',now=NOW)
    with pytest.raises(RuntimeError):b.claim(now=NOW+timedelta(hours=1),source_revision='abc',trigger='schedule')
    b.claim(now=NOW+timedelta(hours=1),source_revision='fixed',trigger='schedule')
    assert memory.value['attempt']==2


def test_completed_claim_is_noop(monkeypatch):
    memory,a,b=claim(monkeypatch);a.transition('published',now=NOW,publication_id='abc')
    assert b.claim(now=NOW+timedelta(hours=1),source_revision='abc',trigger='schedule') is None


def test_completion_monitor_checks_edition_and_preserves_owner_hold(monkeypatch):
    from src.zeitgeist.edition_approval import HELD_EDITIONS
    monkeypatch.setitem(HELD_EDITIONS, '2026-11-03', frozenset())
    from scripts.check_editorial_deadline import assess
    assert assess({'date':'2026-11-02'},datetime(2026,11,3,10,tzinfo=timezone.utc))['status']=='approval_pending'
    assert assess({'date':'2026-11-02'},datetime(2026,11,4,10,tzinfo=timezone.utc))['status']=='publication_missing'
    assert assess({'date':'2026-11-04'},datetime(2026,11,4,10,tzinfo=timezone.utc))['status']=='complete'


def test_director_timeout_never_repeats_paid_call():
    from src.zeitgeist.adk_pipeline import commission_daily_zeitgeist
    calls=[]
    def timeout(*args):
        calls.append(1);raise TimeoutError('provider outcome unknown')
    with pytest.raises(TimeoutError):
        commission_daily_zeitgeist(top_clusters=[],history_words_30d=[],llm_callable=timeout)
    assert len(calls)==1


def at_bound(monkeypatch, failure_category):
    memory,a,b=claim(monkeypatch)
    memory.value.update(stage='blocked',failure_category=failure_category,attempt=3,
                        lease_until=(NOW+timedelta(minutes=20)).isoformat())
    return memory,b


def test_pre_inference_spend_maintenance_blocks_do_not_exhaust_the_attempt_bound(monkeypatch):
    """2026-11-13: three spend-maintenance blocks (no provider call) exhausted the edition."""
    memory,b=at_bound(monkeypatch,'spend_maintenance')
    b.claim(now=NOW+timedelta(minutes=21),source_revision='fixed',trigger='workflow_dispatch')
    assert memory.value['stage']=='preflight' and memory.value['attempt']==4


@pytest.mark.parametrize('category',['pricing','generation','freshness'])
def test_other_failures_still_stop_at_the_attempt_bound(monkeypatch,category):
    memory,b=at_bound(monkeypatch,category)
    with pytest.raises(RuntimeError,match='attempt bound'):
        b.claim(now=NOW+timedelta(minutes=21),source_revision='fixed',trigger='workflow_dispatch')


def test_a_spend_maintenance_block_still_needs_a_new_source_revision(monkeypatch):
    memory,b=at_bound(monkeypatch,'spend_maintenance')
    with pytest.raises(RuntimeError,match='Deterministic failure'):
        b.claim(now=NOW+timedelta(minutes=21),source_revision='abc',trigger='workflow_dispatch')
