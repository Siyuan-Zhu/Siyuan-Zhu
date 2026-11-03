from datetime import datetime,timezone
import httpx,pytest
from scripts.gcp_zeitgeist_bridge import run

NOW=datetime(2026,11,4,9,tzinfo=timezone.utc)


class Response:
    status_code=200
    def __init__(self,value):self.value=value
    def raise_for_status(self):pass
    def json(self):return self.value


class Client:
    def __init__(self,*,unknown=False,failed=False,unchanged=False):
        self.posts=0;self.reads=0;self.unknown=unknown;self.failed=failed;self.unchanged=unchanged
    def post(self,*a,**kw):
        self.posts+=1
        if self.unknown:raise httpx.ReadTimeout('ack lost')
        return Response({})
    def get(self,url,**kwargs):
        if '/runs' in url:return Response({'workflow_runs':[{'display_title':'Zeitgeist test-event','id':123,
            'status':'completed','conclusion':'failure' if self.failed else 'success'}]})
        self.reads+=1
        return Response({'date':'2026-11-04' if self.reads>1 and not self.unchanged else '2026-11-02','publication_id':'abc'})


@pytest.mark.parametrize('unknown',[False,True])
def test_bridge_requires_completed_run_and_actual_edition_even_after_lost_ack(unknown):
    client=Client(unknown=unknown)
    assert run(client,correlation='test-event',now=NOW)['kind']=='publication_complete'
    assert client.posts==1


@pytest.mark.parametrize('kwargs',[{'failed':True},{'unchanged':True}])
def test_green_trigger_or_green_job_does_not_imply_publication(kwargs):
    client=Client(**kwargs)
    with pytest.raises(RuntimeError):run(client,correlation='test-event',now=NOW)
    assert client.posts==1


def test_held_edition_never_dispatches(monkeypatch):
    from src.zeitgeist.edition_approval import HELD_EDITIONS
    monkeypatch.setitem(HELD_EDITIONS, '2026-11-03', frozenset())
    client=Client()
    assert run(client,correlation='test-event',now=datetime(2026,11,3,9,tzinfo=timezone.utc))['status']=='approval_pending'
    assert client.posts==0


def test_deadline_monitor_never_dispatches_or_generates():
    client=Client()
    with pytest.raises(RuntimeError):run(client,correlation='test-event',now=NOW,monitor=True)
    assert client.posts==0


def test_approval_hold_blocks_even_before_delivery_deadline(monkeypatch):
    from src.zeitgeist.edition_approval import HELD_EDITIONS
    monkeypatch.setitem(HELD_EDITIONS, '2026-11-03', frozenset())
    client=Client()
    assert run(client,correlation='test-event',now=datetime(2026,11,3,7,tzinfo=timezone.utc))['status']=='approval_pending'
    assert client.posts==0
