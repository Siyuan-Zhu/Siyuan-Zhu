"""Anonymous GET timing and rate hints without network, real sleeps or inference."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
from src.zeitgeist import public_tariff_http as reads
from src.zeitgeist.spend_guard import SpendBlocked

NOW = datetime(2026,11,5,7,tzinfo=timezone.utc)
URL = 'https://bfl.ai/pricing'


@pytest.mark.parametrize('hint', ['120', format_datetime(NOW+timedelta(seconds=120), usegmt=True)])
def test_long_valid_hint_defers_without_a_second_get_or_long_sleep(monkeypatch, hint):
    calls, sleeps = [], []
    monkeypatch.setattr(reads, 'utc_now', lambda: NOW)
    monkeypatch.setattr(reads.time, 'sleep', sleeps.append)
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={'retry-after':hint})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(reads.TransientPublicRead) as error:
            reads.public_get(client, URL)
    assert len(calls)==1 and sleeps==[]
    assert datetime.fromisoformat(error.value.retry_not_before)==NOW+timedelta(seconds=120)


def test_short_hint_is_honored_before_retry_with_bounded_jitter(monkeypatch):
    clock, monotonic, calls, sleeps = [NOW], [0.0], [], []
    monkeypatch.setattr(reads, 'utc_now', lambda: clock[0])
    monkeypatch.setattr(reads.time, 'monotonic', lambda: monotonic[0])
    monkeypatch.setattr(reads.random, 'uniform', lambda low, high: high)
    def sleep(delay):
        sleeps.append(delay);clock[0]+=timedelta(seconds=delay);monotonic[0]+=delay
    monkeypatch.setattr(reads.time, 'sleep', sleep)
    def handler(request):
        calls.append(clock[0])
        return httpx.Response(429, headers={'retry-after':'2'}) if len(calls)==1 else httpx.Response(200)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert reads.public_get(client, URL).status_code == 200
    assert 2 <= sleeps[0] <= 2.125
    assert calls[1] >= calls[0] + timedelta(seconds=2)


@pytest.mark.parametrize('hint',['negative','-1','99999999999999999999','Fri, 05 Nov 9999 00:00:00 GMT'])
def test_invalid_or_out_of_window_hint_is_held_not_capped_then_retried(monkeypatch, hint):
    calls=[]
    monkeypatch.setattr(reads,'utc_now',lambda:NOW)
    def handler(request):
        calls.append(request);return httpx.Response(503,headers={'retry-after':hint})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SpendBlocked) as error:reads.public_get(client,URL)
    assert not isinstance(error.value,reads.TransientPublicRead) and len(calls)==1


def test_total_deadline_prevents_a_request_beyond_read_budget(monkeypatch):
    clock=[0.0];calls=[]
    monkeypatch.setattr(reads,'utc_now',lambda:NOW)
    monkeypatch.setattr(reads.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(reads.time,'sleep',lambda delay:None)
    def handler(request):
        calls.append(request);clock[0]+=55;raise httpx.ReadTimeout('synthetic timeout',request=request)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(reads.TransientPublicRead):reads.public_get(client,URL)
    assert len(calls)==1


def test_transport_configuration_error_is_held_without_retry(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(reads.time, 'sleep', sleeps.append)
    def handler(request):
        calls.append(request)
        raise httpx.UnsupportedProtocol('synthetic invalid transport configuration')
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.UnsupportedProtocol):
            reads.public_get(client, 'https://bfl.ai/pricing')
    assert len(calls) == 1 and not sleeps


def test_typed_transient_has_deliberate_static_public_diagnostic_without_timestamp():
    failure = reads.TransientPublicRead(not_before=NOW+timedelta(seconds=120), observed_at=NOW)
    assert failure.public_diagnostics == {'code':'transient-anonymous-price-read'}
    assert failure.retry_not_before == (NOW+timedelta(seconds=120)).isoformat()
