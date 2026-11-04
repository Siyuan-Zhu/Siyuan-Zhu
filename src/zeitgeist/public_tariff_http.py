"""Bounded anonymous tariff reads only; never paid requests or mutations."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import random
import re
import time

import httpx
from .spend_guard import SpendBlocked

ALLOWED_URLS = frozenset({'https://ai.google.dev/gemini-api/docs/pricing',
    'https://aistudio.google.com/docs/pricing', 'https://bfl.ai/pricing'})
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_DEFER_SECONDS = 86400
MAX_READ_SECONDS = 55
MAX_INLINE_WAIT_SECONDS = 5


def utc_now():
    return datetime.now(timezone.utc)


def validate_retry_not_before(raw, *, now):
    if now.tzinfo is None or not isinstance(raw, str) or len(raw) > 40:
        raise ValueError('Bounded UTC retry timestamp required')
    value = datetime.fromisoformat(raw)
    if (value.tzinfo is None or value.utcoffset() != timedelta(0)
            or value.isoformat() != raw or value < now - timedelta(seconds=60)
            or value > now + timedelta(seconds=MAX_DEFER_SECONDS)):
        raise ValueError('Retry timestamp lies outside the bounded UTC recovery window')
    return value


class TransientPublicRead(SpendBlocked):
    def __init__(self, *, not_before, observed_at):
        validate_retry_not_before(not_before.isoformat(), now=observed_at)
        super().__init__('Temporary anonymous public pricing read unavailable')
        self.retry_not_before = not_before.isoformat()


def _retry_after(raw, *, now):
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 128:
        raise SpendBlocked('Public pricing retry interval is invalid')
    raw = raw.strip()
    try:
        if re.fullmatch(r'[0-9]{1,6}', raw):
            value = now + timedelta(seconds=int(raw))
        else:
            value = parsedate_to_datetime(raw)
            if value.tzinfo is None:
                raise ValueError('Aware HTTP date required')
            value = value.astimezone(timezone.utc)
        value = max(now, value)
        validate_retry_not_before(value.isoformat(), now=now)
        return value
    except (ValueError, TypeError, OverflowError):
        raise SpendBlocked('Public pricing retry interval is outside the bounded recovery window') from None


def public_get(client, url):
    """At most three GETs per fixed URL within a 55-second retry budget.

    An in-flight HTTP phase has a separate maximum 15-second timeout; no new
    attempt or wait starts after the elapsed retry budget.

    Retry-After is either honored inline or carried to a later edition claim;
    it is never capped followed by an early retry. Unknown headers/long intervals
    remain held. The caller still validates all price bytes, units and redirects.
    """
    sensitive = {'authorization', 'proxy-authorization', 'x-key', 'x-goog-api-key'}
    if (url not in ALLOWED_URLS or getattr(client, 'auth', None) is not None
            or any(str(key).lower() in sensitive for key in getattr(client, 'headers', {}))):
        raise SpendBlocked('Anonymous fixed-origin public pricing client required')
    started = time.monotonic()
    for attempt in range(3):
        remaining = MAX_READ_SECONDS - (time.monotonic() - started)
        now = utc_now()
        if remaining <= 0:
            raise TransientPublicRead(not_before=now, observed_at=now)
        hint = None
        try:
            response = client.get(url, follow_redirects=False, timeout=min(15, remaining))
            if str(response.url) != url:
                raise SpendBlocked('Public pricing response origin changed')
            if response.status_code not in TRANSIENT_STATUSES:
                return response
            hint = _retry_after(response.headers.get('retry-after'), now=utc_now())
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            pass
        now = utc_now()
        base = .5 * (2 ** attempt)
        defer = max(hint or now, now + timedelta(seconds=base))
        wait = max(base, ((hint or now) - now).total_seconds()) + random.uniform(0, base * .25)
        remaining = MAX_READ_SECONDS - (time.monotonic() - started)
        if attempt == 2 or wait > MAX_INLINE_WAIT_SECONDS or wait >= remaining:
            raise TransientPublicRead(not_before=defer, observed_at=now)
        time.sleep(wait)
