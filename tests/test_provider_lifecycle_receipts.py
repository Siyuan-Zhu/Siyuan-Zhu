"""Provider adapters over real HTTP transports; no paid dispatch or ledger writes."""

from datetime import datetime, timezone, timedelta
import json

import httpx
import pytest

from src.zeitgeist import collectors, visual_router, provider_lifecycle as lifecycle
from src.zeitgeist.spend_guard import SpendBlocked


class Handle:
    reservation_id = "reservation-fixture"
    credential_fingerprint = "a" * 16

    def __init__(self, events):
        self.events, self.receipts, self.submissions = events, [], []
        self.submission_error = None

    def submitted(self, request_id, *, submitted_at=None, metadata=None):
        self.events.append("persist-submission")
        if self.submission_error:
            raise self.submission_error
        self.submissions.append((request_id, submitted_at, metadata))

    def receipt(self, receipt):
        self.events.append("receipt")
        self.receipts.append(receipt)


class Guard:
    def __init__(self, events, *, automatic=True):
        self.events, self.automatic = events, automatic
        self.handle = Handle(events)
        self.legacy_calls = 0
        self.requests = []

    def begin_provider_request(self, provider, operation, maximum, digest):
        self.requests.append((provider, operation, maximum, digest))
        if not self.automatic:
            return None
        self.events.append("reserve")
        return self.handle

    def call(self, provider, operation, maximum, function):
        self.legacy_calls += 1
        self.events.append("legacy-reserve")
        return function()


def wire(monkeypatch, handler):
    original = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs)
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _: None)


def run_body(status="SUCCEEDED", **extra):
    return {
        "id": "run123",
        "userId": "account123",
        "actId": lifecycle.APIFY_ACTOR_ID,
        "buildNumber": "0.0.648",
        "status": status,
        "finishedAt": (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat(),
        "usageTotalUsd": "0.0068",
        "chargedEventCounts": {"post": 4},
        "accountedChargedEventCounts": {"post": 4},
        **extra,
    }


def apify_wire(monkeypatch, *, finals=None, start_error=None):
    events, requests = [], []
    guard = Guard(events)
    final = run_body()
    responses = list(finals or [final, final, final])

    def handle(request):
        requests.append(request)
        assert request.headers.get("Authorization") == "Bearer fixture-private-key"
        if request.method == "POST":
            events.append("post")
            if start_error:
                raise start_error("lost response with private-key", request=request)
            body = json.loads(request.content)
            assert body["username"] == ["fixture_creator"]
            assert body["resultsLimit"] == 5
            assert request.url.params["build"] == "0.0.648"
            assert request.url.params["maxTotalChargeUsd"] == "0.01"
            assert request.url.params["timeout"] == "300"
            return httpx.Response(201, json={"data": run_body("RUNNING")})
        assert guard.handle.submissions, "ID must be durable before polling"
        events.append("get")
        assert request.url.path.startswith("/v2/actor-runs/run123")
        if request.url.path.endswith("/dataset/items"):
            return httpx.Response(
                200,
                json=[
                    {
                        "ownerUsername": "fixture_creator",
                        "shortCode": "fixture123",
                        "url": "https://www.instagram.com/p/fixture123/",
                        "caption": "A public craft study.",
                        "timestamp": "2026-10-30T10:00:00Z",
                    }
                ],
            )
        return httpx.Response(200, json={"data": responses.pop(0)})

    wire(monkeypatch, handle)
    return guard, requests, events


def test_apify_exact_run_is_persisted_before_reads_and_stable_cost_receipt(monkeypatch):
    guard, requests, events = apify_wire(monkeypatch)
    posts = collectors.fetch_instagram_batch(
        ["fixture_creator"],
        "fixture-private-key",
        spend_guard=guard,
        actor_build="0.0.648",
    )
    assert len(posts) == 1 and guard.legacy_calls == 0
    assert requests[0].url.path == "/v2/actors/apify~instagram-post-scraper/runs"
    assert events[:3] == ["reserve", "post", "persist-submission"]
    assert len([r for r in requests if r.method == "POST"]) == 1
    receipt = guard.handle.receipts[0]
    assert receipt["provider_request_id"] == "run123"
    assert receipt["accounted_gross_usd"] == "0.0068"
    assert receipt["credential_fingerprint"] == guard.handle.credential_fingerprint
    assert receipt["request_sha256"] == guard.requests[0][3]
    assert "fixture-private-key" not in json.dumps(receipt)
    assert set(receipt) == {
        "schema_version",
        "accounting_policy",
        "reservation_id",
        "provider",
        "operation",
        "credential_fingerprint",
        "provider_request_id",
        "request_sha256",
        "observed_at",
        "accounted_gross_usd",
        "accounting_basis",
        "evidence_sha256",
    }


@pytest.mark.parametrize("cost", [None, "NaN", "-1", True])
def test_apify_missing_or_invalid_price_keeps_hold_but_valid_source_data(
    monkeypatch, cost
):
    final = run_body(usageTotalUsd=cost)
    guard, requests, events = apify_wire(monkeypatch, finals=[final, final, final])
    assert collectors.fetch_instagram_batch(
        ["fixture_creator"],
        "fixture-private-key",
        spend_guard=guard,
        actor_build="0.0.648",
    )
    assert not guard.handle.receipts
    assert len([r for r in requests if r.method == "POST"]) == 1


def test_apify_unstable_or_unaccounted_charges_remain_reserved(monkeypatch):
    first = run_body(usageTotalUsd="0.005", accountedChargedEventCounts={})
    guard, requests, events = apify_wire(
        monkeypatch, finals=[first, run_body(), run_body(usageTotalUsd="0.007")]
    )
    assert collectors.fetch_instagram_batch(
        ["fixture_creator"],
        "fixture-private-key",
        spend_guard=guard,
        actor_build="0.0.648",
    )
    assert not guard.handle.receipts


@pytest.mark.parametrize(
    "change",
    [{"id": "foreignRun"}, {"userId": "foreignAccount"}, {"buildNumber": "0.0.999"}],
)
def test_apify_foreign_or_wrong_build_run_never_settles(monkeypatch, change):
    guard, requests, events = apify_wire(monkeypatch, finals=[run_body(**change)])
    with pytest.raises(collectors.IngestionUnavailable):
        collectors.fetch_instagram_batch(
            ["fixture_creator"],
            "fixture-private-key",
            spend_guard=guard,
            actor_build="0.0.648",
        )
    assert not guard.handle.receipts
    assert len(requests) == 2


def test_apify_lost_start_response_has_no_repeated_post_or_latest_run_guess(
    monkeypatch,
):
    guard, requests, events = apify_wire(monkeypatch, start_error=httpx.ReadTimeout)
    with pytest.raises(collectors.IngestionUnavailable):
        collectors.fetch_instagram_batch(
            ["fixture_creator"],
            "fixture-private-key",
            spend_guard=guard,
            actor_build="0.0.648",
        )
    assert len(requests) == 1 and requests[0].method == "POST"
    assert not guard.handle.submissions and not guard.handle.receipts


def bfl_wire(
    monkeypatch,
    *,
    results=None,
    poll_url="https://api.bfl.ai/v1/get_result?id=task123",
    automatic=True
):
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    events, requests = [], []
    guard = Guard(events, automatic=automatic)
    responses = list(
        results
        or [
            {
                "id": "task123",
                "status": "Ready",
                "cost": 7,
                "result": {"sample": "https://images.example.invalid/accepted.webp"},
            }
        ]
    )

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            events.append("post")
            assert request.headers["x-key"] == "fixture-private-key"
            return httpx.Response(
                200, json={"id": "task123", "polling_url": poll_url, "cost": 1}
            )
        if request.url.host == "images.example.invalid":
            assert "x-key" not in request.headers
            return httpx.Response(200, content=b"accepted-image-bytes")
        if automatic:
            assert guard.handle.submissions
        assert "x-key" not in request.headers and "Authorization" not in request.headers
        assert "Cookie" not in request.headers
        return httpx.Response(200, json=responses.pop(0))

    wire(monkeypatch, handler)
    monkeypatch.setattr(visual_router, "get_spend_guard", lambda **kwargs: guard)
    return guard, requests, events


def test_bfl_receipt_uses_terminal_settled_cost_not_submission_estimate(monkeypatch):
    guard, requests, events = bfl_wire(monkeypatch)
    assert (
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
        == b"accepted-image-bytes"
    )
    assert events[:3] == ["reserve", "post", "persist-submission"]
    receipt = guard.handle.receipts[0]
    assert (
        receipt["accounted_gross_usd"] == "0.07"
        and receipt["provider_request_id"] == "task123"
    )
    assert receipt["accounting_basis"] == "provider_settled_cost"
    assert "fixture-private-key" not in json.dumps(receipt)


def test_bfl_second_read_can_supply_settled_cost_missing_from_first_ready(monkeypatch):
    result = {
        "id": "task123",
        "status": "Ready",
        "result": {"sample": "https://images.example.invalid/accepted.webp"},
    }
    guard, requests, events = bfl_wire(
        monkeypatch, results=[result, dict(result, cost=7)]
    )
    assert (
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
        == b"accepted-image-bytes"
    )
    assert len(guard.handle.receipts) == 1
    assert len([r for r in requests if r.url.path == "/v1/get_result"]) == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://foreign.invalid/v1/get_result?id=task123",
        "http://api.bfl.ai/v1/get_result?id=task123",
        "https://api.bfl.ai/v1/get_result?id=foreign",
        "https://api.bfl.ai/v1/get_result?id=task123&key=private",
        "https://api.bfl.ai/v1/get_result?id=task123#fragment",
        "https://api.eu0.bfl.ai/v1/get_result?id=task123",
        "https://delivery.eu2.bfl.ai/v1/get_result?id=task123",
        "https://api.eu2.bfl.ai.foreign.invalid/v1/get_result?id=task123",
        "https://api.eu2.bfl.ai:444/v1/get_result?id=task123",
        "https://api.isr.bfl.ai.foreign.invalid/v1/get_result?id=task123",
        "https://api.isr.bfl.ai:444/v1/get_result?id=task123",
    ],
)
def test_bfl_never_forwards_key_to_unverified_polling_destination(monkeypatch, url):
    guard, requests, events = bfl_wire(monkeypatch, poll_url=url)
    assert (
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
        is None
    )
    assert len(requests) == 1 and not guard.handle.receipts
    assert (
        guard.handle.submissions[0][0] == "task123"
    )  # Known job remains reconcilable.


def test_bfl_mismatched_response_id_has_no_receipt_or_image_fetch(monkeypatch):
    guard, requests, events = bfl_wire(
        monkeypatch, results=[{"id": "foreign", "status": "Ready", "cost": 0}]
    )
    assert (
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
        is None
    )
    assert not guard.handle.receipts and len(requests) == 2


def test_bfl_legacy_policy_path_stays_full_hold_and_does_not_emit_receipt(monkeypatch):
    guard, requests, events = bfl_wire(monkeypatch, automatic=False)
    assert (
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
        == b"accepted-image-bytes"
    )
    assert (
        guard.legacy_calls == 1
        and not guard.handle.submissions
        and not guard.handle.receipts
    )


@pytest.mark.parametrize('host',['api.bfl.ai','api.eu.bfl.ai','api.us.bfl.ai',
    'api.eu2.bfl.ai','api.us7.bfl.ai','api.us8.bfl.ai','api.us123.bfl.ai','api.isr.bfl.ai'])
def test_bfl_submission_regional_url_is_persisted_and_polled_without_credentials(monkeypatch,host):
    guard,requests,events=bfl_wire(monkeypatch,poll_url='https://'+host+'/v1/get_result?id=task123')
    assert visual_router._generate_live_flux_bytes('Bounded artwork',api_key='fixture-private-key')
    assert len([request for request in requests if request.method=='POST'])==1
    assert guard.handle.submissions[0][2]['polling_host']==host
    poll=next(request for request in requests if request.url.path=='/v1/get_result')
    assert poll.url.host==host and 'x-key' not in poll.headers and 'authorization' not in poll.headers


def test_missing_returned_polling_url_keeps_known_job_without_guess_or_new_post(monkeypatch):
    guard,requests,events=bfl_wire(monkeypatch,poll_url=None)
    assert visual_router._generate_live_flux_bytes('Bounded artwork',api_key='fixture-private-key') is None
    assert len(requests)==1 and len(guard.handle.submissions)==1 and not guard.handle.receipts


def real_bfl_guard(monkeypatch):
    import copy
    from tests.test_automatic_guard import budget
    from src.zeitgeist import automatic_spend as accounting
    guard=budget();operation='generate_image:flux-3-image'
    account=guard.automatic_policy['providers']['bfl']['account']
    account['operation_caps_usd']={operation:'.07'}
    guard.expected_policy_sha=accounting.digest(guard.automatic_policy)
    guard.ledger.document['automatic_policy_sha256']=guard.expected_policy_sha
    guard.ledger.document['providers']['bfl'].update(copy.deepcopy(account))
    monkeypatch.setattr(lifecycle,'utc_now',lambda:guard.now().isoformat())
    return guard


def real_bfl_request(monkeypatch):
    guard=real_bfl_guard(monkeypatch)
    digest=lifecycle.request_digest({'provider':'bfl','shape':'fixed-single-image-fixture'})
    operation='generate_image:flux-3-image'
    handle=guard.begin_provider_request('bfl',operation,None,digest)
    handle.submitted('task123',metadata={'resolution':'1.5k','aspect_ratio':'3:2','polling_host':'api.eu2.bfl.ai'})
    return guard,handle,digest


def test_regional_ready_missing_cost_completes_through_real_guard_with_one_post(monkeypatch):
    guard=real_bfl_guard(monkeypatch)
    ready={'id':'task123','status':'Ready',
        'result':{'sample':'https://images.example.invalid/accepted.webp'}}
    _,requests,_=bfl_wire(monkeypatch,results=[ready],automatic=False,
        poll_url='https://api.us8.bfl.ai/v1/get_result?id=task123')
    monkeypatch.setattr(visual_router,'get_spend_guard',lambda **kwargs:guard)
    assert visual_router._generate_live_flux_bytes('Bounded image',api_key='fixture-private-key')==b'accepted-image-bytes'
    assert len([request for request in requests if request.method=='POST'])==1
    assert len([request for request in requests if request.url.path=='/v1/get_result'])==1
    record=next(iter(guard.ledger.document['reservations'].values()))
    assert record['submission_metadata']['polling_host']=='api.us8.bfl.ai'
    assert record['status']=='settled' and record['actual_gross_usd']=='0.07'
    assert record['usage_receipt']['accounting_basis']=='conservative_response_usage'


def test_ready_without_cost_books_original_full_bound_and_is_idempotent(monkeypatch):
    guard,handle,digest=real_bfl_request(monkeypatch)
    result={'id':'task123','status':'Ready','result':{'sample':'https://delivery.eu2.bfl.ai/private?signature=secret'}}
    before=guard._cash_exposure(guard.ledger.document)
    assert lifecycle.submit_bfl_receipt(handle,digest,'task123',result)
    record=guard.ledger.document['reservations'][handle.reservation_id]
    assert record['status']=='settled' and record['actual_gross_usd']==record['max_gross_usd']=='0.07'
    assert record['usage_receipt']['accounting_basis']=='conservative_response_usage'
    assert 'signature' not in json.dumps(record) and 'delivery.eu2' not in json.dumps(record)
    reference=record['billing_reference']
    assert lifecycle.submit_bfl_receipt(handle,digest,'task123',result)
    assert guard.ledger.document['reservations'][handle.reservation_id]['billing_reference']==reference
    assert guard._cash_exposure(guard.ledger.document)==before
    # Known conservative consumption debits the original prepaid tranche at UTC
    # handoff; it cannot become newly available credit when its row retires.
    from src.zeitgeist import automatic_spend as accounting
    from tests.test_automatic_spend import fixtures
    now=guard.now().replace(month=11,day=1,hour=0,minute=1)
    _,_,observation=fixtures(now)
    observation['gemini']['available_required_models']={'gemini-3.8-flash':['generateContent']}
    plan=accounting.prepare_month_handoff(guard.ledger.document,guard.automatic_policy,observation,
        expected_policy_sha=guard.expected_policy_sha,
        source_identity=dict(accounting.TARGET,document_id='2026-10',update_time=now.isoformat()),now=now)
    assert handle.reservation_id not in plan['destination_after']['reservations']
    assert accounting.money(plan['destination_after']['providers']['bfl']['credit_baseline_usd'])==accounting.money('9.18')


@pytest.mark.parametrize('change',['not_found','missing_id','foreign_id','null_cost','invalid_cost',
    'missing_result','wrong_shape','wrong_digest'])
def test_ready_fallback_never_releases_unknown_invalid_or_unbound_usage(monkeypatch,change):
    import copy
    guard,handle,digest=real_bfl_request(monkeypatch)
    result={'id':'task123','status':'Ready','result':{'sample':'https://delivery.eu2.bfl.ai/private?signature=secret'}}
    if change=='not_found':result['status']='Task not found'
    if change=='missing_id':result.pop('id')
    if change=='foreign_id':result['id']='another'
    if change=='null_cost':result['cost']=None
    if change=='invalid_cost':result['cost']='NaN'
    if change=='missing_result':result.pop('result')
    if change=='wrong_shape':guard.ledger.document['reservations'][handle.reservation_id]['submission_metadata']['resolution']='4k'
    if change=='wrong_digest':digest='b'*64
    before=copy.deepcopy(guard.ledger.document)
    assert not lifecycle.submit_bfl_receipt(handle,digest,'task123',result)
    assert guard.ledger.document==before


def test_later_higher_settled_cost_preserves_overrun_and_freezes_after_full_bound(monkeypatch):
    guard,handle,digest=real_bfl_request(monkeypatch)
    result={'id':'task123','status':'Ready','result':{'sample':'https://delivery.us7.bfl.ai/private'}}
    assert lifecycle.submit_bfl_receipt(handle,digest,'task123',result)
    with pytest.raises(SpendBlocked):lifecycle.submit_bfl_receipt(handle,digest,'task123',dict(result,cost=8))
    record=guard.ledger.document['reservations'][handle.reservation_id]
    assert record['actual_gross_usd']=='0.08' and guard.ledger.document['reconciliation_required'] is True


def test_anonymous_poll_strips_response_cookies_and_refuses_default_credentials(monkeypatch):
    requests=[]
    def handler(request):
        requests.append(request)
        assert 'x-key' not in request.headers and 'cookie' not in request.headers
        return httpx.Response(302,headers={'Location':'https://foreign.invalid','Set-Cookie':'tracking=opaque; Domain=.bfl.ai'})
    with httpx.Client(transport=httpx.MockTransport(handler),follow_redirects=True) as client:
        client.cookies.set('old','private',domain='.bfl.ai')
        response=lifecycle.read_bfl_result(client,'https://api.eu2.bfl.ai/v1/get_result?id=task123','task123')
        assert response.status_code==302 and len(requests)==1
        lifecycle.read_bfl_result(client,'https://api.eu2.bfl.ai/v1/get_result?id=task123','task123')
        assert len(requests)==2
        client.headers['x-key']='must-not-send'
        with pytest.raises(ValueError):lifecycle.read_bfl_result(client,'https://api.eu2.bfl.ai/v1/get_result?id=task123','task123')
        assert len(requests)==2


def test_apify_submission_persistence_failure_blocks_polling_and_preserves_hold(
    monkeypatch,
):
    guard, requests, events = apify_wire(monkeypatch)
    guard.handle.submission_error = SpendBlocked("Billing reconciliation required")
    with pytest.raises(SpendBlocked):
        collectors.fetch_instagram_batch(
            ["fixture_creator"],
            "fixture-private-key",
            spend_guard=guard,
            actor_build="0.0.648",
        )
    assert len(requests) == 1 and not guard.handle.receipts


def test_apify_poll_timeout_never_restarts_actor(monkeypatch):
    guard, requests, events = apify_wire(monkeypatch, finals=[run_body("RUNNING")])
    clock = iter((0.0, 1.0, 331.0))
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: next(clock))
    with pytest.raises(collectors.IngestionUnavailable):
        collectors.fetch_instagram_batch(
            ["fixture_creator"],
            "fixture-private-key",
            spend_guard=guard,
            actor_build="0.0.648",
        )
    assert len([request for request in requests if request.method == "POST"]) == 1
    assert len(requests) == 2 and not guard.handle.receipts


def test_apify_legacy_none_policy_uses_existing_single_sync_call(monkeypatch):
    events = []
    guard = Guard(events, automatic=False)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path.endswith("/run-sync-get-dataset-items")
        return httpx.Response(
            200,
            json=[
                {
                    "ownerUsername": "fixture_creator",
                    "url": "https://www.instagram.com/p/fixture123/",
                    "caption": "Public craft.",
                }
            ],
        )

    wire(monkeypatch, handler)
    assert collectors.fetch_instagram_batch(
        ["fixture_creator"],
        "fixture-private-key",
        spend_guard=guard,
        actor_build="0.0.648",
    )
    assert len(requests) == 1 and guard.legacy_calls == 1
    assert not guard.handle.submissions and not guard.handle.receipts


def test_bfl_submission_persistence_failure_stops_after_one_post(monkeypatch):
    guard, requests, events = bfl_wire(monkeypatch)
    guard.handle.submission_error = SpendBlocked("Billing reconciliation required")
    with pytest.raises(SpendBlocked):
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
    assert len(requests) == 1 and not guard.handle.receipts


def test_invalid_automatic_policy_cannot_fall_back_to_legacy_paid_dispatch(monkeypatch):
    guard, requests, events = bfl_wire(monkeypatch)

    def blocked(*args):
        raise SpendBlocked("Billing reconciliation required")

    guard.begin_provider_request = blocked
    with pytest.raises(SpendBlocked):
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
    assert requests == [] and guard.legacy_calls == 0


@pytest.mark.parametrize("cost", [None, "NaN", -1, True])
def test_bfl_missing_or_invalid_settled_cost_keeps_full_hold(monkeypatch, cost):
    result = {
        "id": "task123",
        "status": "Ready",
        "cost": cost,
        "result": {"sample": "https://images.example.invalid/accepted.webp"},
    }
    guard, requests, events = bfl_wire(monkeypatch, results=[result, result])
    assert (
        visual_router._generate_live_flux_bytes(
            "A bounded public image", api_key="fixture-private-key"
        )
        == b"accepted-image-bytes"
    )
    assert not guard.handle.receipts
    assert len([request for request in requests if request.method == "POST"]) == 1


def test_apify_transitional_missing_build_waits_but_terminal_build_is_required(
    monkeypatch,
):
    final = run_body()
    guard, requests, events = apify_wire(
        monkeypatch, finals=[run_body("RUNNING", buildNumber=None), final, final, final]
    )
    assert collectors.fetch_instagram_batch(
        ["fixture_creator"],
        "fixture-private-key",
        spend_guard=guard,
        actor_build="0.0.648",
    )
    assert len(guard.handle.receipts) == 1
    assert len([request for request in requests if request.method == "POST"]) == 1
