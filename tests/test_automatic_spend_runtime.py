"""Actual REST encoding, fake private server, no provider or cloud operations."""
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from urllib.error import HTTPError

import pytest

from src import state_store as stores
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_automatic_spend import fixtures, NOW


class PrivateWire:
    def __init__(self, monkeypatch, docs):
        self.docs = copy.deepcopy(docs)
        self.versions = {key: 1 for key in docs}
        self.posts = []
        self.reject_post = None
        self.unknown_post = None
        self.unknown_before_post = None
        self.before_post = None
        self.observations = 0
        monkeypatch.setattr(stores.urllib.request, "urlopen", self.urlopen)
        client = stores._FirestoreRestClient("git-readme-features", access_token="private-token")
        self.store = runtime.PrivateFirestoreStore(client)

    def revision(self, key):
        return "2026-11-01T00:00:00.%06dZ" % self.versions[key]

    def urlopen(self, request, **kwargs):
        assert request.full_url.startswith("https://firestore.googleapis.com/v1/projects/git-readme-features/databases/(default)/documents")
        path = request.full_url.split("/documents", 1)[1]
        if request.method == "GET":
            key = path.removeprefix("/spend_control/")
            if key not in self.docs:
                raise HTTPError(request.full_url, 404, "private missing", {}, io.BytesIO())
            body = {"fields": {k: stores._py_to_firestore_value(v) for k,v in self.docs[key].items()},
                    "updateTime": self.revision(key)}
        else:
            assert request.method == "POST" and path == ":commit"
            payload = json.loads(request.data)
            self.posts.append(payload)
            if self.before_post:
                callback, self.before_post = self.before_post, None
                callback(self)
            if len(self.posts) == self.unknown_before_post:
                raise TimeoutError("private uncertain dispatch")
            if len(self.posts) == self.reject_post:
                raise HTTPError(request.full_url, 400, "private detail", {}, io.BytesIO(b"private balance"))
            decoded = []
            for write in payload["writes"]:
                key = write["update"]["name"].split("/documents/spend_control/", 1)[1]
                pre = write["currentDocument"]
                if (pre.get("exists") is False and key in self.docs or
                    "updateTime" in pre and (key not in self.docs or pre["updateTime"] != self.revision(key))):
                    raise HTTPError(request.full_url, 412, "private conflict", {}, io.BytesIO())
                decoded.append((key, {k: stores._firestore_value_to_py(v) for k,v in write["update"]["fields"].items()}))
            for key, value in decoded:
                self.docs[key] = dict(self.docs.get(key, {}), **value)
                self.versions[key] = self.versions.get(key, 0) + 1
            if len(self.posts) == self.unknown_post:
                raise TimeoutError("private accepted response was lost")
            body = {"commitTime": "now"}
        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
        return Response(json.dumps(body).encode())


def setup(monkeypatch, now=NOW, *, legacy=False):
    ledger, policy, obs = fixtures(now)
    if legacy:
        ledger["schema_version"] = 2
    sha = auto.digest(policy)
    wire = PrivateWire(monkeypatch, {"policy_"+sha: policy, ledger["month"]: ledger})
    def observe(policy_copy, observed_now):
        wire.observations += 1
        assert policy_copy == policy and observed_now == now
        return copy.deepcopy(obs)
    return wire, sha, observe, obs


def run(wire, sha, observer, now=NOW, apply=True):
    return runtime.maintain_spend_control(wire.store, expected_policy_sha=sha,
        observer=observer, now=now, apply=apply)


def test_plan_has_no_writes_and_public_status_has_no_private_accounting(monkeypatch):
    wire, sha, observe, _ = setup(monkeypatch)
    result = run(wire, sha, observe, apply=False)
    assert result.public_status() == {"status":"planned", "operation":"refresh", "month":"2026-10"}
    assert not wire.posts and wire.observations == 1
    assert "providers" not in repr(result)


@pytest.mark.parametrize("kind", ["missing_hash", "wrong_hash", "legacy", "policy_tamper"])
def test_unapproved_state_blocks_before_metadata_or_writes(monkeypatch, kind):
    wire, sha, observe, _ = setup(monkeypatch, legacy=kind=="legacy")
    if kind == "missing_hash": sha = ""
    if kind == "wrong_hash": sha = "a"*64
    if kind == "policy_tamper": wire.docs["policy_"+sha]["cash_ceiling_usd"] = "3"
    with pytest.raises(SpendBlocked): run(wire, sha, observe)
    assert not wire.posts and wire.observations == 0


def test_refresh_unknown_ack_reads_back_once_and_repeating_refresh_is_idempotent(monkeypatch):
    wire, sha, observe, _ = setup(monkeypatch)
    wire.unknown_post = 1
    assert run(wire, sha, observe).status == "refreshed"
    after = copy.deepcopy(wire.docs["2026-10"])
    assert len(wire.posts) == 1
    assert run(wire, sha, observe).status == "refreshed"
    assert wire.docs["2026-10"] == after and len(wire.posts) == 2
    write = wire.posts[0]["writes"][0]
    assert write["currentDocument"] == {"updateTime":"2026-11-01T00:00:00.000001Z"}


def test_refresh_http400_does_not_retry_or_mutate_ledger(monkeypatch):
    wire, sha, observe, _ = setup(monkeypatch)
    before = copy.deepcopy(wire.docs["2026-10"])
    wire.reject_post = 1
    with pytest.raises(RuntimeError): run(wire, sha, observe)
    assert wire.docs["2026-10"] == before and len(wire.posts) == 1


def test_refresh_cas_conflict_never_overwrites_a_concurrent_reservation(monkeypatch):
    wire, sha, observe, _ = setup(monkeypatch)
    def reserve(server):
        server.docs["2026-10"]["reservations"]["concurrent"] = {
            "provider":"gemini", "operation":"generate_content:gemini-2.5-flash",
            "max_gross_usd":".20", "status":"reserved"}
        server.versions["2026-10"] += 1
    wire.before_post = reserve
    with pytest.raises(stores.PublicationConflict): run(wire, sha, observe)
    assert "concurrent" in wire.docs["2026-10"]["reservations"] and len(wire.posts) == 1


def test_refresh_unknown_not_proven_by_readback_stops_without_retry(monkeypatch):
    wire, sha, observe, _ = setup(monkeypatch)
    before = copy.deepcopy(wire.docs["2026-10"])
    wire.unknown_before_post = 1
    with pytest.raises(SpendBlocked, match="unknown"):
        run(wire, sha, observe)
    assert len(wire.posts) == 1 and wire.docs["2026-10"] == before


def test_handoff_unknown_ack_readback_completes_one_conditional_atomic_batch(monkeypatch):
    now = datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    wire, sha, observe, _ = setup(monkeypatch, now)
    wire.unknown_post = 2  # Intent succeeds; canonical batch accepted, ack lost.
    result = run(wire, sha, observe, now)
    assert result.status == "handed_off"
    assert len(wire.posts) == 3
    writes = wire.posts[1]["writes"]
    assert len(writes) == 2 and writes[1]["currentDocument"] == {"exists":False}
    assert wire.docs["2026-10"]["closed_for_rollover"] is True
    assert wire.docs["handoff_2026-11"]["status"] == "complete"
    # Later fresh refresh does not create another handoff or duplicate credits.
    before_credit = wire.docs["2026-11"]["providers"]["apify"]["credit_baseline_usd"]
    run(wire, sha, observe, now)
    assert len(wire.posts) == 4
    assert wire.docs["2026-11"]["providers"]["apify"]["credit_baseline_usd"] == before_credit


def test_pending_handoff_intent_never_redispatches_after_restart(monkeypatch):
    now = datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    wire, sha, observe, _ = setup(monkeypatch, now)
    wire.reject_post = 2  # Crash/rejection after private intent but before apply.
    with pytest.raises(RuntimeError): run(wire, sha, observe, now)
    assert "2026-11" not in wire.docs
    before_posts = len(wire.posts)
    before_observations = wire.observations
    with pytest.raises(SpendBlocked, match="do not redispatch"):
        run(wire, sha, observe, now)
    assert len(wire.posts) == before_posts and wire.observations == before_observations


def test_unknown_unapplied_handoff_stays_pending_and_never_redispatches(monkeypatch):
    now = datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    wire, sha, observe, _ = setup(monkeypatch, now)
    wire.unknown_before_post = 2
    with pytest.raises(SpendBlocked, match="do not redispatch"):
        run(wire, sha, observe, now)
    assert wire.docs["handoff_2026-11"]["status"] == "pending"
    before_posts, before_observations = len(wire.posts), wire.observations
    with pytest.raises(SpendBlocked, match="do not redispatch"):
        run(wire, sha, observe, now)
    assert len(wire.posts) == before_posts and wire.observations == before_observations


def test_definitive_handoff_conflict_replans_on_next_invocation_and_keeps_concurrent_hold(monkeypatch):
    now = datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    wire, sha, observe, observation = setup(monkeypatch, now)
    def concurrent_hold(server):
        if len(server.posts) == 1:
            server.before_post = concurrent_hold  # Wait for the canonical batch.
            return
        server.docs["2026-10"]["reservations"]["accepted-concurrently"] = {
            "provider":"gemini", "operation":"generate_content:gemini-2.5-flash",
            "max_gross_usd":".20", "status":"reserved", "created_at":now.isoformat()}
        server.versions["2026-10"] += 1
    wire.before_post = concurrent_hold
    with pytest.raises(stores.PublicationConflict):
        run(wire, sha, observe, now)
    assert len(wire.posts) == 3  # Intent, rejected batch, durable rejection.
    assert "2026-11" not in wire.docs
    rejected = copy.deepcopy(wire.docs["handoff_2026-11"])
    assert rejected["status"] == "known_rejected"
    history = rejected["rejected_plan_history"]
    assert rejected["rejected_plan_history_sha256"] == auto.digest(history)
    assert history[0]["plan_sha256"] == rejected["plan"]["plan_sha256"]
    assert history[0]["source_before_sha256"] == rejected["plan"]["source_before_sha256"]
    later = now+timedelta(seconds=30)
    def fresh_observe(policy, observed_now):
        assert observed_now == later
        wire.observations += 1
        fresh = copy.deepcopy(observation)
        fresh["observed_at"] = later.isoformat()
        for price in fresh["price_snapshots"].values():
            price["observed_at"] = later.isoformat()
        return fresh
    planned = run(wire, sha, fresh_observe, later, apply=False)
    assert len(wire.posts) == 3 and planned.status == "planned"
    assert "accepted-concurrently" in planned.plan["destination_after"]["reservations"]
    assert wire.docs["handoff_2026-11"] == rejected
    result = run(wire, sha, fresh_observe, later)
    assert result.status == "handed_off" and len(wire.posts) == 6
    assert result.plan["plan_sha256"] != rejected["plan"]["plan_sha256"]
    assert result.plan["source_identity"]["update_time"] != rejected["plan"]["source_identity"]["update_time"]
    destination = wire.docs["2026-11"]
    assert destination["reservations"]["accepted-concurrently"]["max_gross_usd"] == ".20"
    assert destination["providers"]["apify"]["credit_baseline_usd"] == "4.50"
    assert destination["providers"]["gemini"]["automatic_observed_at"] == later.isoformat()
    completed = wire.docs["handoff_2026-11"]
    assert completed["status"] == "complete" and completed["rejected_plan_history"] == history
    assert completed["rejected_plan_history_sha256"] == auto.digest(history)


def test_unknown_rejection_marker_does_not_authorize_replan_without_readback(monkeypatch):
    now = datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    wire, sha, observe, _ = setup(monkeypatch, now)
    def conflict(server):
        if len(server.posts) == 1:
            server.before_post = conflict
            return
        server.versions["2026-10"] += 1
    wire.before_post = conflict
    wire.unknown_before_post = 3  # Canonical rejection known, durable marker unknown.
    with pytest.raises(SpendBlocked, match="outcome unknown"):
        run(wire, sha, observe, now)
    assert wire.docs["handoff_2026-11"]["status"] == "pending"
    posts, observations = len(wire.posts), wire.observations
    with pytest.raises(SpendBlocked, match="do not redispatch"):
        run(wire, sha, observe, now)
    assert len(wire.posts) == posts and wire.observations == observations


def test_rejection_marker_lost_ack_can_be_proved_without_repeating_canonical_batch(monkeypatch):
    now = datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    wire, sha, observe, _ = setup(monkeypatch, now)
    def conflict(server):
        if len(server.posts) == 1:
            server.before_post = conflict
            return
        server.versions["2026-10"] += 1
    wire.before_post = conflict
    wire.unknown_post = 3
    with pytest.raises(stores.PublicationConflict):
        run(wire, sha, observe, now)
    assert len(wire.posts) == 3 and wire.docs["handoff_2026-11"]["status"] == "known_rejected"
    assert run(wire, sha, observe, now).status == "handed_off"
    assert len(wire.posts) == 6
    assert len(wire.docs["handoff_2026-11"]["rejected_plan_history"]) == 1


def test_committed_pending_intent_reconciles_without_metadata_or_batch_replay(monkeypatch):
    now = datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    wire, sha, observe, _ = setup(monkeypatch, now)
    wire.reject_post = 3  # Canonical success; completion marker rejected.
    with pytest.raises(RuntimeError): run(wire, sha, observe, now)
    before = len(wire.posts)
    observed = wire.observations
    assert run(wire, sha, observe, now).status == "reconciled"
    assert len(wire.posts) == before+1 and wire.observations == observed
    assert len(wire.posts[-1]["writes"]) == 1


def test_cli_sanitizes_private_errors_and_never_applies_by_default(monkeypatch, capsys):
    from scripts import maintain_spend_control as cli
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("secret-token private balance 9.25")
    monkeypatch.setattr(cli, "maintain_spend_control", fail)
    assert cli.main([]) == 2
    output = capsys.readouterr().out
    assert "secret" not in output and "9.25" not in output and "private" not in output
    assert calls[0]["apply"] is False


def test_failed_dynamic_tariff_never_becomes_fresh_from_old_policy(monkeypatch):
    ledger, policy, obs = fixtures()
    class Client:
        def __init__(self, **kwargs): assert kwargs["follow_redirects"] is False
        def __enter__(self): return self
        def __exit__(self, *args): pass
    from scripts import provider_preflight
    monkeypatch.setattr(runtime.httpx, "Client", Client)
    monkeypatch.setattr(provider_preflight, "run_preflight", lambda *args, **kwargs: obs)
    from src.zeitgeist import gemini_tariff
    def unavailable(*args, **kwargs):
        raise SpendBlocked('Public Gemini tariff observation unavailable; no price inferred')
    monkeypatch.setattr(gemini_tariff, 'observe_gemini_tariff', unavailable)
    policy["providers"]["gemini"]["price_evidence_mode"] = "dynamic_snapshot"
    with pytest.raises(SpendBlocked, match="unavailable"):
        runtime.observe_provider_metadata(policy, {}, NOW)


def test_default_metadata_mapping_uses_actual_actor_tariff_and_preserves_published_evidence(monkeypatch):
    ledger, policy, obs = fixtures()
    from scripts import provider_preflight
    for name in ("gemini", "bfl"):
        pinned = policy["providers"][name]
        pinned["price_evidence_mode"] = "published_validity"
        pinned["published_contract"] = {
            "reference":pinned["price_source_url"], "tariff_sha256":pinned["price_snapshot_sha256"],
            "observed_at":"2026-10-30T15:00:00Z", "valid_from":"2026-10-01T00:00:00Z",
            "valid_until":pinned["price_valid_until"]}
    apify = policy["providers"]["apify"]
    apify["price_source_url"] = provider_preflight.APIFY_BASE+"/actors/"+provider_preflight.APIFY_ACTOR
    apify["price_snapshot_sha256"] = auto.digest(obs["apify"]["pricing"])
    sha = auto.digest(policy)
    ledger["automatic_policy_sha256"] = sha
    before_policy = copy.deepcopy(policy)
    class Client:
        def __init__(self, **kwargs): assert kwargs["follow_redirects"] is False
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def preflight(client, env, **kwargs):
        assert "UNSPLASH_ACCESS_KEY" not in env and "OTHER_SECRET" not in env
        assert kwargs["required_models"] == ["gemini-2.5-flash"]
        return copy.deepcopy(obs)
    monkeypatch.setattr(runtime.httpx, "Client", Client)
    monkeypatch.setattr(provider_preflight, "run_preflight", preflight)
    mapped = runtime.observe_provider_metadata(policy,
        {"UNSPLASH_ACCESS_KEY":"exclude", "OTHER_SECRET":"exclude"}, NOW)
    assert set(mapped["price_snapshots"]) == {"apify"}
    assert mapped["price_snapshots"]["apify"]["sha256"] == auto.digest(obs["apify"]["pricing"])
    after = auto.prepare_refresh(ledger, policy, mapped, expected_policy_sha=sha, now=NOW)["after"]
    assert policy == before_policy
    assert after["providers"]["gemini"]["verified_at"] == ledger["providers"]["gemini"]["verified_at"]


def test_automatic_refresh_observes_bfl_public_tariff_separately_from_authenticated_metadata(monkeypatch):
    ledger, policy, obs = fixtures()
    from scripts import provider_preflight
    from src.zeitgeist import bfl_tariff
    gemini = policy['providers']['gemini']
    gemini['price_evidence_mode'] = 'published_validity'
    gemini['published_contract'] = {'reference':gemini['price_source_url'],
        'tariff_sha256':gemini['price_snapshot_sha256'], 'observed_at':'2026-10-30T15:00:00Z',
        'valid_from':'2026-10-01T00:00:00Z','valid_until':gemini['price_valid_until']}
    apify = policy['providers']['apify']
    apify['price_source_url'] = provider_preflight.APIFY_BASE+'/actors/'+provider_preflight.APIFY_ACTOR
    apify['price_snapshot_sha256'] = auto.digest(obs['apify']['pricing'])
    calls=[]
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['follow_redirects'] is False
            self.headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def tariff(public_client, *, now):
        assert not public_client.headers and now==NOW
        calls.append('public_bfl_tariff')
        return copy.deepcopy(obs['price_snapshots']['bfl'])
    monkeypatch.setattr(runtime.httpx,'Client',Client)
    monkeypatch.setattr(provider_preflight,'run_preflight',lambda *args,**kwargs:copy.deepcopy(obs))
    monkeypatch.setattr(bfl_tariff,'observe_bfl_flux3_tariff',tariff)
    mapped=runtime.observe_provider_metadata(policy,{'BFL_API_KEY':'synthetic-private'},NOW)
    assert calls==['public_bfl_tariff'] and set(mapped['price_snapshots'])=={'bfl','apify'}
    ledger['automatic_policy_sha256']=auto.digest(policy)
    assert auto.prepare_refresh(ledger,policy,mapped,expected_policy_sha=auto.digest(policy),now=NOW)['after']
    mapped['price_snapshots']['bfl']['sha256']='0'*64
    with pytest.raises(SpendBlocked):
        auto.prepare_refresh(ledger,policy,mapped,expected_policy_sha=auto.digest(policy),now=NOW)


def test_gemini_current_tariff_is_observed_without_credentials_or_future_embedding_claims(monkeypatch):
    ledger,policy,obs=fixtures()
    from scripts import provider_preflight
    from src.zeitgeist import gemini_tariff
    for provider in ('apify','bfl'):
        item=policy['providers'][provider];item['price_evidence_mode']='published_validity'
    calls=[]
    class Client:
        def __init__(self,**kwargs):
            assert kwargs['follow_redirects'] is False
            self.headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def tariff(client,*,now):
        assert client.headers=={} and now==NOW
        calls.append('anonymous_gemini_tariff')
        return copy.deepcopy(obs['price_snapshots']['gemini'])
    monkeypatch.setattr(runtime.httpx,'Client',Client)
    monkeypatch.setattr(provider_preflight,'run_preflight',lambda *args,**kwargs:copy.deepcopy(obs))
    monkeypatch.setattr(gemini_tariff,'observe_gemini_tariff',tariff)
    before=copy.deepcopy(policy)
    mapped=runtime.observe_provider_metadata(policy,{'GEMINI_API_KEY':'synthetic-private-key'},NOW)
    assert calls==['anonymous_gemini_tariff'] and set(mapped['price_snapshots'])=={'gemini'}
    assert mapped['price_snapshots']['gemini']==obs['price_snapshots']['gemini']
    assert policy==before and 'published_contract' not in policy['providers']['gemini']
