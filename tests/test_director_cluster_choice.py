"""The chosen-candidate repair: the Director's choice is checked against what it was shown, and the
published keywords and figures come from the candidate the word is about. Synthetic data only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import model_choice as mc
from src.zeitgeist.adk_pipeline import DirectorCommission, commission_daily_zeitgeist
from src.zeitgeist.understanding import LinkCandidate, link_keywords, public_keywords, repair_enabled
from tests.test_understanding_pipeline import _FakeStore

NOW = "2026-11-21T12:00:00Z"


def _item(title, source_id="m1_src_a", bucket="MACRO"):
    return {"url_hash": title, "source_id": source_id, "bucket": bucket, "title": title,
            "published_at": "2026-11-20T10:00:00Z"}


def _group(cid, score, keywords):
    return SimpleNamespace(cluster_id=cid, commission_score=score, velocity=score / 2, orbit_resonance=1.0 + score / 10,
                           breakout_surge=score > 2, silhouette_score=score / 10, cluster_keywords=list(keywords),
                           items=[_item(f"{cid} title {n}") for n in range(3)])


def _groups():
    return [_group(f"g_{n}", 3.0 - n * 0.4, [f"kw{n}a", f"kw{n}b"]) for n in range(6)]


def _link(groups):
    members = (groups[1], groups[3])
    return LinkCandidate("link_1", ("g_1", "g_3"), ("civic", "markets", "tech"), 4, 0.9, members)


def _reply(**over):
    data = {
        "selected_cluster_id": "g_0", "word": "Selvedge", "gloss": "A finished edge that holds.",
        "commissioned_thesis": "Quiet craft converges on durable edges.",
        "visual_anchor": "Woven linen edge in raking light",
        "unsplash_queries": ["linen weave edge", "loom detail light", "woven textile close"],
        "imagen_commission_prompt": "Close study of a woven linen edge.",
        "palette_hex": ["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"],
    }
    data.update(over)
    return json.dumps(data)


def _scripted(replies):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return replies[min(len(prompts), len(replies)) - 1]

    return llm, prompts


def _commission(replies, *, groups=None, links=None, packet=True, **kwargs):
    llm, prompts = _scripted(replies)
    commission = commission_daily_zeitgeist(
        _groups() if groups is None else groups, ["Entropy"], llm_callable=llm, evidence_packet=packet,
        now_utc=NOW, links=links, **kwargs,
    )
    return commission, prompts


@pytest.fixture
def repair_on(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")


@pytest.fixture(autouse=True)
def _switches_off_unless_set(monkeypatch):
    for name in ("ZEITGEIST_CHOSEN_GROUP_FIELDS", "ZEITGEIST_LINKS", "ZEITGEIST_RUN_RECORD", "ZEITGEIST_OWN_GROUP"):
        monkeypatch.delenv(name, raising=False)


# ---- the switch ----


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False), ("true", False)])
def test_the_switch_is_on_only_for_one(value, expected):
    assert repair_enabled({"ZEITGEIST_CHOSEN_GROUP_FIELDS": value}) is expected
    assert repair_enabled({}) is False


# ---- the Director's choice is checked against what it was shown ----


def test_an_unshown_id_is_accepted_at_once_and_mapped_to_the_top_group(repair_on):
    commission, prompts = _commission([_reply(selected_cluster_id="g_5")])
    assert len(prompts) == 1 and commission.attempts_used == 1
    assert commission.word == "Selvedge" and commission.fallback_used is False
    assert commission.selected_cluster_id == "g_0" and pipeline.chosen_id_mapped(commission) is True
    assert "PREVIOUS ATTEMPT REJECTED" not in "".join(prompts)
    assert "chosen_id_mapped" not in json.dumps(commission.to_dict())


def test_a_shown_group_or_link_is_accepted_at_once(repair_on):
    groups = _groups()
    commission, _ = _commission([_reply(selected_cluster_id="g_4")], groups=groups)
    assert commission.selected_cluster_id == "g_4" and commission.attempts_used == 1
    assert pipeline.chosen_id_mapped(commission) is False
    commission, _ = _commission([_reply(selected_cluster_id="link_1")], groups=groups, links=[_link(groups)])
    assert commission.selected_cluster_id == "link_1" and commission.attempts_used == 1


def test_an_extra_group_shown_on_the_packet_path_is_accepted(repair_on):
    groups = _groups()
    commission, prompts = _commission([_reply(selected_cluster_id="g_5")], groups=groups, extra_group=groups[5])
    assert commission.selected_cluster_id == "g_5" and commission.attempts_used == 1
    assert "- Cluster g_5: " in prompts[0]
    assert pipeline.chosen_id_mapped(commission) is False


def test_with_the_switch_off_an_unshown_id_is_accepted_as_today():
    commission, prompts = _commission([_reply(selected_cluster_id="g_nowhere")])
    assert commission.selected_cluster_id == "g_nowhere" and commission.attempts_used == 1 and len(prompts) == 1
    assert not pipeline.chosen_id_mapped(commission)


# ---- the published fields follow the chosen candidate ----


def _staged(tmp_path, monkeypatch, *, choose, links_for=None):
    """An offline staged run; the Director returns `choose(clusters, kwargs)` as its chosen id."""
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    seen = {"render": [], "bonus": [], "scored": []}
    real_score = pipeline.score_and_cluster_signals
    real_collect = pipeline.collect_96h_signals
    real_render = pipeline.select_and_render_pulse_assets

    def score(rows, **kw):
        out = real_score(rows, **kw)
        seen["scored"].append(out)
        return out

    def collect(*a, health=None, **kw):
        out = real_collect(*a, health=health, **kw)
        if health is not None:
            health["apify_batch_succeeded"] = True
        return out

    def render(*a, **kw):
        seen["render"].append(kw)
        return real_render(*a, **kw)

    def bonus(roster, items, *a, **kw):
        seen["bonus"].append(list(items))
        return roster

    def commission(clusters, **kw):
        return DirectorCommission(word="Loom", gloss="steady weaving", commissioned_thesis="Looms keep time.",
                                  selected_cluster_id=choose(clusters, kw))

    monkeypatch.setattr(pipeline, "score_and_cluster_signals", score)
    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    monkeypatch.setattr(pipeline, "apply_winning_cluster_bonus", bonus)
    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", commission)
    if links_for is not None:
        monkeypatch.setenv("ZEITGEIST_LINKS", "1")
        monkeypatch.setattr(pipeline, "link_candidates", lambda clusters, realm_map: links_for(clusters))
    roster = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=_FakeStore(tmp_path, roster))
    staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
    return seen


def _figures(kw):
    return (kw["orbit_resonance_score"], kw["velocity_score"], kw["breakout_surge"], kw["cluster_silhouette"],
            kw["cluster_keywords"])


def _of(cluster, keywords):
    return (float(cluster.orbit_resonance), float(cluster.velocity), bool(getattr(cluster, "breakout_surge", False)),
            float(cluster.silhouette_score), keywords)


@pytest.fixture
def understanding_on(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")


def test_published_fields_follow_a_chosen_second_ranked_group(tmp_path, monkeypatch, repair_on, understanding_on):
    seen = _staged(tmp_path, monkeypatch, choose=lambda clusters, kw: clusters[1].cluster_id)
    clusters = seen["scored"][0]
    assert len(clusters) >= 2
    second = clusters[1]
    [kw] = seen["render"]
    assert _figures(kw) == _of(second, public_keywords(list(second.cluster_keywords), second))


def test_with_the_switch_off_the_published_fields_stay_the_top_groups(tmp_path, monkeypatch, understanding_on):
    seen = _staged(tmp_path, monkeypatch, choose=lambda clusters, kw: clusters[1].cluster_id)
    top = seen["scored"][0][0]
    [kw] = seen["render"]
    assert _figures(kw) == _of(top, public_keywords(list(top.cluster_keywords), top))


def test_an_id_with_no_match_publishes_the_top_group(tmp_path, monkeypatch, repair_on, understanding_on):
    seen = _staged(tmp_path, monkeypatch, choose=lambda clusters, kw: "g_nowhere")
    top = seen["scored"][0][0]
    [kw] = seen["render"]
    assert _figures(kw) == _of(top, public_keywords(list(top.cluster_keywords), top))


def _link_of(clusters):
    members = (clusters[1], clusters[0]) if len(clusters) < 3 else (clusters[1], clusters[2])
    return [LinkCandidate("link_1", tuple(str(m.cluster_id) for m in members), ("civic", "markets", "tech"), 4, 0.9,
                          members)]


def test_a_chosen_link_publishes_its_merged_keywords_and_its_top_members_figures(
        tmp_path, monkeypatch, repair_on, understanding_on):
    seen = _staged(tmp_path, monkeypatch, choose=lambda clusters, kw: "link_1", links_for=_link_of)
    clusters = seen["scored"][0]
    [link] = _link_of(clusters)
    [kw] = seen["render"]
    assert _figures(kw) == _of(link.members[0], link_keywords(link))


def test_the_roster_bonus_still_goes_to_the_same_group(tmp_path, monkeypatch, understanding_on):
    off = _staged(tmp_path / "off", monkeypatch, choose=lambda clusters, kw: clusters[1].cluster_id)
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    on = _staged(tmp_path / "on", monkeypatch, choose=lambda clusters, kw: clusters[1].cluster_id)
    def hashes(seen):
        return [[getattr(item, "url_hash", None) for item in items] for items in seen["bonus"]]

    assert on["bonus"] and hashes(on) == hashes(off)


# ---- photo recovery after a link choice ----


def _photo_recovery_run(tmp_path, monkeypatch, *, revise_calls):
    """The live photo branch with fake clients: the Director chooses link_1, the photo needs one revision."""
    import sys
    import types

    from src.zeitgeist import photo_recovery, spend_guard

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("ZEITGEIST_LINKS", "1")
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    google = sys.modules.get("google")
    if google is None:
        google = types.ModuleType("google")
        monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda client, **kw: client)
    monkeypatch.setattr(pipeline, "link_candidates", lambda clusters, realm_map: _link_of(clusters))

    def commission(clusters, **kw):
        if "links" in kw:
            chosen = "link_1"
        else:
            revise_calls.append([c.cluster_id for c in clusters])
            chosen = clusters[0].cluster_id
        return DirectorCommission(word="Loom", gloss="steady weaving", commissioned_thesis="Looms keep time.",
                                  selected_cluster_id=chosen, attempts_used=1, packet_used=True)

    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", commission)
    real_render = pipeline.select_and_render_pulse_assets
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets",
                        lambda *a, **kw: real_render(*a, **{**kw, "offline_demo": True, "strict_photo_errors": False}))

    def recover(current, *, render, revise, history, photo_required):
        revised = revise(current, change_word=False)
        return revised, render(revised), {"recovered": True}

    monkeypatch.setattr(photo_recovery, "recover_visual", recover)
    real_collect = pipeline.collect_96h_signals
    monkeypatch.setattr(pipeline, "collect_96h_signals",
                        lambda *a, **kw: real_collect(*a, **{**kw, "offline_demo": True}))
    scored = []
    real_score = pipeline.score_and_cluster_signals
    monkeypatch.setattr(pipeline, "score_and_cluster_signals",
                        lambda rows, **kw: scored.append(real_score(rows, **kw)) or scored[-1])
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    roster = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=_FakeStore(tmp_path, roster))
    staged._run_staged("2026-11-21", None, offline_demo=False)
    return staged, scored[0]


@pytest.mark.parametrize("repair", [None, "1"])
def test_photo_recovery_after_a_link_choice_uses_its_top_member_and_does_not_raise(
        tmp_path, monkeypatch, understanding_on, repair):
    if repair:
        monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", repair)
    revise_calls = []
    staged, clusters = _photo_recovery_run(tmp_path, monkeypatch, revise_calls=revise_calls)
    [link] = _link_of(clusters)
    assert revise_calls == [[link.members[0].cluster_id]]
    assert staged.last_commission.selected_cluster_id == "link_1"


# ---- the workflows pass the switch, off by default ----

from pathlib import Path  # noqa: E402

from tests.test_run_record_wiring import _job_lines  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOLLAR = chr(36)
LINKS_LINE = "ZEITGEIST_LINKS: " + DOLLAR + "{{ vars.ZEITGEIST_LINKS || '0' }}"
REPAIR_LINE = "ZEITGEIST_CHOSEN_GROUP_FIELDS: " + DOLLAR + "{{ vars.ZEITGEIST_CHOSEN_GROUP_FIELDS || '0' }}"


def test_the_daily_workflow_passes_the_repair_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition[edition.index(LINKS_LINE) + 1] == REPAIR_LINE
    assert REPAIR_LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(REPAIR_LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_workflows_pass_the_repair_switch(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(REPAIR_LINE) == 1
    assert lines[lines.index(LINKS_LINE) + 1] == REPAIR_LINE


def test_the_provider_canary_does_not_pass_the_repair_switch():
    assert "ZEITGEIST_CHOSEN_GROUP_FIELDS" not in (ROOT / ".github/workflows/provider_canary.yml").read_text()



# ---- a chosen link's published keywords pass the handle filter per member ----


def test_a_chosen_links_published_keywords_drop_a_members_account_handle():
    first = _group("g_first", 3.0, ["Tide", "handle_one", "Kelp", "salt"])
    first.items.append(_item("first account post", source_id="o3_ig_handle_one", bucket="ORBIT"))
    second = _group("g_second", 2.0, ["SALT", "reef", "current", "shoal", "brine", "spray", "foam"])
    groups = [first, second] + _groups()[2:]
    link = LinkCandidate("link_1", ("g_first", "g_second"), ("culture_and_community", "markets", "tech"), 4, 0.9,
                         (first, second))
    # Each member's keywords lose the handles of every member's items, in rank order.
    assert public_keywords(first.cluster_keywords, first) == ["Tide", "Kelp", "salt"]
    assert public_keywords(second.cluster_keywords, second) == second.cluster_keywords
    field_group, published = pipeline._published_candidate("link_1", groups, [link], groups[0])
    assert field_group is first
    # The handle is gone, "SALT" repeats "salt" regardless of case, and only the first eight are kept.
    assert published == ["Tide", "Kelp", "salt", "reef", "current", "shoal", "brine", "spray"]
    assert "handle_one" not in [word.lower() for word in published]



# ---- a reply whose id was not shown is accepted at once: never worse than switches off ----

import random  # noqa: E402

from src.zeitgeist.spend_guard import SpendBlocked  # noqa: E402

LONG_GLOSS = "A gloss far too long to pass the bounded validation of fifty characters"
ECHO = "alpha bravo charlie delta echo foxtrot"
NON_LATIN = chr(0x421) + chr(0x43B) + chr(0x43E) + chr(0x432) + chr(0x43E)


class _ScriptedClient:
    """A fake genai client answering by model and call count."""

    def __init__(self, answer):
        self.answer = answer
        self.models = self
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        reply = self.answer(model, len(self.calls))
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(text=reply)


def _blocked(pre):
    error = SpendBlocked("Unregistered paid SDK method")
    if pre:
        error.refused_before_dispatch = True
    return error


def _echo_groups():
    groups = _groups()
    groups[0].items.append({"url_hash": "orbit", "source_id": "o1_ig_demo", "bucket": "ORBIT", "title": "t",
                            "summary": ECHO})
    return groups


def _play(script, monkeypatch, *, repair, links, groups=None, owner=None, packet=True, extra=False):
    """Run one scripted reply sequence on Hydrogen; the outcome, the models called and the final models.

    With `extra` the sixth group is offered as the reflector's extra candidate.
    """
    if repair:
        monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    else:
        monkeypatch.delenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", raising=False)

    def answer(model, count):
        step = script[count - 1] if count <= len(script) else "not json"
        return step(model) if callable(step) else step

    client = _ScriptedClient(answer)
    run_models = mc.RunModels(mc.HYDROGEN, mc.HYDROGEN, mc.EMBED_001)
    shown = _echo_groups() if groups is None else groups
    try:
        commission = commission_daily_zeitgeist(
            shown, ["Entropy"], genai_client=client,
            model_name=mc.HYDROGEN, run_models=run_models, evidence_packet=packet, now_utc=NOW, links=links,
            owner_override=owner, extra_group=shown[5] if extra else None,
        )
        outcome = ("published", commission.word, commission.fallback_used)
    except Exception as error:  # noqa: BLE001 - the exception type is the outcome
        commission, outcome = None, ("raised", type(error).__name__, None)
    final = (run_models.director, run_models.judge, run_models.fallback_fired)
    return outcome, client.calls, final, commission


def _same_as_off(script, monkeypatch, *, repair=True, links=None, groups=None, owner=None, packet=True, extra=False):
    off = _play(script, monkeypatch, repair=False, links=None, groups=groups, owner=owner, packet=packet)
    on = _play(script, monkeypatch, repair=repair, links=links, groups=groups, owner=owner, packet=packet,
               extra=extra)
    assert on[:3] == off[:3]
    return off, on


def _assert_mapped(on, top="g_0"):
    outcome, calls, final, commission = on
    assert outcome[0] == "published" and outcome[2] is False
    assert commission.selected_cluster_id == top and pipeline.chosen_id_mapped(commission) is True


def test_an_unshown_id_alone(monkeypatch):
    off, on = _same_as_off([_reply(selected_cluster_id="g_nowhere", word="Firstword")], monkeypatch)
    assert on[0] == ("published", "Firstword", False) and on[1] == [mc.HYDROGEN] and on[2][2] is False
    _assert_mapped(on)


def test_an_unshown_id_then_a_spend_block_after_dispatch_never_makes_the_second_call(monkeypatch):
    off, on = _same_as_off([_reply(selected_cluster_id="g_nowhere", word="Firstword"), _blocked(pre=False)],
                           monkeypatch)
    assert on[1] == [mc.HYDROGEN] and on[0] == ("published", "Firstword", False) and on[2][2] is False
    _assert_mapped(on)


def test_an_unshown_id_then_two_transport_failures(monkeypatch):
    off, on = _same_as_off([_reply(selected_cluster_id="g_nowhere", word="Firstword"),
                            RuntimeError("synthetic"), RuntimeError("synthetic")], monkeypatch)
    assert on[1] == [mc.HYDROGEN] and on[2] == (mc.HYDROGEN, mc.HYDROGEN, False)
    _assert_mapped(on)


def test_an_owner_word_night_with_an_unshown_id(monkeypatch):
    owner = {"enabled": True, "override_word": "Selvedge", "personal_note": "A synthetic note."}
    off, on = _same_as_off([_reply(selected_cluster_id="g_nowhere", word="SELVEDGE")], monkeypatch, owner=owner)
    assert on[0] == ("published", "SELVEDGE", False) and on[1] == [mc.HYDROGEN] and on[2][2] is False
    _assert_mapped(on)


def test_a_link_shaped_id_with_links_on_and_repair_off(monkeypatch):
    groups = _groups()
    links = [_link(groups)]
    off, on = _same_as_off([_reply(selected_cluster_id="link_7", word="Firstword")], monkeypatch, repair=False,
                           links=links, groups=groups)
    assert on[1] == [mc.HYDROGEN] and on[2][2] is False
    _assert_mapped(on)
    assert pipeline.recorded_chosen_mapping(on[3], links) is True


def test_hydrogen_handing_over_to_flash_inside_an_attempt_and_flash_replying_an_unshown_id(monkeypatch):
    script = [lambda model: (RuntimeError("synthetic Hydrogen failure") if model == mc.HYDROGEN
                             else _reply(selected_cluster_id="g_nowhere", word="Firstword"))] * 2
    off, on = _same_as_off(script, monkeypatch)
    assert on[1] == [mc.HYDROGEN, mc.FLASH] and on[0] == ("published", "Firstword", False)
    _assert_mapped(on)


def test_with_links_on_and_repair_off_an_unknown_plain_group_id_is_kept_and_not_mapped(monkeypatch):
    # Only a link-shaped id is checked when links alone are on; any other id is kept, as with the switches off.
    groups = _groups()
    links = [_link(groups)]
    off, on = _same_as_off([_reply(selected_cluster_id="g_nowhere", word="Firstword")], monkeypatch, repair=False,
                           links=links, groups=groups)
    assert on[0] == ("published", "Firstword", False) and on[1] == [mc.HYDROGEN]
    assert on[3].selected_cluster_id == off[3].selected_cluster_id == "g_nowhere"
    assert pipeline.chosen_id_mapped(on[3]) is False
    assert pipeline.recorded_chosen_mapping(on[3], links) is False


def test_a_link_id_replied_on_an_older_prompt_attempt_is_accepted_and_mapped(monkeypatch):
    groups = _groups()
    links = [_link(groups)]
    off, on = _same_as_off(["not json", "not json", _reply(selected_cluster_id="link_1", word="Firstword")],
                           monkeypatch, links=links, groups=groups)
    assert on[1] == [mc.HYDROGEN] * 3 and on[3].packet_used is False and on[2][2] is False
    _assert_mapped(on)
    assert off[3].selected_cluster_id == "link_1"


_KINDS = (
    lambda r: _reply(selected_cluster_id=r.choice(["g_0", "g_1", "g_4"]), word="Shownword"),
    lambda r: _reply(selected_cluster_id=r.choice(["g_5", "g_nowhere", "cluster_9"]), word="Unshownword"),
    lambda r: _reply(selected_cluster_id=r.choice(["link_1", "link_7"]), word="Linkword"),
    lambda r: _reply(selected_cluster_id=" g_2 ", word="Paddedword"),
    lambda r: "not json",
    lambda r: _reply(gloss=LONG_GLOSS, word="Glossword"),
    lambda r: _reply(word="Entropy"),
    lambda r: _reply(word=NON_LATIN),
    lambda r: _reply(word="Echoword", commissioned_thesis="They wrote " + ECHO + " here."),
    lambda r: RuntimeError("synthetic failure"),
    lambda r: _blocked(pre=True),
    lambda r: _blocked(pre=False),
)


def test_random_reply_sequences_end_exactly_as_with_the_switches_off(monkeypatch):
    rng = random.Random(20261123)
    # The reflector's extra group (the sixth) is offered only with the repair on, as in the edition.
    configs = (dict(repair=True, links=None), dict(repair=False, links=[]), dict(repair=True, links=[]),
               dict(repair=True, links=None, extra=True), dict(repair=True, links=[], extra=True))
    for _ in range(320):
        script = [rng.choice(_KINDS)(rng) for _ in range(rng.randint(1, 10))]
        packet = rng.random() < 0.8
        off = _play(script, monkeypatch, repair=False, links=None, packet=packet)
        for config in configs:
            on = _play(script, monkeypatch, packet=packet, **config)
            assert on[:3] == off[:3], (script, config)


# ---- no id check on a night with no groups ----


@pytest.mark.parametrize("links", [None, []])
def test_with_no_groups_any_id_is_accepted_as_today(repair_on, links):
    commission, prompts = _commission([_reply(selected_cluster_id="g_anything")], groups=[], links=links)
    assert len(prompts) == 1 and commission.attempts_used == 1
    assert commission.selected_cluster_id == "g_anything" and commission.fallback_used is False
    assert not pipeline.chosen_id_mapped(commission)



# ---- photo recovery's revision is not id-checked ----


def _revision_run(tmp_path, monkeypatch):
    """The live photo branch with fake clients; the main pass picks the top group, and the one revision the
    photo needs comes from the real Director call with a reply about another group."""
    import sys
    import types

    from src.zeitgeist import photo_recovery, spend_guard

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv(mc.DIRECTOR_ENV, mc.HYDROGEN)
    monkeypatch.setenv(mc.JUDGE_ENV, mc.HYDROGEN)
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    google = sys.modules.get("google")
    if google is None:
        google = types.ModuleType("google")
        monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda client, **kw: client)
    client = _ScriptedClient(lambda model, count: _reply(
        selected_cluster_id="g_other_story", word="Loom", gloss="steady weaving"))
    real_commission = commission_daily_zeitgeist  # imported before any patch
    seen = {}

    def commission(clusters, **kw):
        if "max_retries" not in kw:
            seen["run_models"] = kw["run_models"]
            return DirectorCommission(word="Loom", gloss="steady weaving", commissioned_thesis="Looms keep time.",
                                      selected_cluster_id=clusters[0].cluster_id, attempts_used=1)
        return real_commission(clusters, **{**kw, "genai_client": client})

    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", commission)
    real_render = pipeline.select_and_render_pulse_assets
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets",
                        lambda *a, **kw: real_render(*a, **{**kw, "offline_demo": True, "strict_photo_errors": False}))

    def recover(current, *, render, revise, history, photo_required):
        try:
            revise(current, change_word=False)
        except ValueError as error:
            seen["error"] = str(error)
        return current, render(current), {"recovered": False}

    monkeypatch.setattr(photo_recovery, "recover_visual", recover)
    real_collect = pipeline.collect_96h_signals
    monkeypatch.setattr(pipeline, "collect_96h_signals",
                        lambda *a, **kw: real_collect(*a, **{**kw, "offline_demo": True}))
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    roster = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=_FakeStore(tmp_path, roster))
    staged._run_staged("2026-11-21", None, offline_demo=False)
    return seen, client


def test_a_revision_about_another_group_behaves_as_with_both_switches_off(tmp_path, monkeypatch, understanding_on):
    off, off_client = _revision_run(tmp_path / "off", monkeypatch)
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    monkeypatch.setenv("ZEITGEIST_LINKS", "1")
    on, on_client = _revision_run(tmp_path / "on", monkeypatch)
    # One revision call on Hydrogen, refused by the caller as another story; no Flash round, no extra call.
    assert off_client.calls == on_client.calls == [mc.HYDROGEN]
    assert on["error"] == off["error"] == "Same-story photo recovery did not produce a grounded commission"
    for seen in (off, on):
        assert seen["run_models"].fallback_fired is False
        assert seen["run_models"].current("judge") == mc.HYDROGEN



# ---- ids are stored stripped; links are listed only when shown ----


def test_a_padded_group_or_link_id_is_stored_stripped(repair_on):
    groups = _groups()
    commission, _ = _commission([_reply(selected_cluster_id=" g_2 ")], groups=groups)
    assert commission.selected_cluster_id == "g_2"
    links = [_link(groups)]
    commission, _ = _commission([_reply(selected_cluster_id="link_1\n")], groups=groups, links=links)
    assert commission.selected_cluster_id == "link_1"
    assert pipeline._published_candidate(commission.selected_cluster_id, groups, links, groups[0])[0] is groups[1]


def test_with_the_switches_off_a_padded_id_is_kept_as_today():
    commission, _ = _commission([_reply(selected_cluster_id=" g_2 ")])
    assert commission.selected_cluster_id == " g_2 "


def test_links_count_as_shown_only_when_a_packet_attempt_showed_them():
    groups = _groups()
    links = [_link(groups)]
    shown, _ = _commission([_reply()], groups=groups, links=links)
    assert pipeline.links_shown(shown, links) == links
    older, prompts = _commission([_reply()], groups=groups, links=links, packet=False)
    assert "- Link" not in prompts[0] and pipeline.links_shown(older, links) is None
    fallback, _ = _commission(["not json"], groups=groups, links=links, packet=False)
    assert fallback.fallback_used is True and pipeline.links_shown(fallback, links) is None
    none_found, _ = _commission([_reply()], groups=groups, links=[])
    assert pipeline.links_shown(none_found, []) == []
    assert pipeline.links_shown(shown, None) is None



# ---- the reflector's extra group: never worse than switches off ----


def test_a_reply_naming_the_extra_group_is_a_shown_id_and_kept(monkeypatch):
    off, on = _same_as_off([_reply(selected_cluster_id="g_5", word="Extraword")], monkeypatch, extra=True)
    assert on[0] == ("published", "Extraword", False) and on[1] == [mc.HYDROGEN]
    assert on[3].selected_cluster_id == "g_5" and pipeline.chosen_id_mapped(on[3]) is False


def test_the_extra_groups_id_on_an_older_prompt_attempt_is_mapped_as_any_unshown_id(monkeypatch):
    off, on = _same_as_off(["not json", "not json", _reply(selected_cluster_id="g_5", word="Lateword")],
                           monkeypatch, extra=True)
    assert on[1] == [mc.HYDROGEN] * 3 and on[3].packet_used is False
    _assert_mapped(on)


def test_an_unshown_id_with_the_extra_group_offered_is_mapped(monkeypatch):
    off, on = _same_as_off([_reply(selected_cluster_id="cluster_9", word="Firstword")], monkeypatch, extra=True)
    assert on[1] == [mc.HYDROGEN]
    _assert_mapped(on)


@pytest.mark.parametrize("failure", [RuntimeError("synthetic"), _blocked(pre=True), _blocked(pre=False)],
                         ids=["transport", "refused", "blocked"])
def test_failures_with_the_extra_group_offered_end_as_with_the_switches_off(monkeypatch, failure):
    off, on = _same_as_off([failure, _reply(selected_cluster_id="g_5"), failure], monkeypatch, extra=True)
    assert on[0][0] == off[0][0] and len(on[1]) <= len(off[1])


def test_the_extra_group_is_one_more_cluster_line_and_no_sentence(repair_on):
    groups = _groups()
    _, with_extra = _commission([_reply()], groups=groups, extra_group=groups[5])
    _, without = _commission([_reply()], groups=groups)
    added = [line for line in with_extra[0].split("\n") if line not in without[0].split("\n")]
    assert len(added) == 1 and added[0].startswith("- Cluster g_5: ")



def _echoing(groups, at):
    groups[at].items.append({"url_hash": "orbit_%d" % at, "source_id": "o1_ig_demo", "bucket": "ORBIT", "title": "t",
                             "summary": ECHO})
    return groups


def test_a_reply_that_echoes_the_offered_groups_private_text_is_rejected_as_a_top_five_echo_is(repair_on):
    echoing = _reply(word="Echoword", commissioned_thesis="They wrote " + ECHO + " here.")
    replies = [echoing, _reply(word="Cleanword")]
    in_five = _echoing(_groups(), 0)
    five, five_prompts = _commission(replies, groups=in_five)
    extra_groups = _echoing(_groups(), 5)
    extra, extra_prompts = _commission(replies, groups=extra_groups, extra_group=extra_groups[5])
    for commission, prompts in ((five, five_prompts), (extra, extra_prompts)):
        assert commission.echo_rejections == 1 and commission.attempts_used == 2 and commission.word == "Cleanword"
        assert "Echoes source text" in prompts[1]
    # The same group not offered is not shown, so its text is not checked.
    unshown, _ = _commission(replies, groups=_echoing(_groups(), 5))
    assert unshown.echo_rejections == 0 and unshown.word == "Echoword"
