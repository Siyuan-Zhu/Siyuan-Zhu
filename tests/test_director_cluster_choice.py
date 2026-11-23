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
FALLBACK_WORDS = {"Systole"}


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


def test_an_unshown_id_is_retried_through_the_rejection_path(repair_on):
    commission, prompts = _commission([_reply(selected_cluster_id="g_5"), _reply(selected_cluster_id="g_2")])
    assert commission.selected_cluster_id == "g_2" and commission.attempts_used == 2
    assert "PREVIOUS ATTEMPT REJECTED: " in prompts[1] and "not offered" in prompts[1]
    assert not pipeline.chosen_id_mapped(commission)


def test_a_shown_group_or_link_is_accepted_at_once(repair_on):
    groups = _groups()
    commission, _ = _commission([_reply(selected_cluster_id="g_4")], groups=groups)
    assert commission.selected_cluster_id == "g_4" and commission.attempts_used == 1
    commission, _ = _commission([_reply(selected_cluster_id="link_1")], groups=groups, links=[_link(groups)])
    assert commission.selected_cluster_id == "link_1" and commission.attempts_used == 1


def test_an_extra_group_shown_on_the_packet_path_is_accepted(repair_on):
    groups = _groups()
    commission, prompts = _commission([_reply(selected_cluster_id="g_5")], groups=groups, extra_group=groups[5])
    assert commission.selected_cluster_id == "g_5" and commission.attempts_used == 1
    assert "- Cluster g_5: " in prompts[0]


def test_on_the_last_attempt_an_unshown_id_is_mapped_to_the_top_group(repair_on):
    commission, prompts = _commission([_reply(selected_cluster_id="g_nowhere")])
    # Two packet attempts and three on the older prompt: every one but the last is rejected.
    assert len(prompts) == 5 and commission.attempts_used == 5
    assert commission.fallback_used is False and commission.word == "Selvedge"
    assert commission.word not in FALLBACK_WORDS
    assert commission.selected_cluster_id == "g_0"
    assert pipeline.chosen_id_mapped(commission) is True
    assert "selected_cluster_id" in commission.to_dict() and "chosen_id_mapped" not in json.dumps(commission.to_dict())


def test_a_link_id_on_the_older_prompt_is_not_offered(repair_on):
    groups = _groups()
    commission, prompts = _commission(
        ["not json", "not json", _reply(selected_cluster_id="link_1"), _reply(selected_cluster_id="g_1")],
        groups=groups, links=[_link(groups)])
    assert commission.selected_cluster_id == "g_1" and commission.attempts_used == 4
    assert "not offered" in prompts[3]


class _Client:
    def __init__(self, reply):
        self.reply = reply
        self.models = self
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        return SimpleNamespace(text=self.reply)


def test_the_last_attempt_is_the_last_of_the_whole_loop_across_the_flash_round(repair_on):
    client = _Client(_reply(selected_cluster_id="g_nowhere"))
    run_models = mc.RunModels(mc.HYDROGEN, mc.HYDROGEN, mc.EMBED_001)
    commission = commission_daily_zeitgeist(
        _groups(), ["Entropy"], genai_client=client, model_name=mc.HYDROGEN, run_models=run_models,
        evidence_packet=True, now_utc=NOW,
    )
    # Hydrogen's five attempts are all rejected; the Flash round's third and last attempt is accepted and mapped.
    assert client.calls == [mc.HYDROGEN] * 5 + [mc.FLASH] * 3
    assert commission.fallback_used is False and commission.word == "Selvedge"
    assert commission.selected_cluster_id == "g_0" and pipeline.chosen_id_mapped(commission) is True


def test_with_the_switch_off_an_unshown_id_is_accepted_as_today():
    commission, prompts = _commission([_reply(selected_cluster_id="g_nowhere")])
    assert commission.selected_cluster_id == "g_nowhere" and commission.attempts_used == 1 and len(prompts) == 1
    assert not pipeline.chosen_id_mapped(commission)


def test_with_links_on_and_the_switch_off_a_link_id_on_the_last_attempt_still_keeps_the_edition():
    groups = _groups()
    commission, prompts = _commission([_reply(selected_cluster_id="link_7")], groups=groups, links=[_link(groups)])
    assert len(prompts) == 5 and commission.fallback_used is False and commission.word == "Selvedge"
    assert commission.selected_cluster_id == "g_0" and pipeline.chosen_id_mapped(commission) is True


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



# ---- an id that was not shown can never cost the edition ----

LONG_GLOSS = "A gloss far too long to pass the bounded validation of fifty characters"


def _kept_reply(chosen):
    return _reply(selected_cluster_id=chosen, word="Firstword")


def _assert_kept_and_mapped(commission, links=None):
    assert commission.fallback_used is False
    assert commission.word == "Firstword" and commission.word not in FALLBACK_WORDS
    assert commission.selected_cluster_id == "g_0"
    assert pipeline.chosen_id_mapped(commission) is True
    assert pipeline.recorded_chosen_mapping(commission, links) is True


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


def test_a_handover_to_flash_inside_the_last_attempt_keeps_an_unshown_reply(repair_on):
    def answer(model, count):
        if model == mc.HYDROGEN:
            return "not json" if count < 5 else RuntimeError("synthetic Hydrogen failure")
        return _kept_reply("g_nowhere")

    client = _ScriptedClient(answer)
    run_models = mc.RunModels(mc.HYDROGEN, mc.HYDROGEN, mc.EMBED_001)
    commission = commission_daily_zeitgeist(
        _groups(), ["Entropy"], genai_client=client, model_name=mc.HYDROGEN, run_models=run_models,
        evidence_packet=True, now_utc=NOW,
    )
    # The fifth attempt's Hydrogen call fails and Flash answers it; no Flash round follows.
    assert client.calls == [mc.HYDROGEN] * 5 + [mc.FLASH]
    _assert_kept_and_mapped(commission)


def test_an_unshown_id_then_four_malformed_replies_publishes_the_first_reply(repair_on):
    commission, prompts = _commission([_kept_reply("g_nowhere")] + ["not json"] * 4)
    assert len(prompts) == 5
    _assert_kept_and_mapped(commission)


def test_an_unshown_id_then_four_gloss_failures_publishes_the_first_reply(repair_on):
    commission, prompts = _commission([_kept_reply("g_nowhere")] + [_reply(gloss=LONG_GLOSS)] * 4)
    assert len(prompts) == 5
    _assert_kept_and_mapped(commission)


def test_with_links_on_and_the_switch_off_an_unshown_link_id_then_malformed_replies_publishes_the_first():
    groups = _groups()
    links = [_link(groups)]
    commission, prompts = _commission([_kept_reply("link_9")] + ["not json"] * 4, groups=groups, links=links)
    assert len(prompts) == 5
    _assert_kept_and_mapped(commission, links)


def test_a_later_reply_valid_in_every_respect_is_still_preferred(repair_on):
    commission, prompts = _commission([_kept_reply("g_nowhere"), "not json", _reply(selected_cluster_id="g_3")])
    assert commission.word == "Selvedge" and commission.selected_cluster_id == "g_3"
    assert not pipeline.chosen_id_mapped(commission) and len(prompts) == 3


def test_a_reply_that_also_echoes_is_not_kept(repair_on):
    groups = _groups()
    groups[0].items.append({"url_hash": "orbit", "source_id": "o1_ig_demo", "bucket": "ORBIT", "title": "t",
                            "summary": "alpha bravo charlie delta echo foxtrot"})
    echo = _reply(selected_cluster_id="g_nowhere", word="Firstword",
                  commissioned_thesis="They wrote alpha bravo charlie delta echo foxtrot here.")
    commission, _ = _commission([echo] + ["not json"] * 4, groups=groups)
    assert commission.fallback_used is True and commission.word != "Firstword"

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
