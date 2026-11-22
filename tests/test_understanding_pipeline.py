"""Task 3: the ZEITGEIST_UNDERSTANDING_V1 switch wired into the pipeline. Synthetic data only."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist.adk_pipeline import (
    DirectorCommission,
    _build_director_prompt,
    _row_to_cached_item,
    commission_daily_zeitgeist,
)
from src.zeitgeist.methodology import CachedItem
from src.zeitgeist.spend_guard import SpendBlocked

ECHO_TEXT = "alpha bravo charlie delta echo foxtrot"
PACKET_INTRO = "Select the topical cluster most telling of the day's climate and commission a single 1-2 word"
STYLE_LINE = "Prefer a subtle, evocative editorial word grounded in the signals, rather than a blunt literal news or financial-action label."
PRIVATE_LINE = (
    "The evidence lines are private working material. Never quote them, and never name a "
    "social account or a private person that appears in them."
)
LEGACY_PROMPT = (
    "You are the Editorial Director for the Daily Zeitgeist & Personal Orbit Pulse.\n"
    "Select the most compelling topical cluster and commission a single 1-2 word "
    "English or established European loanword (Latin script ONLY) that captures its thesis.\n"
    "Prefer a subtle, evocative editorial word grounded in the signals, rather than a blunt literal news or financial-action label.\n"
    "The gloss MUST be a single line of at most 50 characters (including spaces and punctuation) "
    "and at most 12 words. Count both limits before returning; put the extended explanation in commissioned_thesis.\n"
    "\n"
    "Forbidden 30-day history words: [Entropy, Tessellation]\n"
    "Top Candidate Clusters:\n"
    "- Cluster c_one: commission_score=1.50, velocity=0.25, orbit_resonance=1.10, "
    "keywords=['loom', 'weft'], sample_titles=['Loom study', 'Weft study', 'Private note']\n"
    "- Cluster c_two: commission_score=0.75, velocity=0.50, orbit_resonance=1.00, "
    "keywords=['kiln'], sample_titles=['Kiln study']\n"
    "Return JSON with keys: selected_cluster_id (one of the provided cluster IDs), word, gloss, commissioned_thesis, visual_anchor, "
    "unsplash_queries (3 strings), imagen_commission_prompt, palette_hex (5 hex strings)."
)


def _item(title, summary="", *, bucket="MACRO", source_id="src_a", published_at="2026-11-20T10:00:00Z"):
    return CachedItem(
        url_hash=title, source_id=source_id, bucket=bucket, title=title, summary=summary,
        published_at=published_at,
    )


def _clusters():
    one = SimpleNamespace(
        cluster_id="c_one", commission_score=1.5, velocity=0.25, orbit_resonance=1.1,
        cluster_keywords=["loom", "weft"],
        items=[_item("Loom study"), _item("Weft study"),
               _item("Private note", ECHO_TEXT, bucket="ORBIT", source_id="o1_ig_demo")],
    )
    two = SimpleNamespace(
        cluster_id="c_two", commission_score=0.75, velocity=0.5, orbit_resonance=1.0,
        cluster_keywords=["kiln"], items=[_item("Kiln study")],
    )
    return [one, two]


def _reply(**over):
    data = {
        "selected_cluster_id": "c_one", "word": "Selvedge", "gloss": "A finished edge that holds.",
        "commissioned_thesis": "Quiet craft converges on durable edges.",
        "visual_anchor": "Woven linen edge in raking light",
        "unsplash_queries": ["linen weave edge", "loom detail light", "woven textile close"],
        "imagen_commission_prompt": "Close study of a woven linen edge.",
        "palette_hex": ["#0F172A", "#1E293B", "#38BDF8", "#10B981", "#F8FAFC"],
    }
    data.update(over)
    return json.dumps(data)


def _echo_reply():
    return _reply(commissioned_thesis="They wrote " + ECHO_TEXT + " somewhere.")


def _scripted(replies):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return replies[len(prompts) - 1]

    return llm, prompts


def _is_packet(prompt):
    return PACKET_INTRO in prompt


def _is_legacy(prompt):
    return "most compelling" in prompt and PACKET_INTRO not in prompt


HISTORY = ["Entropy", "Tessellation"]


def test_default_prompt_is_byte_identical():
    assert _build_director_prompt(_clusters(), HISTORY) == LEGACY_PROMPT


def test_row_to_cached_item_copies_mentioned_handles():
    row = {"source_id": "s", "title": "t", "mentioned_handles": ["demo_one", "demo_two"]}
    item = _row_to_cached_item(row, 0, "2026-11-20T00:00:00Z")
    assert item.extra["mentioned_handles"] == ["demo_one", "demo_two"]
    assert _row_to_cached_item({"source_id": "s"}, 0, "2026-11-20T00:00:00Z").extra == {}
    assert _row_to_cached_item({"source_id": "s", "mentioned_handles": []}, 0, "x").extra == {}


def test_packet_prompt_shape():
    prompt = _build_director_prompt(
        _clusters(), HISTORY, evidence_packet=True, now_utc="2026-11-21T12:00:00Z"
    )
    assert PACKET_INTRO in prompt
    assert PRIVATE_LINE in prompt
    assert STYLE_LINE in prompt
    assert "most compelling" not in prompt and "best supported" not in prompt
    assert "sample_titles=['Loom study'" in prompt
    assert "facts:" not in prompt and "evidence:" not in prompt
    assert "Loom study" in prompt
    assert "Forbidden 30-day history words: [Entropy, Tessellation]" in prompt


def test_legacy_when_packet_not_requested():
    llm, prompts = _scripted([_reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm)
    assert len(prompts) == 1 and prompts[0] == LEGACY_PROMPT
    assert commission.packet_used is False and commission.echo_rejections == 0


def test_valid_first_packet_reply():
    llm, prompts = _scripted([_reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 1 and _is_packet(prompts[0])
    assert commission.packet_used is True and commission.echo_rejections == 0
    assert commission.attempts_used == 1


def test_echo_triggers_second_packet_attempt():
    llm, prompts = _scripted([_echo_reply(), _reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 2
    assert _is_packet(prompts[1]) and "PREVIOUS ATTEMPT REJECTED: Echoes source text" in prompts[1]
    assert commission.echo_rejections == 1 and commission.packet_used is True
    assert commission.attempts_used == 2


def test_two_echoes_fall_back_to_legacy_prompt():
    llm, prompts = _scripted([_echo_reply(), _echo_reply(), _reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 3
    assert _is_packet(prompts[0]) and _is_packet(prompts[1]) and _is_legacy(prompts[2])
    assert commission.word == "Selvedge" and commission.fallback_used is False
    assert commission.packet_used is False and commission.echo_rejections == 2


def test_legacy_attempt_has_no_banner_for_packet_rejections():
    for bad in (_echo_reply(), _reply(word="Selvedge", commissioned_thesis="See cluster 7 for more.")):
        llm, prompts = _scripted([bad, bad, _reply()])
        commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
        assert len(prompts) == 3 and _is_legacy(prompts[2])
        assert "PREVIOUS ATTEMPT REJECTED" not in prompts[2]
        assert "PREVIOUS ATTEMPT REJECTED" in prompts[1]


def test_legacy_banner_kept_for_todays_reasons():
    bad = _reply(word="Entropy")
    llm, prompts = _scripted([_echo_reply(), _echo_reply(), bad, _reply()])
    commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert _is_legacy(prompts[3]) and "PREVIOUS ATTEMPT REJECTED" in prompts[3]
    assert "PREVIOUS ATTEMPT REJECTED" not in prompts[2]


def test_packet_attempts_are_additional():
    bad = _reply(word="Entropy")
    llm, prompts = _scripted([_echo_reply(), _echo_reply(), bad, _reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 4
    assert _is_packet(prompts[0]) and _is_packet(prompts[1])
    assert _is_legacy(prompts[2]) and _is_legacy(prompts[3])
    assert commission.word == "Selvedge" and commission.fallback_used is False
    assert commission.attempts_used == 4 and commission.packet_used is False
    assert commission.echo_rejections == 2


def test_all_attempts_fail_falls_back_after_five_calls():
    bad = _reply(word="Entropy")
    llm, prompts = _scripted([_echo_reply(), _echo_reply(), bad, bad, bad])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 5 and commission.fallback_used is True
    assert "PREVIOUS ATTEMPT REJECTED" not in prompts[2]


def test_one_retry_is_one_packet_then_two_legacy():
    bad = _reply(word="Entropy")
    llm, prompts = _scripted([_echo_reply(), bad, bad])
    commission = commission_daily_zeitgeist(
        _clusters(), HISTORY, llm_callable=llm, evidence_packet=True, max_retries=1
    )
    assert len(prompts) == 3 and commission.fallback_used is True
    assert _is_packet(prompts[0]) and _is_legacy(prompts[1]) and _is_legacy(prompts[2])
    assert "PREVIOUS ATTEMPT REJECTED" not in prompts[1]


def test_switch_off_prompts_match_base():
    bad = _reply(word="Entropy")
    llm, prompts = _scripted([bad, bad, _reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm)
    assert len(prompts) == 3 and all(_is_legacy(p) for p in prompts)
    assert prompts[0] == LEGACY_PROMPT and "PREVIOUS ATTEMPT REJECTED" not in prompts[0]
    assert all("PREVIOUS ATTEMPT REJECTED" in p for p in prompts[1:])
    assert commission.attempts_used == 3 and commission.packet_used is False


def test_zero_retries_is_legacy_and_ungated():
    llm, prompts = _scripted([_echo_reply()])
    commission = commission_daily_zeitgeist(
        _clusters(), HISTORY, llm_callable=llm, evidence_packet=True, max_retries=0
    )
    assert len(prompts) == 1 and prompts[0] == LEGACY_PROMPT
    assert commission.fallback_used is False and ECHO_TEXT in commission.commissioned_thesis
    assert commission.packet_used is False and commission.echo_rejections == 0


def test_owner_constrained_call_never_uses_packet():
    owner = {"enabled": True, "override_word": "Selvedge", "override_phrase": "A finished edge that holds."}
    llm, prompts = _scripted([_reply()])
    commission_daily_zeitgeist(
        _clusters(), HISTORY, llm_callable=llm, evidence_packet=True, owner_override=owner
    )
    assert prompts and all(_is_legacy(p.split("\nOWNER EDITORIAL")[0]) for p in prompts)


def test_spend_blocked_and_runtime_errors_propagate():
    def blocked(prompt):
        raise SpendBlocked("blocked")

    def broken(prompt):
        raise RuntimeError("boom")

    with pytest.raises(SpendBlocked):
        commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=blocked, evidence_packet=True)
    with pytest.raises(RuntimeError):
        commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=broken, evidence_packet=True)


def test_packet_build_failure_uses_legacy_prompt_and_skips_echo(monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("packet failure")

    monkeypatch.setattr(pipeline, "render_evidence_packet", explode)
    llm, prompts = _scripted([_echo_reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 1 and prompts[0] == LEGACY_PROMPT
    assert commission.packet_used is False and commission.echo_rejections == 0


def test_packet_build_failure_spends_only_legacy_attempts(monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("packet failure")

    monkeypatch.setattr(pipeline, "render_evidence_packet", explode)
    bad = _reply(word="Entropy")
    llm, prompts = _scripted([bad, bad, bad, bad, bad])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 3 and commission.fallback_used is True
    assert all(_is_legacy(p) for p in prompts)
    assert commission.attempts_used == len(prompts)
    assert "PREVIOUS ATTEMPT REJECTED" not in prompts[0]


def test_echo_check_failure_never_raises(monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("echo failure")

    monkeypatch.setattr(pipeline, "echo_violation", explode)
    llm, prompts = _scripted([_echo_reply()])
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm, evidence_packet=True)
    assert len(prompts) == 1 and commission.packet_used is True and commission.echo_rejections == 0


def test_commission_dict_unchanged_when_defaults():
    plain = DirectorCommission(word="Selvedge", gloss="g", commissioned_thesis="t").to_dict()
    assert "packet_used" not in plain and "echo_rejections" not in plain
    flagged = DirectorCommission(word="Selvedge", gloss="g", commissioned_thesis="t", packet_used=True).to_dict()
    assert flagged["packet_used"] is True


# ---- pipeline wiring ----

LINE_RE = re.compile(
    r"^\[understanding_v1\] top_changed=[01] new_top_legacy_rank=\d+ attempt=\d+ packet=[01] "
    r"echo_rejections=\d+ final_echo=[01]"
)


def _run_offline(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    seen = {"cohesion": [], "bonus": [], "legacy_top": [], "scored": []}
    real_score = pipeline.score_and_cluster_signals
    real_collect = pipeline.collect_96h_signals
    real_legacy = pipeline.legacy_top_cluster

    def score(rows, **kw):
        seen["cohesion"].append(kw.get("cohesion_mode"))
        out = real_score(rows, **kw)
        seen["scored"].append(out)
        return out

    def collect(*a, health=None, **kw):
        out = real_collect(*a, health=health, **kw)
        if health is not None:
            health["apify_batch_succeeded"] = True
        return out

    def legacy(clusters):
        top = real_legacy(clusters)
        seen["legacy_top"].append(top)
        return top

    def bonus(roster, items, *a, **kw):
        seen["bonus"].append(list(items))
        return roster

    monkeypatch.setattr(pipeline, "score_and_cluster_signals", score)
    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)
    monkeypatch.setattr(pipeline, "legacy_top_cluster", legacy)
    monkeypatch.setattr(pipeline, "apply_winning_cluster_bonus", bonus)
    store = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=_FakeStore(tmp_path, store))
    staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
    return seen, staged


class _FakeStore:
    """Real JSON store plus the two roster hooks the bonus path needs."""

    def __init__(self, root, roster):
        from src.state_store import get_state_store

        self._inner = get_state_store(repo_root=root)
        self._roster = roster

    def __getattr__(self, name):
        if name in ("load_orbit_roster", "save_orbit_roster"):
            return getattr(self._roster, name)
        return getattr(self._inner, name)


def _spy_keywords(monkeypatch):
    got = {"public": [], "render": []}
    real_public = pipeline.public_keywords
    real_render = pipeline.select_and_render_pulse_assets

    def public(kws, cluster):
        out = real_public(kws, cluster)
        got["public"].append((list(kws), out))
        return ["PUBLIC_MARK"]

    def render(*a, **kw):
        got["render"].append(kw.get("cluster_keywords"))
        return real_render(*a, **kw)

    monkeypatch.setattr(pipeline, "public_keywords", public)
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    return got


def test_pipeline_public_keywords_on(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    got = _spy_keywords(monkeypatch)
    _run_offline(tmp_path, monkeypatch)
    assert got["public"] and got["render"] and got["render"][0] == ["PUBLIC_MARK"]


def test_pipeline_public_keywords_off(tmp_path, monkeypatch):
    monkeypatch.delenv("ZEITGEIST_UNDERSTANDING_V1", raising=False)
    got = _spy_keywords(monkeypatch)
    _run_offline(tmp_path, monkeypatch)
    assert got["public"] == [] and got["render"] and got["render"][0] != ["PUBLIC_MARK"]


def test_pipeline_switch_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    seen, staged = _run_offline(tmp_path, monkeypatch)
    assert seen["cohesion"] == [None]
    assert seen["scored"] and seen["scored"][0]
    assert all(c.commission_score == c.legacy_commission_score for c in seen["scored"][0])
    assert seen["bonus"] and seen["legacy_top"]
    assert seen["bonus"][0] == list(seen["legacy_top"][0].items)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[understanding_v1]")]
    assert len(lines) == 1 and LINE_RE.fullmatch(lines[0]), lines


def test_pipeline_switch_off(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ZEITGEIST_UNDERSTANDING_V1", raising=False)
    seen, staged = _run_offline(tmp_path, monkeypatch)
    assert seen["cohesion"] == [None]
    assert seen["legacy_top"] == []
    assert "[understanding_v1]" not in capsys.readouterr().out


def _run_photo_recovery(tmp_path, monkeypatch, *, recover):
    """Run with a fake model client so the photo-recovery branch runs; no network."""
    import sys
    import types

    from src.zeitgeist import photo_recovery, spend_guard

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    google = sys.modules.get("google")
    if google is None:
        # Registered through monkeypatch so the stand-in package is removed again after the test.
        google = types.ModuleType("google")
        monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda client, **kw: client)
    monkeypatch.setattr(
        pipeline, "commission_daily_zeitgeist",
        lambda clusters, **kw: DirectorCommission(
            word="Loom", gloss="steady weaving", commissioned_thesis="Looms keep time.",
            selected_cluster_id=clusters[0].cluster_id,
            attempts_used=3, packet_used=True, echo_rejections=2,
        ),
    )
    real_render = pipeline.select_and_render_pulse_assets
    monkeypatch.setattr(
        pipeline, "select_and_render_pulse_assets",
        lambda *a, **kw: real_render(*a, **{**kw, "offline_demo": True, "strict_photo_errors": False}),
    )
    monkeypatch.setattr(photo_recovery, "recover_visual", recover)
    real_collect = pipeline.collect_96h_signals
    monkeypatch.setattr(
        pipeline, "collect_96h_signals", lambda *a, **kw: real_collect(*a, **{**kw, "offline_demo": True}),
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    store = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=_FakeStore(tmp_path, store))
    staged._run_staged("2026-11-21", None, offline_demo=False)
    return staged


def _orbit_text(clusters):
    for cluster in list(clusters)[:5]:
        for it in cluster.items:
            if str(getattr(it, "bucket", "")).upper() == "ORBIT":
                return f"{it.title} {it.summary}"
    return ""


def _recovering(scored):
    def recover(commission, *, render, revise, history, photo_required):
        revised = DirectorCommission(
            word="Loom", gloss="steady weaving", commissioned_thesis=_orbit_text(scored[0]),
            selected_cluster_id=commission.selected_cluster_id,
        )
        return revised, render(revised), {"recovered": True}

    return recover


def _no_recovery(commission, *, render, revise, history, photo_required):
    return commission, render(commission), {"recovered": False}


def _spy_scored(monkeypatch):
    scored = []
    real_score = pipeline.score_and_cluster_signals

    def score(rows, **kw):
        out = real_score(rows, **kw)
        scored.append(out)
        return out

    monkeypatch.setattr(pipeline, "score_and_cluster_signals", score)
    return scored


def _understanding_lines(capsys):
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[understanding_v1]")]


def test_final_echo_measures_recovered_commission(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    scored = _spy_scored(monkeypatch)
    staged = _run_photo_recovery(tmp_path, monkeypatch, recover=_recovering(scored))
    lines = _understanding_lines(capsys)
    assert scored and _orbit_text(scored[0]) and staged.last_commission.commissioned_thesis == _orbit_text(scored[0])
    assert len(lines) == 1 and LINE_RE.fullmatch(lines[0]), lines
    assert lines[0].endswith("final_echo=1")


def test_counters_come_from_original_commission(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    scored = _spy_scored(monkeypatch)
    _run_photo_recovery(tmp_path, monkeypatch, recover=_recovering(scored))
    lines = _understanding_lines(capsys)
    assert len(lines) == 1
    assert "attempt=3 packet=1 echo_rejections=2 final_echo=1" in lines[0], lines


def test_one_line_when_no_recovery_change(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    _run_photo_recovery(tmp_path, monkeypatch, recover=_no_recovery)
    lines = _understanding_lines(capsys)
    assert len(lines) == 1 and LINE_RE.fullmatch(lines[0]), lines
    assert lines[0].endswith("final_echo=0")


def test_no_line_with_switch_off_after_recovery(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ZEITGEIST_UNDERSTANDING_V1", raising=False)
    scored = _spy_scored(monkeypatch)
    _run_photo_recovery(tmp_path, monkeypatch, recover=_recovering(scored))
    assert _understanding_lines(capsys) == []
