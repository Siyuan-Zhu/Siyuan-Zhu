"""Links between groups in different realms: the night's realm map, the candidates and their line.

Synthetic feeds, handles, roster entries and items only.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from src.state_store import JsonFileStateStore
from src.zeitgeist.understanding import REALMS, realm_counts, source_realm_map

REPO_ROOT = Path(__file__).resolve().parents[1]

FEEDS = [
    {"id": "o3_culture_feed", "category": "culture"},
    {"id": "o2_pulse_feed", "category": "cultural_pulse"},
    {"id": "m1_tech_feed", "category": "tech_community"},
    {"id": "m2_markets_feed", "category": "financial_markets"},
    {"id": "o3_design_feed", "category": "design"},
    {"id": "o3_science_feed", "category": "science"},
    {"id": "m2_civic_feed", "category": "policy"},
]


def _map(private=None, roster=None):
    return source_realm_map(FEEDS, private, roster)


# ---- the realm map ----


def test_the_realms_are_the_six_named() -> None:
    assert REALMS == ("culture_and_community", "design", "tech", "markets", "civic", "science")


def test_feeds_take_the_layer_realm_with_culture_renamed() -> None:
    realms = _map()
    assert realms.get("o3_culture_feed") == "culture_and_community"
    assert realms.get("o2_pulse_feed") == "culture_and_community"
    assert realms.get("m1_tech_feed") == "tech"
    assert realms.get("m2_markets_feed") == "markets"
    assert realms.get("o3_design_feed") == "design"
    assert realms.get("o3_science_feed") == "science"
    assert realms.get("m2_civic_feed") == "civic"
    assert "culture" not in set(dict(realms).values())


def test_every_instagram_source_is_culture_and_community() -> None:
    realms = _map()
    for source_id in ("o3_ig_alpha_account", "o4_ig_beta.scout", "o1_ig_gamma"):
        assert realms.get(source_id) == "culture_and_community"
        assert realms[source_id] == "culture_and_community"
        assert source_id in realms


def test_culture_feeds_and_instagram_share_one_realm() -> None:
    realms = _map()
    assert realms.get("o3_culture_feed") == realms.get("o3_ig_alpha_account")


def test_a_private_entry_wins_for_a_feed_by_feed_id() -> None:
    realms = _map({"o3_culture_feed": "science", "m1_tech_feed": "design"})
    assert realms.get("o3_culture_feed") == "science"
    assert realms.get("m1_tech_feed") == "design"
    assert realms.get("o2_pulse_feed") == "culture_and_community"


def test_a_private_entry_wins_for_an_instagram_account_by_bare_handle() -> None:
    realms = _map({"alpha_account": "design"})
    assert realms.get("o3_ig_alpha_account") == "design"
    assert realms.get("o4_ig_alpha_account") == "design"
    assert realms.get("o3_ig_other_account") == "culture_and_community"


def test_a_private_value_outside_the_realms_is_ignored() -> None:
    realms = _map({"alpha_account": "culture", "o3_design_feed": "faith_realm",
                   "m1_tech_feed": 7, "beta_account": "community"})
    assert realms.get("o3_ig_alpha_account") == "culture_and_community"
    assert realms.get("o3_design_feed") == "design"
    assert realms.get("m1_tech_feed") == "tech"
    assert realms.get("o3_ig_beta_account") == "culture_and_community"


@pytest.mark.parametrize("private", [None, "not a map", ["design"], 42, {}])
def test_an_absent_or_unreadable_private_map_leaves_the_defaults(private) -> None:
    realms = _map(private)
    assert realms.get("o3_culture_feed") == "culture_and_community"
    assert realms.get("o3_ig_alpha_account") == "culture_and_community"
    assert realms.get("m1_tech_feed") == "tech"


def _roster(**parents):
    return {"handles": {handle: {"handle": handle, "parent_sources": list(ps)}
                        for handle, ps in parents.items()}}


def test_a_discovered_account_takes_the_realm_its_parents_agree_on() -> None:
    private = {"seed_one": "design", "seed_two": "design", "seed_three": "tech"}
    roster = _roster(found_scout=["o3_ig_seed_one", "o3_ig_seed_two"],
                     lone_scout=["o3_ig_seed_three"])
    realms = _map(private, roster)
    assert realms.get("o4_ig_found_scout") == "design"
    assert realms.get("o4_ig_lone_scout") == "tech"


def test_a_discovered_account_with_differing_parents_is_culture_and_community() -> None:
    private = {"seed_one": "design", "seed_three": "tech"}
    roster = _roster(mixed_scout=["o3_ig_seed_one", "o3_ig_seed_three"])
    assert _map(private, roster).get("o4_ig_mixed_scout") == "culture_and_community"


def test_a_discovered_account_with_unlabelled_parents_is_culture_and_community() -> None:
    roster = _roster(plain_scout=["o3_ig_seed_one", "o3_ig_seed_two"],
                     orphan_scout=[], world_scout=["w_unknown_source"])
    realms = _map({}, roster)
    assert realms.get("o4_ig_plain_scout") == "culture_and_community"
    assert realms.get("o4_ig_orphan_scout") == "culture_and_community"
    assert realms.get("o4_ig_world_scout") == "culture_and_community"


def test_a_discovered_account_with_feed_parents_takes_the_feed_realm() -> None:
    roster = _roster(feed_scout=["m1_tech_feed"])
    assert _map({}, roster).get("o4_ig_feed_scout") == "tech"


def test_a_private_entry_wins_over_a_discovered_account_parents() -> None:
    private = {"seed_one": "design", "seed_two": "design", "found_scout": "markets"}
    roster = _roster(found_scout=["o3_ig_seed_one", "o3_ig_seed_two"])
    assert _map(private, roster).get("o4_ig_found_scout") == "markets"


def test_the_older_parent_field_is_read_too() -> None:
    roster = {"handles": {"old_scout": {"distinct_parent_sources": ["o3_ig_seed_one"]}}}
    assert _map({"seed_one": "science"}, roster).get("o4_ig_old_scout") == "science"


@pytest.mark.parametrize("roster", [None, "bad", {"handles": "bad"}, {"handles": {"x": "bad"}}])
def test_an_unreadable_roster_leaves_the_defaults(roster) -> None:
    realms = _map({}, roster)
    assert realms.get("o4_ig_some_scout") == "culture_and_community"


def test_a_discovered_world_source_gets_no_realm() -> None:
    realms = _map({"w_discovered_site": "tech"}, None)
    assert realms.get("w_discovered_site") is None
    assert "w_discovered_site" not in realms
    with pytest.raises(KeyError):
        realms["w_discovered_site"]


def test_the_map_does_not_show_its_entries_when_printed() -> None:
    realms = _map({"alpha_account": "design", "o3_culture_feed": "science"})
    for text in (repr(realms), str(realms)):
        assert "alpha_account" not in text
        assert "o3_culture_feed" not in text
        assert "design" not in text


# ---- realm counts ----


def test_realm_counts_count_items_by_realm_with_none_for_the_rest() -> None:
    realms = _map({"alpha_account": "design"})
    items = [
        {"source_id": "o3_culture_feed"},
        {"source_id": "o3_ig_beta_account"},
        {"source_id": "o3_ig_alpha_account"},
        {"source_id": "m1_tech_feed"},
        {"source_id": "w_discovered_site"},
        {},
    ]
    assert realm_counts(items, realms) == {
        "culture_and_community": 2, "design": 1, "tech": 1, "none": 2,
    }


def test_realm_counts_of_nothing_is_empty() -> None:
    assert realm_counts([], _map()) == {}


def test_neither_function_prints_or_logs(capsys, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    private = {"alpha_account": "design", "bad_account": "not_a_realm"}
    realms = source_realm_map(FEEDS, private, _roster(found=["o3_ig_alpha_account"]))
    realm_counts([{"source_id": "o4_ig_found"}], realms)
    source_realm_map(FEEDS, "unreadable", "unreadable")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert caplog.records == []


# ---- the private map in the store ----


def test_the_private_map_round_trips_on_the_json_backend(tmp_path) -> None:
    store = JsonFileStateStore(repo_root=tmp_path)
    assert store.load_calibration("orbit_realms") is None
    document = {"alpha_account": "design", "o3_culture_feed": "science"}
    store.save_calibration("orbit_realms", document)
    assert (tmp_path / "data" / "orbit_realms.json").is_file()
    assert store.load_calibration("orbit_realms") == document


def test_git_ignores_the_json_private_map(tmp_path) -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "check-ignore", "-q", "data/orbit_realms.json"],
        check=False,
    )
    assert result.returncode == 0


# ---- Task 3: links offered to the Director ----

import hashlib  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from src.zeitgeist import adk_pipeline as pipeline  # noqa: E402
from src.zeitgeist.adk_pipeline import commission_daily_zeitgeist  # noqa: E402
from src.zeitgeist.understanding import (  # noqa: E402
    LinkCandidate,
    link_candidates,
    link_keywords,
    link_titles,
    links_enabled,
    render_evidence_packet,
)

DIM = 24
NIGHT_FEEDS = [
    {"id": f"m1_tech_{n}", "category": "tech_ai"} for n in (1, 2)
] + [
    {"id": f"m2_markets_{n}", "category": "economics"} for n in (1, 2)
] + [
    {"id": f"o3_design_{n}", "category": "design"} for n in (1, 2)
] + [
    {"id": f"o3_science_{n}", "category": "science"} for n in (1, 2)
] + [
    {"id": f"m2_civic_{n}", "category": "policy"} for n in (1, 2)
] + [
    {"id": f"o3_culture_{n}", "category": "culture"} for n in (1, 2)
]
ECHO_TEXT = "alpha bravo charlie delta echo foxtrot"


def _vector(axis, wobble):
    vector = np.zeros(DIM)
    vector[axis] = 1.0
    vector[wobble] += 0.05
    return list(vector / np.linalg.norm(vector))


def _group(cid, score, sources, axis, wobble, keywords=None, titles=None, orbit_text=None):
    titles = titles or [f"{cid} title {n}" for n in range(3)]
    items = []
    for n, source in enumerate((sources[0], sources[0], sources[1])):
        item = {"url_hash": f"{cid}_{n}", "source_id": source, "title": titles[n],
                "embedding_f32": _vector(axis, wobble)}
        if orbit_text is not None and n == 2:
            item.update(bucket="ORBIT", summary=orbit_text)
        items.append(item)
    return SimpleNamespace(cluster_id=cid, commission_score=score, velocity=1.0 + score, orbit_resonance=1.2,
                           cluster_keywords=list(keywords or [cid + " word"]), items=items)


def _night(third_link=True, ig_pair=("o3_ig_alpha_one", "o3_ig_beta_two")):
    """Ranked groups: link A (tech, markets, Instagram) on axis 0, link B (design, science, civic) on axis 1,
    link C (tech, markets, design) on axis 2 when asked, and three lone groups."""
    groups = [
        _group("g_zeta", 9.0, ("m1_tech_1", "m1_tech_2"), 0, 10, ["Loom", "weft"]),
        _group("g_mid", 8.5, ("o3_design_1", "o3_design_2"), 1, 11),
        _group("g_alpha", 8.0, ig_pair, 0, 12, ["loom", "alpha_one", "kiln"]),
        _group("g_lone_1", 7.5, ("m1_tech_1", "m1_tech_2"), 3, 13),
        _group("g_beta", 7.0, ("m2_markets_1", "m2_markets_2"), 0, 14, ["thread", "Weft", "dye"]),
        _group("g_lone_2", 6.5, ("m2_markets_1", "m2_markets_2"), 4, 15),
        _group("g_sci", 6.0, ("o3_science_1", "o3_science_2"), 1, 16),
        _group("g_civ", 5.5, ("m2_civic_1", "m2_civic_2"), 1, 17, orbit_text=ECHO_TEXT),
        _group("g_lone_3", 5.0, ("o3_culture_1", "o3_culture_2"), 5, 18),
    ]
    if third_link:
        groups += [
            _group("g_c1", 1.0, ("m1_tech_1", "m1_tech_2"), 2, 22),
            _group("g_c2", 0.9, ("m2_markets_1", "m2_markets_2"), 2, 23),
            _group("g_c3", 0.8, ("o3_design_1", "o3_design_2"), 2, 9),
        ]
    return groups


def _night_map(private=None):
    return source_realm_map(NIGHT_FEEDS, private, None)


def test_the_links_switch_is_on_only_for_one(monkeypatch) -> None:
    assert links_enabled({"ZEITGEIST_LINKS": "1"}) is True
    for value in ("0", "true", "", "yes"):
        assert links_enabled({"ZEITGEIST_LINKS": value}) is False
    assert links_enabled({}) is False
    monkeypatch.setenv("ZEITGEIST_LINKS", "1")
    assert links_enabled() is True


def test_at_most_two_links_with_members_in_rank_order() -> None:
    links = link_candidates(_night(), _night_map())
    assert [link.link_id for link in links] == ["link_1", "link_2"]
    first, second = links
    assert first.member_ids == ("g_zeta", "g_alpha", "g_beta")
    assert first.realms == ("culture_and_community", "markets", "tech")
    assert first.source_count == 6
    assert 0.9 < first.mean_link_similarity <= 1.0
    assert [m.cluster_id for m in first.members] == list(first.member_ids)
    assert second.member_ids == ("g_mid", "g_sci", "g_civ")
    assert len(first.items) == 9


def test_no_link_on_a_night_without_one() -> None:
    groups = [g for g in _night(third_link=False) if not g.cluster_id.startswith("g_lone")][:2]
    assert link_candidates(groups, _night_map()) == []
    assert link_candidates([], _night_map()) == []


def test_culture_feeds_and_instagram_cannot_make_a_link_between_them() -> None:
    # Link A's third realm is the Instagram group; with a culture feed in its place it is the same realm
    # as the Instagram group would be, and with both culture and Instagram only two realms remain.
    groups = _night(third_link=False, ig_pair=("o3_culture_1", "o3_culture_2"))
    assert [l.member_ids for l in link_candidates(groups, _night_map())][0] == ("g_zeta", "g_alpha", "g_beta")
    groups = _night(third_link=False)
    groups[4] = _group("g_beta", 7.0, ("o3_culture_1", "o3_culture_2"), 0, 14)
    found = [l.member_ids for l in link_candidates(groups, _night_map())]
    assert ("g_zeta", "g_alpha", "g_beta") not in found


def test_a_private_realm_feeds_the_links() -> None:
    groups = _night(third_link=False)
    groups[4] = _group("g_beta", 7.0, ("o3_culture_1", "o3_culture_2"), 0, 14)
    private = {"alpha_one": "design", "beta_two": "design"}
    found = link_candidates(groups, _night_map(private))
    assert found[0].member_ids == ("g_zeta", "g_alpha", "g_beta")
    assert found[0].realms == ("culture_and_community", "design", "tech")


def test_merged_keywords_follow_rank_drop_handles_and_duplicates_and_keep_eight() -> None:
    [first, _] = link_candidates(_night(), _night_map())
    # "alpha_one" is a handle of the Instagram group's own items, so it is not public.
    assert link_keywords(first) == ["Loom", "weft", "kiln", "thread", "dye"]
    many = [_group(f"g_k{n}", 1.0, ("m1_tech_1", "m1_tech_2"), 0, 10, [f"k{n}_{m}" for m in range(3)])
            for n in range(4)]
    link = LinkCandidate("link_1", tuple(g.cluster_id for g in many), ("tech",), 2, 0.9, tuple(many))
    assert link_keywords(link) == ["k0_0", "k0_1", "k0_2", "k1_0", "k1_1", "k1_2", "k2_0", "k2_1"]


def test_titles_go_round_the_members_until_there_are_four() -> None:
    members = [_group(f"g_t{n}", 1.0, ("m1_tech_1", "m1_tech_2"), 0, 10,
                      titles=[f"member{n} first", f"member{n} second", f"member{n} third"]) for n in range(3)]
    link = LinkCandidate("link_1", ("g_t0", "g_t1", "g_t2"), ("tech",), 2, 0.9, tuple(members))
    titles = link_titles(link)
    assert len(titles) == 4
    assert [t.split()[0] for t in titles] == ["member0", "member1", "member2", "member0"]
    lone = LinkCandidate("link_1", ("g_t0",), ("tech",), 2, 0.9, (members[0],))
    assert len(link_titles(lone)) == 3


def test_a_link_is_one_line_after_the_five_groups() -> None:
    groups = _night()
    links = link_candidates(groups, _night_map())
    plain = render_evidence_packet(groups)
    packet = render_evidence_packet(groups, links=links)
    lines = packet.split("\n")
    assert lines[:5] == plain.split("\n") and len(lines) == 7
    first = links[0]
    assert lines[5] == (
        "- Link link_1: groups=['g_zeta', 'g_alpha', 'g_beta'], "
        "realms=['culture_and_community', 'markets', 'tech'], sources=6, "
        "similarity=%.2f, keywords=%r, sample_titles=%r"
        % (first.mean_link_similarity, link_keywords(first), link_titles(first))
    )
    assert lines[6].startswith("- Link link_2: groups=['g_mid', 'g_sci', 'g_civ'], ")


def test_without_links_the_packet_is_todays() -> None:
    groups = _night()
    assert render_evidence_packet(groups, links=None) == render_evidence_packet(groups)
    assert render_evidence_packet(groups, links=[]) == render_evidence_packet(groups)
    assert render_evidence_packet(groups, links=[], extra_group=None) == render_evidence_packet(groups)


def test_an_extra_group_is_one_more_cluster_line_after_the_links() -> None:
    groups = _night()
    links = link_candidates(groups, _night_map())
    lines = render_evidence_packet(groups, links=links, extra_group=groups[8]).split("\n")
    assert len(lines) == 8 and lines[7].startswith("- Cluster g_lone_3: ")


# ---- the Director and the links ----


def _reply(**over):
    data = {
        "selected_cluster_id": "g_zeta", "word": "Selvedge", "gloss": "A finished edge that holds.",
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


def _commission(replies, links, groups=None):
    llm, prompts = _scripted(replies)
    commission = commission_daily_zeitgeist(
        groups if groups is not None else _night(), ["Entropy"], llm_callable=llm, evidence_packet=True,
        now_utc="2026-11-21T12:00:00Z", links=links,
    )
    return commission, prompts


def test_the_packet_prompt_shows_the_links_after_the_groups() -> None:
    links = link_candidates(_night(), _night_map())
    _, prompts = _commission([_reply()], links)
    block = render_evidence_packet(_night(), now_utc="2026-11-21T12:00:00Z", links=links)
    assert "Top Candidate Clusters:\n" + block + "\n" + LINK_SENTENCE + "\nReturn JSON" in prompts[0]


def test_a_shown_link_id_is_accepted() -> None:
    links = link_candidates(_night(), _night_map())
    commission, prompts = _commission([_reply(selected_cluster_id="link_2")], links)
    assert commission.fallback_used is False and commission.attempts_used == 1
    assert commission.selected_cluster_id == "link_2"


def test_an_unshown_link_id_is_rejected_and_retried() -> None:
    links = link_candidates(_night(), _night_map())[:1]
    commission, prompts = _commission([_reply(selected_cluster_id="link_2"), _reply(selected_cluster_id="link_1")],
                                      links)
    assert commission.selected_cluster_id == "link_1" and commission.attempts_used == 2
    assert "PREVIOUS ATTEMPT REJECTED: " in prompts[1] and "not offered" in prompts[1]


def test_a_link_id_where_no_link_was_shown_is_rejected() -> None:
    commission, prompts = _commission([_reply(selected_cluster_id="link_1"), _reply()], [])
    assert commission.selected_cluster_id == "g_zeta" and commission.attempts_used == 2
    assert "not offered" in prompts[1]


def test_a_link_id_on_the_older_prompt_is_rejected() -> None:
    links = link_candidates(_night(), _night_map())
    commission, prompts = _commission(
        ["not json", "not json", _reply(selected_cluster_id="link_1"), _reply()], links)
    assert "- Link link_1" in prompts[0] and "- Link link_1" not in prompts[2]
    assert commission.packet_used is False and commission.attempts_used == 4
    assert commission.selected_cluster_id == "g_zeta"
    assert "not offered" in prompts[3]


def test_without_links_a_link_id_is_as_today() -> None:
    llm, _ = _scripted([_reply(selected_cluster_id="link_1")])
    commission = commission_daily_zeitgeist(_night(), ["Entropy"], llm_callable=llm, evidence_packet=True,
                                            now_utc="2026-11-21T12:00:00Z")
    assert commission.selected_cluster_id == "link_1" and commission.attempts_used == 1


def test_the_echo_check_sees_the_items_of_every_link_member() -> None:
    links = link_candidates(_night(), _night_map())
    assert "g_civ" in links[1].member_ids  # ranked eighth: not among the five
    echo = _reply(commissioned_thesis="They wrote " + ECHO_TEXT + " somewhere.")
    commission, prompts = _commission([echo, _reply()], links)
    assert commission.echo_rejections == 1 and commission.attempts_used == 2
    assert "Echoes source text" in prompts[1]
    plain, _ = _commission([echo], None)
    assert plain.echo_rejections == 0 and plain.attempts_used == 1


def test_the_packet_builder_passes_links_through() -> None:
    links = link_candidates(_night(), _night_map())
    prompt = pipeline._build_packet_director_prompt(_night(), ["Entropy"], None, "2026-11-21T12:00:00Z",
                                                    links=links)
    assert "- Link link_1: " in prompt and "- Link link_2: " in prompt
    legacy = pipeline._build_director_prompt(_night(), ["Entropy"])
    assert "- Link" not in legacy


def test_night_links_reads_the_private_map_and_roster_and_never_prints(capsys, caplog) -> None:
    caplog.set_level(logging.DEBUG)

    class Store:
        def __init__(self):
            self.reads = []

        def load_calibration(self, name):
            self.reads.append(name)
            return {"alpha_one": "design", "beta_two": "design"}

        def load_orbit_roster(self):
            self.reads.append("roster")
            return {"handles": {}}

    store = Store()
    groups = _night(third_link=False)
    groups[4] = _group("g_beta", 7.0, ("o3_culture_1", "o3_culture_2"), 0, 14)
    links, realm_map = pipeline._night_links(store, {"feeds": NIGHT_FEEDS}, groups)
    assert store.reads == ["orbit_realms", "roster"]
    assert links[0].member_ids == ("g_zeta", "g_alpha", "g_beta")
    assert realm_map.get("o3_ig_alpha_one") == "design"
    out = capsys.readouterr()
    assert out.out == out.err == "" and caplog.records == []


@pytest.mark.parametrize("failure", ["calibration", "roster", "both", "missing"])
def test_night_links_runs_on_the_defaults_when_the_store_fails(failure, capsys, caplog) -> None:
    caplog.set_level(logging.DEBUG)

    class Store:
        def load_calibration(self, name):
            if failure in ("calibration", "both"):
                raise RuntimeError("Zqsecret store failure alpha_one")
            return {"alpha_one": "not_a_realm"}

        def load_orbit_roster(self):
            if failure in ("roster", "both"):
                raise RuntimeError("Zqsecret roster failure")
            return "unreadable"

    store = object() if failure == "missing" else Store()
    links, realm_map = pipeline._night_links(store, {"feeds": NIGHT_FEEDS}, _night())
    assert [link.link_id for link in links] == ["link_1", "link_2"]
    assert realm_map.get("o3_ig_alpha_one") == "culture_and_community"
    out = capsys.readouterr()
    assert out.out == out.err == "" and caplog.records == []


def test_night_links_never_raises() -> None:
    links, realm_map = pipeline._night_links(None, None, None)
    assert links == [] and realm_map.get("o3_ig_alpha_one") == "culture_and_community"
    assert pipeline._night_links(object(), {"feeds": "bad"}, [object()])[0] == []



# ---- the Director is told what a link line is, only when one is shown ----

LINK_SENTENCE = ("A Link line is one more candidate: it joins groups from different realms whose items sit "
                 "unusually close tonight. To choose it, give its id as selected_cluster_id.")


def _packet_prompt(links, **kwargs):
    return pipeline._build_packet_director_prompt(_night(), ["Entropy"], None, "2026-11-21T12:00:00Z", links=links,
                                                  **kwargs)


def test_the_link_sentence_follows_the_candidate_lines_when_a_link_is_shown() -> None:
    links = link_candidates(_night(), _night_map())
    for one_or_two in (links[:1], links):
        prompt = _packet_prompt(one_or_two)
        block = render_evidence_packet(_night(), now_utc="2026-11-21T12:00:00Z", links=one_or_two)
        assert prompt.count(LINK_SENTENCE) == 1
        assert ("Top Candidate Clusters:\n" + block + "\n" + LINK_SENTENCE + "\nReturn JSON with keys") in prompt
    with_extra = _packet_prompt(links, extra_group=_night()[8])
    assert ("- Cluster g_lone_3: " in with_extra
            and with_extra.index("- Cluster g_lone_3: ") < with_extra.index(LINK_SENTENCE))
    slate = _packet_prompt(links, slate=True)
    assert slate.count(LINK_SENTENCE) == 1


def test_the_link_sentence_is_absent_without_a_link() -> None:
    for links in (None, []):
        assert LINK_SENTENCE not in _packet_prompt(links)
    assert LINK_SENTENCE not in _packet_prompt(None, extra_group=_night()[8])
    assert LINK_SENTENCE not in pipeline._build_director_prompt(_night(), ["Entropy"])


def test_the_kept_candidate_block_is_the_candidate_lines_only() -> None:
    links = link_candidates(_night(), _night_map())
    _, block = pipeline._director_prompt_and_block(_night(), ["Entropy"], evidence_packet=True,
                                                   now_utc="2026-11-21T12:00:00Z", links=links)
    assert LINK_SENTENCE not in block
    assert block == render_evidence_packet(_night(), now_utc="2026-11-21T12:00:00Z", links=links)
