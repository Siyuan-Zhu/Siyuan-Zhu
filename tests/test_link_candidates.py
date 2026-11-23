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
