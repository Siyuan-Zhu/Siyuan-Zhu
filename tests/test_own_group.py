"""The own-group check: how close a word and gloss sit to their own candidate. Synthetic vectors only."""

from __future__ import annotations

import math

import pytest

from src.zeitgeist import own_group


def _item(*vector):
    return {"embedding_f32": list(vector)}


def _unit(*vector):
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


# ---- the switch ----


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("true", False), ("1", True)])
def test_the_switch_is_on_only_for_one(value, expected):
    environ = {} if value is None else {"ZEITGEIST_OWN_GROUP": value}
    assert own_group.own_group_enabled(environ) is expected


def test_the_switch_reads_the_process_environment(monkeypatch):
    monkeypatch.delenv("ZEITGEIST_OWN_GROUP", raising=False)
    assert own_group.own_group_enabled() is False
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    assert own_group.own_group_enabled() is True


# ---- the arithmetic ----


def test_the_three_figures_of_a_hand_computed_case():
    own = [_item(1, 0, 0), _item(0.6, 0.8, 0)]
    others = [[_item(0, 1, 0)], [_item(0.8, 0.6, 0), _item(0, 0, 1)]]
    numbers = own_group.own_group_numbers([1, 0, 0], own, others)
    assert numbers["status"] == "ok"
    assert numbers["own"] == pytest.approx(0.8)
    assert numbers["others"] == pytest.approx([0.0, 0.4])
    assert numbers["margin"] == pytest.approx(0.4)
    assert numbers["n_own"] == 2
    assert set(numbers) == {"status", "own", "others", "margin", "n_own"}
    assert all(type(value) is float for value in [numbers["own"], numbers["margin"], *numbers["others"]])


def test_the_text_vector_is_normalised_before_the_cosine():
    numbers = own_group.own_group_numbers([3, 0, 0], [_item(1, 0, 0)], [[_item(0, 1, 0)]])
    assert numbers["own"] == pytest.approx(1.0) and numbers["margin"] == pytest.approx(1.0)


def test_a_text_at_its_own_centre_scores_a_positive_margin():
    own = [_item(*_unit(1, 0.1, 0)), _item(*_unit(1, -0.1, 0))]
    others = [[_item(0, 1, 0), _item(0, 1, 0.1)], [_item(0, 0, 1)]]
    numbers = own_group.own_group_numbers([1, 0, 0], own, others)
    assert numbers["margin"] > 0.5


def test_a_text_that_fits_any_candidate_scores_a_margin_near_zero():
    own = [_item(1, 0, 0)]
    others = [[_item(0, 1, 0)], [_item(0, 0, 1)]]
    numbers = own_group.own_group_numbers(_unit(1, 1, 1), own, others)
    assert numbers["own"] == pytest.approx(1 / math.sqrt(3))
    assert numbers["others"] == pytest.approx([1 / math.sqrt(3)] * 2)
    assert numbers["margin"] == pytest.approx(0.0, abs=1e-6)


def test_one_candidate_only_has_no_others_and_a_null_margin():
    numbers = own_group.own_group_numbers([1, 0, 0], [_item(1, 0, 0)], [])
    assert numbers == {"status": "ok", "own": pytest.approx(1.0), "others": [], "margin": None, "n_own": 1}


def test_empty_own_items_are_skipped():
    assert own_group.own_group_numbers([1, 0, 0], [], [[_item(1, 0, 0)]]) == {"status": "skipped", "reason": "no_items"}


def test_own_items_without_vectors_are_skipped():
    numbers = own_group.own_group_numbers([1, 0, 0], [{"title": "Zqtitle"}, {"embedding_f32": [0, 0, 0]}], [])
    assert numbers == {"status": "skipped", "reason": "no_vector"}


@pytest.mark.parametrize("text_vector", [[0, 0, 0], [float("nan"), 0, 0], []])
def test_a_text_vector_that_cannot_be_used_is_skipped(text_vector):
    numbers = own_group.own_group_numbers(text_vector, [_item(1, 0, 0)], [])
    assert numbers == {"status": "skipped", "reason": "no_vector"}


def test_a_dimension_mismatch_with_the_own_items_is_skipped():
    numbers = own_group.own_group_numbers([1, 0, 0, 0], [_item(1, 0, 0)], [[_item(0, 1, 0)]])
    assert numbers == {"status": "skipped", "reason": "dimension"}


def test_a_dimension_mismatch_with_another_candidate_is_skipped():
    numbers = own_group.own_group_numbers([1, 0, 0], [_item(1, 0, 0)], [[_item(0, 1, 0, 0)]])
    assert numbers == {"status": "skipped", "reason": "dimension"}


def test_items_without_vectors_are_left_out_of_each_mean():
    own = [_item(1, 0, 0), {"title": "no vector"}]
    others = [[_item(0, 1, 0), {"title": "no vector"}], [{"title": "no vector"}]]
    numbers = own_group.own_group_numbers([1, 0, 0], own, others)
    assert numbers["own"] == pytest.approx(1.0) and numbers["n_own"] == 1
    # A candidate with no vector at all has no figure.
    assert numbers["others"] == pytest.approx([0.0])
    assert numbers["margin"] == pytest.approx(1.0)


def test_the_items_may_be_objects_with_an_f16_vector():
    from types import SimpleNamespace

    from src.zeitgeist.collectors import pack_f16_base64

    own = [SimpleNamespace(embedding_f16=pack_f16_base64([1.0, 0.0, 0.0]))]
    numbers = own_group.own_group_numbers([1, 0, 0], own, [])
    assert numbers["own"] == pytest.approx(1.0, abs=1e-3)
