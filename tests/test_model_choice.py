import pytest

from src.zeitgeist import model_choice as mc
from src.zeitgeist.spend_guard import SpendBlocked


def refused(pre=True):
    error = SpendBlocked("Unregistered paid SDK method")
    if pre:
        error.refused_before_dispatch = True
    return error


def test_unset_variables_give_config_values():
    cfg = {"budget_guardrails": {"primary_text_model": mc.HYDROGEN, "multimodal_judge_model": mc.FLASH}}
    rm = mc.resolve_run_models(cfg, {})
    assert (rm.director, rm.judge, rm.embedding) == (mc.HYDROGEN, mc.FLASH, mc.EMBED_001)
    assert not rm.fallback_fired and not rm.unrecognized


def test_empty_config_and_empty_variables_default():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: "", mc.JUDGE_ENV: ""})
    assert (rm.director, rm.judge, rm.embedding) == (mc.FLASH, mc.FLASH, mc.EMBED_001)
    assert not rm.unrecognized


def test_each_variable_overrides_its_role():
    env = {mc.DIRECTOR_ENV: mc.HYDROGEN, mc.JUDGE_ENV: mc.HYDROGEN, mc.EMBEDDING_ENV: mc.EMBED_2}
    rm = mc.resolve_run_models({}, env)
    assert (rm.director, rm.judge, rm.embedding) == (mc.HYDROGEN, mc.HYDROGEN, mc.EMBED_2)
    only_judge = mc.resolve_run_models({}, {mc.JUDGE_ENV: mc.HYDROGEN})
    assert (only_judge.director, only_judge.judge) == (mc.FLASH, mc.HYDROGEN)


def test_unrecognized_value_gives_default_and_flag():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: "gemini-5"})
    assert rm.director == mc.FLASH and rm.unrecognized
    rm = mc.resolve_run_models({}, {mc.EMBEDDING_ENV: mc.FLASH})
    assert rm.embedding == mc.EMBED_001 and rm.unrecognized


def test_hydrogen_refusal_falls_back_and_flips_both_roles():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN, mc.JUDGE_ENV: mc.HYDROGEN})
    seen = []

    def fn(model):
        seen.append(model)
        if model == mc.HYDROGEN:
            raise refused()
        return "ok"

    assert rm.call("director", fn) == "ok"
    assert seen == [mc.HYDROGEN, mc.FLASH]
    assert rm.fallback_fired
    assert rm.current("judge") == mc.FLASH and rm.current("director") == mc.FLASH


def test_judge_refusal_flips_director_too():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN, mc.JUDGE_ENV: mc.HYDROGEN})

    def fn(model):
        if model == mc.HYDROGEN:
            raise refused()
        return model

    assert rm.call("judge", fn) == mc.FLASH
    assert rm.current("director") == mc.FLASH


def test_post_dispatch_block_propagates_after_one_call():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    calls = []

    def fn(model):
        calls.append(model)
        raise refused(pre=False)

    with pytest.raises(SpendBlocked):
        rm.call("director", fn)
    assert calls == [mc.HYDROGEN]
    assert not rm.fallback_fired and rm.current("director") == mc.HYDROGEN


def test_flash_refusal_propagates_after_one_call():
    rm = mc.resolve_run_models({}, {})
    calls = []

    def fn(model):
        calls.append(model)
        raise refused()

    with pytest.raises(SpendBlocked):
        rm.call("director", fn)
    assert calls == [mc.FLASH]
    assert not rm.fallback_fired


def test_other_exception_propagates():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    calls = []

    def fn(model):
        calls.append(model)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        rm.call("director", fn)
    assert calls == [mc.HYDROGEN]


def test_active_is_none_outside_activate():
    assert mc.active() is None
    rm = mc.resolve_run_models({}, {})
    with mc.activate(rm):
        assert mc.active() is rm
    assert mc.active() is None


def test_line_format_is_exact():
    rm = mc.RunModels(director=mc.HYDROGEN, judge=mc.FLASH, embedding=mc.EMBED_2, fallback_fired=True, unrecognized=False)
    assert mc.model_choice_line(rm, 12) == (
        "[model_choice] director=gemini-4-hydrogen judge=gemini-3.8-flash "
        "embedding=gemini-embedding-2 fallback=1 unrecognized=0 slate=12"
    )
