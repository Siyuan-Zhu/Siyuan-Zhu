"""Refusals raised before any provider call carry a marker; synthetic data only."""
from types import SimpleNamespace

import pytest

from src.zeitgeist.spend_guard import SpendBlocked, guard_genai_client, refused_before_dispatch
from tests.test_automatic_guard import budget, client, generate, sdk_result
from tests import test_gemini38_contract as legacy


def test_unpriced_operation_is_marked_and_never_dispatched():
    api, calls = client(budget())
    with pytest.raises(SpendBlocked) as caught:
        api.models.generate_content(model='gemini-4-hydrogen', contents='Synthetic bounded request')
    assert refused_before_dispatch(caught.value) is True
    assert calls == []


def test_input_over_the_bound_is_marked_and_never_dispatched():
    raw = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: pytest.fail('dispatched'),
        count_tokens=lambda **kw: SimpleNamespace(total_tokens=70000)))
    api = guard_genai_client(raw, budget())
    with pytest.raises(SpendBlocked) as caught:
        generate(api)
    assert refused_before_dispatch(caught.value) is True


def test_version_mismatch_after_dispatch_is_not_marked():
    result = sdk_result()
    result.model_version = 'other'
    api, calls = client(budget(), result=result)
    with pytest.raises(SpendBlocked, match='exceeds the admitted request policy') as caught:
        generate(api)
    assert refused_before_dispatch(caught.value) is False
    assert len(calls) == 1


def test_provider_timeout_propagates_unchanged():
    api, calls = client(budget(), failure=TimeoutError('synthetic'))
    with pytest.raises(TimeoutError):
        generate(api)
    assert len(calls) == 1


def test_marker_is_false_for_other_errors():
    assert refused_before_dispatch(ValueError()) is False
    assert refused_before_dispatch(SpendBlocked('x')) is False


def test_legacy_guard_unpriced_operation_is_marked():
    calls = []
    raw = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: calls.append(kw),
        count_tokens=lambda **kw: SimpleNamespace(total_tokens=10)))
    api = guard_genai_client(raw, legacy.guard())
    with pytest.raises(SpendBlocked) as caught:
        api.models.generate_content(model='gemini-4-hydrogen', contents='Synthetic bounded request')
    assert refused_before_dispatch(caught.value) is True
    assert calls == []
