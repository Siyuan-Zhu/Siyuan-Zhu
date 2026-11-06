"""The approved spend policy is the single source of the Apify Instagram build."""
from pathlib import Path
import re

import pytest

from src.zeitgeist.automatic_spend import approved_apify_build
from src.zeitgeist.collectors import IngestionUnavailable
from src.zeitgeist.spend_guard import SpendBlocked
from scripts.check_runtime_configuration import configuration_errors

POLICY = {'providers': {'apify': {'actor_build': '0.0.659', 'actor_build_id': 'Ih1cRmBcVyKVYQlOj'}}}
ROOT = Path(__file__).resolve().parents[1]


def test_policy_build_is_used_when_nothing_is_configured():
    assert approved_apify_build(POLICY, '') == '0.0.659'


def test_a_matching_configured_build_is_accepted():
    assert approved_apify_build(POLICY, '0.0.659') == '0.0.659'


def test_a_configured_build_that_differs_from_the_policy_stops_the_run():
    with pytest.raises(SpendBlocked):
        approved_apify_build(POLICY, '0.0.648')


def test_without_a_policy_a_verified_configured_build_is_still_required():
    assert approved_apify_build(None, '0.0.659') == '0.0.659'
    for bad in ('', 'latest', '0.0'):
        with pytest.raises(IngestionUnavailable):
            approved_apify_build(None, bad)


def daily_env(**extra):
    env = {'GITHUB_REPOSITORY': 'Siyuan-Zhu/Siyuan-Zhu', 'TARGET_REPO_SLUG': 'Siyuan-Zhu/Siyuan-Zhu',
           'GEMINI_API_KEY': 'x', 'UNSPLASH_ACCESS_KEY': 'x', 'APIFY_TOKEN': 'x',
           'ZEITGEIST_LIVE_COLLECT': '1', 'ZEITGEIST_LIVE_LLM': '1', 'ZEITGEIST_LIVE_UNSPLASH': '1',
           'SPEND_POLICY_SHA256': 'a' * 64, 'STATE_STORE_BACKEND': 'firestore',
           'GCP_PROJECT_ID': 'git-readme-features', 'FIRESTORE_DATABASE_ID': '(default)'}
    env.update(extra)
    return env


def build_errors(env):
    return [e for e in configuration_errors(env, daily=True) if 'APIFY_INSTAGRAM_BUILD' in e]


def test_daily_configuration_needs_no_build_variable_under_an_approved_policy():
    assert build_errors(daily_env()) == []
    assert build_errors(daily_env(APIFY_INSTAGRAM_BUILD='latest')) != []


def test_daily_configuration_without_a_policy_still_requires_the_build():
    env = daily_env()
    env.pop('SPEND_POLICY_SHA256')
    assert build_errors(env) != []


def test_no_workflow_sets_the_build_from_a_repository_variable():
    for path in (ROOT / '.github/workflows').glob('*.yml'):
        assert 'vars.APIFY_INSTAGRAM_BUILD' not in path.read_text(), path.name


def test_every_workflow_policy_pin_is_identical():
    pins = set()
    for path in (ROOT / '.github/workflows').glob('*.yml'):
        pins.update(re.findall(r"SPEND_POLICY_SHA256:\s*['\"]([a-f0-9]{64})['\"]", path.read_text()))
    assert len(pins) <= 1, pins
