"""Exercise the owner workflow's actual dispatch script without provider calls."""
import os
from pathlib import Path
import subprocess
import textwrap

import pytest


def _step():
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/mobile_interject.yml"
    return workflow.read_text().split("      - name: Apply Mobile Orbit Update & Rebuild Profile\n", 1)[1].split("      - name:", 1)[0]


def _dispatch(monkeypatch, *, action="interject", mode="auto", fail_preflight=False):
    code = _step().split("python3 - <<'PY'\n", 1)[1].split("          PY\n", 1)[0]
    monkeypatch.setenv("ACTION", action)
    monkeypatch.setenv("VISUAL_MODE", mode)
    monkeypatch.setenv("WORD", "ANCHORED")
    monkeypatch.setenv("PHRASE", "Steady craft.")
    monkeypatch.setenv("NOTE", "Quiet workshop.")
    monkeypatch.setenv("GAME", "connect4")
    commands = []
    def run(command, *, check):
        assert check is True
        commands.append(command)
        if fail_preflight and "scripts/check_runtime_configuration.py" in command:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", run)
    exec(compile(textwrap.dedent(code), "mobile_dispatch", "exec"), {})
    return commands


def test_mobile_commissioning_environment_forwards_live_provider_contract():
    env = _step().split("        env:\n", 1)[1].split("        run: |", 1)[0]
    for name in ("GEMINI_API_KEY", "BFL_API_KEY", "UNSPLASH_ACCESS_KEY", "APIFY_TOKEN", "APIFY_API_TOKEN"):
        assert f"          {name}: {chr(36)}{{{{ secrets." in env
    assert "APIFY_INSTAGRAM_BUILD: " + chr(36) + "{{ vars.APIFY_INSTAGRAM_BUILD }}" in env
    for flag in ("ZEITGEIST_LIVE_LLM", "ZEITGEIST_LIVE_COLLECT", "ZEITGEIST_LIVE_UNSPLASH"):
        assert f'{flag}: "1"' in env
    assert 'ZEITGEIST_OFFLINE_DEMO: "0"' in env


@pytest.mark.parametrize("mode", ["auto", "photo", "blueprint"])
def test_all_owner_visual_modes_use_live_preflight_before_interjection(monkeypatch, mode):
    commands = _dispatch(monkeypatch, mode=mode)
    assert commands[0] == ["python3", "scripts/check_runtime_configuration.py", "--daily"]
    assert commands[1][1:3] == ["scripts/orbit.py", "interject"]
    assert commands[1][commands[1].index("--mode") + 1] == mode
    assert "ANCHORED" in commands[1] and "Steady craft." in commands[1]


def test_reset_uses_live_preflight_but_game_switch_needs_no_generation(monkeypatch):
    clear = _dispatch(monkeypatch, action="clear")
    assert clear[0][-1] == "--daily"
    assert clear[1] == ["python3", "scripts/orbit.py", "clear-interjection"]
    switch = _dispatch(monkeypatch, action="switch-game")
    assert switch[0] == ["python3", "scripts/check_runtime_configuration.py"]
    assert switch[1] == ["python3", "scripts/orbit.py", "switch-game", "connect4"]


def test_failed_provider_preflight_never_dispatches_owner_write(monkeypatch):
    with pytest.raises(subprocess.CalledProcessError):
        _dispatch(monkeypatch, fail_preflight=True)


def test_unknown_repository_dispatch_action_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="Unknown owner action"):
        _dispatch(monkeypatch, action="unexpected")
