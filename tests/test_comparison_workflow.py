"""The dispatch-only comparison run workflow: the private run's job with two inputs reaching the script by env."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import test_apify_build_source as apify_build_source

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/comparison_run.yml"
PRIVATE = ROOT / ".github/workflows/private_run.yml"
D = chr(36)
MAINTENANCE = ("scripts/maintain_spend_control.py", "--apply", "--private-report",
               "/tmp/private-spend-maintenance/evidence.json")
PIN = re.compile(r"SPEND_POLICY_SHA256:\s*['\"]([a-f0-9]{64})['\"]")


def _text():
    return WORKFLOW.read_text(encoding="utf-8")


def _top_level(text, key):
    """The top-level block that starts with key, up to the next top-level line."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(key + ":"))
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line and not line.startswith(" ") and not line.startswith("#"):
            break
        block.append(line)
    return "\n".join(block).rstrip()


def _job_header(text):
    """The job's lines from its condition through the install step."""
    job = text.split("\njobs:\n", 1)[1]
    return job[job.index("    if:"):job.index("      - run: pip install -r requirements.txt")]


def _step_env(text):
    step = text.split("        env:\n", 1)[1]
    lines = []
    for line in step.splitlines():
        if not line.startswith("          "):
            break
        lines.append(line.strip())
    return lines


def _run_body(text):
    body = text.split("        run: |\n", 1)[1]
    lines = []
    for line in body.splitlines():
        if line.strip() and not line.startswith("          "):
            break
        lines.append(line[10:])
    return "\n".join(lines).strip() + "\n"


def test_dispatch_only_with_the_six_inputs():
    on = _top_level(_text(), "on")
    assert "workflow_dispatch:" in on
    for trigger in ("schedule", "push", "pull_request", "workflow_run", "repository_dispatch"):
        assert trigger not in on, trigger
    # The arms are optional now: a dispatch may run the reflector alone.
    assert re.search(r"\n      arms:\n(?: {8}.*\n)*? {8}required: false", on + "\n")
    assert re.search(r"\n      arms:\n(?: {8}.*\n)*? {8}default: \"\"", on + "\n")
    assert re.search(r"\n      repeats:\n(?: {8}.*\n)*? {8}default: \"3\"", on + "\n")
    assert re.search(r"\n      reflect:\n(?: {8}.*\n)*? {8}default: \"false\"", on + "\n")
    assert re.search(r"\n      reflect_arm:\n(?: {8}.*\n)*? {8}default: \"current\"", on + "\n")
    assert re.search(r"\n      reflect_model:\n(?: {8}.*\n)*? {8}default: \"\"", on + "\n")
    assert re.search(r"\n      reflect_allow_same_model:\n(?: {8}.*\n)*? {8}default: \"false\"", on + "\n")
    assert re.findall(r"\n      ([a-z_]+):\n", on) == ["arms", "repeats", "reflect", "reflect_arm", "reflect_model",
                                                       "reflect_allow_same_model"]


def test_condition_permissions_concurrency_timeout_and_setup_match_the_private_run():
    text, private = _text(), PRIVATE.read_text(encoding="utf-8")
    for key in ("permissions", "concurrency"):
        assert _top_level(text, key) == _top_level(private, key), key
    # The same job header but its timeout: a comparison makes up to 12 naming attempts, two photo
    # paths and one re-embedding, so it gets 25 minutes where the private run has 15.
    assert "    timeout-minutes: 15\n" in _job_header(private)
    # It also checks who triggered it, as owner_repair.yml does: a re-run by someone else is refused.
    assert _job_header(text) == _job_header(private).replace(
        "    timeout-minutes: 15\n", "    timeout-minutes: 25\n").replace(
        "github.actor == 'Siyuan-Zhu' &&", "github.actor == 'Siyuan-Zhu' && github.triggering_actor == 'Siyuan-Zhu' &&")
    assert text.count("timeout-minutes:") == 1
    assert "contents: write" not in text and "contents: read" in text
    assert "github.actor == 'Siyuan-Zhu'" in _job_header(text)
    for banned in ("upload-artifact", "publish_readme", "git push", "git commit"):
        assert banned not in text, banned


def test_the_job_checks_the_triggering_actor_beside_the_actor_as_owner_repair_does():
    header = _job_header(_text())
    owner_repair = (ROOT / ".github/workflows/owner_repair.yml").read_text(encoding="utf-8")
    pair = "github.actor == 'Siyuan-Zhu' && github.triggering_actor == 'Siyuan-Zhu' &&"
    assert pair in owner_repair
    assert pair in header and header.count("github.triggering_actor") == 1


def test_the_env_block_is_the_private_runs_plus_the_six_inputs():
    text, private = _text(), PRIVATE.read_text(encoding="utf-8")
    assert _step_env(text) == _step_env(private) + [
        "COMPARISON_ARMS: " + D + "{{ inputs.arms }}",
        "COMPARISON_REPEATS: " + D + "{{ inputs.repeats }}",
        "REFLECT: " + D + "{{ inputs.reflect }}",
        "REFLECT_ARM: " + D + "{{ inputs.reflect_arm }}",
        "REFLECT_MODEL: " + D + "{{ inputs.reflect_model }}",
        "REFLECT_ALLOW_SAME_MODEL: " + D + "{{ inputs.reflect_allow_same_model }}",
    ]
    assert PIN.findall(text) == PIN.findall(private) and len(PIN.findall(text)) == 1


def test_every_workflow_pin_is_still_identical():
    apify_build_source.test_every_workflow_policy_pin_is_identical()


def test_the_inputs_reach_the_script_only_through_the_environment():
    text = _text()
    assert "shell: python\n" in text
    body = _run_body(text)
    assert D not in body and "inputs." not in body
    run_lines = [line for line in text.splitlines() if line.strip().startswith("run:")]
    assert all("inputs." not in line and D not in line for line in run_lines)
    assert sum("inputs." in line for line in text.splitlines()) == 6
    assert ('[sys.executable, "scripts/private_run.py", "--arms", os.environ.get("COMPARISON_ARMS", ""), '
            '"--repeats", os.environ.get("COMPARISON_REPEATS", "")]') in body
    assert ('[sys.executable, "scripts/reflect.py", "--kind", "private_run", "--arm", '
            'os.environ.get("REFLECT_ARM", "")]') in body


def _run_the_body(monkeypatch, codes, *, arms="previous,current", reflect=None, reflect_arm="current",
                  reflect_model=None, allow_same=None):
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, codes[len(calls) - 1])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("COMPARISON_ARMS", arms)
    monkeypatch.setenv("COMPARISON_REPEATS", "2")
    if reflect is None:
        monkeypatch.delenv("REFLECT", raising=False)
    else:
        monkeypatch.setenv("REFLECT", reflect)
    monkeypatch.setenv("REFLECT_ARM", reflect_arm)
    if reflect_model is None:
        monkeypatch.delenv("REFLECT_MODEL", raising=False)
    else:
        monkeypatch.setenv("REFLECT_MODEL", reflect_model)
    if allow_same is None:
        monkeypatch.delenv("REFLECT_ALLOW_SAME_MODEL", raising=False)
    else:
        monkeypatch.setenv("REFLECT_ALLOW_SAME_MODEL", allow_same)
    with pytest.raises(SystemExit) as stopped:
        exec(compile(_run_body(_text()), "comparison_run.yml", "exec"), {"__name__": "__main__"})
    return calls, stopped.value.code


def test_the_body_runs_maintenance_then_the_arms_and_passes_the_exit_code(monkeypatch):
    calls, code = _run_the_body(monkeypatch, [0, 3])
    assert calls == [[sys.executable, *MAINTENANCE],
                     [sys.executable, "scripts/private_run.py", "--arms", "previous,current", "--repeats", "2"]]
    assert code == 3


def test_a_failed_maintenance_stops_before_the_script(monkeypatch):
    calls, code = _run_the_body(monkeypatch, [1])
    assert calls == [[sys.executable, *MAINTENANCE]] and code == 1


def test_a_clean_run_exits_zero(monkeypatch):
    _, code = _run_the_body(monkeypatch, [0, 0])
    assert code == 0


ARMS_COMMAND = [sys.executable, "scripts/private_run.py", "--arms", "previous,current", "--repeats", "2"]


def _reflect_command(arm="current"):
    return [sys.executable, "scripts/reflect.py", "--kind", "private_run", "--arm", arm]


@pytest.mark.parametrize("reflect", [None, "false", "", "True", "1"])
def test_the_reflector_runs_only_when_reflect_is_true(monkeypatch, reflect):
    calls, code = _run_the_body(monkeypatch, [0, 0], reflect=reflect)
    assert calls == [[sys.executable, *MAINTENANCE], ARMS_COMMAND] and code == 0


def test_with_reflect_true_the_reflector_runs_after_the_arms_on_the_named_arm(monkeypatch):
    calls, code = _run_the_body(monkeypatch, [0, 0, 0], reflect="true", reflect_arm="previous")
    assert calls == [[sys.executable, *MAINTENANCE], ARMS_COMMAND, _reflect_command("previous")] and code == 0


def test_the_reflector_runs_after_failed_arms_and_the_arms_exit_code_is_kept(monkeypatch):
    calls, code = _run_the_body(monkeypatch, [0, 3, 0], reflect="true")
    assert calls == [[sys.executable, *MAINTENANCE], ARMS_COMMAND, _reflect_command()] and code == 3


@pytest.mark.parametrize("arms", ["", "  "])
def test_empty_arms_run_no_arms(monkeypatch, arms):
    calls, code = _run_the_body(monkeypatch, [0], arms=arms)
    assert calls == [[sys.executable, *MAINTENANCE]] and code == 0
    calls, code = _run_the_body(monkeypatch, [0, 0], arms=arms, reflect="true")
    assert calls == [[sys.executable, *MAINTENANCE], _reflect_command()] and code == 0


def test_a_failed_maintenance_stops_before_the_arms_and_the_reflector(monkeypatch):
    calls, code = _run_the_body(monkeypatch, [1], reflect="true")
    assert calls == [[sys.executable, *MAINTENANCE]] and code == 1


@pytest.mark.parametrize("model", [None, "", "  "])
def test_without_a_reflect_model_the_reflector_runs_as_before(monkeypatch, model):
    calls, code = _run_the_body(monkeypatch, [0, 0, 0], reflect="true", reflect_model=model)
    assert calls == [[sys.executable, *MAINTENANCE], ARMS_COMMAND, _reflect_command()] and code == 0


def test_a_reflect_model_is_passed_alone(monkeypatch):
    calls, code = _run_the_body(monkeypatch, [0, 0, 0], reflect="true", reflect_arm="previous",
                                reflect_model="gemini-4-hydrogen")
    assert calls[-1] == _reflect_command("previous") + ["--model", "gemini-4-hydrogen"]
    assert code == 0


@pytest.mark.parametrize("allow_same, passed", [(None, False), ("false", False), ("", False), ("True", False),
                                                ("1", False), ("true", True)])
def test_the_same_model_override_is_passed_only_for_true(monkeypatch, allow_same, passed):
    calls, _ = _run_the_body(monkeypatch, [0, 0, 0], reflect="true", allow_same=allow_same)
    assert calls[-1] == _reflect_command() + (["--allow-same-model"] if passed else [])
    calls, _ = _run_the_body(monkeypatch, [0, 0, 0], reflect="true", reflect_model="gemini-4-hydrogen",
                             allow_same=allow_same)
    assert calls[-1] == _reflect_command() + ["--model", "gemini-4-hydrogen"] + (
        ["--allow-same-model"] if passed else [])


def test_a_reflect_model_without_reflect_runs_no_reflector(monkeypatch):
    calls, _ = _run_the_body(monkeypatch, [0, 0], reflect_model="gemini-4-hydrogen")
    assert calls == [[sys.executable, *MAINTENANCE], ARMS_COMMAND]
