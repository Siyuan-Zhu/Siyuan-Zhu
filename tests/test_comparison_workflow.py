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


def test_dispatch_only_with_the_two_inputs():
    on = _top_level(_text(), "on")
    assert "workflow_dispatch:" in on
    for trigger in ("schedule", "push", "pull_request", "workflow_run", "repository_dispatch"):
        assert trigger not in on, trigger
    assert re.search(r"\n      arms:\n(?: {8}.*\n)*? {8}required: true", on + "\n")
    assert re.search(r"\n      arms:\n(?: {8}.*\n)*? {8}default: \"previous,current\"", on + "\n")
    assert re.search(r"\n      repeats:\n(?: {8}.*\n)*? {8}default: \"3\"", on + "\n")


def test_condition_permissions_concurrency_timeout_and_setup_match_the_private_run():
    text, private = _text(), PRIVATE.read_text(encoding="utf-8")
    for key in ("permissions", "concurrency"):
        assert _top_level(text, key) == _top_level(private, key), key
    assert _job_header(text) == _job_header(private)
    assert "contents: write" not in text and "contents: read" in text
    assert "github.actor == 'Siyuan-Zhu'" in _job_header(text)
    for banned in ("upload-artifact", "publish_readme", "git push", "git commit"):
        assert banned not in text, banned


def test_the_env_block_is_the_private_runs_plus_the_two_inputs():
    text, private = _text(), PRIVATE.read_text(encoding="utf-8")
    assert _step_env(text) == _step_env(private) + [
        "COMPARISON_ARMS: " + D + "{{ inputs.arms }}",
        "COMPARISON_REPEATS: " + D + "{{ inputs.repeats }}",
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
    assert sum("inputs." in line for line in text.splitlines()) == 2
    assert ('[sys.executable, "scripts/private_run.py", "--arms", os.environ.get("COMPARISON_ARMS", ""), '
            '"--repeats", os.environ.get("COMPARISON_REPEATS", "")]') in body


def _run_the_body(monkeypatch, codes):
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, codes[len(calls) - 1])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("COMPARISON_ARMS", "previous,current")
    monkeypatch.setenv("COMPARISON_REPEATS", "2")
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
