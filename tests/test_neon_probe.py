"""The probe, parametrised by model and thinking level, runs Neon's calls at the requested level."""
import json
from pathlib import Path
import re

import pytest

from google.genai import errors, types

from scripts import probe_hydrogen as probe
from tests.test_hydrogen_probe import (KEY, SECRET, THINK_RECORD, THINK_USAGE, FakeGenai, FakeModels,
                                       FakeResponse, _strings, fixed_clock, usage)

ROOT = Path(__file__).resolve().parents[1]
NEON = "gemini-4-neon"
NEON_ARGS = ["--confirm-paid-probe", "--model", NEON, "--thinking-level", "medium"]
NEON_VERSION = "synthetic-neon-version"
METHODS = ["generateContent", "countTokens"]


class ListingHttp:
    def __init__(self, models=(NEON,)):
        self.models = models
        self.requests = []

    def request(self, method, url, **kw):
        self.requests.append((method, url))
        return FakeResponse({"models": [{"name": "models/" + m, "supportedGenerationMethods": METHODS}
                                        for m in self.models]})


class RefusingModels(FakeModels):
    """Refuses, like the API would with a 400, any request whose config matches refuse(config)."""

    def __init__(self, refuse, **kw):
        super().__init__([usage(12), usage(1112, image=1100), THINK_USAGE], version=NEON_VERSION, **kw)
        self.refuse = refuse

    def generate_content(self, **kw):
        if self.refuse(kw["config"]):
            self.calls.append(kw)
            raise errors.ClientError(400, {"error": {"message": KEY + SECRET, "status": "INVALID_ARGUMENT"}})
        return super().generate_content(**kw)


def neon_models(**kw):
    return FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE], version=NEON_VERSION,
                      text='{"a": 1}', **kw)


def run(argv=NEON_ARGS, http=None, models=None, capsys=None):
    http = http or ListingHttp()
    models = models or neon_models()
    made = []

    def genai_factory(key):
        made.append(key)
        return FakeGenai(models)

    code = probe.main(list(argv), env={"GEMINI_API_KEY": KEY}, http_factory=lambda: http,
                      genai_factory=genai_factory)
    full = capsys.readouterr().out
    return code, full, [json.loads(line) for line in full.splitlines()], models, http, made


def test_neon_medium_makes_the_hydrogen_calls_at_medium_and_no_embedding(capsys):
    from src.zeitgeist import collectors  # noqa: F401  (the embedder stays unused for Neon)
    code, _, rows, models, http, made = run(capsys=capsys)
    assert code == 0 and made == [KEY] and len(http.requests) == 1
    assert len(models.calls) == 9 and models.embed_calls == []
    assert all(call["model"] == NEON for call in models.calls)
    assert all(call["model"] == NEON for call in models.count_calls) and len(models.count_calls) == 8
    for call in models.calls[:2]:
        assert call["config"] == probe.CALL_CONFIG
    assert models.calls[2]["contents"] == probe.PROBE_TEXT
    assert models.calls[2]["config"] == {**probe.CALL_CONFIG, "temperature": 0.1, "candidate_count": 1}
    prompt = probe.director_fit_prompt()
    for call in models.calls[3:7]:
        assert call["contents"] == prompt
        assert call["config"] == {"max_output_tokens": 15000, "response_mime_type": "application/json",
                                  "thinking_config": {"thinking_level": "medium"}}
    critic = models.calls[7]
    assert sum(not isinstance(part, str) for part in critic["contents"]) == 9
    assert critic["config"] == {"response_mime_type": "application/json", "max_output_tokens": 15000,
                                "thinking_config": {"thinking_level": "medium"},
                                "media_resolution": "MEDIA_RESOLUTION_HIGH"}
    assert models.calls[8]["contents"] == prompt
    assert models.calls[8]["config"] == {"max_output_tokens": 1024, "response_mime_type": "application/json",
                                         "thinking_config": {"thinking_level": "medium"}}


def test_neon_lines_carry_methods_versions_and_the_summary(capsys, monkeypatch):
    fixed_clock(monkeypatch, 1.5)
    code, full, rows, *_ = run(capsys=capsys)
    assert code == 0 and len(rows) == 9
    first = rows[0]
    assert first["status"] == "ok" and first["listed"] == {NEON: METHODS} and first["missing"] == []
    assert first["text"]["model_version"] == first["image"]["model_version"] == NEON_VERSION
    assert first["tokens_per_image"] == 1100
    assert rows[1] == {"call": "temperature_check", "accepted": 1, "error_type": None, "http_status": None,
                       "model_version": NEON_VERSION, "usage_metadata": THINK_RECORD, "seconds": 1.5}
    for row in rows[2:6]:
        assert row == {"thinking_level": "medium", "max_output_tokens": 15000, "finish_reason": "STOP",
                       "prompt_tokens": 2000, "candidates_tokens": 300, "thoughts_tokens": 1500,
                       "total_output": 1800, "json_ok": 1, "counted_tokens": 1990,
                       "usage_metadata": THINK_RECORD, "seconds": 1.5, "model_version": NEON_VERSION}
    assert rows[6]["call"] == "critic" and rows[6]["thinking_level"] == "medium"
    assert rows[6]["model_version"] == NEON_VERSION and rows[6]["thoughts_tokens"] == 1500
    assert rows[7]["call"] == "cap_check" and rows[7]["model_version"] == NEON_VERSION
    assert rows[8] == {"max_total_output_medium": 1800, "max_thoughts_medium": 1500,
                       "max_seconds_medium": 1.5, "critic_total_output": 1800, "critic_thoughts": 1500,
                       "critic_seconds": 1.5, "max_prompt_minus_counted": 10,
                       "thinking_level_accepted": 1, "model_versions": [NEON_VERSION]}


def test_summary_lists_every_distinct_version_seen(capsys):
    versions = iter(["v-a", "v-a", "v-b"] + ["v-a"] * 6)

    class Versioned(FakeModels):
        def generate_content(self, **kw):
            self.version = next(versions)
            return super().generate_content(**kw)

    models = Versioned([usage(12), usage(1112, image=1100), THINK_USAGE], text="{}")
    _, _, rows, *_ = run(models=models, capsys=capsys)
    assert rows[-1]["model_versions"] == ["v-a", "v-b"]


def test_a_refused_temperature_is_recorded_and_never_fatal(capsys):
    models = RefusingModels(lambda config: "temperature" in config, text='{"a": 1}')
    code, full, rows, models, *_ = run(models=models, capsys=capsys)
    assert code == 0 and len(models.calls) == 9 and len(rows) == 9
    assert {k: rows[1][k] for k in ("call", "accepted", "error_type", "http_status", "model_version")} == {
        "call": "temperature_check", "accepted": 0, "error_type": "ClientError", "http_status": 400,
        "model_version": None}
    assert isinstance(rows[1]["seconds"], float)
    assert rows[-1]["thinking_level_accepted"] == 1
    assert KEY not in full and SECRET not in full and "INVALID_ARGUMENT" not in full


def test_a_refused_thinking_level_prints_accepted_0_with_the_error_class_and_stops(capsys):
    models = RefusingModels(lambda config: config.get("thinking_config") == {"thinking_level": "medium"})
    code, full, rows, models, *_ = run(models=models, capsys=capsys)
    assert code == 1 and len(models.calls) == 4 and models.embed_calls == []
    assert rows[-2] == {"thinking_level": "medium", "thinking_level_accepted": 0,
                        "error_type": "ClientError", "http_status": 400}
    assert rows[-1] == {"status": "failed", "phase": "thinking_medium_1", "error_type": "ClientError",
                        "http_status": 400}
    assert KEY not in full and SECRET not in full and "INVALID_ARGUMENT" not in full


def test_a_neon_critic_failure_prints_the_summary_then_the_failed_line(capsys):
    models = RefusingModels(lambda config: "media_resolution" in config and "response_mime_type" in config,
                            text='{"a": 1}')
    code, full, rows, models, *_ = run(models=models, capsys=capsys)
    assert code == 1 and len(models.calls) == 8
    assert rows[-2]["critic_total_output"] is None and rows[-2]["thinking_level_accepted"] == 1
    assert rows[-1] == {"status": "failed", "phase": "critic", "error_type": "ClientError", "http_status": 400}


def test_unlisted_neon_makes_no_paid_call(capsys):
    code, full, rows, models, http, made = run(http=ListingHttp(models=("gemini-4-hydrogen",)), capsys=capsys)
    assert code == 2 and models.calls == [] and models.count_calls == [] and made == []
    assert rows == [{"status": "neon_not_listed", "listed": {}, "missing": [NEON], "text": None, "image": None}]


@pytest.mark.parametrize("argv", [
    ["--confirm-paid-probe", "--model", "gemini-3.8-flash"],
    ["--confirm-paid-probe", "--model", NEON, "--thinking-level", "minimal"],
])
def test_unknown_model_or_level_is_refused_before_any_call(capsys, argv):
    http, models = ListingHttp(), neon_models()
    with pytest.raises(SystemExit) as exit_info:
        probe.main(argv, env={"GEMINI_API_KEY": KEY}, http_factory=lambda: http,
                   genai_factory=lambda key: FakeGenai(models))
    assert exit_info.value.code == 2 and http.requests == [] and models.calls == []
    assert KEY not in capsys.readouterr().out


def hydrogen_run(argv, capsys, monkeypatch):
    from tests.test_hydrogen_probe import FakeHttp
    fixed_clock(monkeypatch, 1.5)
    code, full, rows, models, *_ = run(argv=argv, http=FakeHttp(), capsys=capsys,
                                       models=FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE],
                                                         text='{"a": 1}'))
    return code, full, models


def test_explicit_hydrogen_high_is_the_default_run(capsys, monkeypatch):
    default = hydrogen_run(["--confirm-paid-probe"], capsys, monkeypatch)
    explicit = hydrogen_run(["--confirm-paid-probe", "--model", "gemini-4-hydrogen",
                             "--thinking-level", "high"], capsys, monkeypatch)
    assert default[0] == explicit[0] == 0 and default[1] == explicit[1]
    assert [(c["model"], c["config"]) for c in default[2].calls] == [
        (c["model"], c["config"]) for c in explicit[2].calls]
    assert len(default[2].calls) == 8 and len(default[2].embed_calls) == len(explicit[2].embed_calls) == 1


@pytest.mark.parametrize("level", ["low", "xhigh"])
def test_hydrogen_at_another_level_runs_every_thinking_call_at_it(capsys, monkeypatch, level):
    code, full, models = hydrogen_run(["--confirm-paid-probe", "--thinking-level", level], capsys, monkeypatch)
    assert code == 0 and len(models.calls) == 9 and len(models.embed_calls) == 1
    assert all(c["model"] == "gemini-4-hydrogen" for c in models.calls)
    assert [c["config"]["thinking_config"] for c in models.calls[3:]] == [{"thinking_level": level}] * 6
    assert json.loads(full.splitlines()[-2])["max_total_output_" + level] == 1800


def test_neon_leak_scan_over_all_printed_output(capsys):
    models = RefusingModels(lambda config: "temperature" in config, text=KEY + SECRET)
    code, full, rows, *_ = run(models=models, capsys=capsys)
    assert code == 0 and len(rows) == 9
    for banned in (KEY, SECRET, probe.CRITIC_WORD, probe.CRITIC_THESIS, *probe.FIT_HISTORY):
        assert banned not in full, banned
    public = {"ok", "STOP", "medium", "critic", "cap_check", "temperature_check", "ClientError",
              "IMAGE", "TEXT", NEON, NEON_VERSION, *METHODS}
    for row in rows:
        assert set(_strings(row)) <= public, row


def test_workflow_passes_the_model_and_level_inputs_as_flags_through_env():
    text = (ROOT / ".github/workflows/hydrogen_probe.yml").read_text()
    inputs = text.split("    inputs:\n", 1)[1].split("concurrency:", 1)[0]
    assert re.search(r"      model:\n(?:        .*\n)*        default: gemini-4-hydrogen\n", inputs)
    assert re.search(r"      thinking_level:\n(?:        .*\n)*        default: high\n", inputs)
    assert "- gemini-4-neon" in inputs and "type: choice" in inputs
    for level in probe.THINKING_LEVELS:
        assert "- " + level in inputs
    assert "PROBE_MODEL: ${{ inputs.model }}" in text
    assert "PROBE_THINKING_LEVEL: ${{ inputs.thinking_level }}" in text
    run_line = [line for line in text.splitlines() if line.strip().startswith("run: python3")]
    assert run_line == ['        run: python3 scripts/probe_hydrogen.py --confirm-paid-probe '
                        '--model "$PROBE_MODEL" --thinking-level "$PROBE_THINKING_LEVEL"']
    assert text.count("${{ inputs.") == 2
    assert set(re.findall(r"secrets\.([A-Z0-9_]+)", text)) == {"GEMINI_API_KEY"}
    assert "SPEND_POLICY_SHA256" not in text and "permissions:\n  contents: read\njobs:" in text


def test_model_and_level_choices_match_the_workflow():
    assert probe.PROBE_MODEL_CHOICES == ("gemini-4-hydrogen", NEON)
    assert probe.THINKING_LEVELS == ("low", "medium", "high", "xhigh")
