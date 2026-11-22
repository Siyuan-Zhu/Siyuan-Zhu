"""The Hydrogen probe prints versions and token counts only and refuses unsafe runs."""
from io import BytesIO
import json
from pathlib import Path
import re

from google.genai import types
from PIL import Image

from scripts import probe_hydrogen as probe

ROOT = Path(__file__).resolve().parents[1]
KEY = "synthetic-key"
SECRET = "SECRET-RESPONSE-TEXT"


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.text = json.dumps(body)


class FakeHttp:
    def __init__(self, models=("gemini-4-hydrogen", "gemini-embedding-2"), fail=None):
        self.models = models
        self.fail = fail
        self.requests = []

    def request(self, method, url, **kw):
        self.requests.append((method, url))
        if self.fail:
            raise self.fail
        return FakeResponse({"models": [
            {"name": "models/" + m, "supportedGenerationMethods": ["generateContent"]}
            for m in self.models
        ]})


def usage(prompt, image=None, total=20):
    details = None
    if image is not None:
        details = [types.ModalityTokenCount(modality=types.MediaModality.IMAGE, token_count=image),
                   types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=12)]
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt, candidates_token_count=3, thoughts_token_count=2,
        total_token_count=total, prompt_tokens_details=details)


class FakeModels:
    def __init__(self, usages, version="synthetic-hydrogen-version", fail_at=None,
                 text=SECRET, finish_reason=types.FinishReason.STOP):
        self.text = text
        self.finish_reason = finish_reason
        self.usages = list(usages)
        self.version = version
        self.fail_at = fail_at
        self.calls = []

    def generate_content(self, **kw):
        self.calls.append(kw)
        if self.fail_at == len(self.calls):
            raise RuntimeError("key=" + KEY)
        index = min(len(self.calls), len(self.usages)) - 1
        response = types.GenerateContentResponse(
            model_version=self.version, response_id="r",
            usage_metadata=self.usages[index],
            candidates=[types.Candidate(
                finish_reason=self.finish_reason,
                content=types.Content(parts=[types.Part(text=self.text)], role="model"))])
        return response


class FakeGenai:
    def __init__(self, models):
        self.models = models


def default_models(**kw):
    return FakeModels([usage(12), usage(1112, image=1100)], **kw)


def run_main(argv=("--confirm-paid-probe",), http=None, models=None, env=None, capsys=None):
    http = http or FakeHttp()
    models = models or default_models()
    made = []

    def genai_factory(key):
        made.append(key)
        return FakeGenai(models)

    code = probe.main(list(argv), env={"GEMINI_API_KEY": KEY} if env is None else env,
                      http_factory=lambda: http, genai_factory=genai_factory)
    full = capsys.readouterr().out
    run_main.full = full
    return code, full.splitlines()[0] if full else full, models, http, made


def test_report_has_only_versions_and_counts(capsys):
    report = probe.run_probe(KEY, http_client=FakeHttp(), genai_client=FakeGenai(default_models()))
    assert set(report) == {"listed", "missing", "text", "image"}
    assert set(report["text"]) == {"model_version", "prompt_token_count", "candidates_token_count",
                                   "thoughts_token_count", "total_token_count"}
    assert set(report["image"]) == set(report["text"]) | {"image_prompt_token_count"}
    code, out, *_ = run_main(capsys=capsys)
    assert code == 0
    for banned in (KEY, SECRET, "masked", "fingerprint"):
        assert banned not in run_main.full
    assert json.loads(out)["image"]["model_version"] == "synthetic-hydrogen-version"


def test_two_calls_second_has_384px_image(capsys):
    _, _, models, _, made = run_main(capsys=capsys)
    assert len(models.calls) == 6 and made == [KEY]  # two A8 calls, then four thinking-fit calls
    for call in models.calls[:2]:
        assert call["model"] == "gemini-4-hydrogen"
        assert call["config"] == {"max_output_tokens": 16,
                                  "thinking_config": {"thinking_level": "low"},
                                  "media_resolution": "MEDIA_RESOLUTION_HIGH"}
    assert models.calls[0]["contents"] == probe.PROBE_TEXT
    part, text = models.calls[1]["contents"]
    assert text == probe.PROBE_TEXT and part.inline_data.mime_type == "image/png"
    assert Image.open(BytesIO(part.inline_data.data)).size == (384, 384)


def test_tokens_per_image(capsys):
    _, out, *_ = run_main(capsys=capsys)
    assert json.loads(out)["tokens_per_image"] == 1100
    models = FakeModels([usage(12), usage(1132)])
    _, out, *_ = run_main(models=models, capsys=capsys)
    assert json.loads(out)["tokens_per_image"] == 1120


def test_unlisted_hydrogen_makes_no_paid_call(capsys):
    code, out, models, http, made = run_main(http=FakeHttp(models=("gemini-embedding-2",)), capsys=capsys)
    assert code == 2 and models.calls == [] and made == []
    assert len(http.requests) == 1


def test_failure_prints_type_only(capsys):
    code, out, models, *_ = run_main(models=default_models(fail_at=2), capsys=capsys)
    assert code == 1
    assert json.loads(out) == {"status": "failed", "phase": "image", "error_type": "RuntimeError"}
    assert KEY not in run_main.full and SECRET not in run_main.full
    code, out, *_ = run_main(models=default_models(fail_at=1), capsys=capsys)
    assert json.loads(out)["phase"] == "text"
    assert KEY not in run_main.full and SECRET not in run_main.full
    code, out, models, *_ = run_main(http=FakeHttp(fail=RuntimeError("key=" + KEY)), capsys=capsys)
    assert code == 1 and json.loads(out)["phase"] == "listing" and models.calls == []
    assert KEY not in run_main.full and SECRET not in run_main.full


def test_requires_confirm_flag(capsys):
    code, out, models, http, made = run_main(argv=(), capsys=capsys)
    assert code == 2 and models.calls == [] and http.requests == [] and made == []


def test_missing_key_makes_no_call(capsys):
    code, out, models, http, made = run_main(env={}, capsys=capsys)
    assert code == 2 and models.calls == [] and http.requests == [] and made == []


def test_unrecognized_version(capsys):
    models = default_models(version="bad version!")
    _, out, *_ = run_main(models=models, capsys=capsys)
    body = json.loads(out)
    assert body["text"]["model_version"] == "unrecognized"
    assert "bad version" not in out


def test_bad_counts_print_null(capsys):
    bad = types.GenerateContentResponseUsageMetadata.model_construct(
        prompt_token_count=True, candidates_token_count=-1, thoughts_token_count="3",
        total_token_count=1.5)
    models = FakeModels([bad, usage(-5, image=1100)])
    _, out, *_ = run_main(models=models, capsys=capsys)
    body = json.loads(out)
    assert body["text"]["prompt_token_count"] is None
    assert body["image"]["prompt_token_count"] is None
    assert body["tokens_per_image"] is None or isinstance(body["tokens_per_image"], int)


def test_probe_workflow_is_minimal():
    text = (ROOT / ".github/workflows/hydrogen_probe.yml").read_text()
    assert "up to six Hydrogen calls" in text and "up to 15000 output tokens" in text
    assert "workflow_dispatch:" in text
    for banned in ("schedule:", "push:", "pull_request", "id-token", "write",
                   "SPEND_POLICY_SHA256", "upload-artifact"):
        assert banned not in text
    assert set(re.findall(r"secrets\.([A-Z0-9_]+)", text)) == {"GEMINI_API_KEY"}
    assert "contents: read" in text
    assert "python3 scripts/probe_hydrogen.py --confirm-paid-probe" in text


# ---- Task A8b: thinking fit for the Director call ----

THINK_USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=2000, candidates_token_count=300, thoughts_token_count=1500,
    total_token_count=3800)


def run_fit(capsys, **kw):
    models = FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE], **kw)
    code, _, models, http, made = run_main(models=models, capsys=capsys)
    lines = run_main.full.splitlines()
    return code, lines, models


FIT_PLAN = (("high", 15000), ("high", 15000), ("high", 15000), ("medium", 15000))


def test_thinking_fit_calls_three_high_then_one_medium_at_15000(capsys):
    code, lines, models = run_fit(capsys, text='{"a": 1}')
    assert code == 0 and len(models.calls) == 6
    for call, (level, tokens) in zip(models.calls[2:], FIT_PLAN):
        assert call["model"] == "gemini-4-hydrogen"
        assert call["config"]["max_output_tokens"] == tokens
        assert call["config"]["response_mime_type"] == "application/json"
        assert call["config"]["thinking_config"] == {"thinking_level": level}
        assert isinstance(call["contents"], str)


def test_thinking_fit_line_format(capsys):
    code, lines, _ = run_fit(capsys, text='{"a": 1}')
    assert len(lines) == 6
    assert json.loads(lines[1]) == {
        "thinking_level": "high", "max_output_tokens": 15000, "finish_reason": "STOP",
        "prompt_tokens": 2000, "candidates_tokens": 300, "thoughts_tokens": 1500,
        "total_output": 1800, "json_ok": 1}
    assert [(json.loads(l)["thinking_level"], json.loads(l)["max_output_tokens"])
            for l in lines[1:5]] == list(FIT_PLAN)
    assert json.loads(lines[5]) == {"max_total_output_high": 1800, "max_total_output_medium": 1800}


def test_summary_takes_the_max_per_level_and_null_when_none(capsys):
    usages = [usage(12), usage(1112, image=1100)] + [
        types.GenerateContentResponseUsageMetadata(prompt_token_count=1, candidates_token_count=c,
                                                   thoughts_token_count=t_)
        for c, t_ in ((100, 200), (500, 700), (50, 60), (10, 20))]
    run_main(models=FakeModels(usages, text='{"a": 1}'), capsys=capsys)
    lines = run_main.full.splitlines()
    assert [json.loads(l)["total_output"] for l in lines[1:5]] == [300, 1200, 110, 30]
    assert json.loads(lines[5]) == {"max_total_output_high": 1200, "max_total_output_medium": 30}
    bad = types.GenerateContentResponseUsageMetadata.model_construct(
        prompt_token_count=1, candidates_token_count=5, thoughts_token_count=None)
    run_main(models=FakeModels([usage(12), usage(1112, image=1100), bad], text="{}"), capsys=capsys)
    lines = run_main.full.splitlines()
    assert json.loads(lines[1])["total_output"] is None
    assert json.loads(lines[5]) == {"max_total_output_high": None, "max_total_output_medium": None}


def test_json_ok_rules(capsys):
    _, lines, _ = run_fit(capsys, text='{"a": 1}')
    assert json.loads(lines[1])["json_ok"] == 1
    _, lines, _ = run_fit(capsys, text='{"a": [1, 2')
    assert json.loads(lines[1])["json_ok"] == 0
    _, lines, _ = run_fit(capsys, text='{"a": 1}', finish_reason=types.FinishReason.MAX_TOKENS)
    row = json.loads(lines[1])
    assert row["finish_reason"] == "MAX_TOKENS" and row["json_ok"] == 0


def test_full_output_never_leaks_key_or_secret(capsys):
    code, lines, _ = run_fit(capsys, text=KEY + SECRET)
    assert code == 0 and len(lines) == 6
    for banned in (KEY, SECRET):
        assert banned not in run_main.full
    for row in lines[1:]:
        assert KEY not in row and SECRET not in row


def test_json_ok_requires_object(capsys):
    for text, expected in (('{"a": 1}', 1), ("42", 0), ("[1, 2]", 0), ('"x"', 0), ("null", 0)):
        _, lines, _ = run_fit(capsys, text=text)
        assert json.loads(lines[1])["json_ok"] == expected, text


def test_finish_reason_sanitised_and_counts_null(capsys):
    assert probe.safe_finish("STOP") == "STOP"
    assert probe.safe_finish("bad reason!") == "unrecognized"
    assert probe.safe_finish(None) == "unrecognized"
    assert probe.safe_finish("A" * 33) == "unrecognized"
    bad = types.GenerateContentResponseUsageMetadata.model_construct(
        prompt_token_count=True, candidates_token_count=-1, thoughts_token_count="3",
        total_token_count=1.5)
    models = FakeModels([usage(12), usage(1112, image=1100), bad], text="{}")
    run_main(models=models, capsys=capsys)
    row = json.loads(run_main.full.splitlines()[1])
    assert [row[k] for k in ("prompt_tokens", "candidates_tokens", "thoughts_tokens",
                             "total_output")] == [None] * 4


def test_thinking_fit_never_prints_reply_text(capsys):
    _, lines, _ = run_fit(capsys, text=SECRET)
    assert SECRET not in "\n".join(lines) and json.loads(lines[1])["json_ok"] == 0


def test_no_thinking_calls_when_unlisted_or_unflagged(capsys):
    _, _, models, *_ = run_main(http=FakeHttp(models=("gemini-embedding-2",)), capsys=capsys)
    assert models.calls == []
    _, _, models, *_ = run_main(argv=(), capsys=capsys)
    assert models.calls == []


def test_thinking_failure_is_type_only_and_stops(capsys):
    base = [usage(12), usage(1112, image=1100), THINK_USAGE]
    for fail_at, phase in ((3, "thinking_high_1"), (4, "thinking_high_2"), (5, "thinking_high_3"),
                           (6, "thinking_medium_4")):
        models = FakeModels(base, fail_at=fail_at)
        code, _, models, *_ = run_main(models=models, capsys=capsys)
        lines = run_main.full.splitlines()
        assert code == 1 and len(models.calls) == fail_at and len(lines) == 1 + (fail_at - 3) + 1
        assert json.loads(lines[-1]) == {"status": "failed", "phase": phase, "error_type": "RuntimeError"}
        assert KEY not in run_main.full and SECRET not in run_main.full


def test_director_fit_prompt_is_synthetic_and_sized():
    prompt = probe.director_fit_prompt()
    assert 4000 <= len(prompt) <= 12000
    assert prompt.count("- Cluster ") == 5
    assert "8 to 12" in prompt and "Editorial Director" in prompt
    assert "Return JSON" in prompt
    for banned in (KEY, "http", "@", "api_key", "AIza"):
        assert banned not in prompt
    assert probe.director_fit_prompt() == prompt
