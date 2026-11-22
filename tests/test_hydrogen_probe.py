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
    def __init__(self, usages, version="synthetic-hydrogen-version", fail_at=None):
        self.usages = list(usages)
        self.version = version
        self.fail_at = fail_at
        self.calls = []

    def generate_content(self, **kw):
        self.calls.append(kw)
        if self.fail_at == len(self.calls):
            raise RuntimeError("key=" + KEY)
        response = types.GenerateContentResponse(
            model_version=self.version, response_id="r",
            usage_metadata=self.usages[len(self.calls) - 1],
            candidates=[types.Candidate(content=types.Content(
                parts=[types.Part(text=SECRET)], role="model"))])
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
    return code, capsys.readouterr().out, models, http, made


def test_report_has_only_versions_and_counts(capsys):
    report = probe.run_probe(KEY, http_client=FakeHttp(), genai_client=FakeGenai(default_models()))
    assert set(report) == {"listed", "missing", "text", "image"}
    assert set(report["text"]) == {"model_version", "prompt_token_count", "candidates_token_count",
                                   "thoughts_token_count", "total_token_count"}
    assert set(report["image"]) == set(report["text"]) | {"image_prompt_token_count"}
    code, out, *_ = run_main(capsys=capsys)
    assert code == 0
    for banned in (KEY, SECRET, "masked", "fingerprint"):
        assert banned not in out
    assert json.loads(out)["image"]["model_version"] == "synthetic-hydrogen-version"


def test_two_calls_second_has_384px_image(capsys):
    _, _, models, _, made = run_main(capsys=capsys)
    assert len(models.calls) == 2 and made == [KEY]
    for call in models.calls:
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
    assert KEY not in out
    code, out, *_ = run_main(models=default_models(fail_at=1), capsys=capsys)
    assert json.loads(out)["phase"] == "text" and KEY not in out
    code, out, models, *_ = run_main(http=FakeHttp(fail=RuntimeError("key=" + KEY)), capsys=capsys)
    assert code == 1 and json.loads(out)["phase"] == "listing" and models.calls == []
    assert KEY not in out


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
    assert "workflow_dispatch:" in text
    for banned in ("schedule:", "push:", "pull_request", "id-token", "write",
                   "SPEND_POLICY_SHA256", "upload-artifact"):
        assert banned not in text
    assert set(re.findall(r"secrets\.([A-Z0-9_]+)", text)) == {"GEMINI_API_KEY"}
    assert "contents: read" in text
    assert "python3 scripts/probe_hydrogen.py --confirm-paid-probe" in text
