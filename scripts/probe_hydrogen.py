#!/usr/bin/env python3
"""Manual Hydrogen probe: list models, then make two tiny paid calls.

Prints model versions and token counts only. It never prints the key, response
text, or any credential identity. The two calls are authorised outside the spend
guard by the spec, so the command refuses to run without --confirm-paid-probe.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.provider_preflight import inspect_gemini  # noqa: E402

HYDROGEN = "gemini-4-hydrogen"
PROBE_MODELS = (HYDROGEN, "gemini-embedding-2")
PROBE_TEXT = "A quiet harbor at dawn with three small boats."
VERSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
CALL_CONFIG = {
    "max_output_tokens": 16,
    "thinking_config": {"thinking_level": "low"},
    "media_resolution": "MEDIA_RESOLUTION_HIGH",
}
COUNT_FIELDS = (
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)


class ProbeFailure(Exception):
    def __init__(self, phase, error_type):
        super().__init__(phase)
        self.phase = phase
        self.error_type = error_type


def synthetic_image() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (384, 384), (235, 240, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 220, 344, 344), fill=(40, 90, 140))
    draw.polygon([(192, 60), (292, 220), (92, 220)], fill=(220, 120, 60))
    draw.ellipse((280, 40, 340, 100), fill=(250, 220, 90))
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def safe_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def safe_version(value):
    if isinstance(value, str) and VERSION_PATTERN.fullmatch(value):
        return value
    return "unrecognized"


def summarize(response):
    usage = getattr(response, "usage_metadata", None)
    summary = {"model_version": safe_version(getattr(response, "model_version", None))}
    for field in COUNT_FIELDS:
        summary[field] = safe_count(getattr(usage, field, None))
    return summary, usage


def image_detail_count(usage):
    for detail in getattr(usage, "prompt_tokens_details", None) or []:
        modality = getattr(detail, "modality", None)
        if str(getattr(modality, "value", modality)).upper() == "IMAGE":
            return safe_count(getattr(detail, "token_count", None))
    return None


def tokens_per_image(report):
    text, image = report["text"], report["image"]
    if text is None or image is None:
        return None
    if text["prompt_token_count"] is None or image["prompt_token_count"] is None:
        return None
    return max(
        image["prompt_token_count"] - text["prompt_token_count"],
        image["image_prompt_token_count"] or 0,
    )


def run_probe(key: str, *, http_client, genai_client) -> dict:
    from google.genai import types

    try:
        found = inspect_gemini(http_client, key, PROBE_MODELS)
    except Exception as error:
        raise ProbeFailure("listing", type(error).__name__) from None
    report = {
        "listed": found["available_required_models"],
        "missing": found["missing_models"],
        "text": None,
        "image": None,
    }
    if HYDROGEN in report["missing"]:
        return report
    try:
        response = genai_client.models.generate_content(
            model=HYDROGEN, contents=PROBE_TEXT, config=CALL_CONFIG
        )
        report["text"], _ = summarize(response)
    except Exception as error:
        raise ProbeFailure("text", type(error).__name__) from None
    try:
        part = types.Part.from_bytes(data=synthetic_image(), mime_type="image/png")
        response = genai_client.models.generate_content(
            model=HYDROGEN, contents=[part, PROBE_TEXT], config=CALL_CONFIG
        )
        report["image"], usage = summarize(response)
        report["image"]["image_prompt_token_count"] = image_detail_count(usage)
    except Exception as error:
        raise ProbeFailure("image", type(error).__name__) from None
    return report


def _http_client():
    import httpx

    return httpx.Client(timeout=20, follow_redirects=False)


def _genai_client(key):
    from google import genai

    return genai.Client(api_key=key, http_options={"retry_options": {"attempts": 1}})


class _LazyGenai:
    """Build the paid client only when the first paid call is made."""

    def __init__(self, factory, key):
        self._factory, self._key, self._client = factory, key, None

    @property
    def models(self):
        if self._client is None:
            self._client = self._factory(self._key)
        return self._client.models


def main(argv=None, *, env=None, http_factory=None, genai_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-paid-probe", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    key = env.get("GEMINI_API_KEY", "")
    if not args.confirm_paid_probe or not key:
        print(json.dumps({"status": "refused"}))
        return 2
    try:
        http_client = (http_factory or _http_client)()
        try:
            report = run_probe(
                key,
                http_client=http_client,
                genai_client=_LazyGenai(genai_factory or _genai_client, key),
            )
        finally:
            close = getattr(http_client, "close", None)
            if close:
                close()
    except ProbeFailure as failure:
        print(json.dumps({"status": "failed", "phase": failure.phase,
                          "error_type": failure.error_type}))
        return 1
    except Exception as error:
        print(json.dumps({"status": "failed", "phase": "listing",
                          "error_type": type(error).__name__}))
        return 1
    if HYDROGEN in report["missing"]:
        print(json.dumps({"status": "hydrogen_not_listed", **report}))
        return 2
    print(json.dumps({"status": "ok", **report, "tokens_per_image": tokens_per_image(report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
