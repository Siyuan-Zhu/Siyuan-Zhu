"""Lightweight GCP Cloud Run HTTP Service for the Google ADK Daily Sentiment, Mood Vote Redirect & Mobile Webhook.

Designed to run within the Cloud Run free tier + Cloud Firestore free tier + Gemini 2.5 Flash budget (< 0.50 index units/month):
  - GET  /vote?dir=spot_on|too_sunny|too_heavy : Records a 1-vote-per-day HMAC-deduplicated mood vote
    in StateStore (Firestore or JSON) and returns 302 Found redirecting back to the GitHub profile.
  - POST /daily-cron : Invoked once daily by Cloud Scheduler at 07:00 US/Eastern to execute
    the 3-Agent ADK pipeline (HarvesterAgent -> SemanticAnalystAgent -> VisualDirectorAgent)
    and commit the updated SVG/README to GitHub.
  - POST /interject  : Invoked anytime from an iOS Shortcut / mobile bookmarklet with a Bearer token
    to override today's word, note, or visual mode in <2 seconds.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from typing import Any, Tuple
import urllib.parse

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.orbit import main as run_orbit_cli
from src.state_store import (
    VALID_MOOD_KEYS,
    StateStore,
    compute_voter_hmac,
    get_state_store,
)
from src.zeitgeist.adk_pipeline import run_daily_zeitgeist

PROFILE_REDIRECT_URL = "https://github.com/Siyuan-Zhu"
MAX_POST_BYTES = 65536


def parse_json_body(
    raw_bytes: bytes | str,
    content_length: int | None = None,
    max_bytes: int = MAX_POST_BYTES,
) -> Tuple[int, dict]:
    """Validate POST body size (`<= 65536` bytes) and parse JSON object payload."""
    if content_length is not None and content_length > max_bytes:
        return (
            413,
            {
                "status": "error",
                "error": "payload_too_large",
                "message": f"Payload exceeds {max_bytes} byte limit",
            },
        )
    if isinstance(raw_bytes, str):
        raw_b = raw_bytes.encode("utf-8")
    else:
        raw_b = bytes(raw_bytes)
    if len(raw_b) > max_bytes:
        return (
            413,
            {
                "status": "error",
                "error": "payload_too_large",
                "message": f"Payload exceeds {max_bytes} byte limit",
            },
        )
    if not raw_b.strip():
        return 200, {}
    try:
        decoded = raw_b.decode("utf-8")
        parsed = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (
            400,
            {
                "status": "error",
                "error": "invalid_json",
                "message": "Malformed JSON request body",
            },
        )
    if not isinstance(parsed, dict):
        return (
            400,
            {
                "status": "error",
                "error": "invalid_json",
                "message": "JSON payload must be an object",
            },
        )
    return 200, parsed


def dispatch_get_request(
    path: str,
    client_ip: str = "127.0.0.1",
    user_agent: str = "github-profile-visitor",
    store: StateStore | None = None,
    date_str: str | None = None,
) -> Tuple[int, dict, dict[str, str]]:
    """Pure GET dispatcher supporting `/healthz` and 1-click mood voting (`GET /vote?dir=...`)."""
    parsed = urllib.parse.urlsplit(path)
    route = parsed.path

    if route == "/healthz":
        return 200, {"status": "ok", "service": "siyuan-zhu-zeitgeist-adk"}, {}

    if route == "/vote":
        qs = urllib.parse.parse_qs(parsed.query)
        mood_dir = (qs.get("dir", [""])[0] or "").strip().lower()
        if mood_dir not in VALID_MOOD_KEYS:
            return (
                400,
                {"status": "error", "message": f"Invalid vote direction '{mood_dir}'"},
                {},
            )

        state_store = store if store is not None else get_state_store(repo_root=REPO_ROOT)
        today = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        voter_hmac = compute_voter_hmac(
            client_ip=client_ip,
            user_agent=user_agent,
            date_str=today,
        )
        accepted, tally = state_store.record_mood_vote(
            date_str=today,
            mood_key=mood_dir,
            voter_hmac=voter_hmac,
        )
        redirect_target = f"{PROFILE_REDIRECT_URL}#mood-{mood_dir}"
        return (
            302,
            {
                "status": "recorded_vote" if accepted else "duplicate_vote_ignored",
                "accepted": accepted,
                "vote": mood_dir,
                "tally": tally,
                "redirect": redirect_target,
            },
            {"Location": redirect_target, "Cache-Control": "no-store"},
        )

    return 404, {"status": "error", "message": f"Unknown GET endpoint: {route}"}, {}


def dispatch_request(
    path: str,
    payload: dict | bytes | str | None = None,
    auth_header: str | dict[str, Any] = "",
    expected_token: str | None = None,
    store: StateStore | None = None,
    content_length: int | None = None,
) -> Tuple[int, dict]:
    """Pure POST/GET request dispatcher for easy unit testing without socket bindings."""
    if path.startswith("/vote") or path == "/healthz":
        code, body, _headers = dispatch_get_request(path, store=store)
        return code, body

    resolved_token = (
        os.environ.get("ORBIT_WEBHOOK_TOKEN", "")
        if expected_token is None
        else str(expected_token)
    ).strip()
    auth_str = (
        (auth_header.get("Authorization") or auth_header.get("authorization") or "")
        if isinstance(auth_header, dict)
        else str(auth_header or "")
    ).strip()

    if not resolved_token or not hmac.compare_digest(
        auth_str.encode("utf-8"),
        f"Bearer {resolved_token}".encode("utf-8"),
    ):
        return (
            401,
            {
                "status": "error",
                "error": "unauthorized_or_unconfigured_webhook_token",
                "message": "Unauthorized or unconfigured bearer token",
            },
        )

    if isinstance(payload, (bytes, bytearray, str)):
        parse_code, parsed_or_err = parse_json_body(payload, content_length=content_length)
        if parse_code != 200:
            return parse_code, parsed_or_err
        payload_dict = parsed_or_err
    else:
        parse_code, _ = parse_json_body(b"{}", content_length=content_length)
        if parse_code != 200:
            return parse_code, _
        payload_dict = payload if isinstance(payload, dict) else {}

    if path in ("/daily-cron", "/daily-pulse"):
        result = run_daily_zeitgeist(
            repo_root=REPO_ROOT,
            date_iso=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        return 200, {
            "status": "ok",
            "mode": path.lstrip("/"),
            "word_of_the_day": result["analysis"]["word_of_the_day"],
            "composite_score": result["analysis"]["composite_score"],
            "selected_visual_mode": result["visual_decision"]["selected_mode"],
            "within_budget": result["telemetry"]["within_budget"],
        }

    if path == "/interject":
        argv = ["interject"]
        if payload_dict.get("word"):
            argv.extend(["--word", str(payload_dict["word"])])
        if payload_dict.get("phrase"):
            argv.extend(["--phrase", str(payload_dict["phrase"])])
        if payload_dict.get("note"):
            argv.extend(["--note", str(payload_dict["note"])])
        if payload_dict.get("mode"):
            argv.extend(["--mode", str(payload_dict["mode"])])
        argv.extend(["--channel", "cloud_run_mobile_webhook"])
        run_orbit_cli(argv)

        store = get_state_store(repo_root=REPO_ROOT)
        state = store.load_pulse_state()
        return 200, {
            "status": "ok",
            "mode": "mobile-interject",
            "word_of_the_day": state.get("analysis", {}).get("word_of_the_day", ""),
            "selected_visual_mode": state.get("visual_decision", {}).get("selected_mode", ""),
        }

    return 404, {"status": "error", "message": f"Unknown endpoint: {path}"}


class CloudRunHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        forwarded = self.headers.get("X-Forwarded-For", "")
        client_ip = (forwarded.split(",")[0].strip() if forwarded else "") or self.client_address[0]
        user_agent = self.headers.get("User-Agent", "github-profile-visitor")
        code, response, extra_headers = dispatch_get_request(
            self.path,
            client_ip=client_ip,
            user_agent=user_agent,
        )
        body = json.dumps(response).encode("utf-8")
        self.send_response(code)
        for h_name, h_val in extra_headers.items():
            self.send_header(h_name, h_val)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 0:
            length = 0
        raw_bytes = self.rfile.read(min(length, MAX_POST_BYTES + 1)) if length > 0 else b"{}"
        parse_code, parsed_or_err = parse_json_body(raw_bytes, content_length=length)
        if parse_code != 200:
            code, response = parse_code, parsed_or_err
        else:
            auth = self.headers.get("Authorization", "")
            expected = os.environ.get("ORBIT_WEBHOOK_TOKEN", "")
            code, response = dispatch_request(self.path, parsed_or_err, auth, expected)
        body = json.dumps(response).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), CloudRunHandler)
    print(f"Listening on port {port}...")
    server.serve_forever()
