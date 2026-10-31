"""Fail closed on an incomplete writer configuration, without logging secrets."""
from __future__ import annotations

import argparse
import os
import re
from typing import Mapping


def configuration_errors(env: Mapping[str, str], *, visitor: bool = False, daily: bool = False) -> list[str]:
    errors = []
    repo = env.get("GITHUB_REPOSITORY", "")
    if not repo or env.get("TARGET_REPO_SLUG", "") != repo:
        errors.append("TARGET_REPO_SLUG must match the executing repository")
    if env.get("STATE_STORE_BACKEND") != "firestore":
        errors.append("Writer requires the canonical Firestore backend")
    if env.get("GCP_PROJECT_ID") != "git-readme-features" or env.get("FIRESTORE_DATABASE_ID") != "(default)":
        errors.append("Writer must bind to the reviewed project and database")
    prefix = env.get("FIRESTORE_COLLECTION_PREFIX", "")
    if repo == "Siyuan-Zhu/readme-live-sandbox" and not prefix.startswith("staging_"):
        errors.append("Sandbox writers require an isolated staging_ namespace")
    elif repo == "Siyuan-Zhu/Siyuan-Zhu" and prefix:
        errors.append("Production writer requires the reviewed unprefixed namespace")
    elif repo not in {"Siyuan-Zhu/readme-live-sandbox", "Siyuan-Zhu/Siyuan-Zhu"}:
        errors.append("Unknown writer repository")
    provider = env.get("GCP_WIF_PROVIDER", "")
    if not provider.startswith("//iam.googleapis.com/projects/") or "/providers/" not in provider:
        errors.append("GCP_WIF_PROVIDER must identify the verified workload identity provider")
    if not env.get("FIRESTORE_DATABASE_ID"):
        errors.append("FIRESTORE_DATABASE_ID is required")
    if visitor and not env.get("PULSE_VOTE_SALT"):
        errors.append("PULSE_VOTE_SALT is required")
    if daily:
        if repo == "Siyuan-Zhu/Siyuan-Zhu" and not re.fullmatch(r"[a-f0-9]{64}", env.get("SPEND_POLICY_SHA256", "")):
            errors.append("Production daily publishing requires an approved automatic spending policy")
        for key in ("GEMINI_API_KEY", "UNSPLASH_ACCESS_KEY", "APIFY_TOKEN"):
            if not env.get(key):
                errors.append(f"{key} is required")
        for key in ("ZEITGEIST_LIVE_COLLECT", "ZEITGEIST_LIVE_LLM", "ZEITGEIST_LIVE_UNSPLASH"):
            if env.get(key) != "1":
                errors.append(f"{key} must enable the reviewed live path")
        if not re.fullmatch(r"\d+\.\d+\.\d+", env.get("APIFY_INSTAGRAM_BUILD", "")):
            errors.append("APIFY_INSTAGRAM_BUILD must be a verified numeric build")
        if env.get("ZEITGEIST_OFFLINE_DEMO") == "1":
            errors.append("Daily writers must never use demonstration signals")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visitor", action="store_true")
    parser.add_argument("--daily", action="store_true")
    args = parser.parse_args(argv)
    errors = configuration_errors(os.environ, visitor=args.visitor, daily=args.daily)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
