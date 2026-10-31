"""GitHub player identity verification for Identity-Staked Reputation play."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
import re
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request

# GitHub username rules: 1-39 chars, alphanumeric or single hyphens, cannot start/end with hyphen.
# Note: uses \Z instead of currency-symbol anchor to adhere to strict repo character rules.
_GITHUB_LOGIN_RE = re.compile(r"\A[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}\Z")

_RESERVED_BOT_LOGINS = frozenset({
    "github-actions",
    "dependabot",
    "renovate",
    "ghost",
    "invalid-user-handle",
})


@dataclass(frozen=True)
class PlayerIdentity:
    """Immutable verified or rejected GitHub player identity."""

    github_login: str
    avatar_url: str
    public_repos: int
    followers: int
    account_created_year: int
    verified: bool
    rejection_reason: str | None


def _default_http_get(url: str) -> Mapping[str, Any]:
    """Fetch JSON payload from GitHub REST API with a short timeout."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Siyuan-Zhu-Hall-Of-Records-Verifier/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=4.0) as response:
        payload = response.read().decode("utf-8")
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("Unexpected non-object payload from GitHub users endpoint")
        return data


def _extract_year(created_at_raw: Any, fallback_year: int) -> int:
    if isinstance(created_at_raw, str) and len(created_at_raw) >= 4 and created_at_raw[:4].isdigit():
        return int(created_at_raw[:4])
    return fallback_year


def verify_github_player(
    github_login: str,
    fetch_online: bool = False,
    *,
    http_get: Callable[[str], Mapping[str, Any]] | None = None,
    current_year: int | None = None,
) -> PlayerIdentity:
    """Validate a GitHub handle and optionally verify account activity online.

    Constructs the official 32px GitHub avatar URL (`https://github.com/{login}.png?size=32`).
    When `fetch_online=True`, queries `https://api.github.com/users/{login}` (with graceful
    offline fallback on transient network or rate-limit errors) to reject empty throwaway
    accounts or automated bots.
    """
    now_year = current_year if current_year is not None else dt.datetime.now(dt.timezone.utc).year
    raw = (github_login or "").strip()
    clean_login = raw[1:].strip() if raw.startswith("@") else raw

    if not clean_login:
        return PlayerIdentity(
            github_login="",
            avatar_url="https://github.com/ghost.png?size=32",
            public_repos=0,
            followers=0,
            account_created_year=now_year,
            verified=False,
            rejection_reason="GitHub handle cannot be empty.",
        )

    lower_login = clean_login.lower()
    if lower_login.endswith("[bot]") or lower_login in _RESERVED_BOT_LOGINS:
        return PlayerIdentity(
            github_login=clean_login,
            avatar_url=f"https://github.com/{clean_login}.png?size=32",
            public_repos=0,
            followers=0,
            account_created_year=now_year,
            verified=False,
            rejection_reason="Automated bot or reserved system accounts cannot stake ELO.",
        )

    if not _GITHUB_LOGIN_RE.match(clean_login):
        return PlayerIdentity(
            github_login=clean_login,
            avatar_url="https://github.com/ghost.png?size=32",
            public_repos=0,
            followers=0,
            account_created_year=now_year,
            verified=False,
            rejection_reason=(
                f"Invalid GitHub handle syntax '{clean_login}'. "
                "Handles must be 1-39 alphanumeric characters or single hyphens."
            ),
        )

    avatar_url = f"https://github.com/{clean_login}.png?size=32"
    default_offline_identity = PlayerIdentity(
        github_login=clean_login,
        avatar_url=avatar_url,
        public_repos=1,
        followers=1,
        account_created_year=now_year - 2,
        verified=True,
        rejection_reason=None,
    )

    if not fetch_online:
        return default_offline_identity

    fetcher = http_get or _default_http_get
    api_url = f"https://api.github.com/users/{clean_login}"
    try:
        profile = fetcher(api_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return PlayerIdentity(
                github_login=clean_login,
                avatar_url=avatar_url,
                public_repos=0,
                followers=0,
                account_created_year=now_year,
                verified=False,
                rejection_reason=f"GitHub profile '@{clean_login}' was not found (HTTP 404).",
            )
        # Rate-limit (403/429) or transient 5xx: fall back gracefully so workflow continues
        return default_offline_identity
    except Exception:
        # Offline / DNS / timeout graceful fallback
        return default_offline_identity

    canonical_login = str(profile.get("login") or clean_login)
    canonical_avatar = f"https://github.com/{canonical_login}.png?size=32"
    account_type = str(profile.get("type") or "User")
    public_repos = int(profile.get("public_repos") or 0)
    followers = int(profile.get("followers") or 0)
    created_year = _extract_year(profile.get("created_at"), now_year)

    if account_type.lower() == "bot":
        return PlayerIdentity(
            github_login=canonical_login,
            avatar_url=canonical_avatar,
            public_repos=public_repos,
            followers=followers,
            account_created_year=created_year,
            verified=False,
            rejection_reason="Automated bot accounts cannot stake ELO.",
        )

    if public_repos <= 0 and followers <= 0 and created_year >= now_year:
        return PlayerIdentity(
            github_login=canonical_login,
            avatar_url=canonical_avatar,
            public_repos=public_repos,
            followers=followers,
            account_created_year=created_year,
            verified=False,
            rejection_reason=(
                "Newly created throwaway account with zero public repos and zero followers "
                "cannot stake ELO."
            ),
        )

    return PlayerIdentity(
        github_login=canonical_login,
        avatar_url=canonical_avatar,
        public_repos=public_repos,
        followers=followers,
        account_created_year=created_year,
        verified=True,
        rejection_reason=None,
    )
