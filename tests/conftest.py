"""Keep unit/integration fixtures isolated from the runner's real Issue event."""
import pytest


@pytest.fixture(autouse=True)
def isolate_runner_event(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_OFFLINE_DEMO", "1")
    for flag in ("ZEITGEIST_LIVE_COLLECT", "ZEITGEIST_LIVE_LLM", "ZEITGEIST_LIVE_UNSPLASH"):
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("STATE_STORE_BACKEND", "json")
    # Individual adapter/configuration tests explicitly set and mock their boundaries.
    # A staging runner must not leak its namespace, OIDC authority or API keys
    # into fixtures whose default store is deliberately local.
    for name in (
        "GITHUB_ACTIONS", "GITHUB_EVENT_PATH", "ISSUE_NUMBER", "ISSUE_TITLE", "VISITOR_LOGIN",
        "GITHUB_REPOSITORY", "TARGET_REPO_SLUG", "FIRESTORE_COLLECTION_PREFIX",
        "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "FIRESTORE_PROJECT_ID", "FIRESTORE_DATABASE_ID",
        "GCP_WIF_PROVIDER", "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        "FIREBASE_SERVICE_ACCOUNT", "GCP_SA_KEY", "GCP_ACCESS_TOKEN", "FIRESTORE_ACCESS_TOKEN",
        "GEMINI_API_KEY", "BFL_API_KEY", "APIFY_TOKEN", "APIFY_API_TOKEN", "UNSPLASH_ACCESS_KEY",
        "SPEND_POLICY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
