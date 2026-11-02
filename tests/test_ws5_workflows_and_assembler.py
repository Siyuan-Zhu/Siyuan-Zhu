"""Integration and unit tests for WS5 (#25): GitHub Actions Workflows, requirements.txt, and README Assembler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.build_readme import render_profile_markdown
from src.state_store import InMemoryStateStore, git_commit_and_push_with_retry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_requirements_txt_contains_pinned_runtime_dependencies() -> None:
    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8")
    for pkg in (
        "httpx>=",
        "defusedxml>=",
        "numpy>=",
        "Pillow>=",
        "google-genai>=",
        "google-cloud-firestore>=",
        "pytest>=",
    ):
        assert pkg in content


def test_workflows_share_concurrency_group_and_zero_pip_in_visitor_workflow() -> None:
    daily_yml = (REPO_ROOT / ".github" / "workflows" / "daily_zeitgeist.yml").read_text(
        encoding="utf-8"
    )
    game_yml = (
        REPO_ROOT / ".github" / "workflows" / "community_game_move.yml"
    ).read_text(encoding="utf-8")
    mobile_yml = (
        REPO_ROOT / ".github" / "workflows" / "mobile_interject.yml"
    ).read_text(encoding="utf-8")

    for wf in (daily_yml, game_yml, mobile_yml):
        assert "group: readme-state-writer" in wf
        assert "cancel-in-progress: false" in wf

    assert "pip install -r requirements.txt" in daily_yml
    assert 'cache: "pip"' in daily_yml
    assert "pip install -r requirements.txt" in game_yml
    assert "pip install -r requirements.txt" in mobile_yml
    assert "id-token: write" in daily_yml
    assert "id-token: write" in game_yml


def test_git_commit_and_push_with_retry_sets_reconcile_flag_on_exhaustion(
    tmp_path: Path,
) -> None:
    store = InMemoryStateStore()
    sleep_calls: list[float] = []
    seen_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        seen_cmds.append(list(cmd))
        res = MagicMock()
        if cmd[:2] == ["git", "diff"]:
            res.returncode = 1  # Changes staged
        elif cmd[:3] == ["git", "pull", "--rebase"]:
            res.returncode = 1  # Simulate rebase conflict
        elif cmd[:2] == ["git", "push"]:
            res.returncode = 1
            res.stderr = "rejected - non-fast-forward"
        else:
            res.returncode = 0
        return res

    with patch("subprocess.run", side_effect=fake_run):
        pushed = git_commit_and_push_with_retry(
            commit_message="chore: test push retry",
            paths=["README.md"],
            store=store,
            repo_root=tmp_path,
            sleep_fn=lambda s: sleep_calls.append(float(s)),
        )

    assert pushed is False
    assert sleep_calls == [2.0, 5.0]
    assert store.needs_git_reconcile is True
    assert ["git", "rebase", "--abort"] in seen_cmds
    assert not any(command[:2] == ["git", "push"] for command in seen_cmds)


def test_render_profile_markdown_includes_cache_busting_7_columns_mode_and_moods() -> None:
    from src.game_engine import create_initial_game_state
    store = InMemoryStateStore()
    store.save_game_state(create_initial_game_state())
    md = render_profile_markdown(REPO_ROOT, state_store=store, write_assets=False)
    assert "assets/daily_pulse_light.webp?v=" in md and "#gh-light-mode-only" in md
    assert "assets/daily_pulse_dark.webp?v=" in md and "#gh-dark-mode-only" in md
    assert "assets/game_board.svg?v=" in md
    for col in range(1, 8):
        assert (
            f"title=game:move:connect4:col:{col}:" in md
            or f"title=game:move:connect4:col{col}:" in md
        )
    assert "title=game:mode:connect4:new_pvp:turn=1" in md
    assert "title=pulse:vote:spot_on" in md
    assert "title=pulse:vote:too_sunny" in md
    assert "title=pulse:vote:too_heavy" in md
