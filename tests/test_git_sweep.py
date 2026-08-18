"""Unit tests for the git sweep collectors.

Functions run inside the fail-open recall hook, so every test that matters
here is about degradation: real throwaway repos prove the happy path, and
non-repo directories, timeouts, and missing binaries must all yield the
documented empty value — never an exception.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_memory_mcp.capture.git_sweep import (
    commits_since,
    current_branch,
    edited_files,
    infer_task_key,
    repo_name,
)

OLD_SINCE = "2000-01-01T00:00:00"
# Git's approxidate rejects far-future dates (e.g. 2100) and then matches
# everything, so stay within its parseable range.
FUTURE_SINCE = "2030-01-01T00:00:00"


def _git_env(tmp_path: Path) -> dict[str, str]:
    """Env that isolates git from the user's global config (signing hooks etc.)."""
    return {
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }


def _run_git(repo: Path, env: dict[str, str], *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo on branch ``main`` with two commits and a dirty tree.

    Commit 1 touches ``a.txt`` and ``b.txt``; commit 2 touches ``c.txt``.
    Afterwards ``a.txt`` is modified unstaged, ``b.txt`` is modified and
    staged, and ``d.txt`` is untracked.
    """
    env = _git_env(tmp_path)
    repo_dir = tmp_path / "sweep-repo"
    repo_dir.mkdir()
    _run_git(repo_dir, env, "init", "-q", "-b", "main")
    _run_git(repo_dir, env, "config", "user.email", "test@example.com")
    _run_git(repo_dir, env, "config", "user.name", "Test User")

    (repo_dir / "a.txt").write_text("a1\n")
    (repo_dir / "b.txt").write_text("b1\n")
    _run_git(repo_dir, env, "add", "a.txt", "b.txt")
    _run_git(repo_dir, env, "commit", "-q", "-m", "first: add a and b")

    (repo_dir / "c.txt").write_text("c1\n")
    _run_git(repo_dir, env, "add", "c.txt")
    _run_git(repo_dir, env, "commit", "-q", "-m", "second: add c")

    (repo_dir / "a.txt").write_text("a2\n")  # dirty, unstaged
    (repo_dir / "b.txt").write_text("b2\n")  # dirty, staged
    _run_git(repo_dir, env, "add", "b.txt")
    (repo_dir / "d.txt").write_text("d1\n")  # untracked

    return repo_dir


@pytest.fixture
def non_repo(tmp_path: Path) -> Path:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    return plain


class TestCurrentBranch:
    def test_returns_created_branch(self, repo: Path) -> None:
        assert current_branch(str(repo)) == "main"

    def test_non_repo_returns_none(self, non_repo: Path) -> None:
        assert current_branch(str(non_repo)) is None


class TestRepoName:
    def test_returns_directory_basename(self, repo: Path) -> None:
        assert repo_name(str(repo)) == "sweep-repo"

    def test_non_repo_returns_none(self, non_repo: Path) -> None:
        assert repo_name(str(non_repo)) is None


class TestEditedFiles:
    def test_union_of_dirty_staged_untracked(self, repo: Path) -> None:
        # Porcelain output: unstaged a.txt, staged b.txt, untracked d.txt,
        # in git status order.
        assert edited_files(str(repo)) == ["a.txt", "b.txt", "d.txt"]

    def test_cap_respected(self, repo: Path) -> None:
        assert edited_files(str(repo), cap=1) == ["a.txt"]

    def test_non_repo_returns_empty(self, non_repo: Path) -> None:
        assert edited_files(str(non_repo)) == []

    def test_timeout_returns_empty(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="git", timeout=3.0)

        monkeypatch.setattr(subprocess, "run", _boom)
        assert edited_files(str(repo)) == []

    def test_missing_git_binary_returns_empty(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_git(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", _no_git)
        assert edited_files(str(repo)) == []


class TestCommitsSince:
    def test_returns_both_commits_newest_first(self, repo: Path) -> None:
        commits = commits_since(str(repo), OLD_SINCE)
        assert len(commits) == 2
        newest, oldest = commits
        assert newest["message"] == "second: add c"
        assert newest["files"] == ["c.txt"]
        assert oldest["message"] == "first: add a and b"
        assert oldest["files"] == ["a.txt", "b.txt"]
        for commit in commits:
            assert len(commit["sha"]) == 40
            assert set(commit["sha"]) <= set("0123456789abcdef")
            assert "T" in commit["ts"]  # ISO 8601 from %cI

    def test_empty_commit_yields_no_files(self, repo: Path, tmp_path: Path) -> None:
        env = _git_env(tmp_path)
        # Unstage b.txt first: --allow-empty still commits whatever is staged.
        _run_git(repo, env, "restore", "--staged", "b.txt")
        _run_git(repo, env, "commit", "-q", "--allow-empty", "-m", "merge-ish")
        commits = commits_since(str(repo), OLD_SINCE)
        assert commits[0]["message"] == "merge-ish"
        assert commits[0]["files"] == []
        assert len(commits) == 3

    def test_cap_respected(self, repo: Path) -> None:
        commits = commits_since(str(repo), OLD_SINCE, cap=1)
        assert len(commits) == 1
        assert commits[0]["message"] == "second: add c"

    def test_future_since_returns_empty(self, repo: Path) -> None:
        assert commits_since(str(repo), FUTURE_SINCE) == []

    def test_non_repo_returns_empty(self, non_repo: Path) -> None:
        assert commits_since(str(non_repo), OLD_SINCE) == []

    def test_show_signature_config_does_not_corrupt_parse(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # With log.showSignature=true, GPG lines would interleave in the
        # pretty output and get misread as filenames; --no-show-signature
        # must suppress them. Unsigned commits cannot reproduce the
        # corruption, so also pin the flag in the argv below.
        _run_git(repo, _git_env(tmp_path), "config", "log.showSignature", "true")
        commits = commits_since(str(repo), OLD_SINCE)
        assert [c["message"] for c in commits] == [
            "second: add c",
            "first: add a and b",
        ]
        assert commits[0]["files"] == ["c.txt"]

    def test_log_invocation_disables_signature_display(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[list[str]] = []
        real_run = subprocess.run

        def _record(cmd: list[str], **kwargs: object):
            recorded.append(cmd)
            return real_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", _record)
        commits_since(str(repo), OLD_SINCE)
        assert len(recorded) == 1
        assert "--no-show-signature" in recorded[0]


class TestInferTaskKey:
    def test_env_override_wins(self) -> None:
        env = {"NAM_TASK_KEY": "MUD-999"}
        assert infer_task_key("feature/MUD-395-pivot", env=env) == "MUD-999"

    def test_ticket_pattern_extracted_from_branch(self) -> None:
        assert infer_task_key("feature/MUD-395-pivot", env={}) == "MUD-395"

    def test_plain_branch_falls_through_to_itself(self) -> None:
        assert infer_task_key("main", env={}) == "main"

    def test_none_and_empty_branch_return_none(self) -> None:
        assert infer_task_key(None, env={}) is None
        assert infer_task_key("", env={}) is None

    def test_empty_env_value_does_not_override(self) -> None:
        assert (
            infer_task_key("feature/MUD-395-pivot", env={"NAM_TASK_KEY": ""})
            == "MUD-395"
        )

    def test_env_override_without_branch(self) -> None:
        assert infer_task_key(None, env={"NAM_TASK_KEY": "MUD-1"}) == "MUD-1"

    def test_branch_fallback_scoped_by_repo(self) -> None:
        assert infer_task_key("main", env={}, repo="alpha") == "alpha/main"

    def test_ticket_branch_unaffected_by_repo(self) -> None:
        assert (
            infer_task_key("feature/MUD-395-pivot", env={}, repo="alpha") == "MUD-395"
        )
