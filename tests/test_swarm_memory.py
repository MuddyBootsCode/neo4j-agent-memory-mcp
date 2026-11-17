"""MUD-850 Phase 2: the observe-and-log hooks in ``src/swarm_memory``.

Each rule that decides what is stored or what would have been delivered is
pinned on a small case, against a real git repo with two linked worktrees.
"""

import io
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys

import pytest

from swarm_memory import hook as hook_entry
from swarm_memory import observe, report, rules, store

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "replay"))

import replay  # noqa: E402

T = 1_800_000_000.0
MIN = 60.0
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def git(*args):
    subprocess.run(["git", *args], check=True, capture_output=True, env=GIT_ENV)


@pytest.fixture
def repo(tmp_path):
    """``proj`` on main, with worktrees ``a`` (branch agent/abc-101-a) and ``b`` (agent/abc-102-b)."""
    main = tmp_path / "proj"
    git("init", "-q", "-b", "main", str(main))
    (main / "src").mkdir()
    (main / "src" / "app.py").write_text("x = 1\n")
    (main / "README.md").write_text("hello\n")
    (main / "scripts").mkdir()
    (main / "scripts" / "run.sh").write_text("echo run\n")
    git("-C", str(main), "add", ".")
    git("-C", str(main), "commit", "-q", "-m", "init")
    git("-C", str(main), "worktree", "add", "-q", "-b", "agent/abc-101-a", str(main / ".worktrees" / "a"))
    git("-C", str(main), "worktree", "add", "-q", "-b", "agent/abc-102-b", str(main / ".worktrees" / "b"))
    return main


@pytest.fixture
def db(tmp_path):
    """A store path whose ``config.json`` sets the made-up ticket prefix ABC."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.json").write_text('{"ticket_keys": ["ABC"]}')
    return str(state / "observe.sqlite3")


def wt(repo, name):
    return repo / ".worktrees" / name


def fire(db, event, sid, cwd, ts, **fields):
    payload = {"hook_event_name": event, "session_id": sid, "cwd": str(cwd), "transcript_path": "/dev/null", **fields}
    observe.run(json.dumps(payload), db_arg=db, ts=ts)


def start(db, sid, cwd, ts, source="startup"):
    fire(db, "SessionStart", sid, cwd, ts, source=source)


def tool_ok(db, sid, cwd, ts, tool, tool_input, tuid, **extra):
    """A PostToolUse event. Claude Code sends ``duration_ms`` with every one."""
    fire(db, "PostToolUse", sid, cwd, ts, tool_name=tool, tool_input=tool_input, tool_use_id=tuid,
         **{"tool_response": {}, "duration_ms": 50, **extra})


def tool_fail(db, sid, cwd, ts, tool, tool_input, error, tuid, **extra):
    fire(db, "PostToolUseFailure", sid, cwd, ts, tool_name=tool, tool_input=tool_input, tool_use_id=tuid,
         error=error, is_interrupt=extra.pop("is_interrupt", False), **extra)


def edit(db, sid, cwd, ts, path, tuid):
    tool_ok(db, sid, cwd, ts, "Edit", {"file_path": str(path), "old_string": "x", "new_string": "y"}, tuid)


def q(db, sql, args=()):
    con = sqlite3.connect(db)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def links(db, record_id, typ):
    return sorted(k for (k,) in q(db, "SELECT key FROM link WHERE record_id = ? AND type = ?", (record_id, typ)))


def session_record(db, sid):
    return q(db, "SELECT id, info, status FROM record WHERE kind = 'session' AND session_id = ?", (sid,))[0]


def delivered(db, moment=None):
    sql = "SELECT session_id, moment, records, detail FROM event WHERE kind = 'would_deliver'"
    rows = q(db, sql + (" AND moment = ?" if moment else "") + " ORDER BY id", (moment,) if moment else ())
    return [(s, m, json.loads(r), json.loads(d) if d else None) for s, m, r, d in rows]


class TestLocate:
    def test_the_main_checkout_and_its_worktrees_share_one_repo_key(self, repo):
        keys = {rules.locate(str(p / "src" / "app.py"))[0] for p in (repo, wt(repo, "a"), wt(repo, "b"))}
        assert keys == {os.path.realpath(repo / ".git")}

    def test_a_file_has_the_same_repo_relative_path_in_every_worktree(self, repo):
        rels = {rules.file_location(str(p / "src" / "app.py"))[3] for p in (repo, wt(repo, "a"))}
        assert rels == {"src/app.py"}

    def test_each_worktree_is_its_own_checkout(self, repo):
        assert rules.file_location(str(wt(repo, "a") / "README.md"))[2] == os.path.realpath(wt(repo, "a"))
        assert rules.file_location(str(repo / "README.md"))[2] == os.path.realpath(repo)

    def test_the_branch_and_its_ticket_come_from_the_checkouts_head(self, repo):
        loc = rules.locate(str(wt(repo, "a")))
        assert rules.branch(loc[3]) == "agent/abc-101-a"
        assert rules.branch_ticket(rules.branch(loc[3]), ("ABC",)) == "ABC-101"
        assert rules.branch_ticket("main", ("ABC",)) is None
        assert rules.branch_ticket(rules.branch(loc[3])) is None

    def test_paths_outside_a_repo_or_inside_dot_git_are_not_repo_files(self, repo, tmp_path):
        (tmp_path / "loose.txt").write_text("x")
        assert rules.file_location(str(tmp_path / "loose.txt")) is None
        assert rules.file_location(str(repo / ".git" / "HEAD")) is None


class TestTicketKeys:
    """Ticket prefixes are a local setting beside the store, never part of the repo."""

    def test_prefixes_come_from_the_config_beside_the_store(self, tmp_path):
        store_path = str(tmp_path / "observe.sqlite3")
        assert store.ticket_keys(store_path) == ()
        (tmp_path / "config.json").write_text('{"ticket_keys": ["ABC", "not a key", "abc", "XYZ", "ABC"]}')
        assert store.ticket_keys(store_path) == ("ABC", "XYZ")
        (tmp_path / "config.json").write_text("not json")
        assert store.ticket_keys(store_path) == ()

    def test_the_prefix_rule_matches_the_replay(self):
        for keys in (["ABC", "xy", "X1", "TOOLONGPREFIX1", "A-B"], "ABC, XYZ", None, 7):
            assert rules.parse_ticket_keys(keys) == replay.parse_ticket_keys(keys)

    def test_with_no_prefixes_configured_nothing_names_a_ticket(self, tmp_path, repo):
        bare = str(tmp_path / "bare" / "observe.sqlite3")
        start(bare, "s1", wt(repo, "a"), T)
        fire(bare, "UserPromptSubmit", "s1", wt(repo, "a"), T + 1, prompt="Work on ABC-7")
        rid, info, _ = session_record(bare, "s1")
        assert links(bare, rid, "ticket") == [] and "ticket_from" not in json.loads(info)


class TestReplayParity:
    """The hook's copies of the replay rules must give the replay's answers."""

    ERRORS = [
        "Exit code 1\nls: /tmp/x/y: No such file or directory",
        "Exit code 1\nTraceback (most recent call last):\n  File \"a.py\", line 3\nModuleNotFoundError: No module named 'foo'",
        "Exit code 1\n(eval):3: ===== not found",
        "Exit code 128\nfatal: not a git repository (or any of the parent directories): .git",
        "Exit code 1\nError: Cannot find module 'express'",
        "Exit code 2\nmake: *** [build] Error 2",
        "Permission for this action was denied",
        "Exit code 1\nerror: login failed password=hunterpw for user app",
    ]
    COMMANDS = [
        "cd src && uv run pytest tests/test_x.py -q",
        "git -C /x/y status --short",
        "FOO=1 npm run build",
        "bash -c 'python3 -m ruff check src'",
        "agent-test-budget run -- pytest -x",
        "export A=1; cd /tmp",
        "bash -lc 'npm run test -- --ci'",
        "npm run lint",
        "sh -ec 'cd sub && make build'",
        "yarn run build --prod",
    ]

    PROGRAMS = [
        "echo hi && /opt/homebrew/bin/python3 -I -S /x/report.py --minutes 15",
        "env A=1 timeout 60 .venv/bin/python -m pytest tests/test_x.py",
        "bash -c 'node scripts/gate.mjs --check src/a.ts' | tee out.log",
        "cd sub && ./run.sh a.py; uv run python tools/x.py b.py",
        "timeout 60 npm test",
        "agent-test-budget run -- /bin/zsh -c 'cd /x && cat src/a.ts && bash -e scripts/run.sh'",
        "python3 -c \"print(open('src/a.ts').read())\" && node -e 'require(\"./b.js\")' && ruby -w tools/c.rb",
        "bash -lc 'cat src/a.ts'",
        "zsh -xec 'node scripts/gate.mjs src/a.ts'",
        "bash -o pipefail -c 'make lint'",
    ]

    def test_command_dirs_match_the_replay(self):
        cases = [("make build", "/p"), ("cd /x && make", "/p"), ("bash -lc 'cd sub && make'", "/p"),
                 ("agent-test-budget run -- /bin/zsh -c 'cd /wt/a && pytest -q'", "/p"), ("npm test", None),
                 ("cd .. && cd q && ls", "/p/r"), ("git -C /repo -C .worktrees/a -c core.x=1 log -C", "/p"),
                 ("cd sub && git -C ../q status", "/p")]
        for cmd, cwd in cases:
            assert rules.command_dir(cmd, cwd) == replay.command_dir(cmd, cwd), cmd
        assert rules.command_dir("bash -lc 'cd sub && make'", "/p") == "/p/sub"
        assert rules.command_dir("git -C /repo -C .worktrees/a -c core.x=1 log -C", "/p") == "/repo/.worktrees/a"

    def test_command_identities_match_the_replay(self):
        compound = ["git status --short && npm test", "npm test 2>&1 | tail -20", "make || make clean; make",
                    "bash -lc 'make && make test'", "cd x && FOO=1 npm test && git status"]
        for cmd in self.COMMANDS + self.PROGRAMS + compound:
            assert rules.cmd_identity(cmd) == replay.cmd_identity(cmd), cmd
        assert rules.cmd_identity("git status --short && npm test") == "git status && npm test"

    def test_program_tokens_match_the_replay(self):
        for cmd in self.PROGRAMS:
            assert rules.program_tokens(cmd) == replay.program_tokens(cmd), cmd
            assert rules.cmd_prefix(cmd) == replay.cmd_prefix(cmd), cmd
        assert rules.cmd_prefix("timeout 60 npm test") == "npm test"

    def test_signatures_kinds_and_prefixes_match_the_replay(self):
        for text in self.ERRORS:
            assert rules.failure_signature(text) == replay.failure_signature(text), text
            assert rules.failure_kind(text) == replay.failure_kind(text), text
        for cmd in self.COMMANDS:
            assert rules.cmd_prefix(cmd) == replay.cmd_prefix(cmd), cmd


class TestCommandFiles:
    def test_existing_relative_and_absolute_paths_count(self, repo):
        cwd = str(wt(repo, "a"))
        found = rules.command_files(f"cat src/app.py && wc -l {repo / 'README.md'} missing.txt", cwd)
        assert found == [os.path.join(cwd, "src", "app.py"), str(repo / "README.md")]

    def test_dot_relative_paths_and_dotfiles_count(self, repo):
        """Regression: the replay's token cleaner stripped the leading dot, so
        ``./x`` became ``/x`` and ``.env`` became ``env`` (fixed in both copies)."""
        (repo / ".env.sample").write_text("A=1\n")
        cwd = str(wt(repo, "a"))
        found = rules.command_files(f"cat ./README.md ../../.env.sample && head {repo}/src/app.py.", cwd)
        assert found == [os.path.join(cwd, "README.md"), str(repo / ".env.sample"), str(repo / "src" / "app.py")]
        for tok in ("./x/y.py", "../y.py", ".env", "src/a.py.", "`src/a.py`"):
            assert rules._clean_token(tok) == replay._clean_token(tok)
        assert replay._clean_token("./x/y.py") == "x/y.py"

    def test_a_program_is_never_a_file_mention(self, repo, tmp_path):
        """Regression (live store, 2026-11-16): /opt/homebrew is a git checkout,
        so the interpreter running report.py was logged as a mention of
        ``bin/python3`` in a repo named homebrew. A segment's program, bare or
        absolute, before or after wrappers, is never a mention."""
        brew = tmp_path / "homebrew"
        git("init", "-q", "-b", "main", str(brew))
        (brew / "bin").mkdir()
        py = brew / "bin" / "python3"
        py.write_text("#!/bin/sh\n")
        venv_py = repo / ".venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.write_text("#!/bin/sh\n")
        outside, readme = str(tmp_path), str(repo / "README.md")
        assert rules.command_files(f"echo hi && {py} -I -S {repo}/src/app.py --minutes 15", outside) == []
        assert rules.command_files(f"env A=1 {py} {repo}/src/app.py", outside) == []
        assert rules.command_files(f"bash -c '{py} {repo}/src/app.py {readme}'", outside) == [readme]
        assert rules.command_files("timeout 60 ./scripts/run.sh src/app.py", str(repo)) == [str(repo / "src" / "app.py")]
        assert rules.command_files(".venv/bin/python -m pytest README.md", str(repo)) == [readme]

    def test_each_segment_resolves_from_the_directory_its_cd_moved_to(self, repo):
        """Review finding: the starting directory won when both had the file."""
        a, b = wt(repo, "a"), wt(repo, "b")
        assert rules.command_files(f"cd {b} && cat README.md", str(a)) == [str(b / "README.md")]
        assert rules.command_files(f"cat README.md && cd {b} && cat src/app.py", str(a)) == [
            str(a / "README.md"), str(b / "src" / "app.py")]

    def test_a_cd_inside_a_shell_c_string_moves_only_that_string(self, repo):
        """Review finding: the cd inside ``bash -c '...'`` was ignored, so its
        paths resolved from the outer directory. Every agent-test-budget run
        has this shape."""
        a, b = wt(repo, "a"), wt(repo, "b")
        assert rules.command_files(f"bash -c 'cd {b} && cat README.md'", str(a)) == [str(b / "README.md")]
        wrapped = f"agent-test-budget run -- /bin/zsh -c 'cd {b} && cat src/app.py'"
        assert rules.command_files(wrapped, str(a)) == [str(b / "src" / "app.py")]
        assert rules.command_files(f"bash -c 'cd {b}' && cat README.md", str(a)) == [str(a / "README.md")]

    def test_a_shell_option_cluster_string_is_read_from_its_own_directory(self, repo):
        """Review finding: only a standalone -c was read as a command string."""
        a, b = wt(repo, "a"), wt(repo, "b")
        assert rules.command_files(f"bash -lc 'cd {b} && cat README.md'", str(a)) == [str(b / "README.md")]
        assert rules.command_files("sh -ec 'cat src/app.py'", str(a)) == [str(a / "src" / "app.py")]

    def test_a_path_after_cd_resolves_from_that_directory(self, repo):
        assert rules.command_files("cd src && cat app.py", str(repo)) == [str(repo / "src" / "app.py")]

    def test_the_script_a_command_runs_is_not_counted(self, repo):
        assert rules.command_files("bash scripts/run.sh src/app.py", str(repo)) == [str(repo / "src" / "app.py")]
        os.chmod(repo / "scripts" / "run.sh", 0o755)
        assert rules.command_files("./scripts/run.sh", str(repo)) == []


class TestActivity:
    def test_file_tools_record_edits_apart_from_reads(self, db, repo):
        a = wt(repo, "a")
        edit(db, "s1", a, T, a / "src" / "app.py", "t1")
        tool_ok(db, "s1", a, T + 1, "Read", {"file_path": str(a / "README.md")}, "t2")
        rows = q(db, "SELECT id, tool, action FROM record WHERE kind = 'activity' ORDER BY id")
        assert [(t, act) for _, t, act in rows] == [("Edit", "edit"), ("Read", "read")]
        key = os.path.realpath(repo / ".git")
        assert links(db, rows[0][0], "file") == [f"{key}::src/app.py"]
        assert links(db, rows[0][0], "repo") == [key]
        assert links(db, rows[0][0], "worktree") == [os.path.realpath(a)]

    def test_a_shell_command_that_names_repo_files_is_a_mention_with_its_program(self, db, repo):
        tool_ok(db, "s1", repo, T, "Bash", {"command": "grep -n x src/app.py README.md"}, "t1")
        (rid, action, info), = q(db, "SELECT id, action, info FROM record WHERE kind = 'activity'")
        assert action == "mention" and json.loads(info) == {"program": "grep"}
        assert [k.split("::")[1] for k in links(db, rid, "file")] == ["README.md", "src/app.py"]
        assert links(db, rid, "program") == ["grep"]

    def test_calls_that_touch_no_repo_file_write_no_activity(self, db, repo, tmp_path):
        tool_ok(db, "s1", repo, T, "Bash", {"command": "echo hello"}, "t1")
        (tmp_path / "loose.txt").write_text("x")
        tool_ok(db, "s1", repo, T, "Read", {"file_path": str(tmp_path / "loose.txt")}, "t2")
        tool_ok(db, "s1", repo, T, "Grep", {"pattern": "x", "path": str(repo)}, "t3")
        assert q(db, "SELECT count(*) FROM record WHERE kind = 'activity'") == [(0,)]
        assert q(db, "SELECT count(*) FROM event WHERE kind = 'observed'") == [(3,)]

    def test_the_interpreter_running_a_command_is_not_activity(self, db, repo, tmp_path):
        """The live row through the whole hook: no activity, and the session
        borrows no repo from the interpreter's checkout."""
        brew = tmp_path / "homebrew"
        git("init", "-q", "-b", "main", str(brew))
        (brew / "bin").mkdir()
        (brew / "bin" / "python3").write_text("#!/bin/sh\n")
        outside = tmp_path / "projects"
        outside.mkdir()
        cmd = f"echo hi && {brew}/bin/python3 -I -S {repo}/src/app.py --minutes 15"
        tool_ok(db, "lead", outside, T, "Bash", {"command": cmd}, "t1")
        assert q(db, "SELECT count(*) FROM record WHERE kind = 'activity'") == [(0,)]
        rid, _, _ = session_record(db, "lead")
        assert links(db, rid, "repo") == [] and links(db, rid, "touched_repo") == []

    def test_a_tool_call_is_recorded_once(self, db, repo):
        edit(db, "s1", repo, T, repo / "src" / "app.py", "t1")
        edit(db, "s1", repo, T + 1, repo / "src" / "app.py", "t1")
        assert q(db, "SELECT count(*) FROM record WHERE kind = 'activity'") == [(1,)]

    def test_a_subagent_call_keeps_the_subagent_id_under_its_session(self, db, repo):
        tool_ok(db, "s1", repo, T, "Read", {"file_path": str(repo / "README.md")}, "t1", agent_id="ag1",
                agent_type="Explore")
        assert q(db, "SELECT session_id, agent_id FROM record WHERE kind = 'activity'") == [("s1", "ag1")]


class TestSession:
    def test_session_start_records_repo_worktree_branch_and_ticket(self, db, repo):
        start(db, "s1", wt(repo, "a"), T)
        rid, info, status = session_record(db, "s1")
        assert json.loads(info) == {"branch": "agent/abc-101-a", "ticket_from": "branch"} and status == "active"
        assert links(db, rid, "ticket") == ["ABC-101"]
        assert links(db, rid, "repo") == [os.path.realpath(repo / ".git")]
        assert links(db, rid, "worktree") == [os.path.realpath(wt(repo, "a"))]

    def test_without_a_branch_ticket_the_first_prompt_supplies_it(self, db, repo):
        start(db, "s1", repo, T)
        fire(db, "UserPromptSubmit", "s1", repo, T + 1, prompt="Work on ABC-77 next, not abc-5")
        fire(db, "UserPromptSubmit", "s1", repo, T + 2, prompt="Also ABC-78")
        rid, info, _ = session_record(db, "s1")
        assert links(db, rid, "ticket") == ["ABC-77"]
        assert json.loads(info)["ticket_from"] == "prompt"

    def test_a_session_first_seen_after_it_started_is_still_recorded(self, db, repo):
        edit(db, "s1", wt(repo, "b"), T, wt(repo, "b") / "README.md", "t1")
        rid, info, _ = session_record(db, "s1")
        assert json.loads(info)["branch"] == "agent/abc-102-b"
        assert links(db, rid, "ticket") == ["ABC-102"]

    def test_a_session_outside_any_repo_records_the_repos_it_touches_apart(self, db, repo, tmp_path):
        """A lead often runs from ~/Projects and edits repos by absolute path. Its
        activity goes under the file's repo, and the session gets touched_repo,
        never a repo of its own, yet still shows in that repo's roster."""
        outside = tmp_path / "projects"
        outside.mkdir()
        key = os.path.realpath(repo / ".git")
        start(db, "lead", outside, T)
        edit(db, "lead", outside, T + 1, repo / "src" / "app.py", "t1")
        rid, _, _ = session_record(db, "lead")
        assert (links(db, rid, "repo"), links(db, rid, "worktree"), links(db, rid, "touched_repo")) == ([], [], [key])
        (arid,), = q(db, "SELECT id FROM record WHERE kind = 'activity'")
        assert links(db, arid, "repo") == [key]
        start(db, "w1", wt(repo, "a"), T + MIN)
        edit(db, "w1", wt(repo, "a"), T + 2 * MIN, wt(repo, "a") / "README.md", "t2")
        assert links(db, session_record(db, "w1")[0], "touched_repo") == []
        (sid, _, records, _), = delivered(db, "session_start")
        assert (sid, records) == ("w1", [rid])

    def test_only_a_session_start_brings_an_ended_session_back(self, db, repo):
        """Review finding: a duplicate tool event after SessionEnd set the
        session active again, putting an ended worker back in the roster."""
        start(db, "s1", repo, T)
        tool_ok(db, "s1", repo, T + 1, "Bash", {"command": "cat README.md"}, "t1")
        fire(db, "SessionEnd", "s1", repo, T + 2, reason="other")
        tool_ok(db, "s1", repo, T + 3, "Bash", {"command": "cat README.md"}, "t1")
        tool_ok(db, "s1", repo, T + 4, "Bash", {"command": "cat src/app.py"}, "t2")
        assert q(db, "SELECT status, ts_end FROM record WHERE kind = 'session'") == [("ended", T + 2)]
        start(db, "s1", repo, T + 5, source="resume")
        assert q(db, "SELECT status, ts_end FROM record WHERE kind = 'session'") == [("active", T + 5)]

    def test_session_end_marks_the_session_ended(self, db, repo):
        start(db, "s1", repo, T)
        fire(db, "SessionEnd", "s1", repo, T + 5, reason="other")
        assert q(db, "SELECT status, ts_end FROM record WHERE kind = 'session'") == [("ended", T + 5)]


class TestFailures:
    def test_a_failure_keeps_program_exit_kind_and_a_keyed_signature_hash(self, db, repo):
        cmd = "psql postgresql://app:hunter2pw@db.internal/main -c 'select 1'"
        err = "Exit code 2\npsql: error: connection to server failed: password=hunter2pw for user app"
        tool_fail(db, "s1", repo, T, "Bash", {"command": cmd}, err, "t1")
        (rid, action, status, info), = q(db, "SELECT id, action, status, info FROM record WHERE kind = 'failure'")
        info = json.loads(info)
        assert action == "command or tool failure" and status == "open"
        assert info["program"] == "psql" and info["exit"] == 2 and "head" not in info and "sig" not in info
        with open(store.key_path(db), "rb") as fh:
            assert info["sig_hash"] == store.signature_hash(fh.read(), replay.failure_signature(err))
        assert links(db, rid, "signature") == [info["sig_hash"]]
        assert links(db, rid, "program") == ["psql"]

    def test_a_later_success_with_the_same_command_prefix_resolves_it(self, db, repo):
        tool_fail(db, "s1", repo, T, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2", "t1")
        tool_ok(db, "s1", repo, T + 30, "Bash", {"command": "make build -j4"}, "t2")
        (status, ts_end, info), = q(db, "SELECT status, ts_end, info FROM record WHERE kind = 'failure'")
        assert status == "resolved" and ts_end == T + 30 and json.loads(info)["resolved_by"] == "t2"

    def test_a_success_that_began_before_the_failure_came_back_does_not_resolve_it(self, db, repo):
        """The replay's rule: a concurrent call that started first is not the
        way past a failure. Its start is its result time less duration_ms."""
        tool_fail(db, "s1", repo, T + 10, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2",
                  "t1", duration_ms=10_000)
        tool_ok(db, "s1", repo, T + 12, "Bash", {"command": "make build"}, "t2", duration_ms=11_000)
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", repo, T + 20, "Bash", {"command": "make build"}, "t3", duration_ms=1_000)
        assert q(db, "SELECT status, ts_end FROM record WHERE kind = 'failure'") == [("resolved", T + 20)]

    def test_a_redelivered_success_never_resolves_a_newer_failure(self, db, repo):
        """Review finding: the same success delivered again ran the resolution
        again and closed a failure that came after it."""
        tool_ok(db, "s1", repo, T, "Bash", {"command": "make build"}, "t1")
        tool_fail(db, "s1", repo, T + 10, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2", "t2")
        tool_ok(db, "s1", repo, T + 20, "Bash", {"command": "make build"}, "t1")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        assert q(db, "SELECT count(*) FROM event WHERE kind = 'observed' AND detail LIKE '%duplicate%'") == [(1,)]

    def test_a_background_launch_resolves_nothing(self, db, repo):
        """The replay's rule: a launch answers before its work ends, so its
        outcome is unknown."""
        tool_fail(db, "s1", repo, T, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2", "t1")
        tool_ok(db, "s1", repo, T + 5, "Bash", {"command": "make build", "run_in_background": True}, "t2")
        tool_ok(db, "s1", repo, T + 6, "Agent", {"prompt": "x"}, "t3", tool_response={"status": "async_launched"})
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", repo, T + 9, "Bash", {"command": "make build"}, "t4")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]

    def test_another_prefix_another_session_or_an_interrupt_does_not_resolve(self, db, repo):
        tool_fail(db, "s1", repo, T, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2", "t1")
        tool_ok(db, "s1", repo, T + 1, "Bash", {"command": "make lint"}, "t2")
        tool_ok(db, "s2", repo, T + 2, "Bash", {"command": "make build"}, "t3")
        tool_fail(db, "s1", repo, T + 3, "Bash", {"command": "sleep 99"}, "Interrupted", "t4", is_interrupt=True)
        assert q(db, "SELECT session_id, status FROM record WHERE kind = 'failure'") == [("s1", "open")]

    def test_a_file_tool_failure_is_resolved_by_the_same_tool_on_the_same_file(self, db, repo):
        app = str(repo / "src" / "app.py")
        tool_fail(db, "s1", repo, T, "Write", {"file_path": app}, "EACCES: permission denied, open", "t1")
        tool_ok(db, "s1", repo, T + 1, "Write", {"file_path": str(repo / "README.md")}, "t2")
        edit(db, "s1", repo, T + 2, app, "t3")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", repo, T + 3, "Write", {"file_path": app}, "t4")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]

    def test_the_same_repo_file_in_another_worktree_does_not_resolve_it(self, db, repo):
        """Review finding: a failed Edit in one worktree was resolved by an Edit
        of the same repo file in another."""
        a_app, b_app = wt(repo, "a") / "src" / "app.py", wt(repo, "b") / "src" / "app.py"
        tool_fail(db, "s1", repo, T, "Edit", {"file_path": str(a_app)}, "EACCES: permission denied, open", "t1")
        edit(db, "s1", repo, T + 1, b_app, "t2")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        edit(db, "s1", repo, T + 2, a_app, "t3")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]

    def test_the_same_command_in_another_directory_does_not_resolve_it(self, db, repo):
        """Review finding: shell resolution ignored where the command ran."""
        a, b = wt(repo, "a"), wt(repo, "b")
        tool_fail(db, "s1", a, T, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2", "t1")
        tool_ok(db, "s1", b, T + 1, "Bash", {"command": "make build"}, "t2")
        tool_ok(db, "s1", repo, T + 2, "Bash", {"command": f"cd {b} && make build"}, "t3")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", repo, T + 3, "Bash", {"command": f"bash -lc 'cd {a} && make build'"}, "t4")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]

    def test_a_compound_failure_is_resolved_only_by_the_same_compound(self, db, repo):
        """Review finding: git status --short && npm test failing was resolved
        by a later git status. Which segment failed is unknown, so neither
        segment alone resolves it."""
        tool_fail(db, "s1", repo, T, "Bash", {"command": "git status --short && npm test"},
                  "Exit code 1\nError: 3 tests failed", "t1")
        tool_ok(db, "s1", repo, T + 1, "Bash", {"command": "git status --short"}, "t2")
        tool_ok(db, "s1", repo, T + 2, "Bash", {"command": "npm test"}, "t3")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", repo, T + 3, "Bash", {"command": "git status && npm test"}, "t4")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]

    def test_git_dash_c_in_another_worktree_does_not_resolve_it(self, db, repo):
        """Review finding: git -C targets collapsed into one identity."""
        a, b = wt(repo, "a"), wt(repo, "b")
        tool_fail(db, "s1", repo, T, "Bash", {"command": f"git -C {a} merge topic"},
                  "Exit code 1\nerror: merge conflict in src/app.py", "t1")
        tool_ok(db, "s1", repo, T + 1, "Bash", {"command": f"git -C {b} merge topic"}, "t2")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", a, T + 2, "Bash", {"command": "git merge topic"}, "t3")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]

    def test_a_later_lint_does_not_resolve_failed_tests(self, db, repo):
        """Review finding: npm run test and npm run lint shared the prefix npm run."""
        tool_fail(db, "s1", repo, T, "Bash", {"command": "npm run test"}, "Exit code 1\nError: 3 tests failed", "t1")
        tool_ok(db, "s1", repo, T + 1, "Bash", {"command": "npm run lint"}, "t2")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", repo, T + 2, "Bash", {"command": "npm run test -- --ci"}, "t3")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]

    def test_files_outside_any_repo_resolve_only_their_own_failures(self, db, repo, tmp_path):
        """Review finding: a file outside every checkout had no file key, so a
        failed Write to one such file was resolved by a Write to another."""
        conf, notes = tmp_path / "conf" / "app.json", tmp_path / "notes.txt"
        conf.parent.mkdir()
        tool_fail(db, "s1", repo, T, "Write", {"file_path": str(conf)}, "EACCES: permission denied, open", "t1")
        tool_ok(db, "s1", repo, T + 1, "Write", {"file_path": str(notes)}, "t2")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]
        tool_ok(db, "s1", repo, T + 2, "Write", {"file_path": str(conf)}, "t3")
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("resolved",)]
        stored = "\n".join(str(row) for table in ("event", "record", "link")
                           for row in q(db, f"SELECT * FROM {table}"))  # noqa: S608
        assert "app.json" not in stored and "notes.txt" not in stored

    def test_a_success_without_a_duration_resolves_nothing(self, db, repo):
        """Review finding: with no ``duration_ms`` the start was taken as the
        result time, so a call that overlapped the failure could resolve it."""
        tool_fail(db, "s1", repo, T + 10, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2", "t1")
        fire(db, "PostToolUse", "s1", repo, T + 12, tool_name="Bash", tool_input={"command": "make build"},
             tool_use_id="t2", tool_response={})
        assert q(db, "SELECT status FROM record WHERE kind = 'failure'") == [("open",)]


class TestWouldDeliver:
    def test_session_start_names_the_other_sessions_working_in_the_repo(self, db, repo):
        start(db, "s1", wt(repo, "a"), T)
        start(db, "s2", wt(repo, "b"), T + 10 * MIN)
        (sid, _, records, detail), = delivered(db, "session_start")
        assert sid == "s2" and records == [session_record(db, "s1")[0]]
        assert detail == {"source": "startup", "repo": "proj"}

    def test_session_start_skips_idle_ended_and_other_repo_sessions(self, db, repo, tmp_path):
        other = tmp_path / "other"
        git("init", "-q", "-b", "main", str(other))
        start(db, "idle", wt(repo, "a"), T)
        start(db, "gone", wt(repo, "a"), T + 30 * MIN)
        fire(db, "SessionEnd", "gone", wt(repo, "a"), T + 31 * MIN, reason="other")
        start(db, "elsewhere", other, T + 40 * MIN)
        start(db, "s2", wt(repo, "b"), T + 61 * MIN)
        assert [d for d in delivered(db, "session_start") if d[0] == "s2"] == []

    def test_compaction_logs_the_roster_again(self, db, repo):
        start(db, "s1", wt(repo, "a"), T)
        start(db, "s2", wt(repo, "b"), T + MIN)
        start(db, "s2", wt(repo, "b"), T + 2 * MIN, source="compact")
        assert [d[3]["source"] for d in delivered(db, "session_start")] == ["startup", "compact"]

    def test_the_first_prompt_names_who_holds_its_ticket_or_has_edited_its_files(self, db, repo):
        a = wt(repo, "a")
        start(db, "s1", a, T)
        edit(db, "s1", a, T + MIN, a / "src" / "app.py", "t1")
        start(db, "s2", repo, T + 2 * MIN)
        fire(db, "UserPromptSubmit", "s2", repo, T + 3 * MIN, prompt="Pick up ABC-101 and fix src/app.py")
        fire(db, "UserPromptSubmit", "s2", repo, T + 4 * MIN, prompt="Again: ABC-101, src/app.py")
        (sid, _, records, detail), = delivered(db, "first_prompt")
        activity = q(db, "SELECT id FROM record WHERE kind = 'activity'")[0][0]
        assert sid == "s2" and records == sorted([session_record(db, "s1")[0], activity])
        assert detail == {"tickets": ["ABC-101"], "files": ["src/app.py"]}

    def test_a_swarm_brief_file_is_looked_up_when_the_worker_reads_it(self, db, repo):
        """ollama-swarm and epic-squad workers get "Read ./SWARM_BRIEF.md ..." as
        their prompt, so the brief's ticket and files only appear when it is read."""
        a = wt(repo, "a")
        start(db, "s1", a, T)
        edit(db, "s1", a, T + MIN, a / "src" / "app.py", "t1")
        (repo / "SWARM_BRIEF.md").write_text("Slice 2 of ABC-101.\nChange src/app.py only.\n")
        start(db, "w1", repo, T + 2 * MIN)
        fire(db, "UserPromptSubmit", "w1", repo, T + 2 * MIN, prompt="Read ./SWARM_BRIEF.md in this directory and follow it exactly.")
        assert delivered(db, "first_prompt") == []
        for n in (1, 2):
            tool_ok(db, "w1", repo, T + 3 * MIN, "Read", {"file_path": str(repo / "SWARM_BRIEF.md")}, f"r{n}")
        (sid, _, records, detail), = delivered(db, "brief_file")
        activity = q(db, "SELECT id FROM record WHERE tool_use_id = 't1'")[0][0]
        assert sid == "w1" and records == sorted([session_record(db, "s1")[0], activity])
        assert detail == {"tickets": ["ABC-101"], "files": ["src/app.py"]}
        rid, info, _ = session_record(db, "w1")
        assert links(db, rid, "ticket") == ["ABC-101"] and json.loads(info)["ticket_from"] == "brief"

    def test_a_brief_read_with_a_shell_command_counts_too(self, db, repo):
        a = wt(repo, "a")
        start(db, "s1", a, T)
        (repo / "EPIC_BRIEF.md").write_text("Slice of ABC-101.\n")
        start(db, "w1", repo, T + MIN)
        tool_ok(db, "w1", repo, T + 2 * MIN, "Bash", {"command": "cat ./EPIC_BRIEF.md"}, "t1")
        (sid, _, records, detail), = delivered(db, "brief_file")
        assert sid == "w1" and records == [session_record(db, "s1")[0]] and detail["tickets"] == ["ABC-101"]

    def test_a_brief_read_after_cd_is_that_directorys_brief(self, db, repo):
        """Review finding: with a brief in both places, the starting directory's
        brief was opened instead of the one the command read."""
        a, b = wt(repo, "a"), wt(repo, "b")
        start(db, "s1", a, T)
        (repo / "SWARM_BRIEF.md").write_text("Nothing to do here.\n")
        (b / "SWARM_BRIEF.md").write_text("Slice of ABC-101.\n")
        start(db, "w1", repo, T + MIN)
        tool_ok(db, "w1", repo, T + 2 * MIN, "Bash", {"command": f"cd {b} && cat SWARM_BRIEF.md"}, "t1")
        (_, _, records, detail), = delivered(db, "brief_file")
        assert detail["tickets"] == ["ABC-101"] and records == [session_record(db, "s1")[0]]

    def test_a_brief_read_after_a_cd_inside_a_shell_c_string_is_that_directorys_brief(self, db, repo):
        """Review finding: the wrong brief was opened and marked read for good."""
        a, b = wt(repo, "a"), wt(repo, "b")
        start(db, "s1", a, T)
        (repo / "SWARM_BRIEF.md").write_text("Nothing to do here.\n")
        (b / "SWARM_BRIEF.md").write_text("Slice of ABC-101.\n")
        start(db, "w1", repo, T + MIN)
        tool_ok(db, "w1", repo, T + 2 * MIN, "Bash", {"command": f"zsh -c 'cd {b} && cat SWARM_BRIEF.md'"}, "t1")
        (_, _, records, detail), = delivered(db, "brief_file")
        assert detail["tickets"] == ["ABC-101"] and records == [session_record(db, "s1")[0]]

    def test_whoever_wrote_a_brief_file_is_not_a_holder_of_it(self, db, repo):
        tool_ok(db, "lead", repo, T, "Write", {"file_path": str(repo / "EPIC_BRIEF.md"), "content": "x"}, "t1")
        (repo / "EPIC_BRIEF.md").write_text("Read README.md first.\n")
        start(db, "w1", repo, T + MIN)
        fire(db, "UserPromptSubmit", "w1", repo, T + MIN, prompt="Read ./EPIC_BRIEF.md in this directory, then begin.")
        tool_ok(db, "w1", repo, T + 2 * MIN, "Read", {"file_path": str(repo / "EPIC_BRIEF.md")}, "t2")
        assert delivered(db, "first_prompt") == [] and delivered(db, "brief_file") == []

    def test_a_same_file_edit_is_logged_once_per_session_file_and_other_session(self, db, repo):
        a, b = wt(repo, "a"), wt(repo, "b")
        edit(db, "s1", a, T, a / "src" / "app.py", "t1")
        edit(db, "s2", b, T + 2 * MIN, b / "src" / "app.py", "t2")
        edit(db, "s2", b, T + 3 * MIN, b / "src" / "app.py", "t3")
        edit(db, "s3", repo, T + 4 * MIN, repo / "src" / "app.py", "t4")
        rows = delivered(db, "same_file_edit")
        assert [(s, d["file"]) for s, _, _, d in rows] == [("s2", "src/app.py"), ("s3", "src/app.py"),
                                                          ("s3", "src/app.py")]
        first = q(db, "SELECT id FROM record WHERE tool_use_id = 't1'")[0][0]
        assert rows[0][2] == [first] and rows[0][3]["age_s"] == 120

    def test_an_old_edit_a_read_or_a_mention_is_not_a_same_file_edit(self, db, repo):
        a, b = wt(repo, "a"), wt(repo, "b")
        edit(db, "s1", a, T, a / "src" / "app.py", "t1")
        tool_ok(db, "s1", a, T + 62 * MIN, "Read", {"file_path": str(a / "src" / "app.py")}, "t2")
        tool_ok(db, "s1", a, T + 62 * MIN, "Bash", {"command": "cat src/app.py"}, "t3")
        edit(db, "s2", b, T + 63 * MIN, b / "src" / "app.py", "t4")
        assert delivered(db, "same_file_edit") == []


class TestSafety:
    def test_the_hook_prints_nothing_and_exits_0(self, db, repo, monkeypatch, capsys):
        payload = {"hook_event_name": "SessionStart", "session_id": "s1", "cwd": str(repo), "source": "startup"}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        assert hook_entry.main(["--db", db]) == 0
        assert capsys.readouterr().out == ""
        assert q(db, "SELECT count(*) FROM record WHERE kind = 'session'") == [(1,)]

    def test_the_hook_script_runs_isolated_and_silent(self, db, repo):
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "swarm_memory", "hook.py")
        payload = {"hook_event_name": "PostToolUse", "session_id": "s1", "cwd": str(repo), "tool_name": "Bash",
                   "tool_input": {"command": "cat README.md"}, "tool_use_id": "t1"}
        res = subprocess.run([sys.executable, "-I", "-S", script, "--db", db], input=json.dumps(payload).encode(),
                             capture_output=True, timeout=30)
        assert (res.returncode, res.stdout) == (0, b"")
        assert q(db, "SELECT action FROM record WHERE kind = 'activity'") == [("mention",)]

    def test_a_package_that_will_not_load_still_exits_0_and_is_counted(self, db, tmp_path):
        import shutil
        pkg = tmp_path / "copy" / "swarm_memory"
        shutil.copytree(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "swarm_memory"), pkg,
                        ignore=shutil.ignore_patterns("__pycache__"))
        (pkg / "observe.py").write_text("def broken(:\n")
        res = subprocess.run([sys.executable, "-I", "-S", str(pkg / "hook.py"), "--db", db], input=b"{}",
                             capture_output=True, timeout=30)
        assert (res.returncode, res.stdout) == (0, b"")
        with open(f"{db}.errors.log") as fh:
            assert [ln.rstrip("\n").split("\t")[-1] for ln in fh] == ["SyntaxError"]

    def test_an_exception_is_counted_and_never_raised(self, db, repo, monkeypatch):
        def boom(*_):
            raise RuntimeError("handler broke")
        monkeypatch.setitem(observe._HANDLERS, "PostToolUse", boom)
        tool_ok(db, "s1", repo, T, "Bash", {"command": "ls"}, "t1")
        observe.run("not json", db_arg=db, ts=T)
        rows = q(db, "SELECT hook, detail FROM event WHERE kind = 'hook_error' ORDER BY id")
        assert [(h, json.loads(d)["type"]) for h, d in rows] == [("PostToolUse", "RuntimeError"),
                                                                  (None, "JSONDecodeError")]
        assert q(db, "SELECT count(*) FROM record") == [(0,)]

    def test_a_busy_store_is_skipped_and_counted(self, db, repo):
        start(db, "s1", repo, T)
        holder = sqlite3.connect(db, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            edit(db, "s1", repo, T + 1, repo / "README.md", "t1")
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert q(db, "SELECT count(*) FROM record WHERE kind = 'activity'") == [(0,)]
        with open(store.side_file(db, "skips")) as fh:
            assert [ln.split("\t")[1] for ln in fh] == ["PostToolUse"]

    def test_the_store_is_private_and_in_wal_mode(self, db, repo):
        start(db, "s1", repo, T)
        assert stat.S_IMODE(os.stat(db).st_mode) == 0o600
        assert q(db, "PRAGMA journal_mode") == [("wal",)]

    def test_no_part_of_a_failed_calls_output_reaches_the_store_or_report(self, db, repo):
        """The store keeps a failed call's exit code, kind and a keyed hash of
        its signature, and no text from its output. Each error is a shape that
        a stored head or signature once leaked; none of its made-up words may
        appear in the store, the links or the report."""
        words = ("hunterpw", "zxcvbnmasd", "qwertyui", "plughxyzzy", "zorblax", "wibblewobble", "ploverpw",
                 "wobblegong", "snorkelpw", "snorkeltok", "elauth", "blorptastic", "frobnicate", "grommetpw",
                 "quibbleish", "escapedpw", "zqvmarker", "longtokenab")
        errors = [
            "Exit code 1\nerror: zqvmarker authentication failed password=hunterpw for user app",
            "Exit code 22\ncurl: error 401 with Authorization: Bearer zxcvbnmasd for the api",
            "Exit code 2\nfatal: could not connect to postgresql://app:qwertyui@db.internal/main",
            'Exit code 1\nerror: invalid config {"password": "plughxyzzy", "user": "app"}',
            'Exit code 1\nerror: login failed password="zorblax quixotic" for app',
            "Exit code 1\nerror: connection failed with PGPASSWORD=wibblewobble set",
            "Exit code 1\nerror: dbcli --password ploverpw failed to connect",
            'Exit code 1\nerror: sync --api-key "fizzle wobblegong" refused by the server',
            'Exit code 1\nerror: bad config {"password":"wobble\\"snorkelpw"} rejected by the server',
            'Exit code 1\nerror: bad config {\\"api_key\\":\\"glimmer snorkeltok\\"} rejected',
            'Exit code 1\nerror: header Authorization: "Basic snork\\"elauth" was refused today',
            "Exit code 22\nerror: curl --user app:blorptastic https://api.internal failed",
            "Exit code 22\nerror: curl -uapp:frobnicate -X POST https://api.internal failed",
            "Exit code 1\nerror: login --password 'prefix'\\''grommetpw' was refused",
            "Exit code 1\nfatal: could not connect to https://app:snork@quibbleish@db.internal/main",
            "Exit code 1\nerror: login --password prefix\\ escapedpw failed",
            "Exit code 1\nerror: invalid api key " + "longtoken" + "Ab" * 12 + " for this request",
        ]
        for n, err in enumerate(errors):
            tool_fail(db, "s1", repo, T + n, "Bash", {"command": f"tool{n} run"}, err, f"t{n}")
        infos = [json.loads(i) for (i,) in q(db, "SELECT info FROM record WHERE kind = 'failure'")]
        assert len(infos) == len(errors)
        assert all(set(i) <= {"exit", "sig_hash", "program", "prefix_key", "file", "where"} for i in infos)
        keys = q(db, "SELECT key FROM link WHERE type = 'signature'")
        assert len(keys) == sum(replay.failure_signature(e) is not None for e in errors)
        assert all(re.fullmatch(r"[0-9a-f]{32}", k) for (k,) in keys)
        stored = "\n".join(str(row) for table in ("event", "record", "link")
                           for row in q(db, f"SELECT * FROM {table}"))  # noqa: S608
        text = (stored + report.build(db)).lower()
        for word in words:
            assert word not in text, word

    def test_the_signature_key_is_made_once_private_and_repeats_match(self, db, repo):
        err = "Exit code 2\nmake: *** [build] Error 2"
        tool_fail(db, "s1", repo, T, "Bash", {"command": "make build"}, err, "t1")
        tool_fail(db, "s2", repo, T + 1, "Bash", {"command": "make build"}, err, "t2")
        key_file = store.key_path(db)
        assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600 and os.path.getsize(key_file) == store.KEY_BYTES
        with open(key_file, "rb") as fh:
            expected = store.signature_hash(fh.read(), replay.failure_signature(err))
        assert q(db, "SELECT key FROM link WHERE type = 'signature'") == [(expected,), (expected,)]

    def test_a_missing_key_makes_a_new_one_and_logs_it(self, db, repo):
        """Hashes made under a lost key no longer match new ones, so the new key
        is logged with the number of hashes made before it."""
        err = "Exit code 2\nmake: *** [build] Error 2"
        tool_fail(db, "s1", repo, T, "Bash", {"command": "make build"}, err, "t1")
        os.remove(store.key_path(db))
        tool_fail(db, "s1", repo, T + 1, "Bash", {"command": "make build"}, err, "t2")
        (a,), (b,) = q(db, "SELECT key FROM link WHERE type = 'signature' ORDER BY record_id")
        assert a != b
        (detail,), = q(db, "SELECT detail FROM event WHERE kind = 'signature_key_new'")
        assert json.loads(detail) == {"earlier_signatures": 1}

    def test_a_key_file_of_the_wrong_size_is_an_error_and_is_kept(self, db, repo):
        start(db, "s1", repo, T)
        with open(store.key_path(db), "wb") as fh:
            fh.write(b"short")
        tool_fail(db, "s1", repo, T + 1, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2",
                  "t1")
        assert [json.loads(d)["type"] for (d,) in q(db, "SELECT detail FROM event WHERE kind = 'hook_error'")] == [
            "ValueError"]
        with open(store.key_path(db), "rb") as fh:
            assert fh.read() == b"short"

    def test_a_version_1_store_is_migrated_to_hash_only_once(self, db):
        """Version 1 kept a redacted error head and the signature text. Opening
        it with this code hashes the signature in the record and the link and
        drops the head. A record already without text is left as it is."""
        old_sig = "error: login failed for user <q>"
        con = sqlite3.connect(db, isolation_level=None)
        con.executescript(store.SCHEMA)
        con.execute("PRAGMA user_version=1")
        rows = [(1, {"head": "error: login failed for user 'zorkquill'", "sig": old_sig, "exit": 1,
                     "program": "login"}),
                (2, {"exit": 2, "program": "make"}),
                (3, {"sig_hash": "0" * 32, "exit": 1, "program": "make"})]
        for rid, info in rows:
            con.execute("INSERT INTO record (id, kind, ts, session_id, tool, action, status, tool_use_id, info)"
                        " VALUES (?, 'failure', ?, 's1', 'Bash', 'command or tool failure', 'open', ?, ?)",
                        (rid, T, f"t{rid}", json.dumps(info)))
        con.execute("INSERT INTO link VALUES (1, 'signature', ?, ?)", (old_sig, T))
        con.execute("INSERT INTO link VALUES (3, 'signature', ?, ?)", ("0" * 32, T))
        con.close()
        store.connect(db).close()
        with open(store.key_path(db), "rb") as fh:
            new = store.signature_hash(fh.read(), old_sig)
        assert {rid: json.loads(i) for rid, i in q(db, "SELECT id, info FROM record ORDER BY id")} == {
            1: {"sig_hash": new, "exit": 1, "program": "login"}, 2: {"exit": 2, "program": "make"},
            3: {"sig_hash": "0" * 32, "exit": 1, "program": "make"}}
        assert q(db, "SELECT record_id, key FROM link ORDER BY record_id") == [(1, new), (3, "0" * 32)]
        assert q(db, "PRAGMA user_version") == [(2,)]
        con = sqlite3.connect(db, isolation_level=None)
        assert store.migrate_to_hash_only(con, db) == {}
        con.close()
        dump = "\n".join(str(row) for table in ("event", "record", "link")
                         for row in q(db, f"SELECT * FROM {table}"))  # noqa: S608
        assert "zorkquill" not in dump and "login failed" not in dump

    def test_the_report_shows_a_short_hash_and_the_tool_use_id(self, db, repo):
        tool_fail(db, "s1", repo, T, "Bash", {"command": "make build"}, "Exit code 2\nmake: *** [build] Error 2",
                  "toolu_x1")
        (sig,), = q(db, "SELECT key FROM link WHERE type = 'signature'")
        out = report.build(db)
        assert "toolu_x1" in out and sig[:12] in out and sig not in out and "make: ***" not in out

    def test_no_prompt_command_or_output_text_is_stored(self, db, repo):
        start(db, "s1", repo, T)
        fire(db, "UserPromptSubmit", "s1", repo, T, prompt="zebraprompt words about README.md")
        tool_ok(db, "s1", repo, T, "Bash", {"command": "echo zebracommand >> README.md"}, "t1",
                tool_response={"stdout": "zebraoutput"})
        tool_fail(db, "s1", repo, T, "Bash", {"command": "grep zebrafail README.md"},
                  "Exit code 1\n" + "zebraerrortail " * 40, "t2")
        dump = "\n".join(str(row) for table in ("event", "record", "link")
                         for row in q(db, f"SELECT * FROM {table}"))  # noqa: S608
        for word in ("zebraprompt", "zebracommand", "zebraoutput", "zebrafail"):
            assert word not in dump
        assert "zebraerrortail" not in dump
