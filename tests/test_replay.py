"""Phase 1 replay (MUD-850): the counting rules behind each moment's hits.

A wrong rule here does not error. It changes a count that decides whether
Phases 2 and 3 get built, so each rule is pinned on a small case.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "replay"))

import replay  # noqa: E402

MIN = 60.0


def _session(sid, worktree, spans, repo="r"):
    return {"sid": sid, "repo": repo, "worktree": worktree, "spans": spans}


class TestActivitySpans:
    def test_an_idle_gap_longer_than_the_threshold_splits_a_span(self):
        times = [0, 5 * MIN, 9 * MIN, 40 * MIN, 41 * MIN]
        assert replay.activity_spans(times, gap=15 * MIN) == [(0, 9 * MIN), (40 * MIN, 41 * MIN)]

    def test_unsorted_times_give_the_same_spans(self):
        assert replay.activity_spans([9 * MIN, 0, 5 * MIN], gap=15 * MIN) == [(0, 9 * MIN)]


class TestGroupRuns:
    def test_overlapping_sessions_in_two_worktrees_form_a_run(self):
        runs = replay.group_runs([
            _session("a", "wt1", [(0, 30 * MIN)]),
            _session("b", "wt2", [(20 * MIN, 50 * MIN)]),
        ])
        assert runs == [["a", "b"]]

    def test_sessions_that_never_overlap_while_active_form_no_run(self):
        """First-to-last timestamps overlap here, but the active spans do not:
        one session was idle while the other worked."""
        runs = replay.group_runs([
            _session("a", "wt1", [(0, 10 * MIN), (200 * MIN, 210 * MIN)]),
            _session("b", "wt2", [(60 * MIN, 120 * MIN)]),
        ])
        assert runs == []

    def test_overlap_inside_one_worktree_is_not_a_run(self):
        runs = replay.group_runs([
            _session("a", "wt1", [(0, 30 * MIN)]),
            _session("b", "wt1", [(10 * MIN, 40 * MIN)]),
        ])
        assert runs == []

    def test_different_repos_never_join(self):
        runs = replay.group_runs([
            _session("a", "wt1", [(0, 30 * MIN)], repo="r1"),
            _session("b", "wt2", [(10 * MIN, 40 * MIN)], repo="r2"),
        ])
        assert runs == []

    def test_a_chain_of_overlaps_is_one_run(self):
        runs = replay.group_runs([
            _session("a", "wt1", [(0, 30 * MIN)]),
            _session("b", "wt2", [(20 * MIN, 60 * MIN)]),
            _session("c", "wt3", [(50 * MIN, 90 * MIN)]),
        ])
        assert runs == [["a", "b", "c"]]


class TestCmdPrefix:
    def test_a_leading_cd_is_skipped(self):
        assert replay.cmd_prefix("cd packages/frontend && npm test") == "npm test"

    def test_git_dash_c_is_dropped(self):
        assert replay.cmd_prefix("git -C /Users/x/repo status --short") == "git status"

    def test_env_assignments_and_runners_are_stripped(self):
        assert replay.cmd_prefix("FOO=1 uv run --no-sync pytest tests/test_a.py -q") == "pytest tests/test_a.py"

    def test_python_dash_m_names_the_module(self):
        assert replay.cmd_prefix(".venv/bin/python -m pytest tests/test_a.py") == "pytest tests/test_a.py"

    def test_a_command_that_is_only_cd_keeps_cd(self):
        assert replay.cmd_prefix("cd nowhere") == "cd"


class TestFailureSignature:
    def test_bash_error_uses_the_last_error_line_with_placeholders(self):
        text = "Exit code 1\n(eval):cd:1: no such file or directory: packages/frontend/src"
        sig = replay.failure_signature(text)
        assert sig is not None
        assert "no such file or directory" in sig
        assert "packages/frontend/src" not in sig

    def test_noise_is_not_a_failure(self):
        """The F4 NOISE filter: a rejected or blocked call is not a failure."""
        assert replay.failure_signature("The user doesn't want to proceed with this tool use.") is None

    def test_two_paths_with_the_same_error_share_a_signature(self):
        a = replay.failure_signature("Exit code 1\nzsh: no such file or directory: /a/b/one.py")
        b = replay.failure_signature("Exit code 1\nzsh: no such file or directory: /c/d/two.py")
        assert a == b


class TestSameKind:
    def test_bash_matches_on_command_prefix(self):
        fail = {"tool": "Bash", "command": "cd x && npm test", "file": None}
        ok = {"tool": "Bash", "command": "npm test -- --run", "file": None}
        other = {"tool": "Bash", "command": "npm run build", "file": None}
        assert replay.same_kind(fail, ok)
        assert not replay.same_kind(fail, other)

    def test_file_tools_match_on_tool_and_file(self):
        fail = {"tool": "Edit", "command": None, "file": "src/a.py"}
        assert replay.same_kind(fail, {"tool": "Edit", "command": None, "file": "src/a.py"})
        assert not replay.same_kind(fail, {"tool": "Edit", "command": None, "file": "src/b.py"})
        assert not replay.same_kind(fail, {"tool": "Write", "command": None, "file": "src/a.py"})

    def test_files_outside_any_repo_are_told_apart_by_their_paths(self):
        """Review finding: file-tool calls outside every repo had no file, so a
        failed Write to one file was resolved by a Write to another."""
        fail = {"tool": "Write", "command": None, "file": None, "path": "/Users/me/.config/app/config.json"}
        other = {"tool": "Write", "command": None, "file": None, "path": "/tmp/notes.txt"}
        same = {"tool": "Write", "command": None, "file": None, "path": "/Users/me/.config/app/./config.json"}
        assert not replay.same_kind(fail, other)
        assert replay.same_kind(fail, same)
        no_path = {"tool": "Write", "command": None, "file": None, "path": None}
        assert not replay.same_kind(no_path, dict(no_path))


class TestFirstResolution:
    def test_the_first_later_success_of_the_same_kind_resolves(self):
        calls = [
            {"tool": "Bash", "command": "npm test", "file": None, "ok": False},
            {"tool": "Read", "command": None, "file": "a.ts", "ok": True},
            {"tool": "Bash", "command": "npm test", "file": None, "ok": False},
            {"tool": "Bash", "command": "npm test", "file": None, "ok": True},
        ]
        assert replay.first_resolution(calls, 0) == 3

    def test_no_later_success_means_unresolved(self):
        calls = [
            {"tool": "Bash", "command": "npm test", "file": None, "ok": False},
            {"tool": "Bash", "command": "npm run build", "file": None, "ok": True},
        ]
        assert replay.first_resolution(calls, 0) is None

    def test_a_success_that_began_before_the_failure_came_back_does_not_resolve_it(self):
        """Review finding: a concurrent success counted as the way past a
        failure that had not happened yet."""
        calls = [
            {"tool": "Bash", "command": "npm test", "file": None, "ok": False, "ts": 0.0, "rts": 10 * MIN},
            {"tool": "Bash", "command": "npm test", "file": None, "ok": True, "ts": 1 * MIN, "rts": 2 * MIN},
            {"tool": "Bash", "command": "npm test", "file": None, "ok": True, "ts": 11 * MIN, "rts": 12 * MIN},
        ]
        assert replay.first_resolution(calls, 0) == 2

    def test_calls_already_in_flight_are_not_recovery_effort(self):
        """Review finding: recovery effort counted every call started since the
        failed one, including twenty that were running when it came back."""
        calls = [{"tool": "Bash", "command": "npm test", "file": None, "ok": False, "ts": 0.0, "rts": 10 * MIN}]
        calls += [{"tool": "Read", "command": None, "file": f"f{n}", "ok": True, "ts": n, "rts": n + 1}
                  for n in range(1, 21)]
        calls.append({"tool": "Bash", "command": "npm test", "file": None, "ok": True, "ts": 11 * MIN, "rts": 12 * MIN})
        j = replay.first_resolution(calls, 0)
        assert (j, replay.recovery_calls(calls, 0, j)) == (21, 1)

    def test_calls_begun_while_the_retry_ran_are_recovery_effort(self):
        """Review finding: effort stopped at the retry's start, though the
        retry got past the failure only when it finished."""
        calls = [{"tool": "Bash", "command": "npm test", "file": None, "ok": False, "ts": 0.0, "rts": 1 * MIN},
                 {"tool": "Bash", "command": "npm test", "file": None, "ok": True, "ts": 2 * MIN, "rts": 10 * MIN}]
        calls += [{"tool": "Read", "command": None, "file": f"f{n}", "ok": True, "ts": n * MIN, "rts": n * MIN + 1}
                  for n in range(3, 10)]
        calls.append({"tool": "Read", "command": None, "file": "late", "ok": True, "ts": 11 * MIN, "rts": 11 * MIN + 1})
        j = replay.first_resolution(calls, 0)
        assert (j, replay.recovery_calls(calls, 0, j)) == (1, 8)

    def test_the_first_to_finish_among_later_successes_resolves(self):
        calls = [
            {"tool": "Bash", "command": "npm test", "file": None, "ok": False, "ts": 0.0, "rts": 1 * MIN},
            {"tool": "Bash", "command": "npm test", "file": None, "ok": True, "ts": 2 * MIN, "rts": 9 * MIN},
            {"tool": "Bash", "command": "npm test", "file": None, "ok": True, "ts": 3 * MIN, "rts": 4 * MIN},
        ]
        assert replay.first_resolution(calls, 0) == 2


def _touch(ts, sid, wt, f):
    return {"ts": ts, "sid": sid, "worktree": wt, "file": f}


class TestFileWindowHits:
    def test_a_later_touch_by_another_worktree_inside_the_window_hits(self):
        events = [_touch(0, "a", "wt1", "src/x.py"), _touch(10 * MIN, "b", "wt2", "src/x.py")]
        hits = replay.file_window_hits(events, 15 * MIN)
        assert [h["event"]["sid"] for h in hits] == ["b"]
        assert hits[0]["other"]["sid"] == "a"

    def test_outside_the_window_is_no_hit(self):
        events = [_touch(0, "a", "wt1", "src/x.py"), _touch(20 * MIN, "b", "wt2", "src/x.py")]
        assert replay.file_window_hits(events, 15 * MIN) == []
        assert len(replay.file_window_hits(events, 60 * MIN)) == 1

    def test_the_same_worktree_is_no_hit(self):
        events = [_touch(0, "a", "wt1", "src/x.py"), _touch(5 * MIN, "b", "wt1", "src/x.py")]
        assert replay.file_window_hits(events, 15 * MIN) == []

    def test_a_read_between_two_edits_does_not_hide_the_edit_after_edit(self):
        """Review finding: the latest touch was the first worktree's read-back,
        so its earlier edit did not count, though the hook's rule fires on it."""
        events = [{**_touch(0, "a", "wt1", "src/x.py"), "edit": True, "call": "c1"},
                  {**_touch(1 * MIN, "a", "wt1", "src/x.py"), "edit": False, "call": "c2"},
                  {**_touch(5 * MIN, "b", "wt2", "src/x.py"), "edit": True, "call": "c3"}]
        assert replay.edit_after_edit_calls(events, 15 * MIN) == {"c3"}
        assert replay.edit_after_edit_calls(events, 4 * MIN) == set()

    def test_a_call_that_never_ran_is_no_touch(self):
        """Review finding: an Edit a hook refused, or whose input the tool
        rejected, counted as an edit of the file."""
        denied = {"ok": False, "blocked": False, "error": "PreToolUse:Edit hook error: [gate.sh]: not allowed"}
        rejected = {"ok": False, "blocked": False, "error": "<tool_use_error>String to replace not found</tool_use_error>"}
        failed_run = {"ok": False, "blocked": False, "error": "Exit code 1\nError: 3 failed"}
        assert not replay.call_ran(denied) and not replay.call_ran(rejected)
        assert replay.call_ran(failed_run) and replay.call_ran({"ok": True, "blocked": False})
        assert not replay.call_ran({"ok": None, "blocked": True})

    def test_a_touch_is_timed_by_its_result_and_a_call_without_one_is_left_out(self):
        """Review finding: touches were timed by the call's start, and calls
        that never returned counted. A hook sees a touch when its result comes."""
        assert replay.touch_ts({"ts": 0.0, "rts": 2 * MIN}) == 2 * MIN
        assert replay.touch_ts({"ts": 0.0, "rts": None}) is None
        long_call = replay.touch_ts({"ts": 0.0, "rts": 20 * MIN})
        quick_call = replay.touch_ts({"ts": 1 * MIN, "rts": 2 * MIN})
        events = [_touch(long_call, "a", "wt1", "src/x.py"), _touch(quick_call, "b", "wt2", "src/x.py")]
        hits = replay.file_window_hits(events, 60 * MIN)
        assert [(h["event"]["sid"], h["other"]["sid"]) for h in hits] == [("a", "b")]

    def test_related_files_hit_through_the_relation(self):
        events = [_touch(0, "a", "wt1", "src/x.py"), _touch(5 * MIN, "b", "wt2", "src/y.py")]
        assert replay.file_window_hits(events, 15 * MIN) == []
        related = {"src/y.py": {"src/x.py"}, "src/x.py": {"src/y.py"}}
        hits = replay.file_window_hits(events, 15 * MIN, related=lambda f: related.get(f, set()))
        assert [h["event"]["file"] for h in hits] == ["src/y.py"]


class TestRepoRel:
    def test_a_path_under_the_repo_in_other_letter_case_still_resolves(self):
        """macOS paths are case-insensitive; sessions opened under
        ~/projects and ~/Projects touch the same files."""
        assert replay.repo_rel("/users/me/Proj/src/a.py", "/Users/me/proj") == "src/a.py"

    def test_a_path_in_another_repos_worktree_does_not(self):
        assert replay.repo_rel("/Users/me/other/.claude/worktrees/w/src/a.py", "/Users/me/proj") is None


class TestCommandFiles:
    FILES = {"src/app/main.py", "tests/test_main.py"}

    def test_repo_files_named_in_a_command_are_touches(self):
        cmd = "cd /Users/me/proj && sed -i 's/a/b/' src/app/main.py && pytest tests/test_main.py -q"
        assert replay.command_files(cmd, self.FILES, repo_root="/Users/me/proj") == [
            "src/app/main.py", "tests/test_main.py"]

    def test_a_command_naming_no_repo_file_touches_nothing(self):
        assert replay.command_files("git status --short", self.FILES, repo_root="/Users/me/proj") == []

    def test_a_script_run_as_the_program_is_executed_not_worked_on(self):
        """Every agent runs the same gate script; that is using a tool, not
        working on the file, and it would make every pair of agents collide."""
        files = {"scripts/verify/gate.mjs", "src/a.ts", "tools/check.sh", "src/app/main.py"}
        cmd = "node scripts/verify/gate.mjs --check src/a.ts && bash tools/check.sh && pytest src/app/main.py"
        assert replay.executed_files(cmd, files, repo_root="/Users/me/proj") == {
            "scripts/verify/gate.mjs", "tools/check.sh"}

    def test_a_shell_c_string_is_parsed_as_commands(self):
        """Review finding: in ``zsh -c 'cat src/a.ts'`` the whole string was
        taken as the script the shell runs, so the read of src/a.ts was dropped."""
        files = {"src/a.ts", "scripts/run.sh"}
        root = "/Users/me/proj"
        assert replay.executed_files("zsh -c 'cat src/a.ts'", files, repo_root=root) == set()
        wrapped = "agent-test-budget run -- /bin/zsh -c 'cd /Users/me/proj && bash scripts/run.sh src/a.ts'"
        assert replay.executed_files(wrapped, files, repo_root=root) == {"scripts/run.sh"}
        assert replay.command_files(wrapped, files, repo_root=root) == ["scripts/run.sh", "src/a.ts"]

    def test_inline_code_names_no_script(self):
        files = {"src/a.ts", "tools/x.py", "scripts/run.sh"}
        root = "/Users/me/proj"
        assert replay.executed_files("python3 -c \"print(open('src/a.ts').read())\"", files, repo_root=root) == set()
        assert replay.executed_files("node -e \"require('./src/a.ts')\"", files, repo_root=root) == set()
        assert replay.executed_files("python3 -u tools/x.py src/a.ts", files, repo_root=root) == {"tools/x.py"}
        assert replay.executed_files("bash -e scripts/run.sh", files, repo_root=root) == {"scripts/run.sh"}


class TestSameDirectory:
    def test_siblings_relate_and_a_file_does_not_relate_to_itself(self):
        rel = replay.same_directory_relation(["src/a.py", "src/b.py", "lib/c.py"])
        assert rel("src/a.py") == {"src/b.py"}
        assert rel("lib/c.py") == set()


class TestRepeatFailureHits:
    def _f(self, ts, sid, wt, sig, resolved_ts=None):
        return {"ts": ts, "sid": sid, "worktree": wt, "sig": sig, "resolved_ts": resolved_ts}

    def test_a_signature_another_worktree_already_got_past_hits_with_the_fix(self):
        fs = [self._f(0, "a", "wt1", "s", resolved_ts=2 * MIN), self._f(10 * MIN, "b", "wt2", "s")]
        hits = replay.repeat_failure_hits(fs)
        assert len(hits) == 1
        assert hits[0]["earlier"]["sid"] == "a"
        assert hits[0]["resolved_before"] is True

    def test_an_earlier_failure_not_yet_resolved_hits_without_the_fix(self):
        fs = [self._f(0, "a", "wt1", "s", resolved_ts=30 * MIN), self._f(10 * MIN, "b", "wt2", "s")]
        hits = replay.repeat_failure_hits(fs)
        assert hits[0]["resolved_before"] is False
        assert hits[0]["resolved_ever"] is True

    def test_the_same_worktree_or_another_signature_is_no_hit(self):
        fs = [self._f(0, "a", "wt1", "s"), self._f(5 * MIN, "b", "wt1", "s"), self._f(6 * MIN, "c", "wt2", "t")]
        assert replay.repeat_failure_hits(fs) == []

    def test_a_window_drops_earlier_failures_older_than_it(self):
        fs = [self._f(0, "a", "wt1", "s"), self._f(30 * MIN, "b", "wt2", "s")]
        assert replay.repeat_failure_hits(fs, window_s=15 * MIN) == []
        assert len(replay.repeat_failure_hits(fs, window_s=60 * MIN)) == 1


class TestFailureKind:
    def test_a_hook_block_is_its_own_kind(self):
        text = "PreToolUse:Bash hook error: [agent-test-budget]: local tests share two slots"
        assert replay.failure_kind(text) == "hook block"

    def test_a_rejected_tool_input_is_its_own_kind(self):
        text = "InputValidationError: if your input does not match this schema, correct it"
        assert replay.failure_kind(text) == "tool input rejected"

    def test_anything_else_is_a_command_or_tool_failure(self):
        assert replay.failure_kind("Exit code 1\nzsh: command not found: vulture") == "command or tool failure"


def _write_jsonl(path, records):
    import json

    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(path)


def _rec(kind, sid, ts, content, **extra):
    return {"type": kind, "sessionId": sid, "timestamp": ts, "cwd": "/repo",
            "message": {"role": kind, "content": content}, **extra}


class TestSessionAttribution:
    def test_records_copied_into_a_resumed_session_stay_with_the_session_that_made_them(self, tmp_path):
        """A resumed or forked transcript repeats the earlier session's
        records under their original sessionId. Counting them twice doubles
        prompts and failures and can fake an overlap between worktrees."""
        use = {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "make"}}
        result = {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "Exit code 2\nmake: *** no rule"}
        original = [_rec("assistant", "aaa", "2026-10-01T10:00:00Z", [use]),
                    _rec("user", "aaa", "2026-10-01T10:00:05Z", [result])]
        own_use = {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/repo/x.py"}}
        resumed = original + [_rec("assistant", "bbb", "2026-10-02T09:00:00Z", [own_use])]
        a = replay.scan_transcript(_write_jsonl(tmp_path / "aaa.jsonl", original), 0)
        b = replay.scan_transcript(_write_jsonl(tmp_path / "bbb.jsonl", resumed), 0)

        merged = replay.merge_scans([a, b])

        assert [c["id"] for c in merged["aaa"]["calls"]] == ["t1"]
        assert merged["aaa"]["calls"][0]["ok"] is False
        assert [c["id"] for c in merged["bbb"]["calls"]] == ["t2"]
        assert len(merged["aaa"]["times"]) == 2


class TestSessionBrief:
    def test_a_brief_from_the_session_that_started_this_one_is_a_brief(self):
        rec = {"type": "user", "promptSource": "sdk", "turnOrigin": "sdk",
               "message": {"content": "Background (from the session that started this one): fix ABC-1"}}
        assert replay.session_brief(rec).startswith("Background")

    def test_a_human_prompt_is_not_a_brief(self):
        rec = {"type": "user", "origin": {"kind": "human"}, "message": {"content": "fix ABC-1"}}
        assert replay.session_brief(rec) is None


class TestHumanPrompt:
    def _rec(self, content, **extra):
        return {"type": "user", "message": {"role": "user", "content": content}, **extra}

    def test_a_human_turn_is_a_prompt(self):
        rec = self._rec("fix the bug", origin={"kind": "human"}, promptSource="sdk")
        assert replay.human_prompt(rec) == "fix the bug"

    def test_a_task_notification_is_not(self):
        rec = self._rec("<task-notification>x</task-notification>",
                        origin={"kind": "task-notification"}, promptSource="system")
        assert replay.human_prompt(rec) is None

    def test_a_system_reminder_is_stripped_from_a_human_turn(self):
        rec = self._rec("<system-reminder>You are in a worktree</system-reminder>\nrun the tests",
                        origin={"kind": "human"}, promptSource="sdk")
        assert replay.human_prompt(rec) == "run the tests"

    def test_shell_mode_input_is_not_a_prompt(self):
        rec = self._rec("<bash-input>ls</bash-input>", origin={"kind": "human"}, promptSource="sdk")
        assert replay.human_prompt(rec) is None

    def test_a_session_started_by_another_session_is_not_human(self):
        rec = self._rec("Background (from the session that started this one): do X",
                        promptSource="sdk", turnOrigin="sdk")
        assert replay.human_prompt(rec) is None

    def test_a_tool_result_is_not_a_prompt(self):
        rec = self._rec([{"type": "tool_result", "tool_use_id": "t", "content": "ok"}])
        assert replay.human_prompt(rec) is None


class TestPromptTriggers:
    FILES = {"src/app/main.py", "README.md", "webapp/packages/core/src/a.ts"}

    def test_an_existing_repo_path_is_found(self):
        assert replay.prompt_paths("look at src/app/main.py:42 please", self.FILES) == ["src/app/main.py"]

    def test_an_absolute_worktree_path_is_normalised(self):
        text = "see /Users/me/proj/.claude/worktrees/wt-1/src/app/main.py"
        assert replay.prompt_paths(text, self.FILES, repo_root="/Users/me/proj") == ["src/app/main.py"]

    def test_a_path_relative_to_the_session_cwd_is_found(self):
        assert replay.prompt_paths("open packages/core/src/a.ts", self.FILES, cwd_prefix="webapp") == [
            "webapp/packages/core/src/a.ts"
        ]

    def test_a_path_that_does_not_exist_is_not(self):
        assert replay.prompt_paths("what about src/app/other.py", self.FILES) == []

    def test_ticket_keys_are_the_configured_prefixes_in_upper_case(self):
        text = "Do ABC-850 then XYZ-12, not UTF-8 or abc-1"
        assert replay.prompt_tickets(text, keys=("ABC", "XYZ")) == ["ABC-850", "XYZ-12"]


class TestTicketKeys:
    """Ticket prefixes are a local setting, never part of the repo."""

    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWARM_MEMORY_DB", str(tmp_path / "observe.sqlite3"))
        monkeypatch.delenv("SWARM_MEMORY_TICKET_KEYS", raising=False)
        self.config = tmp_path / "config.json"

    def test_keys_come_from_the_flag_then_the_environment_then_the_store_config(self, monkeypatch):
        self.config.write_text('{"ticket_keys": ["ABC", "not a key", "XYZ", "ABC"]}')
        assert replay.load_ticket_keys() == ("ABC", "XYZ")
        monkeypatch.setenv("SWARM_MEMORY_TICKET_KEYS", "OPS")
        assert replay.load_ticket_keys() == ("OPS",)
        assert replay.load_ticket_keys("QA, ABC") == ("QA", "ABC")

    def test_with_no_keys_configured_no_prompt_names_a_ticket(self):
        assert replay.load_ticket_keys() == ()
        assert replay.prompt_tickets("Do ABC-850 now", keys=()) == []


class TestReadOnly:
    def test_a_repos_own_ontology_library_is_never_run(self, tmp_path, monkeypatch):
        """Review finding: the replay imported ontology/scripts/ontology_lib.py
        from the repo it read. Only the installed plugin's parser may run."""
        repo = tmp_path / "repo"
        (repo / "ontology" / "scripts").mkdir(parents=True)
        marker = tmp_path / "repo-code-ran"
        (repo / "ontology" / "scripts" / "ontology_lib.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
        plugin = tmp_path / "plugin"
        plugin.mkdir()
        (plugin / "ontology_lib.py").write_text(
            "def load_all(root):\n    return []\n\ndef path_governed_by(f, onts):\n    return False\n")
        monkeypatch.setattr(replay, "PLUGIN_ONTOLOGY_LIB", str(plugin))
        assert replay.ontology_rules(str(repo), {"src/a.py"}) == ({}, "plugin 0.1.0")
        assert not marker.exists()


class TestBackgroundLaunch:
    def test_a_background_launch_has_no_known_outcome_and_resolves_nothing(self, tmp_path):
        """Review finding: a background launch's acknowledgement counted as the
        way past a failure, even if the command later failed."""
        import json
        sid, recs = "s-bg", []

        def call(i, cmd, at, result, error=False, background=False):
            inp = {"command": cmd, **({"run_in_background": True} if background else {})}
            recs.append({"type": "assistant", "sessionId": sid, "timestamp": at[0], "cwd": "/tmp/x",
                         "message": {"content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": inp}]}})
            recs.append({"type": "user", "sessionId": sid, "timestamp": at[1], "cwd": "/tmp/x",
                         "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": result,
                                                  "is_error": error}]}})

        call(0, "npm test", ("2026-11-01T10:00:00Z", "2026-11-01T10:01:00Z"), "Exit code 1\nError: 3 failed", error=True)
        call(1, "npm test", ("2026-11-01T10:02:00Z", "2026-11-01T10:02:01Z"),
             "Command running in background with ID: b1. Output is being written to: /tmp/o", background=True)
        call(2, "npm test", ("2026-11-01T10:05:00Z", "2026-11-01T10:06:00Z"), "ok")
        path = tmp_path / f"{sid}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        calls = replay.scan_transcript(str(path), 0)["sessions"][sid]["calls"]
        assert [c["ok"] for c in calls] == [False, None, True]
        assert replay.first_resolution(calls, 0) == 2
        assert replay._ASYNC_LAUNCH.search("Async agent launched successfully. (internal metadata)")


class TestGitRunsNoRepoCode:
    def test_a_configured_fsmonitor_hook_never_runs(self, tmp_path):
        """Review finding: git ls-files ran a repo's core.fsmonitor hook."""
        import subprocess
        repo = tmp_path / "r"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "a.py").write_text("x = 1\n")
        subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
                       check=True)
        marker = tmp_path / "hook-ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        hook.chmod(0o755)
        subprocess.run(["git", "-C", str(repo), "config", "core.fsmonitor", str(hook)], check=True)
        assert "a.py" in replay.repo_file_set(str(repo), "2000-01-01")
        assert not marker.exists()


class TestLinkedWorktreeOutsideTheMainCheckout:
    """Review finding: paths in a linked worktree that is not under the main
    checkout came back as None and dropped out of the counts."""

    def test_its_paths_are_taken_relative_to_its_own_checkout(self):
        assert replay.repo_rel("/projects/app-feature/src/a.py", "/projects/app") is None
        assert replay.checkout_rel("/projects/app-feature/src/a.py", "/projects/app-feature") == "src/a.py"
        assert replay.checkout_rel("/projects/app-other/src/a.py", "/projects/app-feature") is None
        assert replay.checkout_rel("/projects/app-feature/.git/HEAD", "/projects/app-feature") is None

    def test_a_prompt_naming_one_of_its_files_is_found(self):
        assert replay.prompt_paths("see /projects/app-feature/src/a.py", {"src/a.py"}, repo_root="/projects/app",
                                   checkout_root="/projects/app-feature") == ["src/a.py"]

    def test_a_pasted_error_line_is_found_and_prose_is_not(self):
        assert replay.prompt_error_lines("there's an error when I click save") == []
        assert replay.prompt_error_lines("it says\nTypeError: x is undefined\n") == ["TypeError: x is undefined"]
        assert replay.prompt_error_lines("fatal: not a git repository") == ["fatal: not a git repository"]

    def test_a_quoted_error_line_is_found_without_its_quote_marker(self):
        """Pasted terminal output often arrives quoted; the marker would also
        keep its signature from matching the same error from a tool call."""
        assert replay.prompt_error_lines("> fatal: not a git repository") == ["fatal: not a git repository"]


class TestReadRepoFiles:
    def test_every_version_a_touching_branch_holds_is_read(self, tmp_path):
        """An agent on a feature branch saw that branch's file. Reading one
        version only made the import relation depend on which session came
        first."""
        import subprocess

        def git(*args):
            subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (tmp_path / "a.py").write_text("import os\n")
        git("add", "a.py")
        git("commit", "-q", "-m", "main")
        git("checkout", "-q", "-b", "feature")
        (tmp_path / "a.py").write_text("from pkg import b\n")
        git("commit", "-q", "-am", "feature")
        git("checkout", "-q", "main")

        texts = replay.read_repo_files(str(tmp_path), {"a.py": {"feature", "gone-branch"}})

        assert sorted(texts["a.py"]) == ["from pkg import b\n", "import os\n"]


class TestImports:
    FILES = {"src/a.ts", "src/b.ts", "src/c/index.tsx", "pkg/__init__.py", "pkg/x.py", "pkg/m.py", "tools/lib.py",
             "tools/run.py"}

    def test_relative_ts_imports_resolve_with_extensions_and_index(self):
        text = "import { a } from './a';\nimport C from './c';\nexport * from './b.js';\n"
        assert replay.js_imports("src/main.ts", text, self.FILES, {}) == {"src/a.ts", "src/b.ts", "src/c/index.tsx"}

    def test_relative_python_imports_resolve(self):
        idx = replay.build_py_index(self.FILES)
        assert replay.py_imports("pkg/m.py", "from .x import thing\n", idx, self.FILES) == {"pkg/x.py"}

    def test_absolute_python_imports_resolve(self):
        idx = replay.build_py_index(self.FILES)
        assert replay.py_imports("tools/run.py", "import pkg.x\nfrom lib import y\n", idx, self.FILES) == {
            "pkg/x.py", "tools/lib.py"
        }
