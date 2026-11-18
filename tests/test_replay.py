"""Phase 1 replay (MUD-850): the counting rules behind each moment's hits.

A wrong rule here does not error. It changes a count that decides whether
Phases 2 and 3 get built, so each rule is pinned on a small case.
"""

import collections
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


def _call(ts, error=None):
    return {"id": f"c{ts}", "ts": ts, "rts": ts + 1, "line": 0, "tool": "Bash", "sub": False, "path": None,
            "cwd": None, "ok": error is None, "blocked": False, "error": error}


def _merged(sid, cwds, times, calls):
    return {"times": sorted(times), "cwds": cwds, "branches": collections.Counter(), "calls": calls,
            "prompts": [], "briefs": [], "files": {f"/projects/p/{sid}.jsonl"}, "subagent_files": set()}


class TestRepoSwitch:
    def test_a_session_that_moves_to_another_repo_is_left_out_of_runs(self, tmp_path):
        """Review finding (run 13): a session kept the repo of its first cwd,
        so its work in a second repo overlapped another worktree of the first
        and made a run and a failure hit that never happened."""
        a, b = tmp_path / "a", tmp_path / "b"
        for root in (a, b):
            (root / ".git").mkdir(parents=True)
        for name in ("w1", "w2"):
            (a / ".worktrees" / name).mkdir(parents=True)
        err = "Exit code 2\nmake: *** [build] Error 2"
        t0 = 1_800_000_000.0
        later = [t0 + 60 * MIN + k * MIN for k in range(30)]
        merged = {
            "switcher": _merged("switcher", [(t0, str(a / ".worktrees" / "w1")), (t0 + 10 * MIN, str(b))],
                                [t0, t0 + MIN] + later, [_call(t0 + 70 * MIN, err)]),
            "other": _merged("other", [(t0 + 55 * MIN, str(a / ".worktrees" / "w2"))], later,
                             [_call(t0 + 75 * MIN, err)]),
        }
        sessions, _ = replay.build_sessions(merged, 15 * MIN)
        placed, runs = replay.place_in_runs(sessions, 15 * MIN)
        assert runs == [] and [s["sid"] for s in placed] == ["other"]
        assert replay.moment4(sessions, runs)["per_run"] == {}
        assert replay.left_out_counts(sessions) == {"cwd moved between worktrees": 0, "cwd moved to another repo": 1}


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

    def test_a_sessions_own_touches_in_two_worktrees_are_no_hit(self):
        """Touches are attributed to their file's worktree, so one session can
        touch two worktrees; that is not two agents."""
        events = [_touch(0, "a", "wt1", "src/x.py"), _touch(5 * MIN, "a", "wt2", "src/x.py")]
        assert replay.file_window_hits(events, 15 * MIN) == []

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


class TestFileAttribution:
    def test_a_symlink_and_its_target_are_one_file(self, tmp_path):
        """Review finding: a link such as alias.py -> src/app.py was its own
        file, so two sessions editing one physical file never matched."""
        main = tmp_path / "app"
        (main / ".git").mkdir(parents=True)
        (main / "src").mkdir()
        (main / "src" / "app.py").write_text("x = 1\n")
        os.symlink("src/app.py", main / "alias.py")
        os.symlink("src", main / "lib")
        for name in ("alias.py", "lib/app.py", "src/app.py"):
            call = {"tool": "Edit", "path": str(main / name)}
            replay.attribute_call(call, str(main), str(main))
            assert (call["path"], call["file"]) == (str(main / "src" / "app.py"), "src/app.py"), name
        outside = tmp_path / "elsewhere.py"
        outside.write_text("y = 2\n")
        os.symlink(str(outside), main / "out.py")
        assert replay.canonical_path(str(main / "out.py"), (str(main), None)) == str(main / "out.py")

    def test_a_file_tool_call_in_a_linked_worktree_outside_the_main_checkout_counts(self, tmp_path):
        """Review finding: a session in the main checkout reading a file in a
        sibling worktree such as /tmp/app-feature got no file at all."""
        main = tmp_path / "app"
        (main / ".git" / "worktrees" / "feat").mkdir(parents=True)
        ext = tmp_path / "app-feature"
        (ext / "src").mkdir(parents=True)
        (ext / "src" / "a.py").write_text("x = 1\n")
        (ext / ".git").write_text(f"gitdir: {main}/.git/worktrees/feat\n")
        path = str(ext / "src" / "a.py")
        call = {"tool": "Read", "path": path}
        replay.attribute_call(call, str(main), str(main))
        assert call["file"] == "src/a.py"
        assert replay.worktree_of(path, str(main), str(main)) == str(ext).lower()
        other = tmp_path / "other"
        (other / ".git").mkdir(parents=True)
        (other / "a.py").write_text("x = 1\n")
        call = {"tool": "Read", "path": str(other / "a.py")}
        replay.attribute_call(call, str(main), str(main))
        assert call["file"] is None


class TestSameDirectory:
    def test_siblings_relate_and_a_file_does_not_relate_to_itself(self):
        rel = replay.same_directory_relation(["src/a.py", "src/b.py", "lib/c.py"])
        assert rel("src/a.py") == {"src/b.py"}
        assert rel("lib/c.py") == set()


class TestRepeatFailureHits:
    def _f(self, ts, sid, wt, sig):
        return {"ts": ts, "sid": sid, "worktree": wt, "sig": sig}

    def test_a_signature_another_worktree_already_hit_is_a_hit(self):
        fs = [self._f(0, "a", "wt1", "s"), self._f(10 * MIN, "b", "wt2", "s")]
        hits = replay.repeat_failure_hits(fs)
        assert [(h["failure"]["sid"], h["earlier"]["sid"]) for h in hits] == [("b", "a")]
        assert set(hits[0]) == {"failure", "earlier"}

    def test_moment4_counts_the_same_signature_seen_again_and_nothing_about_recovery(self):
        """The exact part kept when recovery matching was removed: a failure is
        a hit when another worktree in the run had the same signature, and it is
        a repeat of another session's signature."""
        err = "Exit code 2\nmake: *** [build] Error 2"

        def session(sid, wt, ts):
            return {"sid": sid, "repo": "/r/.git", "repo_name": "r", "worktree": wt, "worktree_name": wt,
                    "calls": [{"tool": "Bash", "ok": False, "error": err, "ts": ts, "rts": ts + 1}]}

        sessions = {"a": session("a", "wt1", 0.0), "b": session("b", "wt2", 10 * MIN)}
        total = replay.moment4(sessions, [{"run": "run001", "repo": "r", "sessions": ["a", "b"]}])["summary"]["total"]
        assert (total["failures in runs"], total["hits|15"], total["hits|kind|command or tool failure"]) == (2, 1, 1)
        assert total["repeats of another session's signature, all sessions"] == 1
        assert not [k for k in total if "past it" in k]

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
