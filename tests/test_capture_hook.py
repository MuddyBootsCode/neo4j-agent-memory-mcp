"""Tests for the SessionEnd capture hook: transcript parsing and fail-open run."""

import io
import json

from agent_memory_mcp.hook.capture_hook import extract_transcript_text, run


def _jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            if isinstance(record, str):
                fh.write(record + "\n")
            else:
                fh.write(json.dumps(record) + "\n")
    return str(path)


def _user(content):
    return {"type": "user", "message": {"content": content}}


def _assistant(content):
    return {"type": "assistant", "message": {"content": content}}


class TestExtractTranscriptText:
    def test_string_content_renders_role_lines(self, tmp_path):
        path = _jsonl(
            tmp_path / "t.jsonl",
            [_user("fix the bug"), _assistant("done, it was a typo")],
        )
        text = extract_transcript_text(path)
        assert text == "user: fix the bug\nassistant: done, it was a typo"

    def test_block_list_content_keeps_text_blocks_only(self, tmp_path):
        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant(
                    [
                        {"type": "text", "text": "let me check"},
                        {"type": "tool_use", "id": "x", "name": "Bash", "input": {}},
                        {"type": "text", "text": "found it"},
                    ]
                ),
                _user([{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]),
            ],
        )
        text = extract_transcript_text(path)
        # Bash has no command here, so no marker; the ok result is a stub line.
        assert text == "assistant: let me check found it\ntool: [Bash ok] ok"
        assert "tool_use" not in text

    def test_tool_calls_and_errors_are_rendered(self, tmp_path):
        trace = "Traceback (most recent call last):\n  File x.py\nValueError: boom"
        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant([
                    {"type": "text", "text": "running"},
                    {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "pytest  -q\n tests/"}},
                    {"type": "tool_use", "id": "e1", "name": "Edit", "input": {"file_path": "/repo/src/x.py"}},
                    {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/repo/README.md"}},
                ]),
                _user([{"type": "tool_result", "tool_use_id": "b1", "content": trace}]),
                _user([{"type": "tool_result", "tool_use_id": "e1", "content": "The file has been updated."}]),
                _user([{"type": "tool_result", "tool_use_id": "r1", "content": "x" * 1000}]),
            ],
        )
        lines = extract_transcript_text(path).split("\n")
        assert lines[0] == "assistant: running [bash: pytest -q tests/] [edit /repo/src/x.py]"
        assert lines[1].startswith("tool: [error from Bash] Traceback")
        assert "ValueError: boom" in lines[1]
        assert lines[2] == "tool: [Edit ok] The file has been updated."
        # Read calls are not rendered, but their (successful) output is stubbed.
        assert lines[3] == "tool: [Read ok] " + "x" * 200

    def test_is_error_flag_renders_as_error_even_without_keywords(self, tmp_path):
        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant([{"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "ls"}}]),
                _user([{"type": "tool_result", "tool_use_id": "b1", "content": "nope", "is_error": True}]),
            ],
        )
        assert extract_transcript_text(path) == "assistant: [bash: ls]\ntool: [error from Bash] nope"

    def test_long_error_output_is_truncated_in_the_middle(self, tmp_path):
        big = "Error: " + "a" * 5000 + " END"
        path = _jsonl(
            tmp_path / "t.jsonl",
            [_user([{"type": "tool_result", "tool_use_id": "z", "content": big}])],
        )
        text = extract_transcript_text(path)
        assert len(text) < 2200
        assert text.startswith("tool: [error from tool] Error: ")
        assert text.endswith(" END")
        assert " … " in text

    def test_malformed_and_irrelevant_lines_skipped(self, tmp_path):
        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                "not json{{{",
                {"type": "summary", "summary": "irrelevant"},
                {"type": "user"},
                {"type": "user", "message": "not a dict... wait, a string"},
                _user(""),
                _user("real content"),
                "42",
            ],
        )
        text = extract_transcript_text(path)
        assert text == "user: real content"

    def test_meta_entries_excluded_sidechain_included(self, tmp_path):
        meta = {
            "type": "user",
            "isMeta": True,
            "message": {"content": "<local-command-caveat> " + ("skill text " * 500)},
        }
        sidechain = {
            "type": "assistant",
            "isSidechain": True,
            "message": {"content": "subagent decided to use MERGE"},
        }
        path = _jsonl(tmp_path / "t.jsonl", [meta, _user("real ask"), sidechain])
        text = extract_transcript_text(path)
        assert text == "user: real ask\nassistant: subagent decided to use MERGE"
        assert "skill text" not in text

    def test_tail_cap_cuts_at_line_boundary(self, tmp_path):
        records = [_user(f"message number {i} padded out a bit") for i in range(50)]
        path = _jsonl(tmp_path / "t.jsonl", records)
        full = extract_transcript_text(path)
        capped = extract_transcript_text(path, max_chars=200)
        assert len(capped) <= 200
        # Tail: the last message survives, the first is gone.
        assert capped.endswith("message number 49 padded out a bit")
        assert "message number 0 " not in capped
        # Line boundary: every kept line is a complete rendered line.
        full_lines = set(full.split("\n"))
        assert all(line in full_lines for line in capped.split("\n"))

    def test_single_oversized_line_still_returns_tail(self, tmp_path):
        path = _jsonl(tmp_path / "t.jsonl", [_user("x" * 500)])
        capped = extract_transcript_text(path, max_chars=100)
        assert len(capped) == 100
        assert capped == ("user: " + "x" * 500)[-100:]

    def test_missing_file_returns_empty(self, tmp_path):
        assert extract_transcript_text(str(tmp_path / "absent.jsonl")) == ""


class TestTouchedFilesAndErrorSteps:
    def test_normalize_repo_path(self):
        from agent_memory_mcp.hook.capture_hook import normalize_repo_path

        assert normalize_repo_path("/repo/src/x.py", "/repo") == "src/x.py"
        assert normalize_repo_path("./src/x.py", "/repo") == "src/x.py"
        assert normalize_repo_path("/repo/.claude/worktrees/GRA-1/src/y.py", "/repo") == "src/y.py"
        assert normalize_repo_path("/elsewhere/z.py", "/repo") is None
        assert normalize_repo_path("../z.py", "/repo") is None
        assert normalize_repo_path("", "/repo") is None

    def test_transcript_touched_files_in_first_touch_order(self, tmp_path):
        from agent_memory_mcp.hook.capture_hook import transcript_touched_files

        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant([
                    {"type": "tool_use", "id": "1", "name": "Edit", "input": {"file_path": "/repo/b.py"}},
                    {"type": "tool_use", "id": "2", "name": "Read", "input": {"file_path": "/repo/c.py"}},
                ]),
                _assistant([
                    {"type": "tool_use", "id": "3", "name": "Write", "input": {"file_path": "/repo/a.py"}},
                    {"type": "tool_use", "id": "4", "name": "Edit", "input": {"file_path": "/repo/b.py"}},
                    {"type": "tool_use", "id": "5", "name": "Edit", "input": {"file_path": "/other/x.py"}},
                ]),
            ],
        )
        assert transcript_touched_files(path, "/repo") == ["b.py", "a.py"]

    def test_error_steps_pair_results_with_their_calls(self, tmp_path):
        from agent_memory_mcp.hook.capture_hook import error_steps

        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant([
                    {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "make test"}},
                    {"type": "tool_use", "id": "e1", "name": "Edit", "input": {"file_path": "/repo/src/x.py"}},
                ]),
                _user([{"type": "tool_result", "tool_use_id": "b1", "is_error": True,
                        "content": "Exit code 1 FAILED tests/test_x.py::t - assert 1 == 2"}]),
                _user([{"type": "tool_result", "tool_use_id": "e1", "content": "ok"}]),
                _user([{"type": "tool_result", "tool_use_id": "missing", "content": "boom", "is_error": True}]),
            ],
        )
        steps = error_steps(path, "/repo")
        assert steps == [
            {"tool": "Bash", "input": "make test", "file": None,
             "error": "Exit code 1 FAILED tests/test_x.py::t - assert 1 == 2"},
            {"tool": "tool", "input": "", "file": None, "error": "boom"},
        ]

    def test_error_steps_require_the_is_error_flag(self, tmp_path):
        """A successful `cat` of a file that mentions "Traceback" is not a
        dead end. Claude Code flags every failed tool result with is_error
        (130/130 exit-code failures in a sample of 60 real transcripts), so
        the keyword regex only decides rendering, never candidacy."""
        from agent_memory_mcp.hook.capture_hook import error_steps

        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant([{"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "cat notes.md"}}]),
                _user([{"type": "tool_result", "tool_use_id": "c1",
                        "content": "## Known error\nTraceback (most recent call last): ... was fixed in #12"}]),
            ],
        )
        assert error_steps(path, "/repo") == []

    def test_error_steps_skip_permission_and_hook_blocks(self, tmp_path):
        """A denied action is not an attempt that failed; nothing was tried."""
        from agent_memory_mcp.hook.capture_hook import error_steps

        path = _jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant([
                    {"type": "tool_use", "id": "d1", "name": "Bash", "input": {"command": "aws ec2 describe-instances"}},
                    {"type": "tool_use", "id": "d2", "name": "Bash", "input": {"command": "sleep 45"}},
                    {"type": "tool_use", "id": "d3", "name": "Bash", "input": {"command": "rm -rf build"}},
                    {"type": "tool_use", "id": "ok", "name": "Bash", "input": {"command": "make"}},
                ]),
                _user([{"type": "tool_result", "tool_use_id": "d1", "is_error": True,
                        "content": "Permission for this action was denied by the Claude Code auto mode classifier. Reason: Blocked by classifier."}]),
                _user([{"type": "tool_result", "tool_use_id": "d2", "is_error": True,
                        "content": "<tool_use_error>Blocked: sleep 45 followed by: tail -3 /tmp/x.txt</tool_use_error>"}]),
                _user([{"type": "tool_result", "tool_use_id": "d3", "is_error": True,
                        "content": "The user doesn't want to proceed with this tool use. The tool use was rejected."}]),
                _user([{"type": "tool_result", "tool_use_id": "ok", "is_error": True,
                        "content": "Exit code 2 make: *** No rule to make target 'all'.  Stop."}]),
            ],
        )
        steps = error_steps(path, "/repo")
        assert [s["input"] for s in steps] == ["make"]

    def test_error_steps_missing_file_is_empty(self, tmp_path):
        from agent_memory_mcp.hook.capture_hook import error_steps

        assert error_steps(str(tmp_path / "absent.jsonl"), "/repo") == []


def _transcript_file(tmp_path):
    return _jsonl(
        tmp_path / "transcript.jsonl",
        [_user("do the thing"), _assistant("did the thing")],
    )


def _ctx(**overrides):
    ctx = {
        "agent_id": "agent-1",
        "session_id": "sess-1",
        "repo": "my-repo",
        "branch": "feature/x",
        "files": ["a.py"],
        "task_key": "MUD-395",
    }
    ctx.update(overrides)
    return ctx


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, transcript, ctx):
        self.calls.append((transcript, ctx))


class TestRun:
    def test_happy_path_calls_capture_with_context(self, tmp_path, capsys):
        capture = Recorder()
        payload = {
            "session_id": "sess-1",
            "transcript_path": _transcript_file(tmp_path),
            "cwd": str(tmp_path),
            "hook_event_name": "SessionEnd",
        }
        code = run(
            stdin=io.StringIO(json.dumps(payload)),
            capture=capture,
            gather=lambda p: _ctx(),
        )
        assert code == 0
        assert capsys.readouterr().out == ""
        assert len(capture.calls) == 1
        transcript, ctx = capture.calls[0]
        assert transcript == "user: do the thing\nassistant: did the thing"
        assert ctx == _ctx()

    def test_garbage_stdin_fails_open(self):
        capture = Recorder()
        code = run(stdin=io.StringIO("not json{{{"), capture=capture)
        assert code == 0
        assert capture.calls == []

    def test_missing_transcript_path_fails_open(self, tmp_path):
        capture = Recorder()
        payload = {"session_id": "s", "cwd": str(tmp_path)}
        code = run(
            stdin=io.StringIO(json.dumps(payload)),
            capture=capture,
            gather=lambda p: _ctx(),
        )
        assert code == 0
        assert capture.calls == []

    def test_nonexistent_transcript_fails_open(self, tmp_path):
        capture = Recorder()
        payload = {
            "session_id": "s",
            "transcript_path": str(tmp_path / "absent.jsonl"),
            "cwd": str(tmp_path),
        }
        code = run(
            stdin=io.StringIO(json.dumps(payload)),
            capture=capture,
            gather=lambda p: _ctx(),
        )
        assert code == 0
        assert capture.calls == []

    def test_non_repo_cwd_fails_open(self, tmp_path):
        # Real gather: tmp_path is not a git repo, so context comes back
        # None and the capture is skipped.
        capture = Recorder()
        payload = {
            "session_id": "s",
            "transcript_path": _transcript_file(tmp_path),
            "cwd": str(tmp_path),
        }
        code = run(stdin=io.StringIO(json.dumps(payload)), capture=capture)
        assert code == 0
        assert capture.calls == []

    def test_gather_returning_none_skips_capture(self, tmp_path):
        capture = Recorder()
        payload = {
            "session_id": "s",
            "transcript_path": _transcript_file(tmp_path),
            "cwd": str(tmp_path),
        }
        code = run(
            stdin=io.StringIO(json.dumps(payload)),
            capture=capture,
            gather=lambda p: None,
        )
        assert code == 0
        assert capture.calls == []

    def test_capture_asks_for_background_by_default(self, monkeypatch):
        """The server queues the capture and answers at once, so the hook
        (60 s budget) is never mid-call when a 15-minute capture ends."""
        from agent_memory_mcp.hook.capture_hook import capture_call_args

        ctx = {"agent_id": "a", "session_id": "s", "repo": "r", "branch": "b",
               "task_key": None, "files": [], "error_steps": []}
        args = capture_call_args("user: hi", ctx)
        assert args["background"] is True and args["transcript"] == "user: hi"
        monkeypatch.setenv("NAM_CAPTURE_BACKGROUND", "0")
        assert capture_call_args("user: hi", ctx)["background"] is False

    def test_capture_error_fails_open(self, tmp_path, capsys):
        def boom(transcript, ctx):
            raise ConnectionError("server down")

        payload = {
            "session_id": "s",
            "transcript_path": _transcript_file(tmp_path),
            "cwd": str(tmp_path),
        }
        code = run(
            stdin=io.StringIO(json.dumps(payload)),
            capture=boom,
            gather=lambda p: _ctx(),
        )
        assert code == 0
        assert capsys.readouterr().out == ""

    def test_kill_switch_short_circuits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NAM_CAPTURE_DISABLED", "1")
        capture = Recorder()
        gathered = []

        def gather(p):
            gathered.append(p)
            return _ctx()

        payload = {
            "session_id": "s",
            "transcript_path": _transcript_file(tmp_path),
            "cwd": str(tmp_path),
        }
        code = run(
            stdin=io.StringIO(json.dumps(payload)), capture=capture, gather=gather
        )
        assert code == 0
        assert capture.calls == []
        assert gathered == []

    def test_empty_transcript_skips_capture(self, tmp_path):
        capture = Recorder()
        path = _jsonl(tmp_path / "t.jsonl", ["not json", {"type": "summary"}])
        payload = {
            "session_id": "s",
            "transcript_path": path,
            "cwd": str(tmp_path),
        }
        code = run(
            stdin=io.StringIO(json.dumps(payload)),
            capture=capture,
            gather=lambda p: _ctx(),
        )
        assert code == 0
        assert capture.calls == []


class TestSubagentCapture:
    """SubagentStop: a subagent's own transcript reaches the graph.

    SubagentStart carries no additionalContext, so a subagent cannot be fed
    from memory; capturing what it learned is the half that is possible.
    """

    def _payload(self, tmp_path, path, **overrides):
        payload = {
            "session_id": "parent-sess",
            "agent_id": "agent_abc123",
            "agent_type": "Explore",
            "transcript_path": path,
            "cwd": str(tmp_path),
            "hook_event_name": "SubagentStop",
        }
        payload.update(overrides)
        return payload

    def _big_transcript(self, tmp_path, size=4000):
        return _jsonl(
            tmp_path / "sub.jsonl",
            [_user("find the thing"), _assistant("x" * size)],
        )

    def test_detaches_the_capture_instead_of_blocking(self, tmp_path, capsys):
        from agent_memory_mcp.hook.capture_hook import run_subagent

        spawned = []
        payload = self._payload(tmp_path, self._big_transcript(tmp_path))
        code = run_subagent(stdin=io.StringIO(json.dumps(payload)), spawn=spawned.append)

        assert code == 0
        assert capsys.readouterr().out == ""
        assert spawned == [payload]

    def test_short_transcripts_are_not_worth_an_extraction(self, tmp_path):
        from agent_memory_mcp.hook.capture_hook import run_subagent

        spawned = []
        payload = self._payload(tmp_path, _transcript_file(tmp_path))
        assert run_subagent(stdin=io.StringIO(json.dumps(payload)), spawn=spawned.append) == 0
        assert spawned == []

    def test_kill_switches_and_bad_payloads_never_spawn(self, tmp_path, monkeypatch):
        from agent_memory_mcp.hook.capture_hook import run_subagent

        spawned = []
        good = json.dumps(self._payload(tmp_path, self._big_transcript(tmp_path)))

        monkeypatch.setenv("NAM_SUBAGENT_CAPTURE", "0")
        assert run_subagent(stdin=io.StringIO(good), spawn=spawned.append) == 0
        monkeypatch.delenv("NAM_SUBAGENT_CAPTURE")
        monkeypatch.setenv("NAM_CAPTURE_DISABLED", "1")
        assert run_subagent(stdin=io.StringIO(good), spawn=spawned.append) == 0
        monkeypatch.delenv("NAM_CAPTURE_DISABLED")

        assert run_subagent(stdin=io.StringIO("not json{{{"), spawn=spawned.append) == 0
        missing = json.dumps(self._payload(tmp_path, str(tmp_path / "gone.jsonl")))
        assert run_subagent(stdin=io.StringIO(missing), spawn=spawned.append) == 0
        assert spawned == []

    def test_context_keeps_the_parents_repo_and_the_subagents_identity(self, tmp_path, monkeypatch):
        from agent_memory_mcp.hook import capture_hook

        monkeypatch.setattr(capture_hook, "gather_capture_context", lambda p: _ctx())
        ctx = capture_hook.subagent_capture_context(
            self._payload(tmp_path, self._big_transcript(tmp_path))
        )

        assert ctx["agent_id"] == "subagent:Explore"
        assert ctx["session_id"] == "parent-sess:agent_abc123"
        # The git context is the parent's, unchanged.
        assert ctx["repo"] == "my-repo" and ctx["branch"] == "feature/x"
        assert ctx["files"] == ["a.py"] and ctx["task_key"] == "MUD-395"

    def test_context_falls_back_when_the_payload_names_no_agent(self, tmp_path, monkeypatch):
        from agent_memory_mcp.hook import capture_hook

        monkeypatch.setattr(capture_hook, "gather_capture_context", lambda p: _ctx())
        ctx = capture_hook.subagent_capture_context(
            {"session_id": "parent-sess", "cwd": str(tmp_path)}
        )
        assert ctx["agent_id"] == "subagent:subagent"
        assert ctx["session_id"] == "parent-sess"

    def test_context_is_none_when_the_parent_context_is(self, tmp_path, monkeypatch):
        from agent_memory_mcp.hook import capture_hook

        monkeypatch.setattr(capture_hook, "gather_capture_context", lambda p: None)
        assert capture_hook.subagent_capture_context({"cwd": str(tmp_path)}) is None

    def test_worker_captures_under_the_subagent_identity(self, tmp_path, monkeypatch):
        from agent_memory_mcp.hook import capture_hook

        capture = Recorder()
        monkeypatch.setattr(capture_hook, "gather_capture_context", lambda p: _ctx())
        payload = self._payload(tmp_path, self._big_transcript(tmp_path, size=40))
        code = capture_hook.run_subagent_worker(
            stdin=io.StringIO(json.dumps(payload)), capture=capture
        )

        assert code == 0
        assert len(capture.calls) == 1
        transcript, ctx = capture.calls[0]
        assert "find the thing" in transcript
        assert ctx["agent_id"] == "subagent:Explore"
        assert ctx["session_id"] == "parent-sess:agent_abc123"
