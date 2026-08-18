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
        assert text == "assistant: let me check found it"
        assert "tool_use" not in text
        assert "Bash" not in text

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
