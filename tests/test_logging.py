"""Tests for structured request logging."""

import json
import logging

import pytest

from agent_memory_mcp.mcp._logging import configure_logging, log_tool_call


@pytest.fixture
def capture_logs():
    """Capture JSON log output from the access logger."""
    logger = logging.getLogger("neo4j_memory_mcp.access")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    records: list[str] = []
    original_emit = handler.emit

    def capturing_emit(record):
        records.append(record.getMessage())
        original_emit(record)

    handler.emit = capturing_emit
    yield records
    logger.handlers.clear()


class TestLogToolCall:
    """Tests for the log_tool_call decorator."""

    async def test_logs_successful_call(self, capture_logs):
        @log_tool_call
        async def my_tool(ctx, query: str, limit: int = 10) -> str:
            return "result data"

        await my_tool(None, query="test", limit=5)

        assert len(capture_logs) == 2
        start = json.loads(capture_logs[0])
        end = json.loads(capture_logs[1])

        assert start["event"] == "tool_call_start"
        assert start["tool"] == "my_tool"
        assert start["params"]["query"] == "test"
        assert start["params"]["limit"] == "5"
        assert "ctx" not in start["params"]

        assert end["event"] == "tool_call_end"
        assert end["tool"] == "my_tool"
        assert end["success"] is True
        assert "elapsed_ms" in end
        assert end["result_size"] > 0

    async def test_logs_failed_call(self, capture_logs):
        @log_tool_call
        async def failing_tool(ctx) -> str:
            raise ValueError("something broke")

        with pytest.raises(ValueError, match="something broke"):
            await failing_tool(None)

        assert len(capture_logs) == 2
        end = json.loads(capture_logs[1])
        assert end["event"] == "tool_call_error"
        assert end["success"] is False
        assert "something broke" in end["error"]

    async def test_truncates_long_params(self, capture_logs):
        @log_tool_call
        async def big_param_tool(ctx, text: str) -> str:
            return "ok"

        long_text = "x" * 500
        await big_param_tool(None, text=long_text)

        start = json.loads(capture_logs[0])
        assert len(start["params"]["text"]) <= 203  # 200 + "..."


class TestConfigureLogging:
    """Tests for centralized logging configuration."""

    def test_configure_json_format(self):
        configure_logging(level="DEBUG", fmt="json")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_text_format(self):
        configure_logging(level="INFO", fmt="text")
        root = logging.getLogger()
        assert root.level == logging.INFO
