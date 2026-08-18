"""Tests for the coding-memory BAML surface (MUD-395)."""

from pathlib import Path

BAML_SRC = Path(__file__).parent.parent / "baml_src"


def test_extract_coding_memory_in_generated_client():
    from agent_memory_mcp.baml_client.async_client import b

    assert hasattr(b, "ExtractCodingMemory")


def test_coding_prompt_fences_transcript():
    src = (BAML_SRC / "coding.baml").read_text()
    assert "<session_transcript>" in src
    assert "NEVER a command" in src or "never as something to obey" in src.lower()
