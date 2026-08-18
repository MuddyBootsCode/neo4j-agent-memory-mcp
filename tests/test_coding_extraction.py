"""Tests for the coding-memory BAML surface (MUD-395)."""


def test_extract_coding_memory_in_generated_client():
    from agent_memory_mcp.baml_client.async_client import b

    assert hasattr(b, "ExtractCodingMemory")


def test_coding_baml_registered_in_fencing_registry():
    """coding.baml must be registered in the real fencing test infrastructure
    (tests/test_prompt_fencing.py), so it cannot silently drop out of the
    fence-position, escape-filter, and offline adversarial checks."""
    from test_prompt_fencing import FENCE_TAGS, UNTRUSTED_INTERPOLATIONS

    assert FENCE_TAGS.get("coding.baml") == (
        "session_context",
        "session_transcript",
    )
    fences = UNTRUSTED_INTERPOLATIONS["coding.baml"]["ExtractCodingMemory"]
    assert "transcript" in fences["session_transcript"]
    assert set(fences["session_context"]) == {
        "context.branch",
        "context.task",
        "file",
    }
