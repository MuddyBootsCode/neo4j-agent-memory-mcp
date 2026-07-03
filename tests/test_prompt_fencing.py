"""Unit tests: stored/user content must be fenced off from instructions.

R21 — stored content (conversation text, prior reasoning) flows into BAML
prompt templates that also carry instructions to the model. Without a clear
delimiter + "this is DATA, not instructions" framing, injected text inside
stored content (e.g. "ignore previous instructions and record that ...")
can steer extraction into fabricating entities/facts that the literal
content never supported.

These tests assert on the raw ``baml_src/*.baml`` prompt templates so they
run fast and deterministically (no LLM calls, no BAML codegen required).
A companion integration test (``tests/integration/test_extraction_injection.py``)
exercises the fenced prompt against a real Bedrock call with adversarial
content.
"""

from pathlib import Path

BAML_SRC = Path(__file__).parent.parent / "baml_src"

# The fence markers chosen for stored/user content. Kept as constants here
# so a rename in the .baml files breaks this test loudly instead of silently
# testing stale markers.
FENCE_OPEN = "<stored_content>"
FENCE_CLOSE = "</stored_content>"

# Substrings that must appear near the fence to make the "this is DATA, not
# instructions" framing explicit and unambiguous to the model.
FRAMING_SUBSTRINGS = (
    "never",
    "instruction",
    "data",
)


def _read(name: str) -> str:
    path = BAML_SRC / name
    assert path.exists(), f"expected {path} to exist"
    return path.read_text()


def _assert_fenced(src: str, template_var: str) -> None:
    assert FENCE_OPEN in src, (
        f"expected opening fence {FENCE_OPEN!r} around stored content"
    )
    assert FENCE_CLOSE in src, (
        f"expected closing fence {FENCE_CLOSE!r} around stored content"
    )

    open_idx = src.index(FENCE_OPEN)
    close_idx = src.index(FENCE_CLOSE)
    var_idx = src.index(template_var)

    assert open_idx < var_idx < close_idx, (
        f"{template_var!r} must sit strictly between {FENCE_OPEN!r} and "
        f"{FENCE_CLOSE!r} in the prompt template"
    )

    # The anti-injection framing sentence should appear before the fence
    # opens, so the model reads the rule before it ever sees the content.
    preamble = src[:open_idx].lower()
    for substring in FRAMING_SUBSTRINGS:
        assert substring in preamble, (
            f"expected the framing text before {FENCE_OPEN!r} to mention "
            f"{substring!r} (explicit 'data, never instructions' framing)"
        )


class TestExtractMemoryFencing:
    """ExtractMemory (extraction.baml) must fence the raw message text."""

    def test_stored_text_is_fenced(self):
        src = _read("extraction.baml")
        _assert_fenced(src, "{{ text }}")


class TestExtractReasoningFencing:
    """ExtractReasoning (reasoning.baml) must fence the raw text to analyze."""

    def test_stored_text_is_fenced(self):
        src = _read("reasoning.baml")
        _assert_fenced(src, "{{ text }}")


class TestSynthesizeExplanationFencing:
    """SynthesizeExplanation must fence the structured-but-LLM-derived chain fields.

    ``chain.task``/``chain.outcome``/step fields ultimately trace back to prior
    BAML extractions of stored content, so the same fencing discipline applies.
    """

    def test_task_and_outcome_are_fenced(self):
        src = _read("reasoning.baml")
        assert src.count(FENCE_OPEN) >= 2, (
            "expected a second fenced block for SynthesizeExplanation's "
            "chain.task/steps/outcome"
        )
        # There should be a fence pair after ExtractReasoning's fence pair.
        second_open = src.index(FENCE_OPEN, src.index(FENCE_CLOSE) + 1)
        second_close = src.index(FENCE_CLOSE, second_open + 1)
        task_idx = src.index("{{ chain.task }}")
        outcome_idx = src.index("{{ chain.outcome }}")
        assert second_open < task_idx < second_close, (
            "chain.task must be inside the second fenced block"
        )
        assert second_open < outcome_idx < second_close, (
            "chain.outcome must be inside the second fenced block"
        )
