"""Golden coding-session extraction accuracy tests.

Runs the coding-memory BAML extraction (via Bedrock) against hand-labeled
agent-session transcripts and checks that every expected decision, gotcha,
dead end, and preference is recovered with the right file anchors. Tracks
accuracy as a regression signal when the extraction prompt or model changes.

The golden dataset is at tests/fixtures/golden_conversations.json. Each
conversation carries a session context (branch, task, files); expected items
match extracted output via case-insensitive regex (``*_contains`` fields),
and ``anchors_include`` lists files that must appear in the matched item's
``anchor_files``.
"""

import json
import re
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_PATH = FIXTURES_DIR / "golden_conversations.json"

# Minimum share of extracted anchored-type items that survive the
# code-enforced anchor filter. Extra unanchored chatter lowers this;
# a clean extraction keeps it at 1.0.
MIN_ANCHOR_RATE = 0.5


def _load_golden_dataset() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    return data["conversations"]


def _field_matches(item: dict, expected: dict, field_map: dict[str, str]) -> bool:
    """True when every ``*_contains`` regex in ``expected`` hits its field."""
    for pattern_key, item_key in field_map.items():
        pattern = expected.get(pattern_key)
        if pattern is None:
            continue
        value = item.get(item_key) or ""
        if not re.search(pattern, value, re.IGNORECASE):
            return False
    return True


def _anchors_ok(item: dict, expected: dict) -> bool:
    required = expected.get("anchors_include", [])
    return all(path in item.get("anchor_files", []) for path in required)


# (fixture key, result key, {expected pattern field: extracted item field})
MATCH_SPECS = [
    (
        "expected_decisions",
        "decisions",
        {"text_contains": "text", "reason_contains": "reason"},
    ),
    ("expected_gotchas", "gotchas", {"text_contains": "text"}),
    (
        "expected_dead_ends",
        "dead_ends",
        {"attempt_contains": "attempt", "why_failed_contains": "why_failed"},
    ),
    (
        "expected_preferences",
        "preferences",
        {"category_contains": "category", "preference_contains": "preference"},
    ),
]


def _evaluate_conversation(conv: dict, result: dict) -> list[str]:
    """Return a list of failure descriptions for one conversation."""
    failures: list[str] = []
    for fixture_key, result_key, field_map in MATCH_SPECS:
        extracted = result[result_key]
        for expected in conv.get(fixture_key, []):
            content_hits = [
                item for item in extracted if _field_matches(item, expected, field_map)
            ]
            if not content_hits:
                failures.append(
                    f"{conv['id']}: no extracted {result_key} item matched "
                    f"{expected!r}; extracted: {extracted!r}"
                )
                continue
            if result_key != "preferences" and not any(
                _anchors_ok(item, expected) for item in content_hits
            ):
                failures.append(
                    f"{conv['id']}: matched {result_key} item(s) missing required "
                    f"anchors {expected.get('anchors_include')!r}; "
                    f"matched: {content_hits!r}"
                )
    return failures


class TestCodingExtractionGoldenDataset:
    """Run extraction against all golden conversations and check recall."""

    async def test_extraction_accuracy(self, bedrock_credentials):
        """Every hand-labeled coding memory item is recovered, well-anchored."""
        import os

        os.environ.setdefault("AWS_REGION", "us-east-1")
        os.environ.setdefault("AWS_PROFILE", "graphable-aws")

        from agent_memory_mcp.extraction.coding import extract_coding_memory

        conversations = _load_golden_dataset()
        all_failures: list[str] = []

        for conv in conversations:
            context = conv["context"]
            result = await extract_coding_memory(
                conv["transcript"],
                branch=context["branch"],
                task=context["task"],
                files=context["files"],
            )

            print(f"\n{'=' * 70}")
            print(f"GOLDEN CODING EXTRACTION: {conv['id']} ({conv['category']})")
            for key in ("decisions", "gotchas", "dead_ends", "preferences"):
                print(f"  {key}: {json.dumps(result[key], indent=2)}")
            print(
                f"  anchor_rate={result['anchor_rate']} "
                f"dropped_unanchored={result['dropped_unanchored']}"
            )
            print(f"{'=' * 70}\n")

            all_failures.extend(_evaluate_conversation(conv, result))

            # The anchor filter is code-enforced; a healthy extraction keeps
            # most anchored-type items instead of inventing unanchored ones.
            if result["anchor_rate"] is not None:
                assert result["anchor_rate"] >= MIN_ANCHOR_RATE, (
                    f"{conv['id']}: anchor_rate {result['anchor_rate']:.0%} "
                    f"below {MIN_ANCHOR_RATE:.0%} "
                    f"(dropped_unanchored={result['dropped_unanchored']})"
                )

        assert not all_failures, "\n".join(all_failures)
