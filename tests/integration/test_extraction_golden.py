"""Golden dataset extraction accuracy tests.

Runs BAML entity extraction (via Bedrock) against 20 hand-labeled
conversations and computes precision/recall metrics for entities,
relationships, and preferences. Tracks accuracy as a regression signal
when extraction prompts or models change.

The golden dataset is at tests/fixtures/golden_conversations.json.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_PATH = FIXTURES_DIR / "golden_conversations.json"

# Minimum accuracy thresholds — fail the test if below these.
# These are baselines from the initial run (2026-04-13).
# Precision is low (46%) due to over-extraction of EVENT/OBJECT
# entities from noun phrases. Improving the BAML prompt should
# raise precision without hurting recall.
MIN_ENTITY_RECALL = 0.60
MIN_ENTITY_PRECISION = 0.40
MIN_RELATION_RECALL = 0.50


@dataclass
class ExtractionMetrics:
    """Aggregated extraction accuracy metrics."""

    entity_true_positives: int = 0
    entity_false_positives: int = 0
    entity_false_negatives: int = 0
    relation_true_positives: int = 0
    relation_false_positives: int = 0
    relation_false_negatives: int = 0
    preference_true_positives: int = 0
    preference_false_negatives: int = 0
    per_conversation: list = field(default_factory=list)

    @property
    def entity_precision(self) -> float:
        denom = self.entity_true_positives + self.entity_false_positives
        return self.entity_true_positives / denom if denom else 1.0

    @property
    def entity_recall(self) -> float:
        denom = self.entity_true_positives + self.entity_false_negatives
        return self.entity_true_positives / denom if denom else 1.0

    @property
    def relation_recall(self) -> float:
        denom = self.relation_true_positives + self.relation_false_negatives
        return self.relation_true_positives / denom if denom else 1.0

    def summary(self) -> str:
        return (
            f"Entity precision={self.entity_precision:.0%} "
            f"recall={self.entity_recall:.0%} "
            f"(TP={self.entity_true_positives} FP={self.entity_false_positives} "
            f"FN={self.entity_false_negatives}) | "
            f"Relation recall={self.relation_recall:.0%} "
            f"(TP={self.relation_true_positives} FN={self.relation_false_negatives}) | "
            f"Preference TP={self.preference_true_positives} "
            f"FN={self.preference_false_negatives}"
        )


def _load_golden_dataset() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    return data["conversations"]


def _entity_matches(extracted_name: str, expected_name: str) -> bool:
    """Fuzzy entity name matching for golden dataset evaluation.

    Handles cases like:
    - Exact match: "Michael" == "Michael"
    - Substring: "Raj" in "Raj Patel" or "Raj Patel" contains "Raj"
    - Case-insensitive: "datavault" matches "DataVault"
    """
    e_lower = extracted_name.lower().strip()
    x_lower = expected_name.lower().strip()
    return (
        e_lower == x_lower
        or e_lower in x_lower
        or x_lower in e_lower
    )


def _type_matches(extracted_type: str, expected_type: str) -> bool:
    return extracted_type.upper() == expected_type.upper()


def _relation_matches(
    extracted_rel: dict,
    expected_rel: dict,
    extracted_entities: list[dict],
) -> bool:
    """Check if an extracted relation matches an expected one.

    expected_rel has: source, target, relation_contains (regex pattern).
    """
    # Match source
    source_ok = any(
        _entity_matches(extracted_rel["source"], expected_rel["source"])
        for _ in [1]
    )
    # Match target
    target_ok = any(
        _entity_matches(extracted_rel["target"], expected_rel["target"])
        for _ in [1]
    )
    # Match relation type (pattern match)
    pattern = expected_rel["relation_contains"]
    rel_ok = bool(re.search(pattern, extracted_rel["relation_type"], re.IGNORECASE))

    return source_ok and target_ok and rel_ok


def _preference_matches(extracted_pref: dict, expected_pref: dict) -> bool:
    """Check if an extracted preference matches expected."""
    cat_pattern = expected_pref["category_contains"]
    pref_pattern = expected_pref["preference_contains"]
    cat_ok = bool(re.search(cat_pattern, extracted_pref["category"], re.IGNORECASE))
    pref_ok = bool(re.search(pref_pattern, extracted_pref["preference"], re.IGNORECASE))
    return cat_ok and pref_ok


def _evaluate_conversation(conv: dict, result) -> dict:
    """Evaluate extraction result against expected labels for one conversation."""
    extracted_entities = [
        {"name": e.name, "type": e.type, "confidence": e.confidence}
        for e in result.entities
    ]
    extracted_relations = [
        {
            "source": r.source,
            "target": r.target,
            "relation_type": r.relation_type,
            "confidence": r.confidence,
        }
        for r in result.relations
    ]
    extracted_preferences = [
        {
            "category": p.category,
            "preference": p.preference,
            "confidence": p.confidence,
        }
        for p in result.preferences
    ]

    # Entity matching
    expected_entities = conv.get("expected_entities", [])
    entity_tp = 0
    entity_fn = 0
    matched_extracted = set()

    for expected in expected_entities:
        found = False
        for i, extracted in enumerate(extracted_entities):
            if i in matched_extracted:
                continue
            if (_entity_matches(extracted["name"], expected["name"])
                    and _type_matches(extracted["type"], expected["type"])):
                entity_tp += 1
                matched_extracted.add(i)
                found = True
                break
        if not found:
            entity_fn += 1

    entity_fp = len(extracted_entities) - len(matched_extracted)

    # Relation matching
    expected_relations = conv.get("expected_relations", [])
    relation_tp = 0
    relation_fn = 0

    for expected in expected_relations:
        found = any(
            _relation_matches(er, expected, extracted_entities)
            for er in extracted_relations
        )
        if found:
            relation_tp += 1
        else:
            relation_fn += 1

    # Preference matching
    expected_preferences = conv.get("expected_preferences", [])
    pref_tp = 0
    pref_fn = 0

    for expected in expected_preferences:
        found = any(
            _preference_matches(ep, expected)
            for ep in extracted_preferences
        )
        if found:
            pref_tp += 1
        else:
            pref_fn += 1

    return {
        "id": conv["id"],
        "category": conv["category"],
        "entity_tp": entity_tp,
        "entity_fp": entity_fp,
        "entity_fn": entity_fn,
        "relation_tp": relation_tp,
        "relation_fn": relation_fn,
        "relation_fp": len(extracted_relations) - relation_tp,
        "preference_tp": pref_tp,
        "preference_fn": pref_fn,
        "extracted_entities": [e["name"] for e in extracted_entities],
        "expected_entities": [e["name"] for e in expected_entities],
        "extracted_relations": [
            f"{r['source']}-{r['relation_type']}->{r['target']}"
            for r in extracted_relations
        ],
    }


class TestExtractionGoldenDataset:
    """Run extraction against all golden conversations and compute metrics."""

    async def test_extraction_accuracy(self):
        """Extraction precision and recall meet minimum thresholds."""
        import os

        os.environ.setdefault("AWS_REGION", "us-east-1")
        os.environ.setdefault("AWS_PROFILE", "graphable-aws")

        from types import SimpleNamespace

        from neo4j_agent_memory.extraction.unified import extract_memory

        conversations = _load_golden_dataset()
        metrics = ExtractionMetrics()

        for conv in conversations:
            raw = await extract_memory(conv["text"])
            # Adapt the unified dict output to the attribute shape the
            # evaluator expects (.entities/.relations/.preferences of objects).
            result = SimpleNamespace(
                entities=[SimpleNamespace(**e) for e in raw["entities"]],
                relations=[SimpleNamespace(**r) for r in raw["relations"]],
                preferences=[SimpleNamespace(**p) for p in raw["preferences"]],
            )
            eval_result = _evaluate_conversation(conv, result)

            metrics.entity_true_positives += eval_result["entity_tp"]
            metrics.entity_false_positives += eval_result["entity_fp"]
            metrics.entity_false_negatives += eval_result["entity_fn"]
            metrics.relation_true_positives += eval_result["relation_tp"]
            metrics.relation_false_negatives += eval_result["relation_fn"]
            metrics.preference_true_positives += eval_result["preference_tp"]
            metrics.preference_false_negatives += eval_result["preference_fn"]
            metrics.per_conversation.append(eval_result)

        # Print detailed report
        print(f"\n{'='*70}")
        print("GOLDEN DATASET EXTRACTION REPORT")
        print(f"{'='*70}")
        print(f"Conversations: {len(conversations)}")
        print(f"Overall: {metrics.summary()}")
        print(f"{'='*70}")

        # Per-category breakdown
        categories = {}
        for pc in metrics.per_conversation:
            cat = pc["category"]
            if cat not in categories:
                categories[cat] = {"tp": 0, "fp": 0, "fn": 0, "count": 0}
            categories[cat]["tp"] += pc["entity_tp"]
            categories[cat]["fp"] += pc["entity_fp"]
            categories[cat]["fn"] += pc["entity_fn"]
            categories[cat]["count"] += 1

        for cat, counts in sorted(categories.items()):
            total_expected = counts["tp"] + counts["fn"]
            recall = counts["tp"] / total_expected if total_expected else 1.0
            print(
                f"  {cat:25s}  "
                f"recall={recall:.0%} "
                f"(TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}) "
                f"[{counts['count']} convs]"
            )

        # Per-conversation failures
        failures = [
            pc for pc in metrics.per_conversation
            if pc["entity_fn"] > 0
        ]
        if failures:
            print(f"\nMissed entities ({len(failures)} conversations):")
            for pc in failures:
                missed = set(pc["expected_entities"]) - set(pc["extracted_entities"])
                print(f"  {pc['id']} ({pc['category']}): missed {missed}")
                print(f"    extracted: {pc['extracted_entities']}")

        print(f"{'='*70}\n")

        # Assert thresholds
        assert metrics.entity_recall >= MIN_ENTITY_RECALL, (
            f"Entity recall {metrics.entity_recall:.0%} below "
            f"threshold {MIN_ENTITY_RECALL:.0%}. {metrics.summary()}"
        )
        assert metrics.entity_precision >= MIN_ENTITY_PRECISION, (
            f"Entity precision {metrics.entity_precision:.0%} below "
            f"threshold {MIN_ENTITY_PRECISION:.0%}. {metrics.summary()}"
        )
        assert metrics.relation_recall >= MIN_RELATION_RECALL, (
            f"Relation recall {metrics.relation_recall:.0%} below "
            f"threshold {MIN_RELATION_RECALL:.0%}. {metrics.summary()}"
        )
