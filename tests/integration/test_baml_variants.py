"""Test 3 BAML EntityType description variants against the golden dataset.

BAML responds more strongly to enum type descriptions than prompt changes.
This test evaluates precision/recall for each variant to find the best
balance between finding real entities and not over-extracting noun phrases.

Baseline (current): precision=47%, recall=98%
Problem: EVENT and OBJECT descriptions are too broad, causing extraction
of generic noun phrases like "migration task", "embedding pipeline", "team".

Workflow: modify extraction.baml → baml-cli generate → import fresh → test.
"""

import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

BAML_SRC = Path(__file__).parent.parent.parent / "baml_src"
EXTRACTION_BAML = BAML_SRC / "extraction.baml"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_PATH = FIXTURES_DIR / "golden_conversations.json"
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── The 3 Variants ───────────────────────────────────────────────────

# Variant A: Strict proper-noun-only
VARIANT_A = {
    "name": "strict_proper_noun",
    "descriptions": {
        "PERSON": "Named individuals only — extract people by their actual name (e.g. Sarah Chen, Raj), not roles or titles alone",
        "ORGANIZATION": "Named companies, institutions, or formal groups (e.g. Graphable, DataVault Solutions) — not informal references like 'the team' or 'engineering department'",
        "LOCATION": "Named geographic places, specific addresses, or landmarks (e.g. Denver, Larimer Street) — not relative locations like 'the office' or 'headquarters'",
        "EVENT": "Named or dated specific events only (e.g. Q2 Board Presentation, January data breach) — NOT generic activities like meetings, discussions, tasks, sprints, or reviews",
        "OBJECT": "Specific branded or model-identified items (e.g. Tesla Model 3, Surface Pro, Miro) — NOT generic nouns like pipeline, system, document, framework, or approach",
    },
}

# Variant B: Negative examples
VARIANT_B = {
    "name": "negative_examples",
    "descriptions": {
        "PERSON": "Individuals mentioned by name. DO NOT extract unnamed roles ('the engineer'), collective nouns ('the team', 'staff'), or pronouns",
        "ORGANIZATION": "Companies, institutions, named groups. DO NOT extract departments ('engineering team'), project names ('migration project'), or informal groups",
        "LOCATION": "Geographic places, named addresses, landmarks. DO NOT extract relative positions ('the office') unless preceded by a proper place name",
        "EVENT": "Specific incidents or calendar events with a date or proper name. DO NOT extract: meetings, calls, reviews, tasks, sprints, deployments, migrations, integrations, or any process noun",
        "OBJECT": "Branded or model-specific items only. DO NOT extract: pipelines, frameworks, approaches, embeddings, APIs, systems, databases, documents, tools, or any technical concept",
    },
}

# Variant C: Knowledge-graph-salient
VARIANT_C = {
    "name": "kg_salient",
    "descriptions": {
        "PERSON": "A specific person you could look up or contact, identified by name. Extract the fullest name form available",
        "ORGANIZATION": "A company, institution, or group that exists as a real named entity. Must have a proper noun name",
        "LOCATION": "A real place on a map — a city, street, building, or region identified by proper noun",
        "EVENT": "A unique dateable occurrence — not a type of activity. 'The January breach' is an event; 'a meeting' is not",
        "OBJECT": "A uniquely identifiable physical or digital item with a brand, model, or product name. 'Tesla Model 3' is an object; 'a document' is not",
    },
}

VARIANTS = [VARIANT_A, VARIANT_B, VARIANT_C]


# ── Golden dataset evaluation ────────────────────────────────────────

def _load_golden_dataset() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        return json.load(f)["conversations"]


def _entity_matches(extracted_name: str, expected_name: str) -> bool:
    e, x = extracted_name.lower().strip(), expected_name.lower().strip()
    return e == x or e in x or x in e


def _type_matches(a: str, b: str) -> bool:
    return a.upper() == b.upper()


def _relation_matches(ext: dict, exp: dict) -> bool:
    return (
        _entity_matches(ext["source"], exp["source"])
        and _entity_matches(ext["target"], exp["target"])
        and bool(re.search(exp["relation_contains"], ext["relation_type"], re.IGNORECASE))
    )


def _preference_matches(ext: dict, exp: dict) -> bool:
    return (
        bool(re.search(exp["category_contains"], ext["category"], re.IGNORECASE))
        and bool(re.search(exp["preference_contains"], ext["preference"], re.IGNORECASE))
    )


@dataclass
class Metrics:
    entity_tp: int = 0
    entity_fp: int = 0
    entity_fn: int = 0
    relation_tp: int = 0
    relation_fn: int = 0
    preference_tp: int = 0
    preference_fn: int = 0
    fp_by_type: dict = field(default_factory=lambda: {
        "PERSON": 0, "ORGANIZATION": 0, "LOCATION": 0, "EVENT": 0, "OBJECT": 0,
    })

    @property
    def precision(self):
        d = self.entity_tp + self.entity_fp
        return self.entity_tp / d if d else 1.0

    @property
    def recall(self):
        d = self.entity_tp + self.entity_fn
        return self.entity_tp / d if d else 1.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def rel_recall(self):
        d = self.relation_tp + self.relation_fn
        return self.relation_tp / d if d else 1.0


def _evaluate(conv, result):
    extracted = [{"name": e.name, "type": e.type} for e in result.entities]
    e_rels = [{"source": r.source, "target": r.target, "relation_type": r.relation_type} for r in result.relations]
    e_prefs = [{"category": p.category, "preference": p.preference} for p in result.preferences]

    expected = conv.get("expected_entities", [])
    tp, matched = 0, set()
    for exp in expected:
        for i, ext in enumerate(extracted):
            if i not in matched and _entity_matches(ext["name"], exp["name"]) and _type_matches(ext["type"], exp["type"]):
                tp += 1
                matched.add(i)
                break

    fn = len(expected) - tp
    fp = len(extracted) - len(matched)
    fp_types = {}
    for i, ext in enumerate(extracted):
        if i not in matched:
            t = ext["type"].upper()
            fp_types[t] = fp_types.get(t, 0) + 1

    rel_tp = sum(1 for exp in conv.get("expected_relations", []) if any(_relation_matches(er, exp) for er in e_rels))
    rel_fn = len(conv.get("expected_relations", [])) - rel_tp
    pref_tp = sum(1 for exp in conv.get("expected_preferences", []) if any(_preference_matches(ep, exp) for ep in e_prefs))
    pref_fn = len(conv.get("expected_preferences", [])) - pref_tp

    return {"tp": tp, "fp": fp, "fn": fn, "fp_types": fp_types,
            "rel_tp": rel_tp, "rel_fn": rel_fn, "pref_tp": pref_tp, "pref_fn": pref_fn}


# ── BAML file manipulation + regeneration ────────────────────────────

def _read_baml() -> str:
    return EXTRACTION_BAML.read_text()


def _apply_variant(descriptions: dict) -> None:
    """Rewrite EntityType descriptions in extraction.baml and regenerate."""
    content = _read_baml()
    for type_name, new_desc in descriptions.items():
        pattern = rf'({type_name}\s+@description\(")([^"]*?)("\))'
        content = re.sub(pattern, rf'\g<1>{new_desc}\3', content)
    EXTRACTION_BAML.write_text(content)

    # Regenerate BAML Python client
    result = subprocess.run(
        ["uv", "run", "baml-cli", "generate"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"baml-cli generate failed: {result.stderr}")


def _reload_baml_client():
    """Force reimport of the BAML generated client after regeneration."""
    # Remove all cached baml_client modules
    to_remove = [k for k in sys.modules if "baml_client" in k]
    for k in to_remove:
        del sys.modules[k]

    # Also remove the extractor since it imports from baml_client
    to_remove = [k for k in sys.modules if "baml_extractor" in k]
    for k in to_remove:
        del sys.modules[k]


async def _run_variant(name: str, conversations: list[dict]) -> Metrics:
    """Run extraction for all conversations and aggregate metrics."""
    _reload_baml_client()
    from neo4j_agent_memory.extraction.baml_extractor import BamlEntityExtractor

    extractor = BamlEntityExtractor(client_name="Bedrock")
    metrics = Metrics()

    for conv in conversations:
        result = await extractor.extract(conv["text"])
        ev = _evaluate(conv, result)
        metrics.entity_tp += ev["tp"]
        metrics.entity_fp += ev["fp"]
        metrics.entity_fn += ev["fn"]
        metrics.relation_tp += ev["rel_tp"]
        metrics.relation_fn += ev["rel_fn"]
        metrics.preference_tp += ev["pref_tp"]
        metrics.preference_fn += ev["pref_fn"]
        for t, c in ev["fp_types"].items():
            metrics.fp_by_type[t] = metrics.fp_by_type.get(t, 0) + c

    return metrics


# ── Test ─────────────────────────────────────────────────────────────

class TestBAMLVariants:
    """Test 3 EntityType description variants against the golden dataset."""

    async def test_compare_variants(self):
        """Run baseline + 3 variants, report comparative metrics, pick winner."""
        os.environ.setdefault("AWS_REGION", "us-east-1")
        os.environ.setdefault("AWS_PROFILE", "graphable-aws")

        conversations = _load_golden_dataset()
        original_baml = _read_baml()
        all_results = {}

        print(f"\n{'='*80}")
        print("BAML VARIANT COMPARISON — EntityType Descriptions")
        print(f"{'='*80}")

        try:
            # Baseline (current descriptions)
            print("\nRunning baseline...", flush=True)
            baseline = await _run_variant("baseline", conversations)
            all_results["baseline"] = baseline
            print(f"  baseline: P={baseline.precision:.0%} R={baseline.recall:.0%} F1={baseline.f1:.0%} "
                  f"FP={baseline.entity_fp} FN={baseline.entity_fn}")

            # Each variant
            for variant in VARIANTS:
                name = variant["name"]
                print(f"\nRunning {name}...", flush=True)
                _apply_variant(variant["descriptions"])
                metrics = await _run_variant(name, conversations)
                all_results[name] = metrics
                print(f"  {name}: P={metrics.precision:.0%} R={metrics.recall:.0%} F1={metrics.f1:.0%} "
                      f"FP={metrics.entity_fp} FN={metrics.entity_fn}")

                # Restore for next variant
                EXTRACTION_BAML.write_text(original_baml)
                subprocess.run(
                    ["uv", "run", "baml-cli", "generate"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True, timeout=30,
                )

        finally:
            # Always restore original
            EXTRACTION_BAML.write_text(original_baml)
            subprocess.run(
                ["uv", "run", "baml-cli", "generate"],
                cwd=str(PROJECT_ROOT),
                capture_output=True, timeout=30,
            )

        # Final comparison table
        print(f"\n{'='*80}")
        print(f"{'Variant':25s} {'Prec':>6s} {'Recall':>7s} {'F1':>5s} {'FP':>5s} {'FN':>5s} {'RelR':>6s}  FP by type")
        print(f"{'-'*80}")
        for name, m in all_results.items():
            fp_str = " ".join(f"{t}={c}" for t, c in sorted(m.fp_by_type.items()) if c > 0)
            print(f"{name:25s} {m.precision:6.0%} {m.recall:7.0%} {m.f1:5.0%} "
                  f"{m.entity_fp:5d} {m.entity_fn:5d} {m.rel_recall:6.0%}  {fp_str}")
        print(f"{'='*80}")

        winner_name, winner = max(all_results.items(), key=lambda x: x[1].f1)
        print(f"\nWinner: {winner_name} (F1={winner.f1:.0%}, P={winner.precision:.0%}, R={winner.recall:.0%})")
        print(f"{'='*80}\n")

        # Save results
        report = {
            "variants": {
                name: {
                    "precision": round(m.precision, 3),
                    "recall": round(m.recall, 3),
                    "f1": round(m.f1, 3),
                    "entity_tp": m.entity_tp,
                    "entity_fp": m.entity_fp,
                    "entity_fn": m.entity_fn,
                    "relation_recall": round(m.rel_recall, 3),
                    "fp_by_type": {k: v for k, v in m.fp_by_type.items() if v > 0},
                }
                for name, m in all_results.items()
            },
            "winner": winner_name,
        }
        report_path = FIXTURES_DIR / "variant_comparison_results.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Test passes if any variant improves F1 over baseline
        assert winner.f1 >= all_results["baseline"].f1, (
            f"No variant improved over baseline F1={all_results['baseline'].f1:.0%}"
        )
