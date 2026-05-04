"""
LLM routing evaluation test suite.

Runs 25 BAML routing/reranking scenarios against a live LLM API
and writes results to CSV + JSON files for human review.

Usage:
    uv run python tests/test_routing_eval.py
    uv run python tests/test_routing_eval.py --client Anthropic
    uv run python tests/test_routing_eval.py --threshold 0.90
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing BAML (it reads env vars at import time)
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path so we can import the baml_client
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neo4j_agent_memory.baml_client import b
from neo4j_agent_memory.baml_client.types import ResultItem


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Routing Evaluation Suite")
    parser.add_argument(
        "--client", default="OpenAI",
        help="BAML client to use (e.g., OpenAI, Anthropic, Resilient). Default: OpenAI",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Minimum pass rate to exit 0 (for CI). Default: 0.80",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for timestamped results. Default: tests/",
    )
    return parser.parse_args()


@dataclass
class TestCase:
    id: int
    category: str
    name: str
    function: str  # RouteQuery | RouteStorage | RerankResults
    args: dict
    expected: dict  # what we expect to see in the result


@dataclass
class TestResult:
    test_case: TestCase
    passed: bool
    actual_primary: str = ""
    actual_ambiguous: str = ""
    actual_fanout: str = ""
    actual_targets: str = ""
    actual_kept: str = ""
    notes: str = ""
    error: str = ""
    raw_json: str = ""


# ── Test Case Definitions ──────────────────────────────────────────────

ROUTE_QUERY_TESTS = [
    # Category 1: Vertical Identification
    TestCase(
        1, "Vertical ID", "MeetingQuery_Clear", "RouteQuery",
        {"query": "What was discussed in yesterday's standup?", "context": None},
        {"primary_vertical": "MEETINGS", "ambiguous": False},
    ),
    TestCase(
        2, "Vertical ID", "ProjectQuery_Clear", "RouteQuery",
        {"query": "What's the status of the migration project?", "context": None},
        {"primary_vertical": "PROJECTS", "ambiguous": False},
    ),
    TestCase(
        3, "Vertical ID", "ResearchQuery_Clear", "RouteQuery",
        {"query": "What papers did we cite in the NLP analysis?", "context": None},
        {"primary_vertical": "RESEARCH", "ambiguous": False},
    ),
    TestCase(
        4, "Vertical ID", "GeneralQuery_Clear", "RouteQuery",
        {"query": "What's Marcus's email address?", "context": None},
        {"primary_vertical": "GENERAL", "ambiguous": False},
    ),
    TestCase(
        5, "Vertical ID", "MeetingQuery_Implicit", "RouteQuery",
        {"query": "Who attended the sync with DataVault?", "context": None},
        {"primary_vertical": "MEETINGS", "ambiguous": False},
    ),
    TestCase(
        6, "Vertical ID", "ProjectQuery_Implicit", "RouteQuery",
        {"query": "Which tasks are blocked right now?", "context": None},
        {"primary_vertical": "PROJECTS", "ambiguous": False},
    ),
    TestCase(
        7, "Vertical ID", "ResearchQuery_Implicit", "RouteQuery",
        {"query": "What experiments validated the embedding approach?", "context": None},
        {"primary_vertical": "RESEARCH", "ambiguous": False},
    ),
    TestCase(
        8, "Vertical ID", "GeneralQuery_Preference", "RouteQuery",
        {"query": "Does Raj prefer Hyatt or Marriott?", "context": None},
        {"primary_vertical": "GENERAL", "ambiguous": False},
    ),
    # Category 3: Ambiguity Detection
    TestCase(
        14, "Ambiguity", "AmbiguousQuery_Vague", "RouteQuery",
        {"query": "anything new?", "context": None},
        {"ambiguous": True},
    ),
    TestCase(
        15, "Ambiguity", "AmbiguousQuery_PersonRef", "RouteQuery",
        {"query": "What did Sarah say?", "context": None},
        {"ambiguous": True},
    ),
    TestCase(
        16, "Ambiguity", "AmbiguousQuery_WithContext", "RouteQuery",
        {
            "query": "What did Sarah say?",
            "context": "Sarah presented research findings at Monday's team meeting about the migration project",
        },
        {"ambiguous": True, "requires_fanout": True},  # Context spans 3 verticals — still ambiguous
    ),
    TestCase(
        17, "Ambiguity", "AmbiguousQuery_Temporal", "RouteQuery",
        {"query": "what happened last week?", "context": None},
        {"ambiguous": True},
    ),
    # Category 4: Cross-Domain / Fanout
    TestCase(
        18, "Cross-Domain", "CrossDomain_MeetingResearch", "RouteQuery",
        {"query": "What research findings were presented in the project kickoff meeting?", "context": None},
        {"requires_fanout": True},
    ),
    TestCase(
        19, "Cross-Domain", "CrossDomain_ProjectMeeting", "RouteQuery",
        {"query": "What action items came out of the sprint planning meeting?", "context": None},
        {"requires_fanout": True},
    ),
    TestCase(
        20, "Cross-Domain", "CrossDomain_AllVerticals", "RouteQuery",
        {"query": "Give me everything we know about the DataVault migration", "context": None},
        {"requires_fanout": True},
    ),
    TestCase(
        21, "Cross-Domain", "CrossDomain_ResearchProject", "RouteQuery",
        {"query": "Which experiments are blocking the ML pipeline deliverable?", "context": None},
        {"requires_fanout": True},
    ),
]

ROUTE_STORAGE_TESTS = [
    TestCase(
        9, "Storage", "StoreMeeting", "RouteStorage",
        {
            "content": "Team standup 3/5: Sarah demoed the new dashboard. Marcus raised a blocker on the API integration. Action item: Sarah to send API specs to Marcus by EOD Thursday.",
            "memory_type": "message",
            "context": None,
        },
        {"primary_vertical": "MEETINGS"},
    ),
    TestCase(
        10, "Storage", "StoreProject", "RouteStorage",
        {
            "content": "Sprint 12 update: Migration task moved to 'in progress'. Milestone 2 deadline pushed to March 15. The embedding pipeline deliverable is 80% complete. Blocked by: AWS credentials from DataVault.",
            "memory_type": "message",
            "context": None,
        },
        {"primary_vertical": "PROJECTS"},
    ),
    TestCase(
        11, "Storage", "StoreResearch", "RouteStorage",
        {
            "content": "Literature review: Chen et al. 2024 found that graph-based RAG outperforms vector-only retrieval by 23% on multi-hop questions. This supports our hypothesis that knowledge graphs improve agent memory. Need to replicate experiment with our dataset.",
            "memory_type": "message",
            "context": None,
        },
        {"primary_vertical": "RESEARCH"},
    ),
    TestCase(
        12, "Storage", "StoreGeneral", "RouteStorage",
        {
            "content": "Raj Patel is the CTO of DataVault Solutions. He's vegetarian and prefers Hyatt hotels. His email is raj.patel@datavaultsolutions.com.",
            "memory_type": "message",
            "context": None,
        },
        {"primary_vertical": "GENERAL"},
    ),
    TestCase(
        13, "Storage", "StoreAmbiguousContent", "RouteStorage",
        {
            "content": "Talked to the team today about next steps. Things are going well overall.",
            "memory_type": "message",
            "context": None,
        },
        {"primary_vertical": "PROJECTS"},  # "next steps" is project-oriented
    ),
]

RERANK_TESTS = [
    TestCase(
        22, "Reranking", "Rerank_HighRelevance", "RerankResults",
        {
            "query": "What actions came out of Monday's standup?",
            "results": [
                {"id": "1", "content": "Monday standup: Sarah to finish API docs by Wednesday. Marcus to review PR #42. Priya to set up the embedding pipeline staging environment.", "source_db": "meetings", "result_type": "message"},
                {"id": "2", "content": "Monday standup action items: 1) Sarah - API docs by Wed 2) Marcus - review PR #42 3) Priya - staging env setup", "source_db": "meetings", "result_type": "message"},
            ],
        },
        {"all_kept": True, "min_relevance": 0.7},
    ),
    TestCase(
        23, "Reranking", "Rerank_LowRelevance", "RerankResults",
        {
            "query": "What's Raj's hotel preference?",
            "results": [
                {"id": "1", "content": "Raj Patel is the CTO of DataVault Solutions based in Denver.", "source_db": "neo4j", "result_type": "entity"},
                {"id": "2", "content": "DataVault Solutions was founded in 2018 and specializes in data migration.", "source_db": "neo4j", "result_type": "entity"},
                {"id": "3", "content": "The Denver office is located on Larimer Street.", "source_db": "neo4j", "result_type": "entity"},
            ],
        },
        {"all_dropped": True, "max_relevance": 0.4},
    ),
    TestCase(
        24, "Reranking", "Rerank_MixedRelevance", "RerankResults",
        {
            "query": "What actions came out of Monday's standup?",
            "results": [
                {"id": "1", "content": "Monday standup: Sarah to finish API docs by Wednesday. Marcus to review PR #42.", "source_db": "meetings", "result_type": "message"},
                {"id": "2", "content": "Sarah mentioned she prefers morning meetings over afternoon ones.", "source_db": "meetings", "result_type": "preference"},
                {"id": "3", "content": "The API documentation project milestone is due Friday.", "source_db": "projects", "result_type": "message"},
                {"id": "4", "content": "Priya's research on embedding models shows BERT outperforms word2vec for our use case.", "source_db": "research", "result_type": "message"},
            ],
        },
        {"expect_kept_ids": ["1"], "expect_dropped_ids": ["4"]},
    ),
    TestCase(
        25, "Reranking", "Rerank_NoRelevant", "RerankResults",
        {
            "query": "What's the weather forecast for Austin?",
            "results": [
                {"id": "1", "content": "Marcus drives up from San Antonio on Thursday mornings.", "source_db": "neo4j", "result_type": "message"},
                {"id": "2", "content": "The kickoff meeting is at the Sixth Street office in Austin.", "source_db": "meetings", "result_type": "message"},
                {"id": "3", "content": "Raj's Tesla Model 3 needs the parking garage on 7th for the chargers.", "source_db": "neo4j", "result_type": "message"},
            ],
        },
        {"total_kept": 0},
    ),
]

ALL_TESTS = sorted(
    ROUTE_QUERY_TESTS + ROUTE_STORAGE_TESTS + RERANK_TESTS,
    key=lambda t: t.id,
)


def evaluate_routing_result(result, expected: dict) -> tuple[bool, str]:
    """Check a RoutingDecision against expected values. Returns (passed, notes)."""
    issues = []

    if "primary_vertical" in expected:
        actual = result.primary_vertical.value
        if actual != expected["primary_vertical"]:
            issues.append(f"primary_vertical: expected {expected['primary_vertical']}, got {actual}")

    if "ambiguous" in expected:
        if result.ambiguous != expected["ambiguous"]:
            issues.append(f"ambiguous: expected {expected['ambiguous']}, got {result.ambiguous}")

    if "requires_fanout" in expected:
        if result.requires_fanout != expected["requires_fanout"]:
            issues.append(f"requires_fanout: expected {expected['requires_fanout']}, got {result.requires_fanout}")

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "OK"


def evaluate_rerank_result(result, expected: dict) -> tuple[bool, str]:
    """Check a RerankOutput against expected values. Returns (passed, notes)."""
    issues = []
    scored = {r.id: r for r in result.scored_results}

    if "all_kept" in expected and expected["all_kept"]:
        dropped = [r for r in result.scored_results if not r.keep]
        if dropped:
            issues.append(f"Expected all kept, but {len(dropped)} dropped")

    if "min_relevance" in expected:
        low = [r for r in result.scored_results if r.relevance < expected["min_relevance"]]
        if low:
            ids = [r.id for r in low]
            issues.append(f"Results {ids} scored below {expected['min_relevance']}")

    if "all_dropped" in expected and expected["all_dropped"]:
        kept = [r for r in result.scored_results if r.keep]
        if kept:
            issues.append(f"Expected all dropped, but {len(kept)} kept")

    if "max_relevance" in expected:
        high = [r for r in result.scored_results if r.relevance > expected["max_relevance"]]
        if high:
            ids = [r.id for r in high]
            issues.append(f"Results {ids} scored above {expected['max_relevance']}")

    if "total_kept" in expected:
        if result.total_kept != expected["total_kept"]:
            issues.append(f"total_kept: expected {expected['total_kept']}, got {result.total_kept}")

    if "expect_kept_ids" in expected:
        for eid in expected["expect_kept_ids"]:
            if eid in scored and not scored[eid].keep:
                issues.append(f"Expected id={eid} kept, but was dropped (relevance={scored[eid].relevance:.2f})")

    if "expect_dropped_ids" in expected:
        for eid in expected["expect_dropped_ids"]:
            if eid in scored and scored[eid].keep:
                issues.append(f"Expected id={eid} dropped, but was kept (relevance={scored[eid].relevance:.2f})")

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "OK"


async def run_test(tc: TestCase, baml_options: dict) -> TestResult:
    """Run a single test case against the live BAML client."""
    try:
        if tc.function == "RouteQuery":
            result = await b.RouteQuery(**tc.args, baml_options=baml_options)
            passed, notes = evaluate_routing_result(result, tc.expected)
            targets_str = ", ".join(
                f"{t.vertical.value}({t.confidence:.2f})" for t in result.targets
            )
            return TestResult(
                test_case=tc,
                passed=passed,
                actual_primary=result.primary_vertical.value,
                actual_ambiguous=str(result.ambiguous),
                actual_fanout=str(result.requires_fanout),
                actual_targets=targets_str,
                notes=notes,
                raw_json=result.model_dump_json(),
            )

        elif tc.function == "RouteStorage":
            result = await b.RouteStorage(**tc.args, baml_options=baml_options)
            passed, notes = evaluate_routing_result(result, tc.expected)
            targets_str = ", ".join(
                f"{t.vertical.value}({t.confidence:.2f})" for t in result.targets
            )
            return TestResult(
                test_case=tc,
                passed=passed,
                actual_primary=result.primary_vertical.value,
                actual_ambiguous=str(result.ambiguous),
                actual_fanout=str(result.requires_fanout),
                actual_targets=targets_str,
                notes=notes,
                raw_json=result.model_dump_json(),
            )

        elif tc.function == "RerankResults":
            # Convert dict results to ResultItem objects
            items = [ResultItem(**r) for r in tc.args["results"]]
            result = await b.RerankResults(query=tc.args["query"], results=items, baml_options=baml_options)
            passed, notes = evaluate_rerank_result(result, tc.expected)
            kept_ids = [r.id for r in result.scored_results if r.keep]
            scores = ", ".join(
                f"id={r.id}:{r.relevance:.2f}({'keep' if r.keep else 'drop'})"
                for r in result.scored_results
            )
            return TestResult(
                test_case=tc,
                passed=passed,
                actual_kept=f"{result.total_kept}/{result.total_input}",
                actual_targets=scores,
                notes=notes,
                raw_json=result.model_dump_json(),
            )

        else:
            return TestResult(tc, False, error=f"Unknown function: {tc.function}")

    except Exception as e:
        return TestResult(tc, False, error=str(e))


async def run_all_tests(baml_options: dict) -> list[TestResult]:
    """Run all test cases sequentially to avoid API rate limits."""
    results = []
    total = len(ALL_TESTS)
    for i, tc in enumerate(ALL_TESTS, 1):
        print(f"[{i}/{total}] Running {tc.name}...", end=" ", flush=True)
        result = await run_test(tc, baml_options)
        status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
        print(status)
        results.append(result)
    return results


def write_csv(results: list[TestResult], output_path: str):
    """Write test results to CSV."""
    headers = [
        "id",
        "category",
        "test_name",
        "function",
        "input_summary",
        "expected",
        "passed",
        "actual_primary",
        "actual_ambiguous",
        "actual_fanout",
        "actual_targets",
        "actual_kept",
        "notes",
        "error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            tc = r.test_case
            # Summarize input for readability
            if tc.function in ("RouteQuery",):
                input_summary = tc.args["query"]
            elif tc.function == "RouteStorage":
                input_summary = tc.args["content"][:80] + "..."
            else:
                input_summary = tc.args["query"]

            writer.writerow({
                "id": tc.id,
                "category": tc.category,
                "test_name": tc.name,
                "function": tc.function,
                "input_summary": input_summary,
                "expected": json.dumps(tc.expected),
                "passed": r.passed,
                "actual_primary": r.actual_primary,
                "actual_ambiguous": r.actual_ambiguous,
                "actual_fanout": r.actual_fanout,
                "actual_targets": r.actual_targets,
                "actual_kept": r.actual_kept,
                "notes": r.notes,
                "error": r.error,
            })


def write_raw_json(results: list[TestResult], output_path: str, client_name: str = "", pass_rate: float = 0.0):
    """Write full raw LLM responses to a companion JSON file with metadata."""
    data = {
        "metadata": {
            "client": client_name,
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "pass_rate": pass_rate,
        },
        "results": [
            {
                "id": r.test_case.id,
                "name": r.test_case.name,
                "passed": r.passed,
                "raw": json.loads(r.raw_json) if r.raw_json else None,
                "error": r.error or None,
            }
            for r in results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    args = parse_args()
    baml_options = {"client": args.client}

    print("=" * 60)
    print("LLM Routing Evaluation Suite")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Client: {args.client}")
    print(f"Tests: {len(ALL_TESTS)}")
    print(f"Threshold: {args.threshold:.0%}")
    print("=" * 60)

    results = await run_all_tests(baml_options)

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.error)
    errors = sum(1 for r in results if r.error)
    pass_rate = passed / len(results) if results else 0.0

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(results)} passed ({pass_rate:.0%}), {failed} failed, {errors} errors")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.test_case.category
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.passed:
            categories[cat]["passed"] += 1

    for cat, counts in categories.items():
        print(f"  {cat}: {counts['passed']}/{counts['total']}")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    # Timestamped output files
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ts_csv_path = str(output_dir / f"routing_eval_{args.client}_{timestamp}.csv")
    ts_json_path = str(output_dir / f"routing_eval_{args.client}_{timestamp}.json")
    write_csv(results, ts_csv_path)
    write_raw_json(results, ts_json_path, args.client, pass_rate)

    # Always overwrite the "latest" files for quick access
    latest_csv = str(Path(__file__).parent / "routing_eval_results.csv")
    latest_json = str(Path(__file__).parent / "routing_eval_results.json")
    write_csv(results, latest_csv)
    write_raw_json(results, latest_json, args.client, pass_rate)

    print(f"\nTimestamped CSV: {ts_csv_path}")
    print(f"Timestamped JSON: {ts_json_path}")
    print(f"Latest CSV:      {latest_csv}")
    print(f"Latest JSON:     {latest_json}")
    print("=" * 60)

    # CI exit code based on threshold
    if pass_rate < args.threshold:
        print(f"\nFAIL: Pass rate {pass_rate:.0%} below threshold {args.threshold:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
