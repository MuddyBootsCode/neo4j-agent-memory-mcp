"""Result merging utilities for multi-database queries."""

from __future__ import annotations

from typing import Any


def merge_search_results(
    per_db_results: dict[str, dict[str, list]],
) -> dict[str, list]:
    """Merge search results from multiple databases.

    Combines lists by memory type, annotates each result with its
    source database, and deduplicates by ID.
    """
    merged: dict[str, list] = {}
    seen_ids: set[str] = set()

    for db_name, results in per_db_results.items():
        if isinstance(results, dict) and "error" in results:
            continue
        for memory_type, items in results.items():
            if not isinstance(items, list):
                continue
            if memory_type not in merged:
                merged[memory_type] = []
            for item in items:
                item_id = item.get("id", "")
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                item["_source_db"] = db_name
                merged[memory_type].append(item)

    # Sort each list: active facts first, then by similarity/confidence
    for memory_type in merged:
        merged[memory_type].sort(
            key=lambda x: (
                # Active facts first (temporal_status="active" or no status)
                0 if x.get("temporal_status", "active") == "active" else 1,
                # Then by similarity/confidence descending
                -(x.get("similarity") or x.get("confidence") or 0),
            ),
        )

    return merged


def merge_entity_results(
    per_db_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge entity lookup results from multiple databases."""
    found_entities = []
    all_neighbors = []

    for db_name, result in per_db_results.items():
        if isinstance(result, dict) and "error" in result:
            continue
        if result.get("found"):
            entity = result.get("entity", {})
            entity["_source_db"] = db_name
            found_entities.append(entity)
            for neighbor in result.get("neighbors", []):
                neighbor["_source_db"] = db_name
                all_neighbors.append(neighbor)

    if not found_entities:
        return {"found": False}

    return {
        "found": True,
        "entities": found_entities,
        "neighbors": all_neighbors,
        "databases_searched": list(per_db_results.keys()),
    }
