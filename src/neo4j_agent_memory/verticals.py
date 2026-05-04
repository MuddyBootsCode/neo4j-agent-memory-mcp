"""Vertical registry — single source of truth for vertical configuration.

Each vertical is defined once here. All other modules (router, extractor,
database init) derive their mappings from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VerticalConfig:
    """Configuration for a database vertical."""

    name: str  # Neo4j database name, e.g. "meetings"
    baml_enum: str  # BAML QueryVertical value, e.g. "MEETINGS"
    extractor_fn: str  # BAML extraction function name, e.g. "ExtractMeetingEntities"
    description: str  # Human-readable description for documentation


# ── Vertical Definitions ─────────────────────────────────────────────
# To add a new vertical:
# 1. Create baml_src/ontology_<name>.baml with entity types + extraction function
# 2. Add the enum value to QueryVertical in baml_src/routing.baml
# 3. Run `baml-cli generate` to regenerate the client
# 4. Add a VerticalConfig entry below

VERTICALS: dict[str, VerticalConfig] = {}


def _register(config: VerticalConfig) -> None:
    VERTICALS[config.name] = config


_register(VerticalConfig(
    name="meetings",
    baml_enum="MEETINGS",
    extractor_fn="ExtractMeetingEntities",
    description="Meetings, calendars, scheduling, attendees, agendas, action items",
))

_register(VerticalConfig(
    name="projects",
    baml_enum="PROJECTS",
    extractor_fn="ExtractProjectEntities",
    description="Projects, tasks, milestones, deliverables, sprints, dependencies",
))

_register(VerticalConfig(
    name="research",
    baml_enum="RESEARCH",
    extractor_fn="ExtractResearchEntities",
    description="Research notes, papers, findings, citations, experiments",
))


# ── Derived Mappings ─────────────────────────────────────────────────
# These replace the hardcoded dicts in router.py and vertical_extractor.py.

def get_vertical_to_db() -> dict[str, str]:
    """BAML enum value -> Neo4j database name. Includes GENERAL -> neo4j."""
    mapping = {v.baml_enum: v.name for v in VERTICALS.values()}
    mapping["GENERAL"] = "neo4j"
    return mapping


def get_vertical_extractors() -> dict[str, str]:
    """Database name -> BAML extraction function name."""
    return {v.name: v.extractor_fn for v in VERTICALS.values()}


def get_default_vertical_names() -> list[str]:
    """Default list of vertical database names."""
    return list(VERTICALS.keys())
