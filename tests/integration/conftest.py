"""Integration test fixtures requiring a running Neo4j instance.

Provides real MemoryClient connections to a dedicated test database,
with Bedrock for both embeddings (Titan V2) and BAML entity extraction
(Claude Sonnet via aws-bedrock provider).

Usage:
    uv run pytest tests/integration/ -m integration

Environment:
    - Neo4j must be running (docker-compose up)
    - AWS profile 'graphable-aws' must have Bedrock access
    - No OPENAI_API_KEY or ANTHROPIC_API_KEY needed — everything runs via Bedrock
"""

import asyncio
import logging
import os

import pytest

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

# Test database name — isolated from production verticals
TEST_DB_NAME = "testharness"

# Bedrock embedding config
BEDROCK_CONFIG = {
    "provider": "bedrock",
    "model": "amazon.titan-embed-text-v2:0",
    "dimensions": 1024,
    "aws_region": "us-east-1",
    "aws_profile": "graphable-aws",
}


def _neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _neo4j_auth() -> tuple[str, str]:
    return (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "graphmemory"),
    )


# ── Auto-apply integration marker to all tests in this directory ─────


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in tests/integration/ as integration."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ── Session-scoped: Neo4j driver + test database lifecycle ───────────


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def neo4j_driver():
    """Session-scoped async Neo4j driver.

    Skips the entire session if Neo4j is unreachable.
    """
    uri = _neo4j_uri()
    user, password = _neo4j_auth()

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        await driver.verify_connectivity()
    except Exception as exc:
        await driver.close()
        pytest.skip(f"Neo4j not reachable at {uri}: {exc}")

    yield driver
    await driver.close()


@pytest.fixture(scope="session")
async def test_database(neo4j_driver):
    """Create the test database once per session.

    Creates the database if it doesn't exist, then drops and recreates
    it to ensure clean 1024-dim vector indexes (Bedrock Titan V2).
    """
    async with neo4j_driver.session(database="system") as session:
        # Drop to ensure clean index dimensions
        await session.run(f"DROP DATABASE {TEST_DB_NAME} IF EXISTS")

    # Neo4j needs a moment to process the drop
    await asyncio.sleep(2)

    async with neo4j_driver.session(database="system") as session:
        await session.run(f"CREATE DATABASE {TEST_DB_NAME} IF NOT EXISTS")

    # Wait for the database to come online
    await asyncio.sleep(2)

    async with neo4j_driver.session(database="system") as session:
        result = await session.run(
            f"SHOW DATABASE {TEST_DB_NAME} YIELD currentStatus"
        )
        record = await result.single()
        assert record["currentStatus"] == "online", (
            f"Test database '{TEST_DB_NAME}' not online"
        )

    logger.info("Test database '%s' ready", TEST_DB_NAME)
    yield TEST_DB_NAME

    # Teardown: drop the test database
    async with neo4j_driver.session(database="system") as session:
        await session.run(f"DROP DATABASE {TEST_DB_NAME} IF EXISTS")
    logger.info("Test database '%s' dropped", TEST_DB_NAME)


# ── Function-scoped: MemoryClient with Bedrock embeddings ───────────


_ENV_OVERRIDES = {
    "NAM_EXTRACTION__BAML_ENABLED": "true",
    "AWS_REGION": "us-east-1",
    "AWS_PROFILE": "graphable-aws",
}


class _EnvContext:
    """Context manager that sets env vars and restores originals on exit."""

    def __init__(self, overrides: dict[str, str]):
        self._overrides = overrides
        self._originals: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self._overrides.items():
            self._originals[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, original in self._originals.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _apply_patches():
    """Apply Bedrock embedder + BAML extraction factory patches.

    Safe to call multiple times — patches are idempotent.
    Does NOT set env vars — use _EnvContext for that.
    """
    # Patch embedder factory for Bedrock
    from neo4j_agent_memory.mcp._embedder_patch import patch_embedder_factory

    patch_embedder_factory()

    # Patch extraction factory for BAML
    import neo4j_agent_memory.extraction.factory as _factory_mod
    from neo4j_agent_memory.extraction.factory_ext import (
        create_extractor as _ext_create_extractor,
    )

    _factory_mod.create_extractor = _ext_create_extractor


def _create_client_settings(test_database: str):
    """Build MemorySettings for the test database with Bedrock."""
    from neo4j_agent_memory.config.settings import (
        EmbeddingConfig,
        MemorySettings,
        Neo4jConfig,
    )

    return MemorySettings(
        neo4j=Neo4jConfig(
            password=_neo4j_auth()[1],
            database=test_database,
        ),
        embedding=EmbeddingConfig(**BEDROCK_CONFIG),
    )


@pytest.fixture
async def memory_client(test_database):
    """Function-scoped MemoryClient connected to the test database.

    Uses Bedrock for both embeddings (Titan V2) and entity extraction
    (Claude Sonnet via BAML). Wipes all data before each test for isolation.

    Env vars are scoped to this fixture — they don't leak into unit tests.
    """
    _apply_patches()
    from neo4j_agent_memory import MemoryClient

    with _EnvContext(_ENV_OVERRIDES):
        async with MemoryClient(settings=_create_client_settings(test_database)) as client:
            await client.graph.execute_write("MATCH (n) DETACH DELETE n", {})
            yield client


@pytest.fixture
async def memory_client_no_wipe(test_database):
    """MemoryClient that does NOT wipe data between tests.

    Use when tests build on each other (e.g., cross-session entity
    accumulation tests). The test module is responsible for cleanup.
    """
    _apply_patches()
    from neo4j_agent_memory import MemoryClient

    with _EnvContext(_ENV_OVERRIDES):
        async with MemoryClient(settings=_create_client_settings(test_database)) as client:
            yield client


# ── Raw Cypher helper for graph assertions ───────────────────────────


@pytest.fixture
def cypher_session(memory_client):
    """Access to the MemoryClient's graph executor for direct Cypher assertions.

    Use this alongside memory_client when you need to verify graph
    structure directly via Cypher after MCP operations.

    Usage:
        async def test_something(memory_client, cypher_session):
            # ... store via memory_client ...
            rows = await cypher_session.execute_read(
                "MATCH (n:Fact) RETURN count(n) AS c", {}
            )
            assert rows[0]["c"] == 1
    """
    return memory_client.graph
