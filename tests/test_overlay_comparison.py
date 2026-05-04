"""Overlay vs upstream tool comparison tests.

Verifies the overlay MCP implementation is a strict superset of the
upstream neo4j-agent-memory package's tool definitions. If the upstream
adds new tools or parameters, these tests will fail — signaling the
overlay needs updating.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest
from fastmcp import FastMCP

from neo4j_agent_memory.mcp._tools import register_tools

# ---------------------------------------------------------------------------
# Known intentional renames (upstream name → overlay name)
# These exist because the overlay avoids Python reserved words or uses
# clearer naming while preserving the same semantic meaning.
# ---------------------------------------------------------------------------
PARAM_RENAMES: dict[str, dict[str, str]] = {
    "memory_store": {
        "type": "memory_type",       # 'type' is a Python builtin
        "object": "object_value",    # 'object' is a Python builtin
    },
    "entity_lookup": {
        "type": "entity_type",       # 'type' is a Python builtin
    },
}


def _get_upstream_tools() -> dict[str, dict]:
    """Load upstream tool definitions directly from the installed package.

    Uses importlib to bypass the overlay's __path__ manipulation,
    loading the upstream tools.py from site-packages directly.
    """
    # Find the installed package in site-packages
    import neo4j_agent_memory

    installed_dir = None
    for p in neo4j_agent_memory.__path__:
        candidate = Path(p) / "mcp" / "tools.py"
        if candidate.exists():
            installed_dir = candidate
            break

    if installed_dir is None:
        pytest.skip(
            "Upstream neo4j_agent_memory.mcp.tools not found in site-packages. "
            "Is the upstream package installed?"
        )

    spec = importlib.util.spec_from_file_location(
        "neo4j_agent_memory_upstream_tools", str(installed_dir)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return {tool["name"]: tool for tool in mod.MEMORY_TOOLS}


def _get_overlay_tools() -> dict[str, dict]:
    """Register overlay tools on a test FastMCP and extract their metadata."""
    mcp = FastMCP("comparison-test")
    register_tools(mcp)

    tools = {}
    for tool in mcp._tool_manager._tools.values():
        # Extract parameter names from the function signature
        sig = inspect.signature(tool.fn)
        param_names = [
            p.name for p in sig.parameters.values()
            if p.name != "ctx"  # FastMCP Context param
        ]
        tools[tool.name] = {
            "name": tool.name,
            "parameters": param_names,
            "fn": tool.fn,
        }
    return tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def upstream_tools():
    return _get_upstream_tools()


@pytest.fixture(scope="module")
def overlay_tools():
    return _get_overlay_tools()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolNameSuperset:
    """Every upstream tool must exist in the overlay."""

    def test_all_upstream_tools_present(self, upstream_tools, overlay_tools):
        upstream_names = set(upstream_tools.keys())
        overlay_names = set(overlay_tools.keys())
        missing = upstream_names - overlay_names
        assert not missing, (
            f"Upstream tools missing from overlay: {missing}. "
            f"Overlay has: {sorted(overlay_names)}"
        )

    def test_upstream_tool_count(self, upstream_tools):
        assert len(upstream_tools) == 6, (
            f"Upstream tool count changed from 6 to {len(upstream_tools)}. "
            f"Tools: {sorted(upstream_tools.keys())}. "
            f"If upstream added tools, the overlay needs updating."
        )


class TestOverlayExtensions:
    """Overlay should have its additional tools."""

    EXPECTED_OVERLAY_ONLY = {
        "explain_reasoning",
        "extract_reasoning",
        "temporal_query",
        "fact_evolution",
        "knowledge_state",
    }

    def test_overlay_only_tools_present(self, overlay_tools):
        overlay_names = set(overlay_tools.keys())
        missing = self.EXPECTED_OVERLAY_ONLY - overlay_names
        assert not missing, (
            f"Expected overlay-only tools missing: {missing}"
        )

    def test_overlay_has_more_tools(self, upstream_tools, overlay_tools):
        assert len(overlay_tools) > len(upstream_tools), (
            f"Overlay ({len(overlay_tools)}) should have more tools "
            f"than upstream ({len(upstream_tools)})"
        )

    def test_total_overlay_tool_count(self, overlay_tools):
        assert len(overlay_tools) == 11, (
            f"Expected 11 overlay tools, got {len(overlay_tools)}: "
            f"{sorted(overlay_tools.keys())}"
        )


class TestParameterSuperset:
    """For each shared tool, every upstream param must exist in the overlay
    (accounting for known renames)."""

    @pytest.fixture(scope="class")
    def shared_tool_names(self, upstream_tools, overlay_tools):
        return set(upstream_tools.keys()) & set(overlay_tools.keys())

    def _upstream_params(self, tool_def: dict) -> set[str]:
        """Extract parameter names from upstream inputSchema."""
        schema = tool_def.get("inputSchema", {})
        return set(schema.get("properties", {}).keys())

    def test_memory_search_params(self, upstream_tools, overlay_tools):
        upstream_params = self._upstream_params(upstream_tools["memory_search"])
        overlay_params = set(overlay_tools["memory_search"]["parameters"])
        renames = PARAM_RENAMES.get("memory_search", {})

        for up_param in upstream_params:
            expected = renames.get(up_param, up_param)
            assert expected in overlay_params, (
                f"memory_search: upstream param '{up_param}' "
                f"(mapped to '{expected}') missing from overlay. "
                f"Overlay params: {sorted(overlay_params)}"
            )

    def test_memory_store_params(self, upstream_tools, overlay_tools):
        upstream_params = self._upstream_params(upstream_tools["memory_store"])
        overlay_params = set(overlay_tools["memory_store"]["parameters"])
        renames = PARAM_RENAMES.get("memory_store", {})

        for up_param in upstream_params:
            expected = renames.get(up_param, up_param)
            assert expected in overlay_params, (
                f"memory_store: upstream param '{up_param}' "
                f"(mapped to '{expected}') missing from overlay. "
                f"Overlay params: {sorted(overlay_params)}"
            )

    def test_entity_lookup_params(self, upstream_tools, overlay_tools):
        upstream_params = self._upstream_params(upstream_tools["entity_lookup"])
        overlay_params = set(overlay_tools["entity_lookup"]["parameters"])
        renames = PARAM_RENAMES.get("entity_lookup", {})

        for up_param in upstream_params:
            expected = renames.get(up_param, up_param)
            assert expected in overlay_params, (
                f"entity_lookup: upstream param '{up_param}' "
                f"(mapped to '{expected}') missing from overlay. "
                f"Overlay params: {sorted(overlay_params)}"
            )

    def test_conversation_history_params(self, upstream_tools, overlay_tools):
        upstream_params = self._upstream_params(upstream_tools["conversation_history"])
        overlay_params = set(overlay_tools["conversation_history"]["parameters"])
        renames = PARAM_RENAMES.get("conversation_history", {})

        for up_param in upstream_params:
            expected = renames.get(up_param, up_param)
            assert expected in overlay_params, (
                f"conversation_history: upstream param '{up_param}' "
                f"(mapped to '{expected}') missing from overlay. "
                f"Overlay params: {sorted(overlay_params)}"
            )

    def test_graph_query_params(self, upstream_tools, overlay_tools):
        upstream_params = self._upstream_params(upstream_tools["graph_query"])
        overlay_params = set(overlay_tools["graph_query"]["parameters"])
        renames = PARAM_RENAMES.get("graph_query", {})

        for up_param in upstream_params:
            expected = renames.get(up_param, up_param)
            assert expected in overlay_params, (
                f"graph_query: upstream param '{up_param}' "
                f"(mapped to '{expected}') missing from overlay. "
                f"Overlay params: {sorted(overlay_params)}"
            )

    def test_add_reasoning_trace_params(self, upstream_tools, overlay_tools):
        upstream_params = self._upstream_params(upstream_tools["add_reasoning_trace"])
        overlay_params = set(overlay_tools["add_reasoning_trace"]["parameters"])
        renames = PARAM_RENAMES.get("add_reasoning_trace", {})

        for up_param in upstream_params:
            expected = renames.get(up_param, up_param)
            assert expected in overlay_params, (
                f"add_reasoning_trace: upstream param '{up_param}' "
                f"(mapped to '{expected}') missing from overlay. "
                f"Overlay params: {sorted(overlay_params)}"
            )


class TestKnownRenames:
    """Verify that known parameter renames are intentional and documented."""

    def test_memory_store_type_renamed(self, overlay_tools):
        params = set(overlay_tools["memory_store"]["parameters"])
        assert "memory_type" in params, "memory_store should use 'memory_type' not 'type'"
        assert "type" not in params, "memory_store should not have bare 'type' param"

    def test_memory_store_object_renamed(self, overlay_tools):
        params = set(overlay_tools["memory_store"]["parameters"])
        assert "object_value" in params, "memory_store should use 'object_value' not 'object'"
        assert "object" not in params, "memory_store should not have bare 'object' param"

    def test_entity_lookup_type_renamed(self, overlay_tools):
        params = set(overlay_tools["entity_lookup"]["parameters"])
        assert "entity_type" in params, "entity_lookup should use 'entity_type' not 'type'"
        assert "type" not in params, "entity_lookup should not have bare 'type' param"


class TestOverlayNewParameters:
    """Verify overlay adds expected new parameters to shared tools."""

    def test_memory_store_has_temporal_params(self, overlay_tools):
        params = set(overlay_tools["memory_store"]["parameters"])
        temporal_params = {"valid_from", "valid_until", "confidence", "supersedes"}
        missing = temporal_params - params
        assert not missing, f"memory_store missing temporal params: {missing}"

    def test_memory_search_has_include_expired(self, overlay_tools):
        params = set(overlay_tools["memory_search"]["parameters"])
        assert "include_expired" in params, "memory_search should have include_expired param"

    def test_shared_tools_have_database_param(self, upstream_tools, overlay_tools):
        """All shared tools should have the multi-database 'database' param."""
        shared = set(upstream_tools.keys()) & set(overlay_tools.keys())
        missing_db = []
        for name in shared:
            params = set(overlay_tools[name]["parameters"])
            if "database" not in params:
                missing_db.append(name)
        assert not missing_db, (
            f"These shared tools lack 'database' param: {missing_db}"
        )
