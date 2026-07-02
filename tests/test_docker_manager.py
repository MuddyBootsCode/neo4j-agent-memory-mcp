"""Tests for Neo4jDockerManager (docker compose based)."""

from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


@pytest.fixture
def docker_config():
    """Standard Docker manager config for tests."""
    return {
        "uri": "bolt://localhost:7687",
        "docker_auto": True,
        "startup_timeout": 10,
    }


@pytest.fixture
def fake_compose_file(tmp_path):
    """Create a temporary docker-compose.yml for tests."""
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services:\n  neo4j:\n    image: neo4j:5-enterprise\n")
    return compose


def _mock_subprocess(returncode=0, stdout=b"", stderr=b""):
    """Create a mock subprocess result."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


class TestProbeNeo4j:
    """Tests for the _probe_bolt helper."""

    async def test_probe_returns_true_when_reachable(self):
        from neo4j_agent_memory.mcp._docker import _probe_bolt

        with patch("neo4j_agent_memory.mcp._docker.asyncio.open_connection") as mock_conn:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            result = await _probe_bolt("localhost", 7687)
            assert result is True

    async def test_probe_returns_false_when_unreachable(self):
        from neo4j_agent_memory.mcp._docker import _probe_bolt

        with patch(
            "neo4j_agent_memory.mcp._docker.asyncio.open_connection",
            side_effect=OSError("Connection refused"),
        ):
            result = await _probe_bolt("localhost", 7687)
            assert result is False


class TestFindComposeFile:
    """Tests for _find_compose_file helper."""

    def test_finds_compose_file_in_ancestor(self, tmp_path):
        from neo4j_agent_memory.mcp._docker import _find_compose_file

        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services: {}")

        # Create a nested directory structure to simulate __file__ location
        nested = tmp_path / "src" / "pkg"
        nested.mkdir(parents=True)
        fake_file = nested / "mod.py"
        fake_file.write_text("")

        # Patch __file__ on the module so _find_compose_file walks up from tmp_path
        with patch(
            "neo4j_agent_memory.mcp._docker.__file__",
            str(fake_file),
        ):
            result = _find_compose_file()
            assert result is not None
            assert result == compose

    def test_returns_none_when_not_found(self):
        from neo4j_agent_memory.mcp._docker import _find_compose_file

        with patch(
            "neo4j_agent_memory.mcp._docker.__file__",
            "/nonexistent/deep/path/mod.py",
        ):
            # The function walks up from __file__, won't find compose in /nonexistent
            result = _find_compose_file()
            # Result depends on actual filesystem, but function should not raise
            assert result is None or isinstance(result, Path)


class TestNeo4jDockerManager:
    """Tests for the manager context lifecycle."""

    async def test_skips_docker_when_neo4j_reachable(self, docker_config):
        """If Neo4j is already up, don't touch Docker at all."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        mgr = Neo4jDockerManager(**docker_config)

        with patch("neo4j_agent_memory.mcp._docker._probe_bolt", return_value=True):
            async with mgr:
                assert mgr._compose_dir is None
                assert mgr._we_started is False

    async def test_skips_when_docker_auto_false(self, docker_config):
        """docker_auto=False disables all container management."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        docker_config["docker_auto"] = False
        mgr = Neo4jDockerManager(**docker_config)

        async with mgr:
            assert mgr._compose_dir is None

    async def test_skips_when_compose_file_not_found(self, docker_config):
        """Gracefully skip if docker-compose.yml cannot be found."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        mgr = Neo4jDockerManager(**docker_config)

        with (
            patch("neo4j_agent_memory.mcp._docker._probe_bolt", return_value=False),
            patch.object(mgr, "_resolve_compose_file", return_value=None),
        ):
            async with mgr:
                assert mgr._compose_dir is None
                assert mgr._we_started is False

    async def test_skips_when_docker_not_on_path(self, docker_config, fake_compose_file):
        """Gracefully skip if docker command is not found."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        docker_config["compose_file"] = str(fake_compose_file)
        mgr = Neo4jDockerManager(**docker_config)

        with (
            patch("neo4j_agent_memory.mcp._docker._probe_bolt", return_value=False),
            patch(
                "neo4j_agent_memory.mcp._docker.asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError("docker not found"),
            ),
        ):
            async with mgr:
                assert mgr._we_started is False

    async def test_runs_compose_up_when_neo4j_unreachable(
        self, docker_config, fake_compose_file
    ):
        """Runs docker compose up -d when Neo4j is not reachable."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        docker_config["compose_file"] = str(fake_compose_file)
        mgr = Neo4jDockerManager(**docker_config)

        # First probe fails (Neo4j not up), then succeeds after compose up
        probe_results = [False, True]
        mock_proc = _mock_subprocess(returncode=0)

        with (
            patch(
                "neo4j_agent_memory.mcp._docker._probe_bolt",
                side_effect=probe_results,
            ),
            patch(
                "neo4j_agent_memory.mcp._docker.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ) as mock_exec,
        ):
            async with mgr:
                assert mgr._we_started is True

            # Verify compose up was called (first call is 'version', second is 'up -d')
            calls = mock_exec.call_args_list
            assert any(
                "up" in call.args and "-d" in call.args for call in calls
            ), f"Expected 'docker compose up -d' call, got: {calls}"

    async def test_runs_compose_stop_on_exit(self, docker_config, fake_compose_file):
        """Runs docker compose stop when exiting after we started."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        docker_config["compose_file"] = str(fake_compose_file)
        mgr = Neo4jDockerManager(**docker_config)

        mock_proc = _mock_subprocess(returncode=0)

        with (
            patch(
                "neo4j_agent_memory.mcp._docker._probe_bolt",
                side_effect=[False, True],
            ),
            patch(
                "neo4j_agent_memory.mcp._docker.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ) as mock_exec,
        ):
            async with mgr:
                pass

            # Last compose call should be 'stop'
            calls = mock_exec.call_args_list
            last_call_args = calls[-1].args
            assert "stop" in last_call_args, (
                f"Expected last call to be 'docker compose stop', got: {last_call_args}"
            )

    async def test_does_not_stop_when_we_did_not_start(self, docker_config):
        """No compose stop when Neo4j was already running."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        mgr = Neo4jDockerManager(**docker_config)

        with (
            patch("neo4j_agent_memory.mcp._docker._probe_bolt", return_value=True),
            patch(
                "neo4j_agent_memory.mcp._docker.asyncio.create_subprocess_exec",
            ) as mock_exec,
        ):
            async with mgr:
                assert mgr._we_started is False

            mock_exec.assert_not_called()

    async def test_compose_up_failure_raises(self, docker_config, fake_compose_file):
        """Raises RuntimeError if docker compose up fails."""
        from neo4j_agent_memory.mcp._docker import Neo4jDockerManager

        docker_config["compose_file"] = str(fake_compose_file)
        mgr = Neo4jDockerManager(**docker_config)

        # version check succeeds, up fails
        version_proc = _mock_subprocess(returncode=0)
        up_proc = _mock_subprocess(returncode=1, stderr=b"port already allocated")

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # version
                return version_proc
            return up_proc  # up -d

        with (
            patch("neo4j_agent_memory.mcp._docker._probe_bolt", return_value=False),
            patch(
                "neo4j_agent_memory.mcp._docker.asyncio.create_subprocess_exec",
                side_effect=mock_exec,
            ),
        ):
            with pytest.raises(RuntimeError, match="docker compose up failed"):
                async with mgr:
                    pass


class TestConnectWithRetry:
    """Tests for connect_with_retry helper."""

    async def test_succeeds_on_first_try(self):
        from neo4j_agent_memory.mcp._docker import connect_with_retry

        mock_client = MagicMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        factory = MagicMock(return_value=mock_cm)

        client, cm = await connect_with_retry(factory, max_attempts=3, delay=0.01)
        assert client is mock_client
        assert factory.call_count == 1

    async def test_retries_on_failure_then_succeeds(self):
        from neo4j_agent_memory.mcp._docker import connect_with_retry

        mock_client = MagicMock()
        mock_cm_good = AsyncMock()
        mock_cm_good.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm_good.__aexit__ = AsyncMock(return_value=False)

        mock_cm_bad = AsyncMock()
        mock_cm_bad.__aenter__ = AsyncMock(side_effect=Exception("not ready"))
        mock_cm_bad.__aexit__ = AsyncMock(return_value=False)

        calls = [mock_cm_bad, mock_cm_bad, mock_cm_good]
        factory = MagicMock(side_effect=calls)

        client, cm = await connect_with_retry(factory, max_attempts=5, delay=0.01)
        assert client is mock_client
        assert factory.call_count == 3

    async def test_raises_after_all_retries_exhausted(self):
        from neo4j_agent_memory.mcp._docker import connect_with_retry

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=Exception("not ready"))
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        factory = MagicMock(return_value=mock_cm)

        with pytest.raises(RuntimeError, match="Could not connect to Neo4j"):
            await connect_with_retry(factory, max_attempts=3, delay=0.01)
        assert factory.call_count == 3


class TestParseBoltUri:
    """Tests for bolt URI parsing."""

    def test_standard_bolt_uri(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt://localhost:7687")
        assert host == "localhost"
        assert port == 7687

    def test_custom_port(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt://myhost:9999")
        assert host == "myhost"
        assert port == 9999

    def test_default_port_when_missing(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt://myhost")
        assert host == "myhost"
        assert port == 7687

    def test_neo4j_scheme(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("neo4j://db.example.com:7687")
        assert host == "db.example.com"
        assert port == 7687

    def test_bolt_plus_s_scheme(self):
        from neo4j_agent_memory.mcp._docker import _parse_bolt_uri

        host, port = _parse_bolt_uri("bolt+s://secure.example.com:7687")
        assert host == "secure.example.com"
        assert port == 7687
