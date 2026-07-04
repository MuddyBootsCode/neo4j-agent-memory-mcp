"""Docker Compose management for Neo4j Enterprise.

Provides Neo4jDockerManager, an async context manager that ensures
a Neo4j Enterprise container is running via docker compose before
the MCP server connects. Stops the services on exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STARTUP_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _probe_bolt(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to *host*:*port* succeeds."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def _parse_bolt_uri(uri: str) -> tuple[str, int]:
    """Extract host and port from a Neo4j bolt URI."""
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    return host, port


def _find_compose_file() -> Path | None:
    """Locate docker-compose.yml by walking up from the package directory."""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # safety limit
        candidate = current / "docker-compose.yml"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


async def _run_compose(
    compose_dir: Path, *args: str, timeout: float = 30.0
) -> tuple[int, str, str]:
    """Run a docker compose command in the given directory.

    Returns:
        Tuple of (return_code, stdout, stderr).
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        *args,
        cwd=str(compose_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(
            f"docker compose {' '.join(args)} timed out after {timeout}s"
        )
    return proc.returncode or 0, stdout.decode(), stderr.decode()


async def connect_with_retry(
    client_factory,
    *,
    max_attempts: int = 5,
    delay: float = 2.0,
):
    """Create a MemoryClient with retries for Neo4j warmup.

    Args:
        client_factory: Callable returning an async context manager
            (e.g., ``lambda: MemoryClient(settings)``).
        max_attempts: Max connection attempts.
        delay: Seconds between attempts.

    Returns:
        Tuple of (client, context_manager) — caller must ``await cm.__aexit__()``
        when done.

    Raises:
        RuntimeError: After all attempts exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        cm = client_factory()
        try:
            client = await cm.__aenter__()
            return client, cm
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Neo4j not ready yet, retrying in %.0fs (attempt %d/%d): %s",
                delay,
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"Could not connect to Neo4j after {max_attempts} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


@dataclass
class Neo4jDockerManager:
    """Async context manager that ensures Neo4j is running via docker compose.

    Usage::

        async with Neo4jDockerManager(uri=uri) as mgr:
            # Neo4j is reachable at uri
            ...
        # Services stopped (if we started them)
    """

    uri: str = "bolt://localhost:7687"
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT
    docker_auto: bool = True
    compose_file: str | None = None

    # Private state
    _we_started: bool = field(default=False, init=False, repr=False)
    _compose_dir: Path | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> Neo4jDockerManager:
        if not self.docker_auto:
            logger.info("Docker auto-management disabled, skipping")
            return self

        host, port = _parse_bolt_uri(self.uri)

        # Already reachable? Skip Docker entirely.
        if await _probe_bolt(host, port):
            logger.info(
                "Neo4j already reachable at %s:%d, skipping Docker", host, port
            )
            return self

        # Resolve compose file location
        compose_path = self._resolve_compose_file()
        if compose_path is None or not compose_path.is_file():
            logger.warning(
                "docker-compose.yml not found — cannot auto-start Neo4j. "
                "Set NEO4J_COMPOSE_FILE or ensure the file exists in the project root."
            )
            return self

        self._compose_dir = compose_path.parent

        # Check if docker compose is available
        try:
            rc, _, stderr = await _run_compose(self._compose_dir, "version")
            if rc != 0:
                logger.warning("docker compose not available: %s", stderr.strip())
                return self
        except FileNotFoundError:
            logger.warning("docker command not found on PATH")
            return self

        # Start services (idempotent)
        logger.info("Running 'docker compose up -d' from %s", self._compose_dir)
        rc, stdout, stderr = await _run_compose(
            self._compose_dir, "up", "-d", timeout=120.0
        )
        if rc != 0:
            logger.error("docker compose up failed: %s", stderr.strip())
            raise RuntimeError(f"docker compose up failed: {stderr.strip()}")

        self._we_started = True
        await self._wait_for_bolt(host, port)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._we_started and self._compose_dir:
            await self._stop_compose()

    # ----- helpers -----

    def _resolve_compose_file(self) -> Path | None:
        """Resolve the compose file path from explicit config, env var, or auto-detection."""
        if self.compose_file:
            return Path(self.compose_file)
        env_path = os.environ.get("NEO4J_COMPOSE_FILE")
        if env_path:
            return Path(env_path)
        return _find_compose_file()

    async def _wait_for_bolt(self, host: str, port: int) -> None:
        """Poll bolt port until ready or timeout."""
        logger.info(
            "Waiting for Neo4j bolt at %s:%d (timeout=%ds)...",
            host,
            port,
            self.startup_timeout,
        )
        elapsed = 0.0
        interval = 0.5
        while elapsed < self.startup_timeout:
            if await _probe_bolt(host, port):
                logger.info(
                    "Neo4j bolt ready at %s:%d (%.1fs)", host, port, elapsed
                )
                return
            await asyncio.sleep(interval)
            elapsed += interval
            # Exponential backoff capped at 4s
            interval = min(interval * 2, 4.0)

        # Timeout — clean up
        if self._we_started and self._compose_dir:
            await self._stop_compose()
        raise RuntimeError(
            f"Neo4j failed to become ready at {host}:{port} "
            f"within {self.startup_timeout}s"
        )

    async def _stop_compose(self) -> None:
        """Stop compose services. Best-effort, never raises."""
        if not self._compose_dir:
            return
        try:
            logger.info(
                "Running 'docker compose stop' from %s", self._compose_dir
            )
            rc, _, stderr = await _run_compose(
                self._compose_dir, "stop", timeout=30.0
            )
            if rc != 0:
                logger.warning(
                    "docker compose stop returned %d: %s", rc, stderr.strip()
                )
            else:
                logger.info("Docker compose services stopped")
        except Exception as exc:
            logger.warning("Failed to stop compose services: %s", exc)
        finally:
            self._compose_dir = None
