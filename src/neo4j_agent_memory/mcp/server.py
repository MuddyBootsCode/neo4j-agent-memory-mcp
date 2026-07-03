"""MCP Server implementation for Neo4j Agent Memory.

Provides a Model Context Protocol server using FastMCP that exposes
memory capabilities as tools, resources, and prompts for AI platforms.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient

logger = logging.getLogger(__name__)

try:
    import hmac

    from fastmcp import FastMCP
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse

    # -----------------------------------------------------------------------
    # HTTP/SSE transport authentication (R10)
    # -----------------------------------------------------------------------
    #
    # The HTTP transport previously had no authentication at all: anyone who
    # could reach the port got full, unauthenticated tool access (see
    # docs/reviews/2026-07-02-agent-memory-deep-dive.md §2.3). BearerAuthMiddleware
    # gates every HTTP request on a static bearer token configured via
    # NAM_HTTP_TOKEN (or --http-token). assert_safe_http_bind refuses to start
    # a non-loopback bind with no token configured at all, so an operator
    # cannot silently ship an unauthenticated public endpoint.

    _LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

    class BearerAuthMiddleware:
        """ASGI middleware enforcing a static bearer token on every request.

        Applied to the FastMCP HTTP/SSE ASGI app. Requests missing a valid
        ``Authorization: Bearer <token>`` header receive a 401 before the
        request reaches the MCP application.
        """

        def __init__(self, app: Any, token: str) -> None:
            self.app = app
            self._expected = f"Bearer {token}"

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return

            headers = dict(scope.get("headers") or [])
            provided = headers.get(b"authorization", b"").decode("latin-1")

            if not provided or not hmac.compare_digest(provided, self._expected):
                response = JSONResponse({"error": "Unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return

            await self.app(scope, receive, send)

    def assert_safe_http_bind(*, host: str, token: str | None) -> None:
        """Refuse a non-loopback HTTP/SSE bind when no bearer token is configured.

        Binding a non-loopback host (e.g. ``0.0.0.0``) with no token means
        anyone who can reach the port gets full, unauthenticated tool access.
        Loopback binds remain allowed without a token (e.g. behind an
        authenticating reverse proxy or SSH tunnel on localhost).

        Raises:
            RuntimeError: if host is non-loopback and no token is configured.
        """
        if host in _LOOPBACK_HOSTS:
            return
        if not token:
            raise RuntimeError(
                f"Refusing to bind HTTP/SSE transport to non-loopback host "
                f"{host!r} with no bearer token configured. Set NAM_HTTP_TOKEN "
                "(or pass --http-token) before binding a non-loopback address."
            )

    def _http_auth_middleware(token: str | None) -> list[Middleware]:
        """Build the ASGI middleware stack for the HTTP/SSE app."""
        if not token:
            return []
        return [Middleware(BearerAuthMiddleware, token=token)]

    def create_mcp_server(
        settings: Any = None,
        *,
        server_name: str = "neo4j-agent-memory",
    ) -> FastMCP:
        """Create a configured FastMCP server.

        The server uses a lifespan to manage the async MemoryClient lifecycle.
        Tools, resources, and prompts are registered on the returned server.

        Args:
            settings: MemorySettings for Neo4j connection. If None, the server
                is created without a lifespan (useful for testing).
            server_name: Server name for MCP registration.

        Returns:
            Configured FastMCP server instance.

        Example:
            from neo4j_agent_memory import MemorySettings
            from neo4j_agent_memory.mcp import create_mcp_server

            settings = MemorySettings(...)
            server = create_mcp_server(settings)
            server.run()
        """
        lifespan = None
        if settings is not None:

            @asynccontextmanager
            async def lifespan(server: FastMCP):  # noqa: E303
                """Manage the Docker container and single MemoryClient lifecycle.

                Entity/relation/preference extraction is owned by the MCP tools
                (unified ExtractMemory pass), so no upstream extractor factory
                patch is needed — only the Bedrock embedder patch.
                """
                from neo4j_agent_memory import MemoryClient as _MemoryClient
                from neo4j_agent_memory.mcp._database_init import ensure_indexes
                from neo4j_agent_memory.mcp._docker import (
                    Neo4jDockerManager,
                    connect_with_retry,
                )

                # Patch embedder factory to support Bedrock
                def _create_embedder_extended(self):
                    """Extended _create_embedder with Bedrock support."""
                    from neo4j_agent_memory.config.settings import EmbeddingProvider

                    config = self._settings.embedding

                    if config.provider == EmbeddingProvider.OPENAI:
                        from neo4j_agent_memory.embeddings.openai import OpenAIEmbedder

                        return OpenAIEmbedder(
                            model=config.model,
                            api_key=config.api_key.get_secret_value() if config.api_key else None,
                            dimensions=config.dimensions if config.dimensions != 1536 else None,
                            batch_size=config.batch_size,
                        )
                    elif config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                        from neo4j_agent_memory.embeddings.sentence_transformers import (
                            SentenceTransformerEmbedder,
                        )

                        return SentenceTransformerEmbedder(
                            model_name=config.model,
                            device=config.device,
                        )
                    elif config.provider == EmbeddingProvider.BEDROCK:
                        from neo4j_agent_memory.embeddings.bedrock import BedrockEmbedder

                        logger.info(
                            "Creating Bedrock embedder (model=%s, region=%s)",
                            config.model,
                            config.aws_region,
                        )
                        return BedrockEmbedder(
                            model=config.model,
                            region_name=config.aws_region,
                            profile_name=config.aws_profile,
                            batch_size=config.batch_size,
                        )
                    else:
                        return None

                try:
                    _MemoryClient._create_embedder = _create_embedder_extended
                    logger.info("Embedder factory patched with Bedrock support")
                except Exception as e:
                    logger.error("Failed to patch embedder factory: %s", e)

                # Phase 1: Ensure Neo4j container is running
                docker_cfg = getattr(settings, "_docker_config", {})
                neo4j_cfg = settings.neo4j
                docker_mgr = Neo4jDockerManager(
                    uri=str(neo4j_cfg.uri),
                    docker_auto=docker_cfg.get("docker_auto", True),
                    startup_timeout=docker_cfg.get("startup_timeout", 60),
                    compose_file=docker_cfg.get("compose_file"),
                )

                async with docker_mgr:
                    # Phase 2: Connect the MemoryClient with retries
                    try:
                        client, client_cm = await connect_with_retry(
                            lambda: _MemoryClient(settings),
                            max_attempts=5,
                            delay=2.0,
                        )
                    except RuntimeError as exc:
                        logger.error(
                            "Neo4j unavailable — server will start but "
                            "tools will return errors until Neo4j is "
                            "reachable: %s",
                            exc,
                        )
                        yield {"client": None}
                        return

                    # Phase 3: Ensure temporal/graph indexes exist
                    try:
                        await ensure_indexes(client.graph)
                    except Exception as e:
                        logger.warning("Could not create indexes: %s", e)

                    logger.info("MemoryClient ready (database=%s)", neo4j_cfg.database)

                    try:
                        yield {"client": client}
                    finally:
                        await client_cm.__aexit__(None, None, None)

        mcp = FastMCP(
            server_name,
            lifespan=lifespan,
        )

        from neo4j_agent_memory.mcp._prompts import register_prompts
        from neo4j_agent_memory.mcp._resources import register_resources
        from neo4j_agent_memory.mcp._tools import register_tools

        register_tools(mcp)
        register_resources(mcp)
        register_prompts(mcp)

        return mcp

    class Neo4jMemoryMCPServer:
        """MCP server exposing Neo4j Agent Memory capabilities.

        Backward-compatible wrapper that accepts a pre-connected MemoryClient.
        For new code, prefer ``create_mcp_server(settings)`` instead.

        Example:
            from neo4j_agent_memory import MemoryClient, MemorySettings
            from neo4j_agent_memory.mcp import Neo4jMemoryMCPServer

            settings = MemorySettings(...)
            async with MemoryClient(settings) as client:
                server = Neo4jMemoryMCPServer(client)
                await server.run()

        Tools:
            - memory_search: Hybrid vector + graph search
            - memory_store: Store messages, facts, preferences
            - entity_lookup: Get entity with relationships
            - conversation_history: Get conversation for session
            - graph_query: Execute read-only Cypher queries
        """

        def __init__(
            self,
            memory_client: MemoryClient,
            *,
            server_name: str = "neo4j-agent-memory",
        ):
            """Initialize the MCP server with a pre-connected client.

            Args:
                memory_client: Connected MemoryClient instance.
                server_name: Server name for MCP registration.
            """
            self._client = memory_client

            @asynccontextmanager
            async def _preconnected_lifespan(server: FastMCP):
                yield {"client": memory_client}

            self._mcp = FastMCP(
                server_name,
                lifespan=_preconnected_lifespan,
            )

            from neo4j_agent_memory.mcp._prompts import register_prompts
            from neo4j_agent_memory.mcp._resources import register_resources
            from neo4j_agent_memory.mcp._tools import register_tools

            register_tools(self._mcp)
            register_resources(self._mcp)
            register_prompts(self._mcp)

        async def run(self) -> None:
            """Run the MCP server using stdio transport."""
            await self._mcp.run_async(transport="stdio")

        async def run_sse(self, host: str = "127.0.0.1", port: int = 8080) -> None:
            """Run the MCP server using SSE transport.

            Args:
                host: Host to bind to.
                port: Port to listen on.
            """
            await self._mcp.run_async(transport="sse", host=host, port=port)

    async def run_server(
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        transport: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        docker_auto: bool = True,
        docker_startup_timeout: int = 60,
        compose_file: str | None = None,
        http_token: str | None = None,
    ) -> None:
        """Run the MCP server with Neo4j connection.

        Convenience function for CLI usage.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            neo4j_database: Neo4j database name.
            transport: Transport type (stdio, sse, or http).
            host: Host for network transports.
            port: Port for network transports.
            docker_auto: Enable automatic Docker container management.
            docker_startup_timeout: Max seconds to wait for Neo4j startup.
            compose_file: Path to docker-compose.yml (auto-detected if None).
            http_token: Bearer token required for HTTP/SSE transport (env:
                NAM_HTTP_TOKEN). Required when binding a non-loopback host.
        """
        import os

        from pydantic import SecretStr

        from neo4j_agent_memory import MemorySettings
        from neo4j_agent_memory.config.settings import EmbeddingConfig, Neo4jConfig

        # Build embedding config from env vars — defaults to Bedrock
        embedding_provider = os.environ.get("NAM_EMBEDDING_PROVIDER", "bedrock")
        embedding_kwargs: dict[str, Any] = {
            "provider": embedding_provider,
        }
        if embedding_provider == "bedrock":
            embedding_kwargs.update({
                "model": os.environ.get(
                    "NAM_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
                ),
                "dimensions": int(os.environ.get("NAM_EMBEDDING_DIMENSIONS", "1024")),
                "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
                "aws_profile": os.environ.get("AWS_PROFILE"),
            })
        elif embedding_provider == "openai":
            embedding_kwargs.update({
                "model": os.environ.get(
                    "NAM_EMBEDDING_MODEL", "text-embedding-3-small"
                ),
            })

        settings = MemorySettings(
            neo4j=Neo4jConfig(
                uri=neo4j_uri,
                username=neo4j_user,
                password=SecretStr(neo4j_password),
                database=neo4j_database,
            ),
            embedding=EmbeddingConfig(**embedding_kwargs),
        )

        # Attach Docker config as private attr for lifespan to read
        settings._docker_config = {
            "docker_auto": docker_auto,
            "startup_timeout": docker_startup_timeout,
            "compose_file": compose_file,
        }

        server = create_mcp_server(settings, server_name="neo4j-agent-memory")

        http_token = http_token or os.environ.get("NAM_HTTP_TOKEN")

        if transport in ("sse", "http"):
            assert_safe_http_bind(host=host, token=http_token)
            middleware = _http_auth_middleware(http_token)
            await server.run_async(
                transport=transport, host=host, port=port, middleware=middleware
            )
        else:
            await server.run_async(transport="stdio")

except ImportError:
    # FastMCP not installed
    class Neo4jMemoryMCPServer:  # type: ignore[no-redef]
        """Placeholder when FastMCP is not installed."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "FastMCP not installed. Install with: pip install neo4j-agent-memory[mcp]"
            )

    def create_mcp_server(*args: Any, **kwargs: Any) -> Neo4jMemoryMCPServer:  # type: ignore[misc]
        raise ImportError(
            "FastMCP not installed. Install with: pip install neo4j-agent-memory[mcp]"
        )


def main() -> None:
    """CLI entry point for running the MCP server."""
    import argparse
    import os

    from neo4j_agent_memory.mcp._logging import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="Neo4j Agent Memory MCP Server")
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="MCP transport type (env: MCP_TRANSPORT)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_HOST", "127.0.0.1"),
        help="Host for network transports (env: MCP_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_PORT", "8080")),
        help="Port for network transports (env: MCP_PORT)",
    )
    parser.add_argument(
        "--http-token",
        default=os.environ.get("NAM_HTTP_TOKEN"),
        help=(
            "Bearer token required for HTTP/SSE transport (env: NAM_HTTP_TOKEN). "
            "Required when --host is not a loopback address."
        ),
    )
    parser.add_argument(
        "--neo4j-docker-auto",
        default=os.environ.get("NEO4J_DOCKER_AUTO", "true").lower()
        in ("true", "1", "yes"),
        action=argparse.BooleanOptionalAction,
        help="Enable automatic Docker container management (default: true)",
    )
    parser.add_argument(
        "--compose-file",
        default=os.environ.get("NEO4J_COMPOSE_FILE"),
        help="Path to docker-compose.yml (auto-detected if not specified)",
    )
    parser.add_argument(
        "--neo4j-docker-startup-timeout",
        type=int,
        default=int(os.environ.get("NEO4J_DOCKER_STARTUP_TIMEOUT", "60")),
        help="Max seconds to wait for Neo4j startup (default: 60)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_server(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            transport=args.transport,
            host=args.host,
            port=args.port,
            docker_auto=args.neo4j_docker_auto,
            docker_startup_timeout=args.neo4j_docker_startup_timeout,
            compose_file=args.compose_file,
            http_token=args.http_token,
        )
    )


if __name__ == "__main__":
    main()
