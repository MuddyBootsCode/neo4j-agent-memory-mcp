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
    from fastmcp import FastMCP

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
                """Manage Docker container and multi-database MemoryClient lifecycle."""
                import os

                from neo4j_agent_memory import MemoryClient as _MemoryClient
                from neo4j_agent_memory.config.settings import Neo4jConfig
                from neo4j_agent_memory.mcp._database_init import (
                    ensure_databases_exist,
                )
                from neo4j_agent_memory.mcp._docker import (
                    Neo4jDockerManager,
                    connect_with_retry,
                )
                from neo4j_agent_memory.mcp._registry import ClientRegistry

                # Patch factory to support BAML extraction [RFI-R1]
                import neo4j_agent_memory.extraction.factory as _factory_mod
                from neo4j_agent_memory.extraction.factory_ext import (
                    create_extractor as _ext_create_extractor,
                )

                _factory_mod.create_extractor = _ext_create_extractor
                logger.info("Extraction factory patched with BAML support")

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

                registry = ClientRegistry()

                async with docker_mgr:
                    # Phase 2: Connect general MemoryClient with retries
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
                        yield {"client": None, "registry": None, "router": None, "reranker": None}
                        return

                    registry.register("neo4j", client, client_cm)

                    # Phase 3: Ensure vertical databases exist
                    try:
                        driver = client.graph._driver
                        verticals = await ensure_databases_exist(driver)
                    except Exception as e:
                        logger.warning(
                            "Could not create vertical databases: %s. "
                            "Continuing with general DB only.", e
                        )
                        verticals = []

                    # Phase 4: Create clients for each vertical
                    for db_name in verticals:
                        try:
                            from pydantic import SecretStr

                            vertical_settings = type(settings)(
                                neo4j=Neo4jConfig(
                                    uri=neo4j_cfg.uri,
                                    username=neo4j_cfg.username,
                                    password=neo4j_cfg.password,
                                    database=db_name,
                                ),
                                embedding=settings.embedding,
                            )
                            v_client, v_cm = await connect_with_retry(
                                lambda s=vertical_settings: _MemoryClient(s),
                                max_attempts=3,
                                delay=2.0,
                            )
                            registry.register(db_name, v_client, v_cm)
                            logger.info("Client ready for database '%s'", db_name)
                        except Exception as e:
                            logger.warning(
                                "Failed to connect to '%s': %s", db_name, e
                            )

                    # Phase 5: Create router and reranker
                    router = None
                    reranker = None
                    try:
                        from neo4j_agent_memory.routing.router import (
                            QueryRouter,
                            ResultReranker,
                        )

                        router = QueryRouter(available_databases=registry.databases)
                        reranker = ResultReranker(enabled=True)
                        logger.info("Query router ready for databases: %s", registry.databases)
                    except Exception as e:
                        logger.warning("Could not initialize router: %s", e)

                    logger.info(
                        "ClientRegistry ready with databases: %s",
                        registry.databases,
                    )

                    try:
                        # Verify BAML patch took effect [RFI-R1]
                        baml_enabled = os.environ.get(
                            "NAM_EXTRACTION__BAML_ENABLED", ""
                        ).lower() in ("true", "1", "yes")
                        if baml_enabled:
                            _ext = getattr(client, "_extractor", None)
                            _ext_name = getattr(_ext, "name", str(type(_ext)))
                            if _ext and "Baml" in str(_ext_name):
                                logger.info(
                                    "BAML extraction active: %s", _ext_name
                                )
                            else:
                                logger.error(
                                    "BAML enabled but extractor is %s "
                                    "— patch may have failed",
                                    _ext_name,
                                )

                        yield {
                            "client": client,
                            "registry": registry,
                            "router": router,
                            "reranker": reranker,
                        }
                    finally:
                        await registry.close_all()

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

        if transport == "sse":
            await server.run_async(transport="sse", host=host, port=port)
        elif transport == "http":
            await server.run_async(transport="http", host=host, port=port)
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
        )
    )


if __name__ == "__main__":
    main()
