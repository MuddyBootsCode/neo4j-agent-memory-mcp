FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install runtime dependencies only. INSTALL_EXTRAS names an optional
# dependency group to include (e.g. "local" for sentence-transformers,
# used by the Anthropic-direct local mode in docker-compose.yml).
ARG INSTALL_EXTRAS=""
RUN uv sync --frozen --no-dev ${INSTALL_EXTRAS:+--extra ${INSTALL_EXTRAS}}

# Copy BAML source and application code
COPY baml_src/ baml_src/
COPY src/ src/

# Generate BAML client at build time
RUN uv run baml-cli generate

EXPOSE 8080

# Disable Docker auto-management (Neo4j runs as separate container)
ENV NEO4J_DOCKER_AUTO=false

# Default to SSE transport on all interfaces
ENV MCP_TRANSPORT=sse
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8080

CMD ["uv", "run", "neo4j-memory-mcp"]
