"""Structured request logging for MCP tool calls.

Provides:
- log_tool_call: Decorator that logs every tool invocation with timing,
  parameters, result size, and errors as structured JSON.
- configure_logging: Centralized logging setup with JSON or text format,
  controllable via LOG_LEVEL and LOG_FORMAT env vars.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time

access_logger = logging.getLogger("neo4j_memory_mcp.access")


def log_tool_call(func):
    """Decorator that logs MCP tool calls with structured JSON.

    Logs two events per call:
    - tool_call_start: tool name and sanitized parameters
    - tool_call_end / tool_call_error: timing, result size, success/failure
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start = time.monotonic()

        # Sanitize kwargs: skip ctx, truncate long values
        log_kwargs = {}
        for k, v in kwargs.items():
            if k == "ctx":
                continue
            s = str(v)
            log_kwargs[k] = s[:200] + "..." if len(s) > 200 else s

        access_logger.info(
            json.dumps(
                {
                    "event": "tool_call_start",
                    "tool": tool_name,
                    "params": log_kwargs,
                }
            )
        )

        try:
            result = await func(*args, **kwargs)
            elapsed = time.monotonic() - start
            access_logger.info(
                json.dumps(
                    {
                        "event": "tool_call_end",
                        "tool": tool_name,
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "result_size": len(str(result)) if result else 0,
                        "success": True,
                    }
                )
            )
            return result
        except Exception as e:
            elapsed = time.monotonic() - start
            access_logger.error(
                json.dumps(
                    {
                        "event": "tool_call_error",
                        "tool": tool_name,
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "error": str(e),
                        "success": False,
                    }
                )
            )
            raise

    return wrapper


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure root logging with JSON or text format.

    Reads defaults from environment:
    - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    - LOG_FORMAT: json, text (default: json)

    Args:
        level: Override log level (e.g., "DEBUG").
        fmt: Override format ("json" or "text").
    """
    log_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    log_format = (fmt or os.environ.get("LOG_FORMAT", "json")).lower()

    if log_format == "json":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(handler)
