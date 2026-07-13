"""Temporal context extraction from text using BAML."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def extract_temporal_context(
    content: str,
    reference_time: datetime | None = None,
) -> dict[str, str | bool | None]:
    """Extract temporal context from text using BAML.

    Returns dict with keys: valid_at, temporal_qualifier, is_current_state.
    Returns defaults (no temporal info) on failure.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    ref_iso = reference_time.isoformat()

    try:
        from agent_memory_mcp.baml_client.async_client import b as baml

        result = await baml.ExtractTemporalContext(
            text=content,
            reference_time=ref_iso,
        )

        return {
            "valid_at": result.valid_at,
            "temporal_qualifier": result.temporal_qualifier,
            "is_current_state": result.is_current_state,
        }

    except Exception as e:
        logger.warning("Temporal extraction failed: %s", e)
        return {
            "valid_at": None,
            "temporal_qualifier": None,
            "is_current_state": True,
        }
