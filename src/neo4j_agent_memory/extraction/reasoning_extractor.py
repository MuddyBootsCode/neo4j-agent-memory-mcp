"""BAML-based reasoning extraction from conversation text."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BamlReasoningExtractor:
    """Extract structured reasoning chains from conversation text using BAML.

    Uses the ExtractReasoning BAML function to identify hypotheses,
    decisions, evidence, and conclusions from raw text.
    """

    def __init__(
        self,
        *,
        client_name: str = "Bedrock",
        client_registry: Any | None = None,
    ):
        self._client_name = client_name
        self._baml_options: dict[str, Any] = {}

        if client_registry:
            self._baml_options["client_registry"] = client_registry
        elif client_name != "Bedrock":
            try:
                from baml_py import ClientRegistry

                registry = ClientRegistry()
                registry.set_primary(client_name)
                self._baml_options["client_registry"] = registry
            except ImportError:
                logger.warning("baml-py not installed, client_name override ignored")

    async def extract_reasoning(self, text: str) -> dict[str, Any]:
        """Extract structured reasoning from conversation text.

        Args:
            text: Conversation transcript or text to analyze.

        Returns:
            Dict with task, steps (thought/action/observation), conclusion, success.
        """
        if not text or not text.strip():
            return {"task": "", "steps": [], "final_conclusion": "", "success": False}

        from neo4j_agent_memory.baml_client.async_client import b

        result = await b.ExtractReasoning(
            text=text,
            baml_options=self._baml_options if self._baml_options else {},
        )

        return {
            "task": result.task,
            "steps": [
                {
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "alternatives_considered": step.alternatives_considered,
                    "confidence": max(0.0, min(1.0, step.confidence)),
                }
                for step in result.steps
            ],
            "final_conclusion": result.final_conclusion,
            "success": result.success,
        }

    async def synthesize_explanation(
        self,
        task: str,
        steps: list[dict[str, str]],
        outcome: str,
    ) -> str:
        """Synthesize a natural-language explanation from a reasoning chain.

        Args:
            task: The task that was solved.
            steps: List of {thought, action, observation} dicts.
            outcome: The final outcome.

        Returns:
            Natural-language explanation string.
        """
        from neo4j_agent_memory.baml_client.async_client import b
        from neo4j_agent_memory.baml_client.types import (
            ReasoningChainInput,
            ReasoningStepInput,
        )

        chain = ReasoningChainInput(
            task=task,
            steps=[
                ReasoningStepInput(
                    thought=s.get("thought", ""),
                    action=s.get("action", ""),
                    observation=s.get("observation", ""),
                )
                for s in steps
            ],
            outcome=outcome,
        )

        return await b.SynthesizeExplanation(
            chain=chain,
            baml_options=self._baml_options if self._baml_options else {},
        )
