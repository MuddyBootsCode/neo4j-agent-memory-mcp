"""Tests for BamlReasoningExtractor."""

from unittest.mock import AsyncMock, MagicMock


class TestBamlReasoningExtractorEmptyInput:
    """Verify empty/whitespace text returns empty result without calling BAML."""

    async def test_empty_string(self):
        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        result = await extractor.extract_reasoning("")
        assert result["task"] == ""
        assert result["steps"] == []
        assert result["final_conclusion"] == ""
        assert result["success"] is False

    async def test_whitespace_only(self):
        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        result = await extractor.extract_reasoning("   \n  ")
        assert result["task"] == ""
        assert result["steps"] == []

    async def test_none_text(self):
        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        result = await extractor.extract_reasoning(None)
        assert result["task"] == ""
        assert result["steps"] == []


class TestBamlReasoningExtractorExtraction:
    """Verify BAML extraction converts results properly."""

    async def test_extract_reasoning(self, monkeypatch):
        # Mock BAML response
        step_mock = MagicMock()
        step_mock.thought = "Maybe the config file is wrong"
        step_mock.action = "Read config.yaml"
        step_mock.observation = "Found a typo in the database URL"
        step_mock.alternatives_considered = "Could also be a network issue"
        step_mock.confidence = 0.9

        result_mock = MagicMock()
        result_mock.task = "Debug connection error"
        result_mock.steps = [step_mock]
        result_mock.final_conclusion = "Fixed typo in config"
        result_mock.success = True

        mock_fn = AsyncMock(return_value=result_mock)
        monkeypatch.setattr(
            "neo4j_agent_memory.baml_client.async_client.b.ExtractReasoning",
            mock_fn,
        )

        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        result = await extractor.extract_reasoning("Some conversation text")

        assert result["task"] == "Debug connection error"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["thought"] == "Maybe the config file is wrong"
        assert result["steps"][0]["action"] == "Read config.yaml"
        assert result["steps"][0]["observation"] == "Found a typo in the database URL"
        assert result["steps"][0]["alternatives_considered"] == "Could also be a network issue"
        assert result["steps"][0]["confidence"] == 0.9
        assert result["final_conclusion"] == "Fixed typo in config"
        assert result["success"] is True

        mock_fn.assert_called_once_with(text="Some conversation text", baml_options={})

    async def test_confidence_clamped(self, monkeypatch):
        """Confidence values outside [0,1] are clamped."""
        step_mock = MagicMock()
        step_mock.thought = "test"
        step_mock.action = "test"
        step_mock.observation = "test"
        step_mock.alternatives_considered = None
        step_mock.confidence = 1.5

        result_mock = MagicMock()
        result_mock.task = "test"
        result_mock.steps = [step_mock]
        result_mock.final_conclusion = "done"
        result_mock.success = True

        monkeypatch.setattr(
            "neo4j_agent_memory.baml_client.async_client.b.ExtractReasoning",
            AsyncMock(return_value=result_mock),
        )

        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        result = await extractor.extract_reasoning("test text")

        assert result["steps"][0]["confidence"] == 1.0

    async def test_confidence_clamped_negative(self, monkeypatch):
        """Negative confidence values are clamped to 0."""
        step_mock = MagicMock()
        step_mock.thought = "test"
        step_mock.action = "test"
        step_mock.observation = "test"
        step_mock.alternatives_considered = None
        step_mock.confidence = -0.5

        result_mock = MagicMock()
        result_mock.task = "test"
        result_mock.steps = [step_mock]
        result_mock.final_conclusion = "done"
        result_mock.success = True

        monkeypatch.setattr(
            "neo4j_agent_memory.baml_client.async_client.b.ExtractReasoning",
            AsyncMock(return_value=result_mock),
        )

        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        result = await extractor.extract_reasoning("test text")

        assert result["steps"][0]["confidence"] == 0.0


class TestBamlReasoningExtractorSynthesis:
    """Verify SynthesizeExplanation integration."""

    async def test_synthesize_explanation(self, monkeypatch):
        mock_fn = AsyncMock(return_value="I first checked the config and found a typo.")
        monkeypatch.setattr(
            "neo4j_agent_memory.baml_client.async_client.b.SynthesizeExplanation",
            mock_fn,
        )

        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        result = await extractor.synthesize_explanation(
            task="Fix connection error",
            steps=[
                {"thought": "Check config", "action": "Read file", "observation": "Found typo"},
            ],
            outcome="Fixed the typo",
        )

        assert result == "I first checked the config and found a typo."
        mock_fn.assert_called_once()


class TestBamlReasoningExtractorClientOptions:
    """Verify client name override behavior."""

    def test_default_client_name(self):
        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        extractor = BamlReasoningExtractor()
        assert extractor._client_name == "Bedrock"
        assert extractor._baml_options == {}

    def test_custom_client_registry(self):
        from neo4j_agent_memory.extraction.reasoning_extractor import BamlReasoningExtractor

        registry = MagicMock()
        extractor = BamlReasoningExtractor(client_registry=registry)
        assert extractor._baml_options["client_registry"] is registry
