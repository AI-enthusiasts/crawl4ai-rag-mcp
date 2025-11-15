"""Integration tests for agentic search with real OpenAI API.

This test suite uses gpt-4.1-nano (CHEAP model ~$0.00015/1K tokens) for
cost-effective integration testing WITHOUT mocks.

Prerequisites:
    - OPENAI_API_KEY or TEST_OPENAI_API_KEY must be set
    - TEST_MODEL_CHOICE=gpt-4.1-nano (or falls back to this)
    - Qdrant running at localhost:6333
    - SearXNG running at localhost:8080 (optional for full test)
    - AGENTIC_SEARCH_ENABLED=true

Cost estimate per test run: ~$0.001-0.002 USD with gpt-4.1-nano

Run with: pytest tests/test_agentic_search_integration.py -v
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from config import get_settings, reset_settings
from core.context import Crawl4AIContext, initialize_global_context
from services.agentic_search import (
    AgenticSearchConfig,
    AgenticSearchService,
    SelectiveCrawler,
    LocalKnowledgeEvaluator,
    URLRanker,
    agentic_search_impl,
    get_agentic_search_service,
)

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_settings():
    """Configure settings for integration tests."""
    # Reset settings to reload from environment
    reset_settings()

    # Set test environment variables
    os.environ["AGENTIC_SEARCH_ENABLED"] = "true"
    os.environ["AGENTIC_SEARCH_COMPLETENESS_THRESHOLD"] = "0.8"  # Lower for testing
    os.environ["AGENTIC_SEARCH_MAX_ITERATIONS"] = "2"  # Fewer iterations for speed
    os.environ["AGENTIC_SEARCH_MAX_URLS_PER_ITERATION"] = "2"
    os.environ["AGENTIC_SEARCH_URL_SCORE_THRESHOLD"] = "0.6"

    # Use test model if available, fallback to gpt-4.1-nano
    if not os.getenv("TEST_MODEL_CHOICE"):
        os.environ["MODEL_CHOICE"] = "gpt-4.1-nano"
    else:
        os.environ["MODEL_CHOICE"] = os.getenv("TEST_MODEL_CHOICE")

    # Use test API key if available
    if os.getenv("TEST_OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("TEST_OPENAI_API_KEY")

    settings = get_settings()

    # Validate required settings
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")

    yield settings

    # Cleanup
    reset_settings()


@pytest.fixture
async def app_context(test_settings):
    """Initialize application context for tests."""
    try:
        ctx = await initialize_global_context()
        yield ctx
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")


@pytest.fixture
async def mock_fastmcp_context():
    """Create a mock FastMCP Context for testing."""
    ctx = MagicMock(spec=Context)
    return ctx


class TestAgenticSearchService:
    """Test AgenticSearchService with real OpenAI API."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, test_settings):
        """Test that service initializes correctly with Pydantic AI agents."""
        # Force settings reload to pick up test environment variables
        reset_settings()

        # Create components
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)
        ranker = URLRanker(config)
        crawler = SelectiveCrawler(config)

        # Create service
        service = AgenticSearchService(
            evaluator=evaluator,
            ranker=ranker,
            crawler=crawler,
            config=config,
        )

        # Verify components are set
        assert service.evaluator is not None
        assert service.ranker is not None
        assert service.crawler is not None
        assert service.openai_model is not None

        # Get fresh settings after reload
        current_settings = get_settings()
        assert service.model_name == current_settings.model_choice
        assert service.temperature == current_settings.agentic_search_llm_temperature
        assert (
            service.completeness_threshold
            == current_settings.agentic_search_completeness_threshold
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_completeness_evaluation_with_real_llm(
        self, test_settings, app_context
    ):
        """Test completeness evaluation with real gpt-4.1-nano API call.

        Cost: ~$0.0001 USD per call with gpt-4.1-nano
        """
        # Force settings reload
        reset_settings()

        # Create components
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)

        # Test with empty results - should score low
        from services.agentic_models import RAGResult

        empty_results = []
        evaluation = await evaluator._evaluate_completeness(
            query="What is Python?", results=empty_results
        )

        assert evaluation.score >= 0.0
        assert evaluation.score <= 1.0
        assert evaluation.reasoning
        assert isinstance(evaluation.gaps, list)
        # Empty results should have low completeness
        assert evaluation.score < 0.5

        # Test with mock results - should score higher
        mock_results = [
            RAGResult(
                content="Python is a high-level, interpreted programming language known for its simplicity and readability.",
                url="https://example.com/python",
                similarity_score=0.95,
                chunk_index=0,
            ),
            RAGResult(
                content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                url="https://example.com/python-features",
                similarity_score=0.92,
                chunk_index=0,
            ),
        ]

        evaluation_with_results = await evaluator._evaluate_completeness(
            query="What is Python?", results=mock_results
        )

        assert evaluation_with_results.score >= 0.0
        assert evaluation_with_results.score <= 1.0
        assert evaluation_with_results.reasoning
        # With relevant results, score should be higher
        assert evaluation_with_results.score > evaluation.score

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_url_ranking_with_real_llm(self, test_settings):
        """Test URL ranking with real gpt-4.1-nano API call.

        Cost: ~$0.0002 USD per call with gpt-4.1-nano
        """
        # Force settings reload
        reset_settings()

        # Create components
        config = AgenticSearchConfig()
        ranker = URLRanker(config)

        mock_search_results = [
            {
                "title": "Official Python Tutorial",
                "url": "https://docs.python.org/3/tutorial/",
                "snippet": "The Python Tutorial — Python 3.12 documentation. This tutorial introduces the reader informally to the basic concepts...",
            },
            {
                "title": "Python Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "snippet": "Python is a high-level, interpreted programming language with dynamic semantics...",
            },
            {
                "title": "Python Snake Care Guide",
                "url": "https://pets.example.com/python-care",
                "snippet": "Learn how to care for your pet python snake. Housing, feeding, and health tips...",
            },
        ]

        rankings = await ranker._rank_urls(
            query="Python programming language tutorial",
            gaps=["basic syntax", "getting started"],
            search_results=mock_search_results,
        )

        assert len(rankings) == 3
        assert all(0.0 <= r.score <= 1.0 for r in rankings)
        assert all(r.reasoning for r in rankings)

        # Should be sorted by score descending
        for i in range(len(rankings) - 1):
            assert rankings[i].score >= rankings[i + 1].score

        # Programming content should rank higher than snake care
        python_doc_score = next(r.score for r in rankings if "docs.python.org" in r.url)
        snake_care_score = next(r.score for r in rankings if "pets.example.com" in r.url)
        assert python_doc_score > snake_care_score

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_query_refinement_with_real_llm(self, test_settings):
        """Test query refinement with real gpt-4.1-nano API call.

        Cost: ~$0.0001 USD per call with gpt-4.1-nano
        """
        # Force settings reload
        reset_settings()

        # Create components
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)
        ranker = URLRanker(config)
        crawler = SelectiveCrawler(config)

        service = AgenticSearchService(
            evaluator=evaluator,
            ranker=ranker,
            crawler=crawler,
            config=config,
        )

        refinement = await service._stage4_query_refinement(
            original_query="What is Python?",
            current_query="What is Python?",
            gaps=["type system", "performance characteristics", "use cases"],
        )

        assert refinement.original_query == "What is Python?"
        assert refinement.current_query == "What is Python?"
        assert len(refinement.refined_queries) > 0
        assert len(refinement.refined_queries) <= 3
        assert refinement.reasoning

        # Refined queries should be different from original
        assert any(q != "What is Python?" for q in refinement.refined_queries)


class TestAgenticSearchIntegration:
    """Integration tests for full agentic search pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_agentic_search_with_mock_components(
        self, test_settings, app_context, mock_fastmcp_context
    ):
        """Test agentic search with mocked web search and crawling.

        This tests the core LLM evaluation logic without hitting external services.
        Cost: ~$0.0005 USD per run with gpt-4.1-nano
        """
        # Mock the web search to avoid hitting SearXNG
        with patch("services.agentic_search._search_searxng") as mock_search:
            mock_search.return_value = [
                {
                    "title": "Test Result",
                    "url": "https://example.com/test",
                    "snippet": "Test snippet",
                }
            ]

            # Mock the crawling to avoid hitting real URLs
            with patch("services.agentic_search.process_urls_for_mcp") as mock_crawl:
                mock_crawl.return_value = json.dumps(
                    {
                        "success": True,
                        "results": [
                            {
                                "success": True,
                                "url": "https://example.com/test",
                                "chunks_stored": 5,
                            }
                        ],
                    }
                )

                # Execute agentic search
                result_json = await agentic_search_impl(
                    ctx=mock_fastmcp_context,
                    query="What is FastMCP?",
                    completeness_threshold=0.9,  # High threshold to trigger search
                    max_iterations=1,  # Just one iteration for speed
                    max_urls_per_iteration=1,
                    url_score_threshold=0.5,
                )

                result = json.loads(result_json)

                # Verify result structure
                assert result["success"] is True
                assert result["query"] == "What is FastMCP?"
                assert result["iterations"] >= 1
                assert "completeness" in result
                assert "results" in result
                assert "search_history" in result
                assert result["status"] in [
                    "complete",
                    "max_iterations_reached",
                    "error",
                ]

                # Verify search history has expected actions
                assert len(result["search_history"]) > 0
                actions = [item["action"] for item in result["search_history"]]
                assert "local_check" in actions

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.getenv("RUN_EXPENSIVE_TESTS"),
        reason="Expensive test - set RUN_EXPENSIVE_TESTS=true to run",
    )
    async def test_full_agentic_search_pipeline(
        self, test_settings, app_context, mock_fastmcp_context
    ):
        """Test full agentic search pipeline with real services.

        WARNING: This test hits real OpenAI API, SearXNG, and crawls real URLs.
        Cost: ~$0.002-0.005 USD per run with gpt-4.1-nano

        Prerequisites:
            - Qdrant running
            - SearXNG running
            - Set RUN_EXPENSIVE_TESTS=true to enable
        """
        result_json = await agentic_search_impl(
            ctx=mock_fastmcp_context,
            query="What is pytest?",  # Simple query for testing
            completeness_threshold=0.85,
            max_iterations=2,
            max_urls_per_iteration=2,
            url_score_threshold=0.6,
        )

        result = json.loads(result_json)

        # Verify comprehensive result
        assert result["success"] is True
        assert result["query"] == "What is pytest?"
        assert result["iterations"] >= 1
        assert result["iterations"] <= 2
        assert 0.0 <= result["completeness"] <= 1.0
        assert isinstance(result["results"], list)
        assert len(result["search_history"]) > 0

        # Verify it attempted multiple stages
        actions = {item["action"] for item in result["search_history"]}
        assert "local_check" in actions  # Stage 1 always runs

        # If completeness was low, should have attempted web search
        if result["completeness"] < 0.85:
            assert "web_search" in actions or result["status"] == "max_iterations_reached"


@pytest.mark.asyncio
async def test_settings_validation_for_agentic_search():
    """Test that settings validation works correctly."""
    reset_settings()

    # Test with disabled agentic search
    os.environ["AGENTIC_SEARCH_ENABLED"] = "false"
    settings = get_settings()

    assert settings.agentic_search_enabled is False

    # Test with enabled agentic search
    os.environ["AGENTIC_SEARCH_ENABLED"] = "true"
    reset_settings()
    settings = get_settings()

    assert settings.agentic_search_enabled is True

    # Test threshold validation
    os.environ["AGENTIC_SEARCH_COMPLETENESS_THRESHOLD"] = "0.95"
    reset_settings()
    settings = get_settings()

    assert settings.agentic_search_completeness_threshold == 0.95

    # Test invalid threshold (should clamp to 0-1)
    os.environ["AGENTIC_SEARCH_COMPLETENESS_THRESHOLD"] = "1.5"
    reset_settings()
    settings = get_settings()

    assert settings.agentic_search_completeness_threshold <= 1.0

    # Cleanup
    reset_settings()


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_agentic_search_integration.py -v
    pytest.main([__file__, "-v", "-s"])
