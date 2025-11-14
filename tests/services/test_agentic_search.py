"""
Unit tests for agentic search service.

Tests the AgenticSearchService that implements the complete agentic search pipeline
with LLM-driven decisions for iterative search refinement.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.exceptions import DatabaseError, LLMError, MCPToolError
from services.agentic_models import (
    ActionType,
    AgenticSearchResult,
    CompletenessEvaluation,
    QueryRefinement,
    RAGResult,
    SearchIteration,
    SearchStatus,
    URLRanking,
    URLRankingList,
)
from services.agentic_search import (
    AgenticSearchService,
    agentic_search_impl,
    get_agentic_search_service,
)


# Helper to patch dependencies for AgenticSearchService
@pytest.fixture
def mock_agentic_dependencies():
    """Patch OpenAIModel and Agent for AgenticSearchService."""
    with patch("services.agentic_search.config.settings") as mock_settings:
        mock_settings.model_choice = "gpt-4o-mini"
        mock_settings.openai_api_key = "test-key"
        mock_settings.agentic_search_llm_temperature = 0.3
        mock_settings.agentic_search_completeness_threshold = 0.85
        mock_settings.agentic_search_max_iterations = 3
        mock_settings.agentic_search_max_urls_per_iteration = 5
        mock_settings.agentic_search_max_pages_per_iteration = 50
        mock_settings.agentic_search_url_score_threshold = 0.7
        mock_settings.agentic_search_use_search_hints = False
        mock_settings.agentic_search_enable_url_filtering = True
        mock_settings.agentic_search_max_urls_to_rank = 20
        mock_settings.agentic_search_max_qdrant_results = 10

        with patch("services.agentic_search.config.OpenAIModel"):
            with patch("services.agentic_search.config.Agent") as mock_agent:
                mock_agent.return_value = MagicMock()
                yield mock_settings


class TestAgenticSearchServiceInitialization:
    """Test AgenticSearchService initialization and agent setup."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for initialization."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            yield mock_settings

    def test_service_initialization(self, mock_settings):
        """Test service initializes with correct configuration."""
        # Mock OpenAIModel and Agent to avoid API key validation
        with patch("services.agentic_search.config.OpenAIModel") as mock_openai_model:
            with patch("services.agentic_search.config.Agent") as mock_agent:
                mock_openai_model.return_value = MagicMock()
                mock_agent.return_value = MagicMock()

                service = AgenticSearchService()

                # Check configuration parameters
                assert service.model_name == "gpt-4o-mini"
                assert service.temperature == 0.3
                assert service.completeness_threshold == 0.85
                assert service.max_iterations == 3
                assert service.max_urls_per_iteration == 5
                assert service.max_pages_per_iteration == 50
                assert service.url_score_threshold == 0.7
                assert service.use_search_hints is False
                assert service.enable_url_filtering is True
                assert service.max_urls_to_rank == 20
                assert service.max_qdrant_results == 10

                # Check agents are initialized
                assert service.completeness_agent is not None
                assert service.ranking_agent is not None
                assert service.openai_model is not None
                assert service.base_model_settings is not None
                assert service.refinement_model_settings is not None

    def test_service_agents_have_correct_settings(self, mock_settings):
        """Test that agents are configured with correct model settings."""
        with patch("services.agentic_search.config.OpenAIModel"):
            with patch("services.agentic_search.config.Agent"):
                service = AgenticSearchService()

                # Base model settings
                assert service.base_model_settings.temperature == 0.3
                assert service.base_model_settings.timeout == 60

                # Refinement settings (more creative)
                assert service.refinement_model_settings.temperature == 0.5
                assert service.refinement_model_settings.timeout == 60


class TestCompletenessEvaluation:
    """Test completeness evaluation stage."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.fixture
    def sample_rag_results(self):
        """Sample RAG results."""
        return [
            RAGResult(
                content="Python is a high-level programming language",
                url="https://example.com/python",
                similarity_score=0.92,
                chunk_index=0,
            ),
            RAGResult(
                content="Python supports multiple programming paradigms",
                url="https://example.com/python",
                similarity_score=0.88,
                chunk_index=1,
            ),
        ]

    @pytest.mark.asyncio
    async def test_evaluate_completeness_success(self, mock_service, sample_rag_results):
        """Test successful completeness evaluation."""
        # Mock the agent run
        mock_result = MagicMock()
        mock_result.output = CompletenessEvaluation(
            score=0.85,
            reasoning="Information is comprehensive and answers the query fully",
            gaps=[],
        )

        with patch.object(mock_service.completeness_agent, "run", return_value=mock_result):
            evaluation = await mock_service._evaluate_completeness(
                query="What is Python?",
                results=sample_rag_results,
            )

            assert evaluation.score == 0.85
            assert "comprehensive" in evaluation.reasoning.lower()
            assert len(evaluation.gaps) == 0

    @pytest.mark.asyncio
    async def test_evaluate_completeness_with_gaps(self, mock_service, sample_rag_results):
        """Test completeness evaluation with knowledge gaps."""
        mock_result = MagicMock()
        mock_result.output = CompletenessEvaluation(
            score=0.5,
            reasoning="Partial information available, missing key details",
            gaps=["Performance characteristics", "Memory management"],
        )

        with patch.object(mock_service.completeness_agent, "run", return_value=mock_result):
            evaluation = await mock_service._evaluate_completeness(
                query="What is Python?",
                results=sample_rag_results,
            )

            assert evaluation.score == 0.5
            assert len(evaluation.gaps) == 2
            assert "Performance characteristics" in evaluation.gaps

    @pytest.mark.asyncio
    async def test_evaluate_completeness_llm_error(self, mock_service, sample_rag_results):
        """Test completeness evaluation with LLM error."""
        with patch.object(
            mock_service.completeness_agent,
            "run",
            side_effect=UnexpectedModelBehavior("Model failed after retries"),
        ):
            with pytest.raises(LLMError, match="LLM completeness evaluation failed"):
                await mock_service._evaluate_completeness(
                    query="What is Python?",
                    results=sample_rag_results,
                )

    @pytest.mark.asyncio
    async def test_evaluate_completeness_unexpected_error(self, mock_service, sample_rag_results):
        """Test completeness evaluation with unexpected error."""
        with patch.object(
            mock_service.completeness_agent,
            "run",
            side_effect=Exception("Unexpected error"),
        ):
            with pytest.raises(LLMError, match="Completeness evaluation failed"):
                await mock_service._evaluate_completeness(
                    query="What is Python?",
                    results=sample_rag_results,
                )


class TestStage1LocalCheck:
    """Test Stage 1: Local knowledge check."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.fixture
    def mock_ctx(self):
        """Mock FastMCP context."""
        return MagicMock()

    @pytest.fixture
    def mock_app_context(self):
        """Mock application context with database client."""
        mock_ctx = MagicMock()
        mock_ctx.database_client = AsyncMock()
        mock_ctx.database_client.search_code_examples = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_stage1_local_check_success(self, mock_service, mock_ctx, mock_app_context):
        """Test successful local knowledge check."""
        # Mock get_app_context
        with patch("services.agentic_search.get_app_context", return_value=mock_app_context):
            # Mock perform_rag_query
            rag_response = json.dumps({
                "success": True,
                "results": [
                    {
                        "chunk": "Python is a high-level programming language",
                        "url": "https://example.com/python",
                        "similarity_score": 0.92,
                        "chunk_index": 0,
                    },
                ],
            })

            with patch("services.agentic_search.perform_rag_query", return_value=rag_response):
                # Mock completeness evaluation
                with patch.object(
                    mock_service,
                    "_evaluate_completeness",
                    return_value=CompletenessEvaluation(
                        score=0.85,
                        reasoning="Good information",
                        gaps=[],
                    ),
                ):
                    search_history = []
                    evaluation, results = await mock_service._stage1_local_check(
                        ctx=mock_ctx,
                        query="What is Python?",
                        iteration=1,
                        search_history=search_history,
                    )

                    assert evaluation.score == 0.85
                    assert len(results) == 1
                    assert results[0].content == "Python is a high-level programming language"
                    assert len(search_history) == 1
                    assert search_history[0].action == ActionType.LOCAL_CHECK

    @pytest.mark.asyncio
    async def test_stage1_local_check_no_database(self, mock_service, mock_ctx):
        """Test local check with no database client available."""
        with patch("services.agentic_search.get_app_context", return_value=None):
            with pytest.raises(MCPToolError, match="Database client not available"):
                await mock_service._stage1_local_check(
                    ctx=mock_ctx,
                    query="test query",
                    iteration=1,
                    search_history=[],
                )

    @pytest.mark.asyncio
    async def test_stage1_local_check_is_recheck(self, mock_service, mock_ctx, mock_app_context):
        """Test local check with is_recheck flag (should not add to history)."""
        with patch("services.agentic_search.get_app_context", return_value=mock_app_context):
            rag_response = json.dumps({"success": True, "results": []})

            with patch("services.agentic_search.perform_rag_query", return_value=rag_response):
                with patch.object(
                    mock_service,
                    "_evaluate_completeness",
                    return_value=CompletenessEvaluation(score=0.5, reasoning="Test", gaps=[]),
                ):
                    search_history = []
                    await mock_service._stage1_local_check(
                        ctx=mock_ctx,
                        query="test",
                        iteration=1,
                        search_history=search_history,
                        is_recheck=True,
                    )

                    # Should not add to history on recheck
                    assert len(search_history) == 0


class TestURLRanking:
    """Test URL ranking stage."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results from SearXNG."""
        return [
            {
                "title": "Python Tutorial",
                "url": "https://example.com/python-tutorial",
                "snippet": "Learn Python programming from scratch",
            },
            {
                "title": "Python Documentation",
                "url": "https://docs.python.org",
                "snippet": "Official Python documentation",
            },
        ]

    @pytest.mark.asyncio
    async def test_rank_urls_success(self, mock_service, sample_search_results):
        """Test successful URL ranking."""
        mock_result = MagicMock()
        mock_result.output = URLRankingList(
            rankings=[
                URLRanking(
                    url="https://docs.python.org",
                    title="Python Documentation",
                    snippet="Official Python documentation",
                    score=0.9,
                    reasoning="Official documentation is highly relevant",
                ),
                URLRanking(
                    url="https://example.com/python-tutorial",
                    title="Python Tutorial",
                    snippet="Learn Python programming from scratch",
                    score=0.7,
                    reasoning="Tutorial is somewhat relevant",
                ),
            ],
        )

        with patch.object(mock_service.ranking_agent, "run", return_value=mock_result):
            rankings = await mock_service._rank_urls(
                query="Python documentation",
                gaps=["Installation guide"],
                search_results=sample_search_results,
            )

            assert len(rankings) == 2
            # Should be sorted by score descending
            assert rankings[0].score == 0.9
            assert rankings[0].url == "https://docs.python.org"
            assert rankings[1].score == 0.7

    @pytest.mark.asyncio
    async def test_rank_urls_llm_error(self, mock_service, sample_search_results):
        """Test URL ranking with LLM error."""
        with patch.object(
            mock_service.ranking_agent,
            "run",
            side_effect=UnexpectedModelBehavior("Model failed"),
        ):
            with pytest.raises(LLMError, match="LLM URL ranking failed"):
                await mock_service._rank_urls(
                    query="test",
                    gaps=[],
                    search_results=sample_search_results,
                )

    @pytest.mark.asyncio
    async def test_rank_urls_unexpected_error(self, mock_service, sample_search_results):
        """Test URL ranking with unexpected error."""
        with patch.object(
            mock_service.ranking_agent,
            "run",
            side_effect=Exception("Unexpected error"),
        ):
            with pytest.raises(LLMError, match="URL ranking failed"):
                await mock_service._rank_urls(
                    query="test",
                    gaps=[],
                    search_results=sample_search_results,
                )


class TestStage2WebSearch:
    """Test Stage 2: Web search and URL ranking."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.mark.asyncio
    async def test_stage2_web_search_success(self, mock_service):
        """Test successful web search and ranking."""
        search_results = [
            {"title": "Result 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
            {"title": "Result 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
        ]

        rankings = [
            URLRanking(
                url="https://example.com/1",
                title="Result 1",
                snippet="Snippet 1",
                score=0.9,
                reasoning="Highly relevant",
            ),
            URLRanking(
                url="https://example.com/2",
                title="Result 2",
                snippet="Snippet 2",
                score=0.6,
                reasoning="Somewhat relevant",
            ),
        ]

        with patch("services.agentic_search._search_searxng", return_value=search_results):
            with patch.object(mock_service, "_rank_urls", return_value=rankings):
                search_history = []
                promising_urls = await mock_service._stage2_web_search(
                    query="test query",
                    gaps=["gap1"],
                    max_urls=5,
                    url_threshold=0.7,
                    iteration=1,
                    search_history=search_history,
                )

                # Only URL with score >= 0.7 should be included
                assert len(promising_urls) == 1
                assert promising_urls[0] == "https://example.com/1"
                assert len(search_history) == 1
                assert search_history[0].action == ActionType.WEB_SEARCH

    @pytest.mark.asyncio
    async def test_stage2_web_search_no_results(self, mock_service):
        """Test web search with no results."""
        with patch("services.agentic_search._search_searxng", return_value=[]):
            search_history = []
            promising_urls = await mock_service._stage2_web_search(
                query="test query",
                gaps=[],
                max_urls=5,
                url_threshold=0.7,
                iteration=1,
                search_history=search_history,
            )

            assert len(promising_urls) == 0
            assert len(search_history) == 1
            assert search_history[0].urls_found == 0

    @pytest.mark.asyncio
    async def test_stage2_web_search_limits_urls_to_rank(self, mock_service):
        """Test that web search limits URLs sent to ranking."""
        # Create 30 search results
        search_results = [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"Snippet {i}"}
            for i in range(30)
        ]

        with patch("services.agentic_search._search_searxng", return_value=search_results):
            with patch.object(mock_service, "_rank_urls", return_value=[]) as mock_rank:
                await mock_service._stage2_web_search(
                    query="test",
                    gaps=[],
                    max_urls=5,
                    url_threshold=0.7,
                    iteration=1,
                    search_history=[],
                )

                # Should only rank max_urls_to_rank (20) URLs
                call_args = mock_rank.call_args[1]
                assert len(call_args["search_results"]) == 20


class TestStage3SelectiveCrawl:
    """Test Stage 3: Selective crawling."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.fixture
    def mock_ctx(self):
        """Mock FastMCP context."""
        return MagicMock()

    @pytest.fixture
    def mock_app_context(self):
        """Mock application context."""
        mock_ctx = MagicMock()
        mock_ctx.database_client = AsyncMock()
        mock_ctx.database_client.url_exists = AsyncMock(return_value=False)
        return mock_ctx

    @pytest.mark.asyncio
    async def test_stage3_selective_crawl_success(self, mock_service, mock_ctx, mock_app_context):
        """Test successful selective crawling."""
        urls = ["https://example.com/1", "https://example.com/2"]
        crawl_result = {
            "urls_crawled": 2,
            "urls_stored": 2,
            "chunks_stored": 10,
            "urls_filtered": 0,
        }

        with patch("services.agentic_search.get_app_context", return_value=mock_app_context):
            with patch(
                "services.agentic_search.crawl_urls_for_agentic_search",
                return_value=crawl_result,
            ):
                search_history = []
                urls_stored = await mock_service._stage3_selective_crawl(
                    ctx=mock_ctx,
                    urls=urls,
                    query="test query",
                    use_hints=False,
                    iteration=1,
                    search_history=search_history,
                )

                assert urls_stored == 2
                assert len(search_history) == 1
                assert search_history[0].action == ActionType.CRAWL
                assert search_history[0].urls_stored == 2

    @pytest.mark.asyncio
    async def test_stage3_selective_crawl_duplicate_detection(
        self, mock_service, mock_ctx, mock_app_context
    ):
        """Test duplicate URL detection."""
        urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]

        # Mock url_exists to return True for first URL
        async def url_exists_side_effect(url):
            return url == "https://example.com/1"

        mock_app_context.database_client.url_exists = AsyncMock(
            side_effect=url_exists_side_effect,
        )

        crawl_result = {"urls_crawled": 2, "urls_stored": 2, "chunks_stored": 10, "urls_filtered": 0}

        with patch("services.agentic_search.get_app_context", return_value=mock_app_context):
            with patch(
                "services.agentic_search.crawl_urls_for_agentic_search",
                return_value=crawl_result,
            ) as mock_crawl:
                await mock_service._stage3_selective_crawl(
                    ctx=mock_ctx,
                    urls=urls,
                    query="test",
                    use_hints=False,
                    iteration=1,
                    search_history=[],
                )

                # Should only crawl 2 URLs (skipping the duplicate)
                call_args = mock_crawl.call_args[1]
                assert len(call_args["urls"]) == 2
                assert "https://example.com/1" not in call_args["urls"]

    @pytest.mark.asyncio
    async def test_stage3_selective_crawl_all_duplicates(
        self, mock_service, mock_ctx, mock_app_context
    ):
        """Test when all URLs are duplicates."""
        urls = ["https://example.com/1", "https://example.com/2"]

        # All URLs exist
        mock_app_context.database_client.url_exists = AsyncMock(return_value=True)

        with patch("services.agentic_search.get_app_context", return_value=mock_app_context):
            with patch(
                "services.agentic_search.crawl_urls_for_agentic_search",
            ) as mock_crawl:
                urls_stored = await mock_service._stage3_selective_crawl(
                    ctx=mock_ctx,
                    urls=urls,
                    query="test",
                    use_hints=False,
                    iteration=1,
                    search_history=[],
                )

                # Should not call crawl
                mock_crawl.assert_not_called()
                assert urls_stored == 0

    @pytest.mark.asyncio
    async def test_stage3_selective_crawl_database_error_fail_open(
        self, mock_service, mock_ctx, mock_app_context
    ):
        """Test that database errors fail open (include URL)."""
        urls = ["https://example.com/1"]

        # Database error on duplicate check
        mock_app_context.database_client.url_exists = AsyncMock(
            side_effect=DatabaseError("DB error"),
        )

        crawl_result = {"urls_crawled": 1, "urls_stored": 1, "chunks_stored": 5, "urls_filtered": 0}

        with patch("services.agentic_search.get_app_context", return_value=mock_app_context):
            with patch(
                "services.agentic_search.crawl_urls_for_agentic_search",
                return_value=crawl_result,
            ) as mock_crawl:
                await mock_service._stage3_selective_crawl(
                    ctx=mock_ctx,
                    urls=urls,
                    query="test",
                    use_hints=False,
                    iteration=1,
                    search_history=[],
                )

                # Should still crawl (fail open)
                call_args = mock_crawl.call_args[1]
                assert len(call_args["urls"]) == 1


class TestStage4QueryRefinement:
    """Test Stage 4: Query refinement."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.mark.asyncio
    async def test_stage4_query_refinement_success(self, mock_service):
        """Test successful query refinement."""
        # Mock Agent class and its instance
        mock_agent = AsyncMock()
        mock_result = MagicMock()
        mock_result.output = MagicMock()
        mock_result.output.refined_queries = [
            "Python performance optimization",
            "Python memory management techniques",
        ]
        mock_result.output.reasoning = "Focusing on performance aspects"
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("services.agentic_search.config.Agent", return_value=mock_agent):
            refinement = await mock_service._stage4_query_refinement(
                original_query="Python programming",
                current_query="Python basics",
                gaps=["Performance optimization", "Memory management"],
            )

            assert refinement.original_query == "Python programming"
            assert refinement.current_query == "Python basics"
            assert len(refinement.refined_queries) == 2
            assert "performance" in refinement.refined_queries[0].lower()
            assert "Focusing on performance aspects" == refinement.reasoning

    @pytest.mark.asyncio
    async def test_stage4_query_refinement_llm_error(self, mock_service):
        """Test query refinement with LLM error."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=UnexpectedModelBehavior("Model failed"))

        with patch("services.agentic_search.config.Agent", return_value=mock_agent):
            with pytest.raises(LLMError, match="LLM query refinement failed"):
                await mock_service._stage4_query_refinement(
                    original_query="test",
                    current_query="test",
                    gaps=[],
                )

    @pytest.mark.asyncio
    async def test_stage4_query_refinement_unexpected_error(self, mock_service):
        """Test query refinement with unexpected error."""
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Unexpected error"))

        with patch("services.agentic_search.config.Agent", return_value=mock_agent):
            with pytest.raises(LLMError, match="Query refinement failed"):
                await mock_service._stage4_query_refinement(
                    original_query="test",
                    current_query="test",
                    gaps=[],
                )


class TestExecuteSearch:
    """Test the complete execute_search pipeline."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.fixture
    def mock_ctx(self):
        """Mock FastMCP context."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_execute_search_threshold_met_first_iteration(self, mock_service, mock_ctx):
        """Test search completes on first iteration when threshold is met."""
        evaluation = CompletenessEvaluation(score=0.9, reasoning="Complete", gaps=[])
        rag_results = [
            RAGResult(content="Result 1", url="https://example.com", similarity_score=0.9, chunk_index=0),
        ]

        with patch.object(
            mock_service,
            "_stage1_local_check",
            return_value=(evaluation, rag_results),
        ):
            result = await mock_service.execute_search(
                ctx=mock_ctx,
                query="test query",
                completeness_threshold=0.85,
            )

            assert result.success is True
            assert result.status == SearchStatus.COMPLETE
            assert result.iterations == 1
            assert result.completeness == 0.9
            assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_execute_search_max_iterations_reached(self, mock_service, mock_ctx):
        """Test search reaches max iterations."""
        evaluation = CompletenessEvaluation(score=0.5, reasoning="Incomplete", gaps=["gap1"])
        rag_results = []

        with patch.object(
            mock_service,
            "_stage1_local_check",
            return_value=(evaluation, rag_results),
        ):
            with patch.object(mock_service, "_stage2_web_search", return_value=[]):
                result = await mock_service.execute_search(
                    ctx=mock_ctx,
                    query="test query",
                    max_iterations=2,
                )

                assert result.success is True
                assert result.status == SearchStatus.MAX_ITERATIONS_REACHED
                assert result.iterations == 2
                assert result.completeness == 0.5

    @pytest.mark.asyncio
    async def test_execute_search_with_crawling_and_recheck(self, mock_service, mock_ctx):
        """Test search with crawling and re-check."""
        # First check: incomplete
        evaluation1 = CompletenessEvaluation(score=0.5, reasoning="Incomplete", gaps=["gap1"])
        rag_results1 = []

        # After crawling: complete
        evaluation2 = CompletenessEvaluation(score=0.9, reasoning="Complete", gaps=[])
        rag_results2 = [
            RAGResult(content="New result", url="https://example.com", similarity_score=0.9, chunk_index=0),
        ]

        stage1_calls = [
            (evaluation1, rag_results1),  # First call
            (evaluation2, rag_results2),  # Re-check after crawling
        ]

        with patch.object(
            mock_service,
            "_stage1_local_check",
            side_effect=stage1_calls,
        ):
            with patch.object(
                mock_service,
                "_stage2_web_search",
                return_value=["https://example.com/page"],
            ):
                with patch.object(mock_service, "_stage3_selective_crawl", return_value=1):
                    result = await mock_service.execute_search(
                        ctx=mock_ctx,
                        query="test query",
                    )

                    assert result.success is True
                    assert result.status == SearchStatus.COMPLETE
                    assert result.completeness == 0.9

    @pytest.mark.asyncio
    async def test_execute_search_no_promising_urls_triggers_refinement(self, mock_service, mock_ctx):
        """Test that no promising URLs triggers query refinement."""
        evaluation = CompletenessEvaluation(score=0.5, reasoning="Incomplete", gaps=["gap1"])
        rag_results = []

        refinement = QueryRefinement(
            original_query="test query",
            current_query="test query",
            refined_queries=["refined test query"],
            reasoning="More specific",
        )

        with patch.object(
            mock_service,
            "_stage1_local_check",
            return_value=(evaluation, rag_results),
        ):
            with patch.object(mock_service, "_stage2_web_search", return_value=[]):
                with patch.object(mock_service, "_stage4_query_refinement", return_value=refinement):
                    result = await mock_service.execute_search(
                        ctx=mock_ctx,
                        query="test query",
                        max_iterations=2,
                    )

                    assert result.iterations == 2
                    # Should have attempted refinement

    @pytest.mark.asyncio
    async def test_execute_search_score_improvement_skips_refinement(self, mock_service, mock_ctx):
        """Test that significant score improvement skips refinement."""
        # First iteration: low score
        evaluation1 = CompletenessEvaluation(score=0.3, reasoning="Low", gaps=["gap1"])
        rag_results1 = []

        # After crawling: improved score but still below threshold
        evaluation2 = CompletenessEvaluation(score=0.6, reasoning="Improved", gaps=["gap2"])
        rag_results2 = []

        stage1_calls = [
            (evaluation1, rag_results1),
            (evaluation2, rag_results2),
            (evaluation2, rag_results2),  # Next iteration
        ]

        with patch.object(
            mock_service,
            "_stage1_local_check",
            side_effect=stage1_calls,
        ):
            with patch.object(
                mock_service,
                "_stage2_web_search",
                return_value=["https://example.com"],
            ):
                with patch.object(mock_service, "_stage3_selective_crawl", return_value=1):
                    with patch.object(
                        mock_service,
                        "_stage4_query_refinement",
                    ) as mock_refinement:
                        result = await mock_service.execute_search(
                            ctx=mock_ctx,
                            query="test",
                            max_iterations=2,
                        )

                        # Should skip refinement due to score improvement (0.3 -> 0.6 = +0.3)
                        # This is above SCORE_IMPROVEMENT_THRESHOLD (0.1)
                        mock_refinement.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_search_handles_exception(self, mock_service, mock_ctx):
        """Test that execute_search handles exceptions gracefully."""
        with patch.object(
            mock_service,
            "_stage1_local_check",
            side_effect=Exception("Unexpected error"),
        ):
            result = await mock_service.execute_search(
                ctx=mock_ctx,
                query="test query",
            )

            assert result.success is False
            assert result.status == SearchStatus.ERROR
            assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_execute_search_with_parameter_overrides(self, mock_service, mock_ctx):
        """Test execute_search with parameter overrides."""
        evaluation = CompletenessEvaluation(score=0.95, reasoning="Complete", gaps=[])
        rag_results = []

        with patch.object(
            mock_service,
            "_stage1_local_check",
            return_value=(evaluation, rag_results),
        ):
            result = await mock_service.execute_search(
                ctx=mock_ctx,
                query="test",
                completeness_threshold=0.9,
                max_iterations=5,
                max_urls_per_iteration=10,
                url_score_threshold=0.8,
                use_search_hints=True,
            )

            # Should use overridden threshold
            assert result.completeness >= 0.9


class TestAgenticSearchImpl:
    """Test the main entry point agentic_search_impl."""

    @pytest.fixture
    def mock_ctx(self):
        """Mock FastMCP context."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_agentic_search_impl_success(self, mock_ctx):
        """Test successful agentic search implementation."""
        mock_result = AgenticSearchResult(
            success=True,
            query="test query",
            iterations=1,
            completeness=0.9,
            results=[],
            search_history=[],
            status=SearchStatus.COMPLETE,
        )

        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.agentic_search_enabled = True

            with patch(
                "services.agentic_search.get_agentic_search_service",
            ) as mock_get_service:
                mock_service = AsyncMock()
                mock_service.execute_search = AsyncMock(return_value=mock_result)
                mock_get_service.return_value = mock_service

                result_json = await agentic_search_impl(ctx=mock_ctx, query="test query")
                result = json.loads(result_json)

                assert result["success"] is True
                assert result["query"] == "test query"
                assert result["completeness"] == 0.9

    @pytest.mark.asyncio
    async def test_agentic_search_impl_disabled(self, mock_ctx):
        """Test agentic search when disabled."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.agentic_search_enabled = False

            with pytest.raises(MCPToolError, match="Agentic search is not enabled"):
                await agentic_search_impl(ctx=mock_ctx, query="test")

    @pytest.mark.asyncio
    async def test_agentic_search_impl_handles_exception(self, mock_ctx):
        """Test agentic search implementation handles exceptions."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.agentic_search_enabled = True

            with patch(
                "services.agentic_search.get_agentic_search_service",
                side_effect=Exception("Service error"),
            ):
                with pytest.raises(MCPToolError, match="Agentic search failed"):
                    await agentic_search_impl(ctx=mock_ctx, query="test")


class TestSingletonPattern:
    """Test the singleton pattern for service instances."""

    def test_get_agentic_search_service_singleton(self):
        """Test that get_agentic_search_service returns singleton."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10

            # Reset singleton
            import services.agentic_search
            services.agentic_search._agentic_search_service = None

            service1 = get_agentic_search_service()
            service2 = get_agentic_search_service()

            # Should be the same instance
            assert service1 is service2


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked agents."""
        with patch("services.agentic_search.config.settings") as mock_settings:
            mock_settings.model_choice = "gpt-4o-mini"
            mock_settings.openai_api_key = "test-key"
            mock_settings.agentic_search_llm_temperature = 0.3
            mock_settings.agentic_search_completeness_threshold = 0.85
            mock_settings.agentic_search_max_iterations = 3
            mock_settings.agentic_search_max_urls_per_iteration = 5
            mock_settings.agentic_search_max_pages_per_iteration = 50
            mock_settings.agentic_search_url_score_threshold = 0.7
            mock_settings.agentic_search_use_search_hints = False
            mock_settings.agentic_search_enable_url_filtering = True
            mock_settings.agentic_search_max_urls_to_rank = 20
            mock_settings.agentic_search_max_qdrant_results = 10
            
            with patch("services.agentic_search.config.OpenAIModel"):
                with patch("services.agentic_search.config.Agent"):
                    service = AgenticSearchService()
                    yield service

    @pytest.mark.asyncio
    async def test_evaluate_completeness_empty_results(self, mock_service):
        """Test completeness evaluation with empty results."""
        mock_result = MagicMock()
        mock_result.output = CompletenessEvaluation(
            score=0.0,
            reasoning="No information available",
            gaps=["All information missing"],
        )

        with patch.object(mock_service.completeness_agent, "run", return_value=mock_result):
            evaluation = await mock_service._evaluate_completeness(query="test", results=[])

            assert evaluation.score == 0.0
            assert len(evaluation.gaps) >= 1

    @pytest.mark.asyncio
    async def test_rank_urls_empty_search_results(self, mock_service):
        """Test URL ranking with empty search results."""
        mock_result = MagicMock()
        mock_result.output = URLRankingList(rankings=[])

        with patch.object(mock_service.ranking_agent, "run", return_value=mock_result):
            rankings = await mock_service._rank_urls(
                query="test",
                gaps=[],
                search_results=[],
            )

            assert len(rankings) == 0

    @pytest.mark.asyncio
    async def test_stage3_crawl_no_content_stored(self, mock_service):
        """Test stage 3 when no content is stored."""
        mock_ctx = MagicMock()
        mock_app_context = MagicMock()
        mock_app_context.database_client = AsyncMock()
        mock_app_context.database_client.url_exists = AsyncMock(return_value=False)

        crawl_result = {"urls_crawled": 1, "urls_stored": 0, "chunks_stored": 0, "urls_filtered": 1}

        with patch("services.agentic_search.get_app_context", return_value=mock_app_context):
            with patch(
                "services.agentic_search.crawl_urls_for_agentic_search",
                return_value=crawl_result,
            ):
                urls_stored = await mock_service._stage3_selective_crawl(
                    ctx=mock_ctx,
                    urls=["https://example.com"],
                    query="test",
                    use_hints=False,
                    iteration=1,
                    search_history=[],
                )

                assert urls_stored == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
