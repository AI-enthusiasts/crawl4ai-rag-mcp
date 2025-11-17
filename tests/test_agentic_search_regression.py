"""Regression test for agentic search completeness evaluation failure.

This test reproduces the bug where agentic_search tool is available in mcpproxy
but fails with "Completeness evaluation failed" error.

Bug report:
- Tool found: crawl4ai-rag:agentic_search
- Error: Completeness evaluation failed
- Response format:
  {
    "success": false,
    "query": "What is LLM reasoning?",
    "iterations": 1,
    "completeness": 0.0,
    "results": [],
    "search_history": [],
    "status": "error",
    "error": "Completeness evaluation failed"
  }

This can happen when:
1. OpenAI API is unavailable/misconfigured
2. LLM returns invalid response that fails Pydantic validation
3. Network timeout during LLM call
4. Rate limiting or authentication errors
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from src.core.exceptions import LLMError


# Mark as integration test
pytestmark = pytest.mark.integration


@pytest.fixture
def mock_fastmcp_context():
    """Create a mock FastMCP Context for testing."""
    ctx = MagicMock(spec=Context)
    return ctx


class TestAgenticSearchCompletenessEvaluationFailure:
    """Regression tests for completeness evaluation failures."""

    @pytest.mark.asyncio
    async def test_completeness_evaluation_failure_returns_proper_error_response(
        self, mock_fastmcp_context
    ):
        """Test that LLM failure during completeness evaluation returns proper error format.

        This reproduces the bug where agentic_search fails with:
        - status: error
        - error: "Completeness evaluation failed"
        - completeness: 0.0
        - iterations: 1
        - success: false
        """
        # Arrange: Set up environment for agentic search
        os.environ["AGENTIC_SEARCH_ENABLED"] = "true"
        os.environ["OPENAI_API_KEY"] = "sk-test-fake-key"

        from src.config import reset_settings
        reset_settings()

        # Mock database client to avoid Qdrant dependency
        mock_db_client = MagicMock()
        mock_rag_response = json.dumps({
            "results": [],
            "query": "What is LLM reasoning?",
            "total_results": 0,
        })

        # Mock the settings check to ensure agentic search is enabled
        with patch("src.services.agentic_search.mcp_wrapper.settings") as mock_settings:
            mock_settings.agentic_search_enabled = True

            with patch("src.services.agentic_search.evaluator.get_app_context") as mock_get_ctx:
                # Set up mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = mock_db_client
                mock_get_ctx.return_value = mock_app_ctx

                with patch("src.services.agentic_search.evaluator.perform_rag_query") as mock_rag:
                    mock_rag.return_value = mock_rag_response

                    # Mock the completeness agent to raise an exception
                    # This simulates LLM API failure, network timeout, or invalid response
                    with patch("src.services.agentic_search.config.Agent") as mock_agent_class:
                        # Create mock agent that fails on run()
                        mock_agent = AsyncMock()
                        mock_agent.run.side_effect = Exception("API connection timeout")
                        mock_agent_class.return_value = mock_agent

                        # Reset singleton to force re-initialization with mocked agent
                        import src.services.agentic_search
                        src.services.agentic_search._service_instance = None

                        # Act: Execute agentic search
                        from src.services.agentic_search.mcp_wrapper import agentic_search_impl

                        result_json = await agentic_search_impl(
                            ctx=mock_fastmcp_context,
                            query="What is LLM reasoning?",
                            completeness_threshold=0.8,
                            max_iterations=3,
                        )

                    result = json.loads(result_json)

                    # Assert: Verify error response format matches bug report
                    assert result["success"] is False, "Search should fail"
                    assert result["status"] == "error", "Status should be 'error'"
                    assert "Completeness evaluation failed" in result["error"], \
                        f"Error message should mention completeness evaluation, got: {result['error']}"
                    assert result["query"] == "What is LLM reasoning?"
                    assert result["iterations"] >= 1, "Should have at least 1 iteration"
                    assert result["completeness"] == 0.0, "Completeness should be 0.0 when evaluation fails"
                    assert isinstance(result["results"], list), "Results should be a list"
                    assert isinstance(result["search_history"], list), "search_history should be a list"

    @pytest.mark.asyncio
    async def test_completeness_evaluation_pydantic_validation_failure(
        self, mock_fastmcp_context
    ):
        """Test that Pydantic validation errors during completeness evaluation are handled.

        This tests the case where LLM returns invalid JSON or wrong schema.
        """
        # Arrange
        os.environ["AGENTIC_SEARCH_ENABLED"] = "true"
        os.environ["OPENAI_API_KEY"] = "sk-test-fake-key"

        from src.config import reset_settings
        reset_settings()

        from pydantic_ai.exceptions import UnexpectedModelBehavior

        # Mock database
        mock_db_client = MagicMock()
        mock_rag_response = json.dumps({
            "results": [],
            "query": "test query",
            "total_results": 0,
        })

        # Mock the settings check to ensure agentic search is enabled
        with patch("src.services.agentic_search.mcp_wrapper.settings") as mock_settings:
            mock_settings.agentic_search_enabled = True

            with patch("src.services.agentic_search.evaluator.get_app_context") as mock_get_ctx:
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = mock_db_client
                mock_get_ctx.return_value = mock_app_ctx

                with patch("src.services.agentic_search.evaluator.perform_rag_query") as mock_rag:
                    mock_rag.return_value = mock_rag_response

                    # Mock agent to raise UnexpectedModelBehavior (Pydantic AI validation failure)
                    with patch("src.services.agentic_search.config.Agent") as mock_agent_class:
                        mock_agent = AsyncMock()
                        mock_agent.run.side_effect = UnexpectedModelBehavior(
                            "LLM returned invalid JSON after 3 retries"
                        )
                        mock_agent_class.return_value = mock_agent

                        # Reset singleton
                        import src.services.agentic_search
                        src.services.agentic_search._service_instance = None

                        # Act
                        from src.services.agentic_search.mcp_wrapper import agentic_search_impl

                        result_json = await agentic_search_impl(
                            ctx=mock_fastmcp_context,
                            query="test query",
                        )

                    result = json.loads(result_json)

                    # Assert: Should handle Pydantic AI validation errors gracefully
                    assert result["success"] is False
                    assert result["status"] == "error"
                    assert "completeness evaluation failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_completeness_evaluation_with_openai_auth_error(
        self, mock_fastmcp_context
    ):
        """Test that OpenAI authentication errors are handled gracefully.

        This tests the case where OPENAI_API_KEY is invalid or missing.
        """
        # Arrange: Use invalid API key
        os.environ["AGENTIC_SEARCH_ENABLED"] = "true"
        os.environ["OPENAI_API_KEY"] = "sk-invalid-key-12345"

        from src.config import reset_settings
        reset_settings()

        # Mock database
        mock_db_client = MagicMock()
        mock_rag_response = json.dumps({
            "results": [],
            "query": "test query",
            "total_results": 0,
        })

        # Mock the settings check to ensure agentic search is enabled
        with patch("src.services.agentic_search.mcp_wrapper.settings") as mock_settings:
            mock_settings.agentic_search_enabled = True

            with patch("src.services.agentic_search.evaluator.get_app_context") as mock_get_ctx:
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = mock_db_client
                mock_get_ctx.return_value = mock_app_ctx

                with patch("src.services.agentic_search.evaluator.perform_rag_query") as mock_rag:
                    mock_rag.return_value = mock_rag_response

                    # Mock agent to raise authentication error
                    with patch("src.services.agentic_search.config.Agent") as mock_agent_class:
                        mock_agent = AsyncMock()
                        mock_agent.run.side_effect = Exception("Invalid API key")
                        mock_agent_class.return_value = mock_agent

                        # Reset singleton
                        import src.services.agentic_search
                        src.services.agentic_search._service_instance = None

                        # Act
                        from src.services.agentic_search.mcp_wrapper import agentic_search_impl

                        result_json = await agentic_search_impl(
                            ctx=mock_fastmcp_context,
                            query="test query",
                        )

                    result = json.loads(result_json)

                    # Assert: Should return error response, not crash
                    assert result["success"] is False
                    assert result["status"] == "error"
                    assert result["completeness"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
