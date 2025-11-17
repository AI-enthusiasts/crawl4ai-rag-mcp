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

NO MOCKS - Real integration test that will FAIL if bug exists.
"""

import asyncio
import json
import os

import pytest

from src.config import get_settings, reset_settings
from src.core.context import initialize_global_context


# Mark as integration test
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_settings():
    """Configure settings for integration tests."""
    # Set test environment variables BEFORE importing anything
    os.environ["AGENTIC_SEARCH_ENABLED"] = "true"
    os.environ["AGENTIC_SEARCH_COMPLETENESS_THRESHOLD"] = "0.8"
    os.environ["AGENTIC_SEARCH_MAX_ITERATIONS"] = "1"  # Just 1 iteration for speed
    os.environ["AGENTIC_SEARCH_MAX_URLS_PER_ITERATION"] = "1"
    os.environ["USE_QDRANT"] = "true"

    # Use test model if available
    if not os.getenv("TEST_MODEL_CHOICE"):
        os.environ["MODEL_CHOICE"] = "gpt-4.1-nano"
    else:
        os.environ["MODEL_CHOICE"] = os.getenv("TEST_MODEL_CHOICE")

    # Use test API key if available
    if os.getenv("TEST_OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("TEST_OPENAI_API_KEY")

    # Force reload of settings module
    import sys
    if 'src.config.settings' in sys.modules:
        del sys.modules['src.config.settings']
    if 'src.config' in sys.modules:
        del sys.modules['src.config']

    reset_settings()
    settings = get_settings()

    # Skip if no API key
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")

    # Check if Qdrant is available
    import httpx
    try:
        response = httpx.get(f"{settings.qdrant_url}/collections", timeout=2.0)
        if response.status_code >= 500:
            pytest.skip(f"Qdrant not available at {settings.qdrant_url}")
    except Exception:
        pytest.skip(f"Qdrant not available at {settings.qdrant_url} - start with docker-compose up qdrant")

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


class TestAgenticSearchCompletenessEvaluationFailure:
    """Regression test for completeness evaluation failures - NO MOCKS, REAL TEST."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_agentic_search_basic_query(self, test_settings, app_context):
        """Real integration test: Execute agentic_search with actual OpenAI API.

        This test will FAIL if the bug exists (Completeness evaluation failed).
        When working correctly, should return success=true with results.

        Currently reproduces bug from issue:
        - Error: "Completeness evaluation failed"
        - status: "error"
        - success: false
        """
        # Force reload of mcp_wrapper to pick up test settings
        import sys
        if 'src.services.agentic_search.mcp_wrapper' in sys.modules:
            del sys.modules['src.services.agentic_search.mcp_wrapper']

        from src.services.agentic_search import agentic_search_impl

        # Create a simple mock context (FastMCP Context not needed for internal call)
        class SimpleContext:
            pass

        ctx = SimpleContext()

        # Execute real agentic search with real LLM, real Qdrant, real everything
        result_json = await agentic_search_impl(
            ctx=ctx,
            query="What is LLM reasoning?",
            completeness_threshold=0.8,
            max_iterations=1,  # Just 1 iteration to speed up test
            max_urls_per_iteration=1,
        )

        result = json.loads(result_json)

        # These assertions will FAIL if bug exists
        assert result["success"] is True, \
            f"BUG REPRODUCED: {result.get('error')} - status={result.get('status')}"

        assert result["status"] != "error", \
            f"BUG: Got error status: {result.get('error')}"

        assert result["query"] == "What is LLM reasoning?"
        assert result["iterations"] >= 1

        # Should not have error field when successful
        if "error" in result:
            assert result["error"] is None, \
                f"BUG: Unexpected error: {result['error']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
