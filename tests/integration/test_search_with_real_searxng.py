"""
Integration tests for search service with REAL SearXNG instance.

NO MOCKS POLICY: All tests use real SearXNG service.
Tests will skip if SearXNG is not available at localhost:8080.
"""

import json

import httpx
import pytest

from src.config import get_settings
from src.core.context import initialize_global_context
from src.services.search import _search_searxng, search_and_process

# ========================================
# Fixtures
# ========================================


@pytest.fixture
async def searxng_available():
    """Check if SearXNG is available before running tests."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.searxng_url}/healthz")
            if response.status_code != 200:
                pytest.skip(f"SearXNG not available at {settings.searxng_url}")
    except Exception as e:
        pytest.skip(f"SearXNG not available: {e}")

    return True


@pytest.fixture
def simple_context():
    """Create simple context for tests."""

    class SimpleContext:
        pass

    return SimpleContext()


# ========================================
# Tests for _search_searxng with real SearXNG
# ========================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_searxng_real_query(searxng_available):
    """Test _search_searxng with real SearXNG query."""
    results = await _search_searxng("Python programming language", num_results=3)

    # Real SearXNG should return results for a common query
    assert isinstance(results, list)
    assert len(results) > 0, "Expected at least one search result"
    assert len(results) <= 3, "Expected at most 3 results"

    # Verify result structure
    for result in results:
        assert "title" in result
        assert "url" in result
        assert "snippet" in result
        assert isinstance(result["title"], str)
        assert isinstance(result["url"], str)
        assert result["url"].startswith(("http://", "https://"))


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_searxng_limit_results(searxng_available):
    """Test that num_results parameter limits results correctly."""
    results = await _search_searxng("test query", num_results=5)

    assert isinstance(results, list)
    assert len(results) <= 5, "Results should be limited to requested amount"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_searxng_obscure_query(searxng_available):
    """Test search with obscure query that may return no results."""
    # Use a very specific nonsense query unlikely to have results
    results = await _search_searxng(
        "xyzabc123nonexistentquery456def", num_results=3,
    )

    assert isinstance(results, list)
    # May be empty or have very few results
    assert len(results) <= 3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_searxng_special_characters(searxng_available):
    """Test search with special characters in query."""
    results = await _search_searxng("C++ programming", num_results=3)

    assert isinstance(results, list)
    # Should handle special characters without errors


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_searxng_unicode_query(searxng_available):
    """Test search with Unicode characters."""
    results = await _search_searxng("Python 日本語", num_results=3)

    assert isinstance(results, list)
    # Should handle Unicode without errors


# ========================================
# Tests for search_and_process with real services
# ========================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_and_process_integration(searxng_available, simple_context):
    """Test search_and_process with real SearXNG and crawling.

    This test verifies the full pipeline:
    1. Search with real SearXNG
    2. Crawl returned URLs (with Crawl4AI service)
    3. Combine results

    Note: This test may take longer as it performs real web requests.
    """
    # Initialize app context for crawling
    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    # Use a simple, reliable query
    result_json = await search_and_process(
        ctx=simple_context,
        query="Python programming language tutorial",
        return_raw_markdown=False,
        num_results=2,  # Keep small for test speed
        batch_size=5,
    )

    result = json.loads(result_json)

    # Verify response structure
    assert "success" in result
    assert "query" in result
    assert "total_results" in result
    assert "results" in result

    # Should have some results for common query
    if result["success"]:
        assert result["query"] == "Python programming language tutorial"
        assert isinstance(result["results"], list)

        # Verify result structure
        for item in result["results"]:
            assert "title" in item
            assert "url" in item
            assert "snippet" in item
            # May have "stored" and "chunks" if crawling succeeded


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_and_process_with_markdown(
    searxng_available, simple_context,
):
    """Test search_and_process with raw markdown return."""
    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    result_json = await search_and_process(
        ctx=simple_context,
        query="Python tutorial",
        return_raw_markdown=True,
        num_results=1,  # Just one for speed
    )

    result = json.loads(result_json)

    assert "success" in result
    assert "results" in result

    # If successful, should have markdown
    if result["success"] and len(result["results"]) > 0:
        for item in result["results"]:
            if "markdown" in item:
                assert isinstance(item["markdown"], str)
                # Should not have storage info
                assert "stored" not in item
                assert "chunks" not in item


# ========================================
# Error handling tests
# ========================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_with_invalid_url(simple_context):
    """Test search_and_process when SearXNG URL is invalid."""
    from unittest.mock import patch

    from src.core import MCPToolError

    # Temporarily override settings to invalid URL
    with patch("src.services.search.settings") as mock_settings:
        mock_settings.searxng_url = None

        with pytest.raises(MCPToolError) as exc_info:
            await search_and_process(
                ctx=simple_context,
                query="test query",
            )

        assert "SearXNG URL not configured" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_searxng_with_bad_url():
    """Test _search_searxng with completely invalid URL."""
    from unittest.mock import patch

    with patch("src.services.search.settings") as mock_settings:
        mock_settings.searxng_url = "http://nonexistent-searxng-server-xyz:9999"
        mock_settings.searxng_user_agent = "Test"
        mock_settings.searxng_timeout = 2
        mock_settings.searxng_default_engines = ""

        results = await _search_searxng("test query", num_results=3)

        # Should return empty list on connection failure
        assert results == []
