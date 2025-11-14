"""
Comprehensive unit tests for search service (src/services/search.py).

This module tests:
- search_and_process() - MCP tool wrapper
- _search_searxng() - SearXNG integration
- Error handling (SearchError, NetworkError, FetchError)
- HTTP failure scenarios
- Malformed responses
- Empty results handling
- Configuration validation
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from fastmcp import Context

from src.core import MCPToolError
from src.core.exceptions import FetchError, SearchError
from src.services.search import _search_searxng, search_and_process


# ========================================
# Fixtures
# ========================================


@pytest.fixture
def mock_settings():
    """Mock settings for search tests."""
    with patch("src.services.search.settings") as mock:
        mock.searxng_url = "http://localhost:8080"
        mock.searxng_user_agent = "Test-Agent/1.0"
        mock.searxng_timeout = 30
        mock.searxng_default_engines = "google,bing"
        yield mock


@pytest.fixture
def mock_context():
    """Mock FastMCP context."""
    ctx = MagicMock(spec=Context)
    ctx.request_context = MagicMock()
    return ctx


@pytest.fixture
def sample_html_response():
    """Sample HTML response from SearXNG."""
    return """
    <html>
        <body>
            <article class="result">
                <h3><a href="https://example.com/1">First Result</a></h3>
                <p class="content">This is the first result snippet</p>
            </article>
            <article class="result">
                <h3><a href="https://example.com/2">Second Result</a></h3>
                <p class="content">This is the second result snippet</p>
            </article>
            <article class="result">
                <h3><a href="https://example.com/3">Third Result</a></h3>
                <p class="content">This is the third result snippet</p>
            </article>
        </body>
    </html>
    """


@pytest.fixture
def sample_search_results():
    """Sample search results for mocking."""
    return [
        {
            "title": "First Result",
            "url": "https://example.com/1",
            "snippet": "This is the first result snippet",
        },
        {
            "title": "Second Result",
            "url": "https://example.com/2",
            "snippet": "This is the second result snippet",
        },
    ]


# ========================================
# Tests for _search_searxng()
# ========================================


@pytest.mark.asyncio
class TestSearchSearxng:
    """Test suite for _search_searxng() function."""

    async def test_search_success(self, mock_settings, sample_html_response):
        """Test successful search with valid HTML response."""
        # Mock aiohttp response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=sample_html_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock session.get
        mock_get = MagicMock(return_value=mock_response)

        # Mock session
        mock_session = MagicMock()
        mock_session.get = mock_get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert len(results) == 3
        assert results[0]["title"] == "First Result"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["snippet"] == "This is the first result snippet"

    async def test_search_empty_results(self, mock_settings):
        """Test search with no results found."""
        empty_html = "<html><body></body></html>"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=empty_html)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("nonexistent query", num_results=5)

        assert results == []

    async def test_search_http_error_status(self, mock_settings):
        """Test search with HTTP error status code."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert results == []

    async def test_search_http_404_status(self, mock_settings):
        """Test search with 404 status code."""
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert results == []

    async def test_search_timeout(self, mock_settings):
        """Test search with network timeout."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError("Connection timeout"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert results == []

    async def test_search_connection_error(self, mock_settings):
        """Test search with connection error."""
        mock_session = MagicMock()
        # Use OSError directly instead of ClientConnectorError to avoid ssl attribute issue
        mock_session.get = MagicMock(side_effect=OSError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert results == []

    async def test_search_fetch_error(self, mock_settings):
        """Test search with FetchError exception."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=FetchError("Failed to fetch"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert results == []

    async def test_search_search_error(self, mock_settings):
        """Test search with SearchError exception."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=SearchError("Search failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert results == []

    async def test_search_generic_exception(self, mock_settings):
        """Test search with generic exception."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Unexpected error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert results == []

    async def test_search_invalid_url_config(self):
        """Test search with invalid SearXNG URL configuration."""
        with patch("src.services.search.settings") as mock_settings:
            mock_settings.searxng_url = None

            results = await _search_searxng("test query", num_results=5)

            assert results == []

    async def test_search_empty_url_config(self):
        """Test search with empty SearXNG URL."""
        with patch("src.services.search.settings") as mock_settings:
            mock_settings.searxng_url = ""

            results = await _search_searxng("test query", num_results=5)

            assert results == []

    async def test_search_malformed_html(self, mock_settings):
        """Test search with malformed HTML response."""
        malformed_html = "<html><body><article class='result'><h3>Missing closing tags"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=malformed_html)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        # BeautifulSoup should handle malformed HTML gracefully
        assert isinstance(results, list)

    async def test_search_missing_title(self, mock_settings):
        """Test search with results missing title elements."""
        html_no_title = """
        <html><body>
            <article class="result">
                <p class="content">Content without title</p>
            </article>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=html_no_title)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        # Should skip results without URLs
        assert results == []

    async def test_search_missing_snippet(self, mock_settings):
        """Test search with results missing snippet/content."""
        html_no_snippet = """
        <html><body>
            <article class="result">
                <h3><a href="https://example.com/1">Title Only</a></h3>
            </article>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=html_no_snippet)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=5)

        assert len(results) == 1
        assert results[0]["title"] == "Title Only"
        assert results[0]["url"] == "https://example.com/1"
        # Snippet key may not exist or be empty
        assert "snippet" not in results[0] or results[0]["snippet"] == ""

    async def test_search_num_results_limit(self, mock_settings):
        """Test that num_results parameter limits the number of results."""
        html_many_results = """
        <html><body>
            <article class="result">
                <h3><a href="https://example.com/1">Result 1</a></h3>
                <p class="content">Snippet 1</p>
            </article>
            <article class="result">
                <h3><a href="https://example.com/2">Result 2</a></h3>
                <p class="content">Snippet 2</p>
            </article>
            <article class="result">
                <h3><a href="https://example.com/3">Result 3</a></h3>
                <p class="content">Snippet 3</p>
            </article>
            <article class="result">
                <h3><a href="https://example.com/4">Result 4</a></h3>
                <p class="content">Snippet 4</p>
            </article>
            <article class="result">
                <h3><a href="https://example.com/5">Result 5</a></h3>
                <p class="content">Snippet 5</p>
            </article>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=html_many_results)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=2)

        # BeautifulSoup's find_all with limit parameter
        assert len(results) <= 2

    async def test_search_url_stripping(self, mock_settings):
        """Test that trailing slashes are stripped from SearXNG URL."""
        with patch("src.services.search.settings") as mock_settings:
            mock_settings.searxng_url = "http://localhost:8080/"
            mock_settings.searxng_user_agent = "Test-Agent/1.0"
            mock_settings.searxng_timeout = 30
            mock_settings.searxng_default_engines = ""

            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<html><body></body></html>")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                await _search_searxng("test query", num_results=5)

            # Verify URL was properly constructed
            call_args = mock_session.get.call_args
            assert "http://localhost:8080/search" in str(call_args)


# ========================================
# Tests for search_and_process()
# ========================================


@pytest.mark.asyncio
class TestSearchAndProcess:
    """Test suite for search_and_process() function."""

    async def test_search_and_process_success(
        self,
        mock_context,
        mock_settings,
        sample_search_results,
    ):
        """Test successful search and process workflow."""
        # Mock _search_searxng
        with patch(
            "src.services.search._search_searxng",
            return_value=sample_search_results,
        ):
            # Mock process_urls_for_mcp
            crawl_response = {
                "success": True,
                "results": [
                    {
                        "success": True,
                        "chunks_stored": 5,
                    },
                    {
                        "success": True,
                        "chunks_stored": 3,
                    },
                ],
            }

            with patch(
                "src.services.search.process_urls_for_mcp",
                return_value=json.dumps(crawl_response),
            ):
                result = await search_and_process(
                    mock_context,
                    "test query",
                    return_raw_markdown=False,
                    num_results=6,
                )

        data = json.loads(result)
        assert data["success"] is True
        assert data["query"] == "test query"
        assert data["total_results"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "First Result"
        assert data["results"][0]["url"] == "https://example.com/1"
        assert data["results"][0]["stored"] is True
        assert data["results"][0]["chunks"] == 5

    async def test_search_and_process_with_markdown(
        self,
        mock_context,
        mock_settings,
        sample_search_results,
    ):
        """Test search and process with raw markdown return."""
        with patch(
            "src.services.search._search_searxng",
            return_value=sample_search_results,
        ):
            crawl_response = {
                "success": True,
                "results": [
                    {
                        "markdown": "# First Page\n\nContent here",
                    },
                    {
                        "markdown": "# Second Page\n\nMore content",
                    },
                ],
            }

            with patch(
                "src.services.search.process_urls_for_mcp",
                return_value=json.dumps(crawl_response),
            ):
                result = await search_and_process(
                    mock_context,
                    "test query",
                    return_raw_markdown=True,
                    num_results=6,
                )

        data = json.loads(result)
        assert data["success"] is True
        assert data["results"][0]["markdown"] == "# First Page\n\nContent here"
        assert "stored" not in data["results"][0]
        assert "chunks" not in data["results"][0]

    async def test_search_and_process_no_results(self, mock_context, mock_settings):
        """Test search and process with no search results."""
        with patch("src.services.search._search_searxng", return_value=[]):
            result = await search_and_process(mock_context, "nonexistent query")

        data = json.loads(result)
        assert data["success"] is False
        assert data["message"] == "No search results found"
        assert data["results"] == []

    async def test_search_and_process_no_searxng_url(self, mock_context):
        """Test search and process without SearXNG URL configured."""
        with patch("src.services.search.settings") as mock_settings:
            mock_settings.searxng_url = None

            with pytest.raises(MCPToolError) as exc_info:
                await search_and_process(mock_context, "test query")

            assert "SearXNG URL not configured" in str(exc_info.value)

    async def test_search_and_process_crawl_failure(
        self,
        mock_context,
        mock_settings,
        sample_search_results,
    ):
        """Test search and process when crawling fails."""
        with patch(
            "src.services.search._search_searxng",
            return_value=sample_search_results,
        ):
            crawl_response = {
                "success": False,
                "message": "Crawl failed",
                "results": [],
            }

            with patch(
                "src.services.search.process_urls_for_mcp",
                return_value=json.dumps(crawl_response),
            ):
                result = await search_and_process(
                    mock_context,
                    "test query",
                    return_raw_markdown=False,
                )

        data = json.loads(result)
        assert data["success"] is True
        # Should still return search metadata even if crawl fails
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "First Result"

    async def test_search_and_process_partial_crawl_results(
        self,
        mock_context,
        mock_settings,
        sample_search_results,
    ):
        """Test search and process when crawl returns fewer results than search."""
        with patch(
            "src.services.search._search_searxng",
            return_value=sample_search_results,  # 2 results
        ):
            crawl_response = {
                "success": True,
                "results": [
                    {
                        "success": True,
                        "chunks_stored": 5,
                    },
                    # Only 1 crawl result instead of 2
                ],
            }

            with patch(
                "src.services.search.process_urls_for_mcp",
                return_value=json.dumps(crawl_response),
            ):
                result = await search_and_process(
                    mock_context,
                    "test query",
                    return_raw_markdown=False,
                )

        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) == 2
        # First result should have crawl data
        assert data["results"][0]["chunks"] == 5
        # Second result won't have crawl data but should still have search metadata
        assert data["results"][1]["title"] == "Second Result"

    async def test_search_and_process_exception_handling(
        self,
        mock_context,
        mock_settings,
    ):
        """Test search and process exception handling."""
        with patch(
            "src.services.search._search_searxng",
            side_effect=Exception("Unexpected error"),
        ):
            with pytest.raises(MCPToolError) as exc_info:
                await search_and_process(mock_context, "test query")

            assert "Search processing failed" in str(exc_info.value)

    async def test_search_and_process_custom_batch_size(
        self,
        mock_context,
        mock_settings,
        sample_search_results,
    ):
        """Test search and process with custom batch size."""
        with patch(
            "src.services.search._search_searxng",
            return_value=sample_search_results,
        ):
            crawl_response = {
                "success": True,
                "results": [
                    {"success": True, "chunks_stored": 5},
                    {"success": True, "chunks_stored": 3},
                ],
            }

            with patch(
                "src.services.search.process_urls_for_mcp",
                return_value=json.dumps(crawl_response),
            ) as mock_process:
                await search_and_process(
                    mock_context,
                    "test query",
                    batch_size=10,
                )

                # Verify batch_size was passed through
                call_args = mock_process.call_args
                assert call_args.kwargs["batch_size"] == 10

    async def test_search_and_process_custom_num_results(
        self,
        mock_context,
        mock_settings,
    ):
        """Test search and process with custom number of results."""
        with patch(
            "src.services.search._search_searxng",
            return_value=[],
        ) as mock_search:
            with patch("src.services.search.process_urls_for_mcp"):
                await search_and_process(
                    mock_context,
                    "test query",
                    num_results=10,
                )

                # Verify num_results was passed through
                call_args = mock_search.call_args
                assert call_args[0][1] == 10  # Second positional argument (query, num_results)

    async def test_search_and_process_url_extraction(
        self,
        mock_context,
        mock_settings,
    ):
        """Test that URLs are properly extracted from search results."""
        search_results = [
            {"title": "Test 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
            {"title": "Test 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
            {"title": "Test 3", "url": "https://example.com/3", "snippet": "Snippet 3"},
        ]

        with patch("src.services.search._search_searxng", return_value=search_results):
            crawl_response = {"success": True, "results": []}

            with patch(
                "src.services.search.process_urls_for_mcp",
                return_value=json.dumps(crawl_response),
            ) as mock_process:
                await search_and_process(mock_context, "test query")

                # Verify correct URLs were passed to process_urls_for_mcp
                call_args = mock_process.call_args
                urls = call_args.kwargs["urls"]
                assert urls == [
                    "https://example.com/1",
                    "https://example.com/2",
                    "https://example.com/3",
                ]

    async def test_search_and_process_missing_snippet(
        self,
        mock_context,
        mock_settings,
    ):
        """Test search results with missing snippet field."""
        search_results = [
            {"title": "Test 1", "url": "https://example.com/1"},  # No snippet
        ]

        with patch("src.services.search._search_searxng", return_value=search_results):
            crawl_response = {
                "success": True,
                "results": [{"success": True, "chunks_stored": 5}],
            }

            with patch(
                "src.services.search.process_urls_for_mcp",
                return_value=json.dumps(crawl_response),
            ):
                result = await search_and_process(mock_context, "test query")

        data = json.loads(result)
        assert data["success"] is True
        assert data["results"][0]["snippet"] == ""  # Default empty string


# ========================================
# Edge Cases and Integration Tests
# ========================================


@pytest.mark.asyncio
class TestSearchEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_search_special_characters_in_query(self, mock_settings):
        """Test search with special characters in query."""
        special_queries = [
            "C++ programming",
            "email@example.com",
            "price: $100-$200",
            "JavaScript (ES6)",
            "Python & ML",
        ]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html><body></body></html>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            for query in special_queries:
                results = await _search_searxng(query, num_results=5)
                assert isinstance(results, list)

    async def test_search_unicode_characters(self, mock_settings):
        """Test search with Unicode characters."""
        unicode_query = "日本語 Chinese 한국어 Русский العربية"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html><body></body></html>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng(unicode_query, num_results=5)
            assert isinstance(results, list)

    async def test_search_very_long_query(self, mock_settings):
        """Test search with very long query string."""
        long_query = "test query " * 100  # 1000+ characters

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html><body></body></html>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng(long_query, num_results=5)
            assert isinstance(results, list)

    async def test_search_zero_results_requested(self, mock_settings):
        """Test search with zero results requested."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html><body></body></html>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await _search_searxng("test query", num_results=0)
            assert isinstance(results, list)
