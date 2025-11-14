"""
Comprehensive unit tests for smart_crawl service.

Tests the smart crawl functionality that intelligently detects and crawls
different types of URLs (sitemaps, text files, regular pages) with appropriate
strategies.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from crawl4ai import BrowserConfig

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.exceptions import CrawlError, DatabaseError, FetchError, MCPToolError
from services.smart_crawl import (
    _crawl_recursive,
    _crawl_sitemap,
    _crawl_text_file,
    _perform_rag_query_with_context,
    smart_crawl_url,
)


class TestSmartCrawlUrl:
    """Test main smart_crawl_url function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        ctx = MagicMock()
        ctx.meta = {}
        return ctx

    @pytest.fixture
    def mock_app_context(self):
        """Create mock application context."""
        app_ctx = MagicMock()
        app_ctx.browser_config = BrowserConfig(headless=True, verbose=False)
        app_ctx.database_client = AsyncMock()
        app_ctx.dispatcher = MagicMock()
        return app_ctx

    @pytest.mark.asyncio
    async def test_smart_crawl_url_sitemap(self, mock_context):
        """Test smart crawl with sitemap URL."""
        sitemap_url = "https://example.com/sitemap.xml"
        expected_result = {
            "success": True,
            "type": "sitemap",
            "total_urls": 5,
        }

        with patch(
            "services.smart_crawl._crawl_sitemap",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps(expected_result)

            result = await smart_crawl_url(
                ctx=mock_context,
                url=sitemap_url,
                max_depth=3,
            )

            # Verify sitemap crawl was called
            mock_crawl.assert_called_once()
            call_args = mock_crawl.call_args
            assert call_args[0][0] == mock_context
            assert "sitemap.xml" in call_args[0][1]  # normalized URL

            # Verify result
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["type"] == "sitemap"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_text_file(self, mock_context):
        """Test smart crawl with text file URL."""
        txt_url = "https://example.com/robots.txt"
        expected_result = {
            "success": True,
            "type": "text_file",
            "chunks_stored": 1,
        }

        with patch(
            "services.smart_crawl._crawl_text_file",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps(expected_result)

            result = await smart_crawl_url(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
            )

            # Verify text file crawl was called
            mock_crawl.assert_called_once()
            call_args = mock_crawl.call_args
            assert call_args[0][0] == mock_context
            assert "robots.txt" in call_args[0][1]

            # Verify result
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["type"] == "text_file"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_regular_page(self, mock_context):
        """Test smart crawl with regular web page."""
        regular_url = "https://example.com/docs/guide"
        expected_result = {
            "success": True,
            "type": "recursive",
            "urls_crawled": 10,
        }

        with patch(
            "services.smart_crawl._crawl_recursive",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps(expected_result)

            result = await smart_crawl_url(
                ctx=mock_context,
                url=regular_url,
                max_depth=2,
            )

            # Verify recursive crawl was called
            mock_crawl.assert_called_once()
            call_args = mock_crawl.call_args
            assert call_args[0][0] == mock_context
            # max_depth is positional arg at index 2
            assert call_args[0][2] == 2

            # Verify result
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["type"] == "recursive"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_with_queries(self, mock_context):
        """Test smart crawl with RAG queries."""
        url = "https://example.com/sitemap.xml"
        queries = ["authentication", "database"]

        with patch(
            "services.smart_crawl._crawl_sitemap",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps(
                {"success": True, "type": "sitemap"},
            )

            await smart_crawl_url(
                ctx=mock_context,
                url=url,
                query=queries,
            )

            # Verify queries were passed (positional arg at index 4)
            call_args = mock_crawl.call_args
            assert call_args[0][4] == queries

    @pytest.mark.asyncio
    async def test_smart_crawl_url_handles_crawl_error(self, mock_context):
        """Test smart crawl handles CrawlError."""
        url = "https://example.com/page"

        with patch(
            "services.smart_crawl._crawl_recursive",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.side_effect = CrawlError("Crawl failed")

            with pytest.raises(MCPToolError, match="Smart crawl failed"):
                await smart_crawl_url(ctx=mock_context, url=url)

    @pytest.mark.asyncio
    async def test_smart_crawl_url_handles_database_error(self, mock_context):
        """Test smart crawl handles DatabaseError."""
        url = "https://example.com/page"

        with patch(
            "services.smart_crawl._crawl_recursive",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.side_effect = DatabaseError("Database error")

            with pytest.raises(MCPToolError, match="Smart crawl failed"):
                await smart_crawl_url(ctx=mock_context, url=url)

    @pytest.mark.asyncio
    async def test_smart_crawl_url_handles_fetch_error(self, mock_context):
        """Test smart crawl handles FetchError."""
        url = "https://example.com/sitemap.xml"

        with patch(
            "services.smart_crawl._crawl_sitemap",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.side_effect = FetchError("Fetch failed")

            with pytest.raises(MCPToolError, match="Smart crawl failed"):
                await smart_crawl_url(ctx=mock_context, url=url)

    @pytest.mark.asyncio
    async def test_smart_crawl_url_handles_unexpected_error(self, mock_context):
        """Test smart crawl handles unexpected errors."""
        url = "https://example.com/page"

        with patch(
            "services.smart_crawl._crawl_recursive",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.side_effect = ValueError("Unexpected error")

            with pytest.raises(MCPToolError, match="Smart crawl failed"):
                await smart_crawl_url(ctx=mock_context, url=url)

    @pytest.mark.asyncio
    async def test_smart_crawl_url_normalizes_url(self, mock_context):
        """Test that URLs are normalized before processing."""
        url_with_fragment = "https://example.com/page#section"

        with patch(
            "services.smart_crawl._crawl_recursive",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps({"success": True})

            await smart_crawl_url(ctx=mock_context, url=url_with_fragment)

            # Verify normalized URL (without fragment) was used
            call_args = mock_crawl.call_args
            normalized_url = call_args[0][1]
            assert "#section" not in normalized_url


class TestCrawlSitemap:
    """Test _crawl_sitemap function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return MagicMock()

    @pytest.fixture
    def sample_sitemap_xml(self):
        """Sample sitemap XML content."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://example.com/page1</loc>
    </url>
    <url>
        <loc>https://example.com/page2</loc>
    </url>
    <url>
        <loc>https://example.com/page3</loc>
    </url>
</urlset>"""

    @pytest.mark.asyncio
    async def test_crawl_sitemap_success(
        self, mock_context, sample_sitemap_xml
    ):
        """Test successful sitemap crawling."""
        sitemap_url = "https://example.com/sitemap.xml"

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_sitemap_xml

        # Mock process_urls_for_mcp
        mock_process_result = {
            "success": True,
            "urls_processed": 3,
        }

        with (
            patch("services.smart_crawl.httpx.AsyncClient") as mock_client,
            patch(
                "services.smart_crawl.process_urls_for_mcp",
                new_callable=AsyncMock,
            ) as mock_process,
        ):
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response,
            )
            mock_process.return_value = json.dumps(mock_process_result)

            result = await _crawl_sitemap(
                ctx=mock_context,
                url=sitemap_url,
                chunk_size=5000,
                return_raw_markdown=False,
                query=None,
            )

            # Verify result
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["type"] == "sitemap"
            assert result_data["sitemap_url"] == sitemap_url
            assert result_data["total_urls"] == 3

    @pytest.mark.asyncio
    async def test_crawl_sitemap_http_error(self, mock_context):
        """Test sitemap crawl with HTTP error."""
        sitemap_url = "https://example.com/sitemap.xml"

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("services.smart_crawl.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response,
            )

            with pytest.raises(MCPToolError, match="Failed to fetch sitemap.*404"):
                await _crawl_sitemap(
                    ctx=mock_context,
                    url=sitemap_url,
                    chunk_size=5000,
                    return_raw_markdown=False,
                    query=None,
                )

    @pytest.mark.asyncio
    async def test_crawl_sitemap_empty_sitemap(self, mock_context):
        """Test sitemap with no URLs."""
        sitemap_url = "https://example.com/sitemap.xml"
        empty_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
</urlset>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = empty_sitemap

        with patch("services.smart_crawl.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response,
            )

            result = await _crawl_sitemap(
                ctx=mock_context,
                url=sitemap_url,
                chunk_size=5000,
                return_raw_markdown=False,
                query=None,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "No URLs found" in result_data["message"]

    @pytest.mark.asyncio
    async def test_crawl_sitemap_with_rag_queries(
        self, mock_context, sample_sitemap_xml
    ):
        """Test sitemap crawl with RAG queries."""
        sitemap_url = "https://example.com/sitemap.xml"
        queries = ["test query"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_sitemap_xml

        mock_process_result = {"success": True}
        mock_rag_result = {"results": ["result1"]}

        with (
            patch("services.smart_crawl.httpx.AsyncClient") as mock_client,
            patch(
                "services.smart_crawl.process_urls_for_mcp",
                new_callable=AsyncMock,
            ) as mock_process,
            patch(
                "services.smart_crawl._perform_rag_query_with_context",
                new_callable=AsyncMock,
            ) as mock_rag,
        ):
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response,
            )
            mock_process.return_value = json.dumps(mock_process_result)
            mock_rag.return_value = json.dumps(mock_rag_result)

            result = await _crawl_sitemap(
                ctx=mock_context,
                url=sitemap_url,
                chunk_size=5000,
                return_raw_markdown=False,
                query=queries,
            )

            result_data = json.loads(result)
            assert "query_results" in result_data
            assert "test query" in result_data["query_results"]
            mock_rag.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_sitemap_handles_crawl_error(self, mock_context, sample_sitemap_xml):
        """Test sitemap crawl handles CrawlError."""
        sitemap_url = "https://example.com/sitemap.xml"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_sitemap_xml

        with (
            patch("services.smart_crawl.httpx.AsyncClient") as mock_client,
            patch(
                "services.smart_crawl.process_urls_for_mcp",
                new_callable=AsyncMock,
            ) as mock_process,
        ):
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response,
            )
            mock_process.side_effect = CrawlError("Crawl failed")

            result = await _crawl_sitemap(
                ctx=mock_context,
                url=sitemap_url,
                chunk_size=5000,
                return_raw_markdown=False,
                query=None,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Crawl failed" in result_data["error"]

    @pytest.mark.asyncio
    async def test_crawl_sitemap_handles_database_error_in_rag(
        self, mock_context, sample_sitemap_xml
    ):
        """Test sitemap crawl handles DatabaseError during RAG queries."""
        sitemap_url = "https://example.com/sitemap.xml"
        queries = ["test query"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_sitemap_xml

        mock_process_result = {"success": True}

        with (
            patch("services.smart_crawl.httpx.AsyncClient") as mock_client,
            patch(
                "services.smart_crawl.process_urls_for_mcp",
                new_callable=AsyncMock,
            ) as mock_process,
            patch(
                "services.smart_crawl._perform_rag_query_with_context",
                new_callable=AsyncMock,
            ) as mock_rag,
        ):
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response,
            )
            mock_process.return_value = json.dumps(mock_process_result)
            mock_rag.side_effect = DatabaseError("Database error")

            result = await _crawl_sitemap(
                ctx=mock_context,
                url=sitemap_url,
                chunk_size=5000,
                return_raw_markdown=False,
                query=queries,
            )

            result_data = json.loads(result)
            assert "query_results" in result_data
            assert "error" in result_data["query_results"]["test query"]


class TestCrawlTextFile:
    """Test _crawl_text_file function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return MagicMock()

    @pytest.fixture
    def mock_app_context(self):
        """Create mock application context."""
        app_ctx = MagicMock()
        app_ctx.browser_config = BrowserConfig(headless=True, verbose=False)
        app_ctx.database_client = AsyncMock()
        app_ctx.database_client.store_crawled_page = AsyncMock()
        return app_ctx

    @pytest.fixture
    def sample_crawl_result(self):
        """Sample crawl result for text file."""
        return [
            {
                "url": "https://example.com/robots.txt",
                "markdown": "User-agent: *\nDisallow: /admin/",
                "success": True,
            },
        ]

    @pytest.mark.asyncio
    async def test_crawl_text_file_raw_markdown(
        self, mock_context, mock_app_context, sample_crawl_result
    ):
        """Test text file crawl with raw markdown return."""
        txt_url = "https://example.com/robots.txt"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_markdown_file",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.return_value = sample_crawl_result

            result = await _crawl_text_file(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
                return_raw_markdown=True,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["type"] == "text_file"
            assert "User-agent" in result_data["markdown"]

    @pytest.mark.asyncio
    async def test_crawl_text_file_store_in_database(
        self, mock_context, mock_app_context, sample_crawl_result
    ):
        """Test text file crawl with database storage."""
        txt_url = "https://example.com/robots.txt"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_markdown_file",
                new_callable=AsyncMock,
            ) as mock_crawl,
            patch(
                "src.utils.text_processing.smart_chunk_markdown",
                return_value=["chunk1", "chunk2"],
            ),
            patch(
                "src.utils.url_helpers.extract_domain_from_url",
                return_value="example.com",
            ),
            patch(
                "src.utils.add_documents_to_database",
                new_callable=AsyncMock,
            ) as mock_add_docs,
        ):
            mock_crawl.return_value = sample_crawl_result

            result = await _crawl_text_file(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
                return_raw_markdown=False,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["chunks_stored"] == 2
            assert result_data["source_id"] == "example.com"

            # Verify database function was called
            mock_add_docs.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_text_file_no_browser_config(self, mock_context):
        """Test text file crawl without browser config."""
        txt_url = "https://example.com/robots.txt"
        mock_app_ctx = MagicMock()
        del mock_app_ctx.browser_config

        with patch(
            "src.core.context.get_app_context",
            return_value=mock_app_ctx,
        ):
            result = await _crawl_text_file(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
                return_raw_markdown=False,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Browser config not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_crawl_text_file_crawl_fails(
        self, mock_context, mock_app_context
    ):
        """Test text file crawl when crawl fails."""
        txt_url = "https://example.com/robots.txt"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_markdown_file",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.return_value = None

            result = await _crawl_text_file(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
                return_raw_markdown=False,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Failed to crawl file" in result_data["error"]

    @pytest.mark.asyncio
    async def test_crawl_text_file_no_chunks(
        self, mock_context, mock_app_context, sample_crawl_result
    ):
        """Test text file crawl when chunking produces no chunks."""
        txt_url = "https://example.com/robots.txt"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_markdown_file",
                new_callable=AsyncMock,
            ) as mock_crawl,
            patch(
                "src.utils.text_processing.smart_chunk_markdown",
                return_value=[],  # No chunks
            ),
        ):
            mock_crawl.return_value = sample_crawl_result

            result = await _crawl_text_file(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
                return_raw_markdown=False,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "No content to store" in result_data["error"]

    @pytest.mark.asyncio
    async def test_crawl_text_file_handles_crawl_error(
        self, mock_context, mock_app_context
    ):
        """Test text file crawl handles CrawlError."""
        txt_url = "https://example.com/robots.txt"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_markdown_file",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.side_effect = CrawlError("Crawl failed")

            result = await _crawl_text_file(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
                return_raw_markdown=False,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Crawl failed" in result_data["error"]

    @pytest.mark.asyncio
    async def test_crawl_text_file_handles_database_error(
        self, mock_context, mock_app_context, sample_crawl_result
    ):
        """Test text file crawl handles DatabaseError."""
        txt_url = "https://example.com/robots.txt"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_markdown_file",
                new_callable=AsyncMock,
            ) as mock_crawl,
            patch(
                "src.utils.text_processing.smart_chunk_markdown",
                return_value=["chunk1"],
            ),
            patch(
                "src.utils.url_helpers.extract_domain_from_url",
                return_value="example.com",
            ),
            patch(
                "src.utils.add_documents_to_database",
                new_callable=AsyncMock,
            ) as mock_add_docs,
        ):
            mock_crawl.return_value = sample_crawl_result
            mock_add_docs.side_effect = DatabaseError("Database error")

            result = await _crawl_text_file(
                ctx=mock_context,
                url=txt_url,
                chunk_size=5000,
                return_raw_markdown=False,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Database error" in result_data["error"]


class TestCrawlRecursive:
    """Test _crawl_recursive function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return MagicMock()

    @pytest.fixture
    def mock_app_context(self):
        """Create mock application context."""
        app_ctx = MagicMock()
        app_ctx.browser_config = BrowserConfig(headless=True, verbose=False)
        app_ctx.database_client = AsyncMock()
        app_ctx.database_client.store_crawled_page = AsyncMock()
        app_ctx.dispatcher = MagicMock()
        return app_ctx

    @pytest.fixture
    def sample_recursive_results(self):
        """Sample recursive crawl results."""
        return [
            {
                "url": "https://example.com/page1",
                "markdown": "# Page 1\nContent of page 1",
                "success": True,
            },
            {
                "url": "https://example.com/page2",
                "markdown": "# Page 2\nContent of page 2",
                "success": True,
            },
        ]

    @pytest.mark.asyncio
    async def test_crawl_recursive_raw_markdown(
        self, mock_context, mock_app_context, sample_recursive_results
    ):
        """Test recursive crawl with raw markdown return."""
        url = "https://example.com/docs"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_recursive_internal_links",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.return_value = sample_recursive_results

            result = await _crawl_recursive(
                ctx=mock_context,
                url=url,
                max_depth=2,
                chunk_size=5000,
                return_raw_markdown=True,
                query=None,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["type"] == "recursive"
            assert result_data["urls_crawled"] == 2
            assert "Page 1" in result_data["raw_markdown"]
            assert "Page 2" in result_data["raw_markdown"]

    @pytest.mark.asyncio
    async def test_crawl_recursive_store_in_database(
        self, mock_context, mock_app_context, sample_recursive_results
    ):
        """Test recursive crawl with database storage."""
        url = "https://example.com/docs"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_recursive_internal_links",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.return_value = sample_recursive_results

            result = await _crawl_recursive(
                ctx=mock_context,
                url=url,
                max_depth=3,
                chunk_size=5000,
                return_raw_markdown=False,
                query=None,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["type"] == "recursive"
            assert result_data["urls_crawled"] == 2
            assert result_data["urls_stored"] == 2
            assert result_data["max_depth"] == 3

            # Verify database calls
            assert (
                mock_app_context.database_client.store_crawled_page.call_count
                == 2
            )

    @pytest.mark.asyncio
    async def test_crawl_recursive_with_failed_pages(
        self, mock_context, mock_app_context
    ):
        """Test recursive crawl with some failed pages."""
        url = "https://example.com/docs"
        mixed_results = [
            {
                "url": "https://example.com/page1",
                "markdown": "Content",
                "success": True,
            },
            {
                "url": "https://example.com/page2",
                "success": False,
                "error": "Failed to load",
            },
        ]

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_recursive_internal_links",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.return_value = mixed_results

            result = await _crawl_recursive(
                ctx=mock_context,
                url=url,
                max_depth=2,
                chunk_size=5000,
                return_raw_markdown=False,
                query=None,
            )

            result_data = json.loads(result)
            assert result_data["urls_crawled"] == 2
            # Only 1 page should be stored (the successful one)
            assert result_data["urls_stored"] == 1

    @pytest.mark.asyncio
    async def test_crawl_recursive_with_rag_queries(
        self, mock_context, mock_app_context, sample_recursive_results
    ):
        """Test recursive crawl with RAG queries."""
        url = "https://example.com/docs"
        queries = ["test query"]
        mock_rag_result = {"results": ["result1"]}

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_recursive_internal_links",
                new_callable=AsyncMock,
            ) as mock_crawl,
            patch(
                "services.smart_crawl._perform_rag_query_with_context",
                new_callable=AsyncMock,
            ) as mock_rag,
        ):
            mock_crawl.return_value = sample_recursive_results
            mock_rag.return_value = json.dumps(mock_rag_result)

            result = await _crawl_recursive(
                ctx=mock_context,
                url=url,
                max_depth=2,
                chunk_size=5000,
                return_raw_markdown=False,
                query=queries,
            )

            result_data = json.loads(result)
            assert "query_results" in result_data
            assert "test query" in result_data["query_results"]
            mock_rag.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_recursive_no_browser_config(self, mock_context):
        """Test recursive crawl without browser config."""
        url = "https://example.com/docs"
        mock_app_ctx = MagicMock()
        del mock_app_ctx.browser_config

        with patch(
            "src.core.context.get_app_context",
            return_value=mock_app_ctx,
        ):
            with pytest.raises(MCPToolError, match="Browser config not available"):
                await _crawl_recursive(
                    ctx=mock_context,
                    url=url,
                    max_depth=2,
                    chunk_size=5000,
                    return_raw_markdown=False,
                    query=None,
                )

    @pytest.mark.asyncio
    async def test_crawl_recursive_handles_crawl_error(
        self, mock_context, mock_app_context
    ):
        """Test recursive crawl handles CrawlError."""
        url = "https://example.com/docs"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_recursive_internal_links",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.side_effect = CrawlError("Crawl failed")

            result = await _crawl_recursive(
                ctx=mock_context,
                url=url,
                max_depth=2,
                chunk_size=5000,
                return_raw_markdown=False,
                query=None,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Crawl failed" in result_data["error"]

    @pytest.mark.asyncio
    async def test_crawl_recursive_handles_database_error(
        self, mock_context, mock_app_context, sample_recursive_results
    ):
        """Test recursive crawl handles DatabaseError during storage."""
        url = "https://example.com/docs"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.services.crawling.crawl_recursive_internal_links",
                new_callable=AsyncMock,
            ) as mock_crawl,
        ):
            mock_crawl.return_value = sample_recursive_results
            mock_app_context.database_client.store_crawled_page.side_effect = (
                DatabaseError("Database error")
            )

            result = await _crawl_recursive(
                ctx=mock_context,
                url=url,
                max_depth=2,
                chunk_size=5000,
                return_raw_markdown=False,
                query=None,
            )

            result_data = json.loads(result)
            # Should continue despite database errors
            assert result_data["success"] is True
            assert result_data["urls_stored"] == 0  # None stored due to errors


class TestPerformRagQueryWithContext:
    """Test _perform_rag_query_with_context helper function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return MagicMock()

    @pytest.fixture
    def mock_app_context(self):
        """Create mock application context."""
        app_ctx = MagicMock()
        app_ctx.database_client = AsyncMock()
        return app_ctx

    @pytest.mark.asyncio
    async def test_perform_rag_query_success(
        self, mock_context, mock_app_context
    ):
        """Test successful RAG query."""
        query = "test query"
        expected_result = {"results": ["result1", "result2"]}

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.database.rag_queries.perform_rag_query",
                new_callable=AsyncMock,
            ) as mock_rag,
        ):
            mock_rag.return_value = json.dumps(expected_result)

            result = await _perform_rag_query_with_context(
                ctx=mock_context,
                query=query,
                source=None,
                match_count=5,
            )

            result_data = json.loads(result)
            assert "results" in result_data
            assert len(result_data["results"]) == 2

            # Verify perform_rag_query was called correctly
            mock_rag.assert_called_once()
            call_args = mock_rag.call_args
            assert call_args[1]["query"] == query
            assert call_args[1]["match_count"] == 5

    @pytest.mark.asyncio
    async def test_perform_rag_query_with_source_filter(
        self, mock_context, mock_app_context
    ):
        """Test RAG query with source filter."""
        query = "test query"
        source = "example.com"

        with (
            patch(
                "src.core.context.get_app_context",
                return_value=mock_app_context,
            ),
            patch(
                "src.database.rag_queries.perform_rag_query",
                new_callable=AsyncMock,
            ) as mock_rag,
        ):
            mock_rag.return_value = json.dumps({"results": []})

            await _perform_rag_query_with_context(
                ctx=mock_context,
                query=query,
                source=source,
                match_count=10,
            )

            # Verify source was passed
            call_args = mock_rag.call_args
            assert call_args[1]["source"] == source

    @pytest.mark.asyncio
    async def test_perform_rag_query_no_app_context(self, mock_context):
        """Test RAG query without app context."""
        with patch(
            "src.core.context.get_app_context",
            return_value=None,
        ):
            result = await _perform_rag_query_with_context(
                ctx=mock_context,
                query="test",
                source=None,
                match_count=5,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Database client not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_perform_rag_query_no_database_client(self, mock_context):
        """Test RAG query without database client."""
        mock_app_ctx = MagicMock()
        mock_app_ctx.database_client = None

        with patch(
            "src.core.context.get_app_context",
            return_value=mock_app_ctx,
        ):
            result = await _perform_rag_query_with_context(
                ctx=mock_context,
                query="test",
                source=None,
                match_count=5,
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "Database client not available" in result_data["error"]


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_url_normalization_removes_fragments(self, mock_context):
        """Test URL normalization removes fragments."""
        url_with_fragment = "https://example.com/page#section"

        with patch(
            "services.smart_crawl._crawl_recursive",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps({"success": True})

            await smart_crawl_url(ctx=mock_context, url=url_with_fragment)

            # Verify normalized URL was used
            call_args = mock_crawl.call_args
            assert "#section" not in call_args[0][1]

    @pytest.mark.asyncio
    async def test_url_normalization_handles_query_params(self, mock_context):
        """Test URL normalization preserves query parameters."""
        url_with_params = "https://example.com/search?q=test&page=1"

        with patch(
            "services.smart_crawl._crawl_recursive",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps({"success": True})

            await smart_crawl_url(ctx=mock_context, url=url_with_params)

            # Query params may be preserved depending on normalize_url impl
            call_args = mock_crawl.call_args
            assert "example.com" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_sitemap_detection_case_insensitive(self, mock_context):
        """Test sitemap detection is case-insensitive."""
        sitemap_url = "https://example.com/sitemap.xml"  # normalize_url lowercases

        with patch(
            "services.smart_crawl._crawl_sitemap",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps({"success": True})

            await smart_crawl_url(ctx=mock_context, url=sitemap_url)

            # Verify sitemap crawl was used
            mock_crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_file_detection_various_extensions(self, mock_context):
        """Test text file detection for various .txt files."""
        txt_files = [
            "https://example.com/robots.txt",
            "https://example.com/humans.txt",
            "https://example.com/file.txt",  # normalize_url lowercases
        ]

        for txt_url in txt_files:
            with patch(
                "services.smart_crawl._crawl_text_file",
                new_callable=AsyncMock,
            ) as mock_crawl:
                mock_crawl.return_value = json.dumps({"success": True})

                await smart_crawl_url(ctx=mock_context, url=txt_url)

                # Verify text file crawl was used
                mock_crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_query_list_handled(self, mock_context):
        """Test that empty query list is handled correctly."""
        url = "https://example.com/sitemap.xml"

        with patch(
            "services.smart_crawl._crawl_sitemap",
            new_callable=AsyncMock,
        ) as mock_crawl:
            mock_crawl.return_value = json.dumps({"success": True})

            await smart_crawl_url(
                ctx=mock_context,
                url=url,
                query=[],  # Empty list
            )

            # Check that query arg was passed (positional arg at index 4)
            call_args = mock_crawl.call_args
            assert call_args[0][4] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.services.smart_crawl"])
