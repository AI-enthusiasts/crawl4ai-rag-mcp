"""
Unit tests for src/services/crawling.py

Tests all crawling service functions with proper mocking to avoid
actual network calls and browser automation.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crawl4ai import BrowserConfig, CacheMode, CrawlerRunConfig, MemoryAdaptiveDispatcher

from src.core.exceptions import CrawlError, DatabaseError
from src.services.crawling import (
    crawl_batch,
    crawl_markdown_file,
    crawl_recursive_internal_links,
    crawl_urls_for_agentic_search,
    process_urls_for_mcp,
    should_filter_url,
    track_memory,
)


class TestTrackMemory:
    """Test memory tracking context manager"""

    @pytest.mark.asyncio
    async def test_track_memory_basic(self):
        """Test basic memory tracking without results"""
        with patch("src.services.crawling.get_memory_stats") as mock_memory:
            # Mock memory stats: (percent, available_gb, total_gb)
            mock_memory.side_effect = [
                (50.0, 8.0, 16.0),  # Start
                (55.0, 7.5, 16.0),  # End
            ]

            async with track_memory("test_operation") as context:
                assert context == {"results": None}

            # Verify memory stats were called twice (start and end)
            assert mock_memory.call_count == 2

    @pytest.mark.asyncio
    async def test_track_memory_with_results(self):
        """Test memory tracking with crawl results"""
        with patch("src.services.crawling.get_memory_stats") as mock_memory:
            mock_memory.side_effect = [
                (50.0, 8.0, 16.0),  # Start
                (60.0, 7.0, 16.0),  # End
            ]

            # Create mock results with dispatch stats
            mock_result = MagicMock()
            mock_result.dispatch_result = MagicMock()
            mock_result.dispatch_result.memory_usage = 512.0
            mock_result.dispatch_result.peak_memory = 1024.0

            async with track_memory("test_operation") as context:
                context["results"] = [mock_result]

            # Verify memory delta was logged
            assert mock_memory.call_count == 2

    @pytest.mark.asyncio
    async def test_track_memory_empty_dispatch_stats(self):
        """Test memory tracking with empty dispatch stats"""
        with patch("src.services.crawling.get_memory_stats") as mock_memory:
            mock_memory.side_effect = [
                (50.0, 8.0, 16.0),
                (55.0, 7.5, 16.0),
            ]

            async with track_memory("test_operation") as context:
                context["results"] = []

            assert mock_memory.call_count == 2


class TestCrawlMarkdownFile:
    """Test crawl_markdown_file function"""

    @pytest.mark.asyncio
    async def test_crawl_markdown_file_success(self):
        """Test successful markdown file crawling"""
        browser_config = BrowserConfig(headless=True)
        url = "https://example.com/file.md"

        # Mock AsyncWebCrawler
        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        # Mock successful result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test Content"
        mock_result.error_message = None
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler):
            result = await crawl_markdown_file(browser_config, url)

        assert len(result) == 1
        assert result[0]["url"] == url
        assert result[0]["markdown"] == "# Test Content"

    @pytest.mark.asyncio
    async def test_crawl_markdown_file_failure(self):
        """Test markdown file crawling failure"""
        browser_config = BrowserConfig(headless=True)
        url = "https://example.com/file.md"

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        # Mock failed result
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.markdown = None
        mock_result.error_message = "Connection failed"
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler):
            result = await crawl_markdown_file(browser_config, url)

        assert result == []

    @pytest.mark.asyncio
    async def test_crawl_markdown_file_no_markdown(self):
        """Test crawling when no markdown content is returned"""
        browser_config = BrowserConfig(headless=True)
        url = "https://example.com/file.md"

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = ""  # Empty markdown
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler):
            result = await crawl_markdown_file(browser_config, url)

        assert result == []


class TestCrawlBatch:
    """Test crawl_batch function"""

    @pytest.mark.asyncio
    async def test_crawl_batch_success(self):
        """Test successful batch crawling"""
        browser_config = BrowserConfig(headless=True)
        urls = ["https://example.com/page1", "https://example.com/page2"]
        dispatcher = MemoryAdaptiveDispatcher()

        # Mock crawler
        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        # Mock successful results
        mock_result1 = MagicMock()
        mock_result1.success = True
        mock_result1.url = urls[0]
        mock_result1.markdown = "# Page 1"
        mock_result1.links = {"internal": [], "external": []}

        mock_result2 = MagicMock()
        mock_result2.success = True
        mock_result2.url = urls[1]
        mock_result2.markdown = "# Page 2"
        mock_result2.links = {"internal": [], "external": []}

        mock_crawler.arun_many = AsyncMock(return_value=[mock_result1, mock_result2])

        # Mock URL validation to pass
        with (
            patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
            patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
            patch("src.utils.validation.validate_urls_for_crawling") as mock_validate,
        ):
            mock_validate.return_value = {"valid": True, "urls": urls}
            results = await crawl_batch(browser_config, urls, dispatcher)

        assert len(results) == 2
        assert results[0]["url"] == urls[0]
        assert results[0]["markdown"] == "# Page 1"
        assert results[1]["url"] == urls[1]
        assert results[1]["markdown"] == "# Page 2"

    @pytest.mark.asyncio
    async def test_crawl_batch_partial_failure(self):
        """Test batch crawling with some failures"""
        browser_config = BrowserConfig(headless=True)
        urls = ["https://example.com/page1", "https://example.com/page2"]
        dispatcher = MemoryAdaptiveDispatcher()

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        # One success, one failure
        mock_result1 = MagicMock()
        mock_result1.success = True
        mock_result1.url = urls[0]
        mock_result1.markdown = "# Page 1"
        mock_result1.links = {"internal": []}

        mock_result2 = MagicMock()
        mock_result2.success = False
        mock_result2.url = urls[1]
        mock_result2.markdown = None

        mock_crawler.arun_many = AsyncMock(return_value=[mock_result1, mock_result2])

        with (
            patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
            patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
            patch("src.utils.validation.validate_urls_for_crawling") as mock_validate,
        ):
            mock_validate.return_value = {"valid": True, "urls": urls}
            results = await crawl_batch(browser_config, urls, dispatcher)

        # Only successful result returned
        assert len(results) == 1
        assert results[0]["url"] == urls[0]

    @pytest.mark.asyncio
    async def test_crawl_batch_invalid_urls(self):
        """Test batch crawling with invalid URLs"""
        browser_config = BrowserConfig(headless=True)
        urls = ["not-a-url", "also-invalid"]
        dispatcher = MemoryAdaptiveDispatcher()

        # Mock validation to fail
        with patch("src.utils.validation.validate_urls_for_crawling") as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "error": "Invalid URLs",
                "invalid_urls": urls,
            }

            with pytest.raises(ValueError, match="Invalid URLs for crawling"):
                await crawl_batch(browser_config, urls, dispatcher)

    @pytest.mark.asyncio
    async def test_crawl_batch_crawl_error(self):
        """Test batch crawling with CrawlError"""
        browser_config = BrowserConfig(headless=True)
        urls = ["https://example.com/page1"]
        dispatcher = MemoryAdaptiveDispatcher()

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()
        mock_crawler.arun_many = AsyncMock(side_effect=CrawlError("Crawling failed"))

        with (
            patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
            patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
            patch("src.utils.validation.validate_urls_for_crawling") as mock_validate,
        ):
            mock_validate.return_value = {"valid": True, "urls": urls}
            with pytest.raises(CrawlError, match="Crawling failed for 1 URLs"):
                await crawl_batch(browser_config, urls, dispatcher)

    @pytest.mark.asyncio
    async def test_crawl_batch_unexpected_error(self):
        """Test batch crawling with unexpected error"""
        browser_config = BrowserConfig(headless=True)
        urls = ["https://example.com/page1"]
        dispatcher = MemoryAdaptiveDispatcher()

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()
        mock_crawler.arun_many = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with (
            patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
            patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
            patch("src.utils.validation.validate_urls_for_crawling") as mock_validate,
        ):
            mock_validate.return_value = {"valid": True, "urls": urls}
            with pytest.raises(CrawlError, match="Crawling failed for 1 URLs"):
                await crawl_batch(browser_config, urls, dispatcher)


class TestCrawlRecursiveInternalLinks:
    """Test crawl_recursive_internal_links function"""

    @pytest.mark.asyncio
    async def test_crawl_recursive_success(self):
        """Test successful recursive crawling"""
        browser_config = BrowserConfig(headless=True)
        start_urls = ["https://example.com/page1"]
        dispatcher = MemoryAdaptiveDispatcher()

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        # First depth: return page1 with link to page2
        mock_result1 = MagicMock()
        mock_result1.success = True
        mock_result1.url = "https://example.com/page1"
        mock_result1.markdown = "# Page 1"
        mock_result1.links = {"internal": [{"href": "https://example.com/page2"}]}

        # Second depth: return page2 with no links
        mock_result2 = MagicMock()
        mock_result2.success = True
        mock_result2.url = "https://example.com/page2"
        mock_result2.markdown = "# Page 2"
        mock_result2.links = {"internal": []}

        mock_crawler.arun_many = AsyncMock(side_effect=[[mock_result1], [mock_result2]])

        with (
            patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
            patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
            patch("src.services.crawling.normalize_url", side_effect=lambda x: x),
        ):
            results = await crawl_recursive_internal_links(
                browser_config, start_urls, dispatcher, max_depth=2
            )

        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/page1"
        assert results[1]["url"] == "https://example.com/page2"

    @pytest.mark.asyncio
    async def test_crawl_recursive_max_depth(self):
        """Test recursive crawling respects max_depth"""
        browser_config = BrowserConfig(headless=True)
        start_urls = ["https://example.com/page1"]
        dispatcher = MemoryAdaptiveDispatcher()

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        # Each call returns a page with a new link
        def make_result(url, next_url=None):
            result = MagicMock()
            result.success = True
            result.url = url
            result.markdown = f"# {url}"
            result.links = {"internal": [{"href": next_url}] if next_url else []}
            return result

        # Will be called 3 times (max_depth=3)
        mock_crawler.arun_many = AsyncMock(
            side_effect=[
                [make_result("https://example.com/page1", "https://example.com/page2")],
                [make_result("https://example.com/page2", "https://example.com/page3")],
                [make_result("https://example.com/page3", "https://example.com/page4")],
            ]
        )

        with (
            patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
            patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
            patch("src.services.crawling.normalize_url", side_effect=lambda x: x),
        ):
            results = await crawl_recursive_internal_links(
                browser_config, start_urls, dispatcher, max_depth=3
            )

        # Should have exactly 3 pages (depth 0, 1, 2)
        assert len(results) == 3
        assert mock_crawler.arun_many.call_count == 3

    @pytest.mark.asyncio
    async def test_crawl_recursive_no_new_urls(self):
        """Test recursive crawling stops when no new URLs"""
        browser_config = BrowserConfig(headless=True)
        start_urls = ["https://example.com/page1"]
        dispatcher = MemoryAdaptiveDispatcher()

        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock()

        # Single page with no internal links
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.url = "https://example.com/page1"
        mock_result.markdown = "# Page 1"
        mock_result.links = {"internal": []}

        mock_crawler.arun_many = AsyncMock(return_value=[mock_result])

        with (
            patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
            patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
            patch("src.services.crawling.normalize_url", side_effect=lambda x: x),
        ):
            results = await crawl_recursive_internal_links(
                browser_config, start_urls, dispatcher, max_depth=3
            )

        # Only one page crawled, stops early
        assert len(results) == 1
        assert mock_crawler.arun_many.call_count == 1


class TestShouldFilterUrl:
    """Test should_filter_url function"""

    def test_filter_disabled(self):
        """Test filtering when disabled"""
        url = "https://github.com/user/repo/commits/main"
        assert should_filter_url(url, enable_filtering=False) is False

    def test_filter_github_commits(self):
        """Test filtering GitHub commit URLs"""
        urls = [
            "https://github.com/user/repo/commits/main",
            "https://github.com/user/repo/commit/abc123",
        ]
        for url in urls:
            assert should_filter_url(url, enable_filtering=True) is True

    def test_no_filter_normal_urls(self):
        """Test normal URLs are not filtered"""
        urls = [
            "https://example.com/docs/guide",
            "https://github.com/user/repo/blob/main/README.md",
            "https://example.com/api/reference",
        ]
        for url in urls:
            assert should_filter_url(url, enable_filtering=True) is False


class TestProcessUrlsForMcp:
    """Test process_urls_for_mcp function"""

    @pytest.mark.asyncio
    async def test_process_urls_success(self):
        """Test successful URL processing for MCP"""
        # Create mock context
        mock_ctx = MagicMock()
        mock_crawl4ai_ctx = MagicMock()
        mock_crawl4ai_ctx.browser_config = BrowserConfig(headless=True)
        mock_crawl4ai_ctx.database_client = AsyncMock()
        mock_crawl4ai_ctx.dispatcher = MemoryAdaptiveDispatcher()
        mock_ctx.crawl4ai_context = mock_crawl4ai_ctx

        urls = ["https://example.com/page1"]

        # Mock crawl_batch
        with (
            patch("src.services.crawling.crawl_batch") as mock_crawl,
            patch("src.services.crawling.smart_chunk_markdown") as mock_chunk,
            patch("src.services.crawling.extract_domain_from_url") as mock_domain,
            patch("src.services.crawling.add_documents_to_database") as mock_add,
        ):
            mock_crawl.return_value = [
                {"url": urls[0], "markdown": "# Test Content"}
            ]
            mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
            mock_domain.return_value = "example.com"

            result = await process_urls_for_mcp(mock_ctx, urls)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["total_urls"] == 1
        assert len(result_data["results"]) == 1
        assert result_data["results"][0]["success"] is True
        assert result_data["results"][0]["chunks_stored"] == 2

    @pytest.mark.asyncio
    async def test_process_urls_return_raw_markdown(self):
        """Test URL processing with raw markdown return"""
        mock_ctx = MagicMock()
        mock_crawl4ai_ctx = MagicMock()
        mock_crawl4ai_ctx.browser_config = BrowserConfig(headless=True)
        mock_crawl4ai_ctx.database_client = AsyncMock()
        mock_crawl4ai_ctx.dispatcher = MemoryAdaptiveDispatcher()
        mock_ctx.crawl4ai_context = mock_crawl4ai_ctx

        urls = ["https://example.com/page1"]

        with patch("src.services.crawling.crawl_batch") as mock_crawl:
            mock_crawl.return_value = [
                {"url": urls[0], "markdown": "# Test Content"}
            ]

            result = await process_urls_for_mcp(
                mock_ctx, urls, return_raw_markdown=True
            )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["results"]) == 1
        assert result_data["results"][0]["markdown"] == "# Test Content"

    @pytest.mark.asyncio
    async def test_process_urls_no_context(self):
        """Test URL processing without context"""
        mock_ctx = MagicMock()
        mock_ctx.crawl4ai_context = None

        with patch("src.core.context.get_app_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await process_urls_for_mcp(mock_ctx, ["https://example.com"])

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "context not available" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_process_urls_database_error(self):
        """Test URL processing with database error"""
        mock_ctx = MagicMock()
        mock_crawl4ai_ctx = MagicMock()
        mock_crawl4ai_ctx.browser_config = BrowserConfig(headless=True)
        mock_crawl4ai_ctx.database_client = AsyncMock()
        mock_crawl4ai_ctx.dispatcher = MemoryAdaptiveDispatcher()
        mock_ctx.crawl4ai_context = mock_crawl4ai_ctx

        urls = ["https://example.com/page1"]

        with (
            patch("src.services.crawling.crawl_batch") as mock_crawl,
            patch("src.services.crawling.smart_chunk_markdown") as mock_chunk,
            patch("src.services.crawling.add_documents_to_database") as mock_add,
        ):
            mock_crawl.return_value = [
                {"url": urls[0], "markdown": "# Test Content"}
            ]
            mock_chunk.return_value = ["Chunk 1"]
            mock_add.side_effect = DatabaseError("Database connection failed")

            result = await process_urls_for_mcp(mock_ctx, urls)

        result_data = json.loads(result)
        assert result_data["success"] is True  # Overall success
        assert result_data["results"][0]["success"] is False  # Individual failure
        assert "Database connection failed" in result_data["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_process_urls_no_chunks(self):
        """Test URL processing when no chunks generated"""
        mock_ctx = MagicMock()
        mock_crawl4ai_ctx = MagicMock()
        mock_crawl4ai_ctx.browser_config = BrowserConfig(headless=True)
        mock_crawl4ai_ctx.database_client = AsyncMock()
        mock_crawl4ai_ctx.dispatcher = MemoryAdaptiveDispatcher()
        mock_ctx.crawl4ai_context = mock_crawl4ai_ctx

        urls = ["https://example.com/page1"]

        with (
            patch("src.services.crawling.crawl_batch") as mock_crawl,
            patch("src.services.crawling.smart_chunk_markdown") as mock_chunk,
        ):
            mock_crawl.return_value = [
                {"url": urls[0], "markdown": "# Test Content"}
            ]
            mock_chunk.return_value = []  # No chunks

            result = await process_urls_for_mcp(mock_ctx, urls)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["results"][0]["success"] is False
        assert result_data["results"][0]["error"] == "No content to store"

    @pytest.mark.asyncio
    async def test_process_urls_exception(self):
        """Test URL processing with general exception"""
        mock_ctx = MagicMock()
        mock_crawl4ai_ctx = MagicMock()
        mock_crawl4ai_ctx.browser_config = BrowserConfig(headless=True)
        mock_crawl4ai_ctx.database_client = AsyncMock()
        mock_crawl4ai_ctx.dispatcher = MemoryAdaptiveDispatcher()
        mock_ctx.crawl4ai_context = mock_crawl4ai_ctx

        urls = ["https://example.com/page1"]

        with patch("src.services.crawling.crawl_batch") as mock_crawl:
            mock_crawl.side_effect = Exception("Unexpected error")

            result = await process_urls_for_mcp(mock_ctx, urls)

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Unexpected error" in result_data["error"]


class TestCrawlUrlsForAgenticSearch:
    """Test crawl_urls_for_agentic_search function"""

    @pytest.mark.asyncio
    async def test_agentic_crawl_success(self):
        """Test successful agentic search crawling"""
        mock_ctx = MagicMock()

        # Mock context
        with patch("src.core.context.get_app_context") as mock_get_ctx:
            mock_app_ctx = MagicMock()
            mock_app_ctx.browser_config = BrowserConfig(headless=True)
            mock_app_ctx.database_client = AsyncMock()
            mock_app_ctx.dispatcher = MemoryAdaptiveDispatcher()
            mock_get_ctx.return_value = mock_app_ctx

            # Mock crawler
            mock_crawler = MagicMock()
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock()

            # Mock crawl result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.url = "https://example.com/page1"
            mock_result.markdown = "# Test Content"
            mock_result.links = {"internal": []}

            mock_crawler.arun_many = AsyncMock(return_value=[mock_result])

            with (
                patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
                patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
                patch("src.services.crawling.smart_chunk_markdown", return_value=["Chunk 1"]),
                patch("src.services.crawling.extract_domain_from_url", return_value="example.com"),
                patch("src.services.crawling.add_documents_to_database") as mock_add,
                patch("src.services.crawling.normalize_url", side_effect=lambda x: x),
            ):
                result = await crawl_urls_for_agentic_search(
                    mock_ctx, ["https://example.com/page1"], max_pages=10
                )

        assert result["success"] is True
        assert result["urls_crawled"] == 1
        assert result["urls_stored"] == 1
        assert result["urls_filtered"] == 0

    @pytest.mark.asyncio
    async def test_agentic_crawl_with_filtering(self):
        """Test agentic search with URL filtering"""
        mock_ctx = MagicMock()

        with patch("src.core.context.get_app_context") as mock_get_ctx:
            mock_app_ctx = MagicMock()
            mock_app_ctx.browser_config = BrowserConfig(headless=True)
            mock_app_ctx.database_client = AsyncMock()
            mock_app_ctx.dispatcher = MemoryAdaptiveDispatcher()
            mock_get_ctx.return_value = mock_app_ctx

            mock_crawler = MagicMock()
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock()

            # Page 1 with link to commits page (should be filtered)
            mock_result1 = MagicMock()
            mock_result1.success = True
            mock_result1.url = "https://github.com/user/repo"
            mock_result1.markdown = "# Repo"
            mock_result1.links = {
                "internal": [{"href": "https://github.com/user/repo/commits"}]
            }

            mock_crawler.arun_many = AsyncMock(return_value=[mock_result1])

            with (
                patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
                patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
                patch("src.services.crawling.smart_chunk_markdown", return_value=["Chunk 1"]),
                patch("src.services.crawling.extract_domain_from_url", return_value="github.com"),
                patch("src.services.crawling.add_documents_to_database"),
                patch("src.services.crawling.normalize_url", side_effect=lambda x: x),
            ):
                result = await crawl_urls_for_agentic_search(
                    mock_ctx,
                    ["https://github.com/user/repo"],
                    max_pages=10,
                    enable_url_filtering=True,
                )

        assert result["success"] is True
        assert result["urls_filtered"] == 1

    @pytest.mark.asyncio
    async def test_agentic_crawl_no_context(self):
        """Test agentic search without context"""
        mock_ctx = MagicMock()

        with patch("src.core.context.get_app_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await crawl_urls_for_agentic_search(
                mock_ctx, ["https://example.com"]
            )

        assert result["success"] is False
        assert "context not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_agentic_crawl_exception(self):
        """Test agentic search with general exception"""
        mock_ctx = MagicMock()

        with patch("src.core.context.get_app_context") as mock_get_ctx:
            mock_app_ctx = MagicMock()
            mock_app_ctx.browser_config = BrowserConfig(headless=True)
            mock_app_ctx.database_client = AsyncMock()
            mock_app_ctx.dispatcher = MemoryAdaptiveDispatcher()
            mock_get_ctx.return_value = mock_app_ctx

            mock_crawler = MagicMock()
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock()
            mock_crawler.arun_many = AsyncMock(side_effect=RuntimeError("Unexpected error"))

            with (
                patch("src.services.crawling.AsyncWebCrawler", return_value=mock_crawler),
                patch("src.services.crawling.get_memory_stats", return_value=(50.0, 8.0, 16.0)),
                patch("src.services.crawling.normalize_url", side_effect=lambda x: x),
            ):
                result = await crawl_urls_for_agentic_search(
                    mock_ctx, ["https://example.com/page1"], max_pages=10
                )

        assert result["success"] is False
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
