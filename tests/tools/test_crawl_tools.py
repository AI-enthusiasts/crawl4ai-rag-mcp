"""
Unit tests for crawl MCP tools.

Tests the following tools:
- scrape_urls: Scrape one or more URLs and store content
- smart_crawl_url: Intelligently crawl URLs with type detection
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from src.core import MCPToolError
from src.tools.crawl import register_crawl_tools


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP instance."""
    mcp = MagicMock()
    mcp.tool = lambda: lambda func: func
    return mcp


@pytest.fixture
def mock_context():
    """Create a mock FastMCP Context."""
    ctx = MagicMock(spec=Context)
    return ctx


class TestScrapeUrlsTool:
    """Tests for the scrape_urls tool."""

    @pytest.mark.asyncio
    async def test_scrape_urls_single_url_success(self, mock_mcp, mock_context):
        """Test scrape_urls tool with single URL."""
        with patch("src.tools.crawl.process_urls_for_mcp") as mock_process:
            with patch("src.tools.crawl.clean_url") as mock_clean:
                mock_clean.return_value = "https://example.com"
                mock_process.return_value = json.dumps(
                    {
                        "success": True,
                        "results": [
                            {
                                "url": "https://example.com",
                                "success": True,
                                "chunks_stored": 10,
                            }
                        ],
                    }
                )

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                    register_crawl_tools(mcp_instance)

                scrape_func = registered_funcs["scrape_urls"]

                result = await scrape_func(
                    ctx=mock_context,
                    url="https://example.com",
                    batch_size=20,
                    return_raw_markdown=False,
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert len(result_data["results"]) == 1

    @pytest.mark.asyncio
    async def test_scrape_urls_multiple_urls_list(self, mock_mcp, mock_context):
        """Test scrape_urls tool with list of URLs."""
        with patch("src.tools.crawl.process_urls_for_mcp") as mock_process:
            with patch("src.tools.crawl.clean_url") as mock_clean:
                mock_clean.side_effect = lambda x: x  # Return as-is
                mock_process.return_value = json.dumps(
                    {
                        "success": True,
                        "results": [
                            {"url": "https://example1.com", "success": True},
                            {"url": "https://example2.com", "success": True},
                        ],
                    }
                )

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                    register_crawl_tools(mcp_instance)

                scrape_func = registered_funcs["scrape_urls"]

                result = await scrape_func(
                    ctx=mock_context,
                    url=["https://example1.com", "https://example2.com"],
                    batch_size=20,
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert len(result_data["results"]) == 2

    @pytest.mark.asyncio
    async def test_scrape_urls_json_array_string(self, mock_mcp, mock_context):
        """Test scrape_urls tool with JSON array string."""
        with patch("src.tools.crawl.process_urls_for_mcp") as mock_process:
            with patch("src.tools.crawl.clean_url") as mock_clean:
                mock_clean.side_effect = lambda x: x
                mock_process.return_value = json.dumps(
                    {"success": True, "results": []}
                )

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                    register_crawl_tools(mcp_instance)

                scrape_func = registered_funcs["scrape_urls"]

                # Test with JSON array string
                result = await scrape_func(
                    ctx=mock_context,
                    url='["https://example1.com", "https://example2.com"]',
                )

                # Verify clean_url was called with both URLs
                assert mock_clean.call_count == 2

    @pytest.mark.asyncio
    async def test_scrape_urls_no_valid_urls(self, mock_mcp, mock_context):
        """Test scrape_urls tool with no valid URLs."""
        with patch("src.tools.crawl.clean_url") as mock_clean:
            mock_clean.return_value = None  # Invalid URL

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            scrape_func = registered_funcs["scrape_urls"]

            with pytest.raises(MCPToolError) as exc_info:
                await scrape_func(
                    ctx=mock_context,
                    url="invalid-url",
                )

            assert "No valid URLs" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_scrape_urls_input_too_large(self, mock_mcp, mock_context):
        """Test scrape_urls tool with input size limit."""
        with patch("src.tools.crawl.clean_url") as mock_clean:
            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            scrape_func = registered_funcs["scrape_urls"]

            # Create a very large input (over 50KB)
            large_url = "https://example.com/" + "a" * 60000

            with pytest.raises(MCPToolError) as exc_info:
                await scrape_func(
                    ctx=mock_context,
                    url=large_url,
                )

            assert "Input too large" in str(exc_info.value) or "Scraping failed" in str(
                exc_info.value
            )

    @pytest.mark.asyncio
    async def test_scrape_urls_with_raw_markdown(self, mock_mcp, mock_context):
        """Test scrape_urls tool with raw markdown return."""
        with patch("src.tools.crawl.process_urls_for_mcp") as mock_process:
            with patch("src.tools.crawl.clean_url") as mock_clean:
                mock_clean.return_value = "https://example.com"
                mock_process.return_value = "# Markdown Content\n\nRaw markdown here"

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                    register_crawl_tools(mcp_instance)

                scrape_func = registered_funcs["scrape_urls"]

                result = await scrape_func(
                    ctx=mock_context,
                    url="https://example.com",
                    return_raw_markdown=True,
                )

                assert "# Markdown Content" in result
                assert "Raw markdown here" in result

    @pytest.mark.asyncio
    async def test_scrape_urls_error_handling(self, mock_mcp, mock_context):
        """Test scrape_urls tool error handling."""
        with patch("src.tools.crawl.process_urls_for_mcp") as mock_process:
            with patch("src.tools.crawl.clean_url") as mock_clean:
                mock_clean.return_value = "https://example.com"
                mock_process.side_effect = Exception("Network error")

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                    register_crawl_tools(mcp_instance)

                scrape_func = registered_funcs["scrape_urls"]

                with pytest.raises(MCPToolError) as exc_info:
                    await scrape_func(
                        ctx=mock_context,
                        url="https://example.com",
                    )

                assert "Scraping failed" in str(exc_info.value)


class TestSmartCrawlUrlTool:
    """Tests for the smart_crawl_url tool."""

    @pytest.mark.asyncio
    async def test_smart_crawl_url_success(self, mock_mcp, mock_context):
        """Test smart_crawl_url tool with successful crawl."""
        with patch(
            "src.tools.crawl.smart_crawl_url_service_impl"
        ) as mock_smart_crawl:
            mock_smart_crawl.return_value = json.dumps(
                {
                    "success": True,
                    "url": "https://example.com",
                    "pages_crawled": 10,
                    "chunks_stored": 50,
                }
            )

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            smart_crawl_func = registered_funcs["smart_crawl_url"]

            result = await smart_crawl_func(
                ctx=mock_context,
                url="https://example.com",
                max_depth=3,
                chunk_size=5000,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["pages_crawled"] == 10

    @pytest.mark.asyncio
    async def test_smart_crawl_url_with_query_list(self, mock_mcp, mock_context):
        """Test smart_crawl_url tool with query parameter as list."""
        with patch(
            "src.tools.crawl.smart_crawl_url_service_impl"
        ) as mock_smart_crawl:
            mock_smart_crawl.return_value = json.dumps(
                {
                    "success": True,
                    "url": "https://example.com",
                    "rag_results": [{"query": "test1"}, {"query": "test2"}],
                }
            )

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            smart_crawl_func = registered_funcs["smart_crawl_url"]

            result = await smart_crawl_func(
                ctx=mock_context,
                url="https://example.com",
                query=["test1", "test2"],
            )

            # Verify the query parameter was passed correctly
            mock_smart_crawl.assert_called_once()
            call_kwargs = mock_smart_crawl.call_args[1]
            assert call_kwargs["query"] == ["test1", "test2"]

    @pytest.mark.asyncio
    async def test_smart_crawl_url_with_query_json_string(
        self, mock_mcp, mock_context
    ):
        """Test smart_crawl_url tool with query parameter as JSON string."""
        with patch(
            "src.tools.crawl.smart_crawl_url_service_impl"
        ) as mock_smart_crawl:
            mock_smart_crawl.return_value = json.dumps({"success": True})

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            smart_crawl_func = registered_funcs["smart_crawl_url"]

            result = await smart_crawl_func(
                ctx=mock_context,
                url="https://example.com",
                query='["test1", "test2"]',
            )

            # Verify the JSON string was parsed correctly
            mock_smart_crawl.assert_called_once()
            call_kwargs = mock_smart_crawl.call_args[1]
            assert call_kwargs["query"] == ["test1", "test2"]

    @pytest.mark.asyncio
    async def test_smart_crawl_url_with_raw_markdown(self, mock_mcp, mock_context):
        """Test smart_crawl_url tool with raw markdown return."""
        with patch(
            "src.tools.crawl.smart_crawl_url_service_impl"
        ) as mock_smart_crawl:
            mock_smart_crawl.return_value = json.dumps(
                {
                    "success": True,
                    "raw_markdown": "# Page Title\n\nContent here",
                }
            )

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            smart_crawl_func = registered_funcs["smart_crawl_url"]

            result = await smart_crawl_func(
                ctx=mock_context,
                url="https://example.com",
                return_raw_markdown=True,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert "raw_markdown" in result_data

    @pytest.mark.asyncio
    async def test_smart_crawl_url_all_parameters(self, mock_mcp, mock_context):
        """Test smart_crawl_url tool with all optional parameters."""
        with patch(
            "src.tools.crawl.smart_crawl_url_service_impl"
        ) as mock_smart_crawl:
            mock_smart_crawl.return_value = json.dumps({"success": True})

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            smart_crawl_func = registered_funcs["smart_crawl_url"]

            await smart_crawl_func(
                ctx=mock_context,
                url="https://example.com",
                max_depth=5,
                chunk_size=10000,
                return_raw_markdown=True,
                query=["test1", "test2"],
            )

            # Verify all parameters were passed
            mock_smart_crawl.assert_called_once()
            call_kwargs = mock_smart_crawl.call_args[1]
            assert call_kwargs["url"] == "https://example.com"
            assert call_kwargs["max_depth"] == 5
            assert call_kwargs["chunk_size"] == 10000
            assert call_kwargs["return_raw_markdown"] is True
            assert call_kwargs["query"] == ["test1", "test2"]

    @pytest.mark.asyncio
    async def test_smart_crawl_url_error_handling(self, mock_mcp, mock_context):
        """Test smart_crawl_url tool error handling."""
        with patch(
            "src.tools.crawl.smart_crawl_url_service_impl"
        ) as mock_smart_crawl:
            mock_smart_crawl.side_effect = Exception("Crawl failed")

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            smart_crawl_func = registered_funcs["smart_crawl_url"]

            with pytest.raises(MCPToolError) as exc_info:
                await smart_crawl_func(
                    ctx=mock_context,
                    url="https://example.com",
                )

            assert "Smart crawl failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_smart_crawl_url_sitemap_detection(self, mock_mcp, mock_context):
        """Test smart_crawl_url tool with sitemap URL."""
        with patch(
            "src.tools.crawl.smart_crawl_url_service_impl"
        ) as mock_smart_crawl:
            mock_smart_crawl.return_value = json.dumps(
                {
                    "success": True,
                    "url_type": "sitemap",
                    "urls_extracted": 50,
                    "pages_crawled": 50,
                }
            )

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.crawl.track_request", lambda x: lambda f: f):
                register_crawl_tools(mcp_instance)

            smart_crawl_func = registered_funcs["smart_crawl_url"]

            result = await smart_crawl_func(
                ctx=mock_context,
                url="https://example.com/sitemap.xml",
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
