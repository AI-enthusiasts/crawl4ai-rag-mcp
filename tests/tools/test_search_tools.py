"""
Unit tests for search MCP tools.

Tests the following tools:
- search: Basic web search with SearXNG integration
- agentic_search: Advanced autonomous search with iterative refinement
- analyze_code_cross_language: Cross-language code analysis
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from src.core import MCPToolError
from src.core.exceptions import DatabaseError, SearchError
from src.tools.search import register_search_tools


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


class TestSearchTool:
    """Tests for the search tool."""

    @pytest.mark.asyncio
    async def test_search_success(self, mock_mcp, mock_context):
        """Test search tool with successful search."""
        # Mock the search_and_process function
        with patch("src.tools.search.search_and_process") as mock_search:
            mock_search.return_value = json.dumps(
                {
                    "success": True,
                    "query": "test query",
                    "results": [
                        {
                            "url": "https://example.com",
                            "title": "Test Result",
                            "content": "Test content",
                        }
                    ],
                }
            )

            # Re-register to get the actual function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            # Mock track_request decorator
            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            search_func = registered_funcs["search"]

            # Call the search function
            result = await search_func(
                ctx=mock_context,
                query="test query",
                return_raw_markdown=False,
                num_results=6,
                batch_size=20,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["query"] == "test query"
            assert len(result_data["results"]) == 1

    @pytest.mark.asyncio
    async def test_search_error_handling_search_error(self, mock_mcp, mock_context):
        """Test search tool error handling with SearchError."""
        with patch("src.tools.search.search_and_process") as mock_search:
            mock_search.side_effect = SearchError("Search service unavailable")

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            search_func = registered_funcs["search"]

            # Call should raise MCPToolError
            with pytest.raises(MCPToolError) as exc_info:
                await search_func(
                    ctx=mock_context,
                    query="test query",
                    return_raw_markdown=False,
                    num_results=6,
                    batch_size=20,
                )

            assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_error_handling_database_error(self, mock_mcp, mock_context):
        """Test search tool error handling with DatabaseError."""
        with patch("src.tools.search.search_and_process") as mock_search:
            mock_search.side_effect = DatabaseError("Database connection failed")

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            search_func = registered_funcs["search"]

            # Call should raise MCPToolError
            with pytest.raises(MCPToolError) as exc_info:
                await search_func(
                    ctx=mock_context,
                    query="test query",
                    return_raw_markdown=False,
                    num_results=6,
                    batch_size=20,
                )

            assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_with_raw_markdown(self, mock_mcp, mock_context):
        """Test search tool with raw markdown return."""
        with patch("src.tools.search.search_and_process") as mock_search:
            mock_search.return_value = "# Test Markdown\n\nContent here"

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            search_func = registered_funcs["search"]

            result = await search_func(
                ctx=mock_context,
                query="test query",
                return_raw_markdown=True,
                num_results=6,
                batch_size=20,
            )

            assert "# Test Markdown" in result
            assert "Content here" in result


class TestAgenticSearchTool:
    """Tests for the agentic_search tool."""

    @pytest.mark.asyncio
    async def test_agentic_search_success(self, mock_mcp, mock_context):
        """Test agentic search tool with successful search."""
        with patch("src.tools.search.agentic_search_impl") as mock_agentic:
            mock_agentic.return_value = json.dumps(
                {
                    "success": True,
                    "query": "test query",
                    "completeness": 0.95,
                    "iterations": 1,
                    "results": [{"url": "https://example.com", "score": 0.9}],
                    "search_history": [{"action": "local_check", "status": "success"}],
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

            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            agentic_search_func = registered_funcs["agentic_search"]

            result = await agentic_search_func(
                ctx=mock_context,
                query="test query",
                completeness_threshold=0.95,
                max_iterations=3,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["completeness"] == 0.95
            assert result_data["iterations"] == 1

    @pytest.mark.asyncio
    async def test_agentic_search_with_all_parameters(self, mock_mcp, mock_context):
        """Test agentic search with all optional parameters."""
        with patch("src.tools.search.agentic_search_impl") as mock_agentic:
            mock_agentic.return_value = json.dumps(
                {
                    "success": True,
                    "query": "test query",
                    "completeness": 0.85,
                    "iterations": 2,
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

            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            agentic_search_func = registered_funcs["agentic_search"]

            await agentic_search_func(
                ctx=mock_context,
                query="test query",
                completeness_threshold=0.85,
                max_iterations=5,
                max_urls_per_iteration=10,
                url_score_threshold=0.7,
                use_search_hints=True,
            )

            # Verify all parameters were passed
            mock_agentic.assert_called_once()
            call_kwargs = mock_agentic.call_args[1]
            assert call_kwargs["query"] == "test query"
            assert call_kwargs["completeness_threshold"] == 0.85
            assert call_kwargs["max_iterations"] == 5
            assert call_kwargs["max_urls_per_iteration"] == 10
            assert call_kwargs["url_score_threshold"] == 0.7
            assert call_kwargs["use_search_hints"] is True

    @pytest.mark.asyncio
    async def test_agentic_search_error_handling(self, mock_mcp, mock_context):
        """Test agentic search error handling."""
        with patch("src.tools.search.agentic_search_impl") as mock_agentic:
            mock_agentic.side_effect = SearchError("Agentic search failed")

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            agentic_search_func = registered_funcs["agentic_search"]

            with pytest.raises(MCPToolError) as exc_info:
                await agentic_search_func(
                    ctx=mock_context,
                    query="test query",
                )

            assert "Agentic search failed" in str(exc_info.value)


class TestAnalyzeCodeCrossLanguageTool:
    """Tests for the analyze_code_cross_language tool."""

    @pytest.mark.asyncio
    async def test_analyze_code_cross_language_success(self, mock_mcp, mock_context):
        """Test cross-language code analysis with successful search."""
        with patch("src.core.context.get_app_context") as mock_get_ctx:
            with patch("src.tools.search.get_available_sources") as mock_sources:
                with patch("src.tools.search.perform_rag_query") as mock_rag:
                    # Mock app context
                    mock_app_ctx = MagicMock()
                    mock_app_ctx.database_client = MagicMock()
                    mock_get_ctx.return_value = mock_app_ctx

                    # Mock sources
                    mock_sources.return_value = json.dumps(
                        {
                            "success": True,
                            "sources": [{"source_id": "test-repo", "summary": "Test"}],
                        }
                    )

                    # Mock RAG query
                    mock_rag.return_value = json.dumps(
                        {
                            "success": True,
                            "results": [
                                {
                                    "content": "def hello(): pass",
                                    "url": "test.py",
                                    "similarity_score": 0.95,
                                    "metadata": {"language": "python"},
                                    "source": "test-repo",
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

                    with patch("src.tools.search.track_request", lambda x: lambda f: f):
                        register_search_tools(mcp_instance)

                    analyze_func = registered_funcs["analyze_code_cross_language"]

                    result = await analyze_func(
                        ctx=mock_context,
                        query="authentication logic",
                        languages=["python"],
                        match_count=10,
                    )

                    result_data = json.loads(result)
                    assert result_data["success"] is True
                    assert "results_by_language" in result_data

    @pytest.mark.asyncio
    async def test_analyze_code_no_app_context(self, mock_mcp, mock_context):
        """Test cross-language analysis when app context is not available."""
        with patch("src.core.context.get_app_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.search.track_request", lambda x: lambda f: f):
                register_search_tools(mcp_instance)

            analyze_func = registered_funcs["analyze_code_cross_language"]

            result = await analyze_func(
                ctx=mock_context,
                query="authentication logic",
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_analyze_code_with_json_string_languages(
        self, mock_mcp, mock_context
    ):
        """Test cross-language analysis with JSON string language list."""
        with patch("src.core.context.get_app_context") as mock_get_ctx:
            with patch("src.tools.search.get_available_sources") as mock_sources:
                with patch("src.tools.search.perform_rag_query") as mock_rag:
                    # Mock app context
                    mock_app_ctx = MagicMock()
                    mock_app_ctx.database_client = MagicMock()
                    mock_get_ctx.return_value = mock_app_ctx

                    # Mock sources
                    mock_sources.return_value = json.dumps(
                        {"success": True, "sources": []}
                    )

                    # Mock RAG query
                    mock_rag.return_value = json.dumps(
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

                    with patch("src.tools.search.track_request", lambda x: lambda f: f):
                        register_search_tools(mcp_instance)

                    analyze_func = registered_funcs["analyze_code_cross_language"]

                    # Test with JSON string
                    result = await analyze_func(
                        ctx=mock_context,
                        query="test query",
                        languages='["python", "javascript"]',
                    )

                    result_data = json.loads(result)
                    assert result_data["success"] is True

    @pytest.mark.asyncio
    async def test_analyze_code_database_error(self, mock_mcp, mock_context):
        """Test cross-language analysis with database error."""
        with patch("src.core.context.get_app_context") as mock_get_ctx:
            with patch("src.tools.search.get_available_sources") as mock_sources:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock database error
                mock_sources.side_effect = DatabaseError("Database connection failed")

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.search.track_request", lambda x: lambda f: f):
                    register_search_tools(mcp_instance)

                analyze_func = registered_funcs["analyze_code_cross_language"]

                result = await analyze_func(
                    ctx=mock_context,
                    query="test query",
                )

                result_data = json.loads(result)
                assert result_data["success"] is False
                assert "Database error" in result_data["error"]
