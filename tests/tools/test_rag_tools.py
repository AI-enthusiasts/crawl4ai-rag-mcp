"""
Unit tests for RAG MCP tools.

Tests the following tools:
- get_available_sources: List all indexed sources
- perform_rag_query: Semantic search over indexed content
- search_code_examples: Search for code examples in vector database
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from src.core import MCPToolError
from src.core.exceptions import DatabaseError
from src.tools.rag import register_rag_tools


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


class TestGetAvailableSourcesTool:
    """Tests for the get_available_sources tool."""

    @pytest.mark.asyncio
    async def test_get_available_sources_success(self, mock_mcp, mock_context):
        """Test get_available_sources tool with successful retrieval."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.rag.get_available_sources"
            ) as mock_get_sources_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock get_available_sources function
                mock_get_sources_func.return_value = json.dumps(
                    {
                        "success": True,
                        "sources": [
                            {
                                "source_id": "example.com",
                                "summary": "Test website",
                                "total_word_count": 5000,
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

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                get_sources_func = registered_funcs["get_available_sources"]

                result = await get_sources_func(ctx=mock_context)

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert len(result_data["sources"]) == 1
                assert result_data["sources"][0]["source_id"] == "example.com"

    @pytest.mark.asyncio
    async def test_get_available_sources_no_database_client(
        self, mock_mcp, mock_context
    ):
        """Test get_available_sources when database client is not available."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            # Mock app context without database_client
            mock_app_ctx = MagicMock()
            mock_app_ctx.database_client = None
            mock_get_ctx.return_value = mock_app_ctx

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                register_rag_tools(mcp_instance)

            get_sources_func = registered_funcs["get_available_sources"]

            result = await get_sources_func(ctx=mock_context)

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_available_sources_database_error(self, mock_mcp, mock_context):
        """Test get_available_sources with database error."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.rag.get_available_sources"
            ) as mock_get_sources_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock database error
                mock_get_sources_func.side_effect = DatabaseError(
                    "Connection failed"
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

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                get_sources_func = registered_funcs["get_available_sources"]

                with pytest.raises(MCPToolError) as exc_info:
                    await get_sources_func(ctx=mock_context)

                assert "Failed to get sources" in str(exc_info.value)


class TestPerformRagQueryTool:
    """Tests for the perform_rag_query tool."""

    @pytest.mark.asyncio
    async def test_perform_rag_query_success(self, mock_mcp, mock_context):
        """Test perform_rag_query tool with successful search."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch("src.tools.rag.perform_rag_query") as mock_rag_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock RAG query
                mock_rag_func.return_value = json.dumps(
                    {
                        "success": True,
                        "query": "test query",
                        "results": [
                            {
                                "content": "Relevant content",
                                "url": "https://example.com",
                                "similarity_score": 0.95,
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

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                rag_query_func = registered_funcs["perform_rag_query"]

                result = await rag_query_func(
                    ctx=mock_context,
                    query="test query",
                    source=None,
                    match_count=5,
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert len(result_data["results"]) == 1

    @pytest.mark.asyncio
    async def test_perform_rag_query_with_source_filter(self, mock_mcp, mock_context):
        """Test perform_rag_query tool with source filter."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch("src.tools.rag.perform_rag_query") as mock_rag_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock RAG query
                mock_rag_func.return_value = json.dumps(
                    {
                        "success": True,
                        "query": "test query",
                        "source_filter": "example.com",
                        "results": [],
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

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                rag_query_func = registered_funcs["perform_rag_query"]

                await rag_query_func(
                    ctx=mock_context,
                    query="test query",
                    source="example.com",
                    match_count=10,
                )

                # Verify source parameter was passed
                mock_rag_func.assert_called_once()
                call_args = mock_rag_func.call_args[0]
                call_kwargs = mock_rag_func.call_args[1]
                assert call_kwargs["source"] == "example.com"

    @pytest.mark.asyncio
    async def test_perform_rag_query_no_database_client(self, mock_mcp, mock_context):
        """Test perform_rag_query when database client is not available."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            # Mock app context without database_client
            mock_app_ctx = MagicMock()
            mock_app_ctx.database_client = None
            mock_get_ctx.return_value = mock_app_ctx

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                register_rag_tools(mcp_instance)

            rag_query_func = registered_funcs["perform_rag_query"]

            result = await rag_query_func(
                ctx=mock_context,
                query="test query",
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_perform_rag_query_database_error(self, mock_mcp, mock_context):
        """Test perform_rag_query with database error."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch("src.tools.rag.perform_rag_query") as mock_rag_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock database error
                mock_rag_func.side_effect = DatabaseError("Query failed")

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                rag_query_func = registered_funcs["perform_rag_query"]

                with pytest.raises(MCPToolError) as exc_info:
                    await rag_query_func(
                        ctx=mock_context,
                        query="test query",
                    )

                assert "RAG query failed" in str(exc_info.value)


class TestSearchCodeExamplesTool:
    """Tests for the search_code_examples tool."""

    @pytest.mark.asyncio
    async def test_search_code_examples_success(self, mock_mcp, mock_context):
        """Test search_code_examples tool with successful search."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch("src.tools.rag.search_code_examples_db") as mock_search_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock code example search
                mock_search_func.return_value = json.dumps(
                    {
                        "success": True,
                        "query": "authentication",
                        "results": [
                            {
                                "code": "def authenticate(user, password): ...",
                                "summary": "User authentication function",
                                "language": "python",
                                "similarity_score": 0.92,
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

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                search_code_func = registered_funcs["search_code_examples"]

                result = await search_code_func(
                    ctx=mock_context,
                    query="authentication",
                    source_id=None,
                    match_count=5,
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert len(result_data["results"]) == 1

    @pytest.mark.asyncio
    async def test_search_code_examples_with_source_filter(
        self, mock_mcp, mock_context
    ):
        """Test search_code_examples tool with source filter."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch("src.tools.rag.search_code_examples_db") as mock_search_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock code example search
                mock_search_func.return_value = json.dumps(
                    {
                        "success": True,
                        "query": "authentication",
                        "source_filter": "my-repo",
                        "results": [],
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

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                search_code_func = registered_funcs["search_code_examples"]

                await search_code_func(
                    ctx=mock_context,
                    query="authentication",
                    source_id="my-repo",
                    match_count=10,
                )

                # Verify source_id parameter was passed
                mock_search_func.assert_called_once()
                call_kwargs = mock_search_func.call_args[1]
                assert call_kwargs["source_id"] == "my-repo"

    @pytest.mark.asyncio
    async def test_search_code_examples_no_database_client(
        self, mock_mcp, mock_context
    ):
        """Test search_code_examples when database client is not available."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            # Mock app context without database_client
            mock_app_ctx = MagicMock()
            mock_app_ctx.database_client = None
            mock_get_ctx.return_value = mock_app_ctx

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                register_rag_tools(mcp_instance)

            search_code_func = registered_funcs["search_code_examples"]

            result = await search_code_func(
                ctx=mock_context,
                query="authentication",
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_search_code_examples_database_error(self, mock_mcp, mock_context):
        """Test search_code_examples with database error."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch("src.tools.rag.search_code_examples_db") as mock_search_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock database error
                mock_search_func.side_effect = DatabaseError("Search failed")

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                search_code_func = registered_funcs["search_code_examples"]

                with pytest.raises(MCPToolError) as exc_info:
                    await search_code_func(
                        ctx=mock_context,
                        query="authentication",
                    )

                assert "Code example search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_code_examples_custom_match_count(
        self, mock_mcp, mock_context
    ):
        """Test search_code_examples with custom match_count."""
        with patch("src.tools.rag.get_app_context") as mock_get_ctx:
            with patch("src.tools.rag.search_code_examples_db") as mock_search_func:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock code example search
                mock_search_func.return_value = json.dumps(
                    {"success": True, "query": "test", "results": []}
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

                with patch("src.tools.rag.track_request", lambda x: lambda f: f):
                    register_rag_tools(mcp_instance)

                search_code_func = registered_funcs["search_code_examples"]

                await search_code_func(
                    ctx=mock_context,
                    query="test",
                    match_count=20,
                )

                # Verify match_count parameter was passed
                mock_search_func.assert_called_once()
                call_kwargs = mock_search_func.call_args[1]
                assert call_kwargs["match_count"] == 20
