"""
Unit tests for validation MCP tools.

Tests the following tools:
- extract_and_index_repository_code: Index code from Neo4j to Qdrant
- smart_code_search: Validated semantic code search
- check_ai_script_hallucinations_enhanced: Enhanced hallucination detection
- get_script_analysis_info: Helper for script analysis setup
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from src.core import MCPToolError
from src.core.exceptions import DatabaseError, KnowledgeGraphError, ValidationError
from src.tools.validation import register_validation_tools


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


class TestExtractAndIndexRepositoryCodeTool:
    """Tests for the extract_and_index_repository_code tool."""

    @pytest.mark.asyncio
    async def test_extract_and_index_success(self, mock_mcp, mock_context):
        """Test extract_and_index_repository_code tool with successful indexing."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.validation.extract_repository_code"
            ) as mock_extract:
                with patch(
                    "src.tools.validation.create_embeddings_batch"
                ) as mock_embeddings:
                    # Mock app context
                    mock_app_ctx = MagicMock()
                    mock_app_ctx.repo_extractor = MagicMock()
                    mock_app_ctx.database_client = MagicMock()
                    mock_app_ctx.database_client.delete_repository_code_examples = (
                        AsyncMock()
                    )
                    mock_app_ctx.database_client.add_code_examples = AsyncMock()
                    mock_app_ctx.database_client.update_source_info = AsyncMock()
                    mock_get_ctx.return_value = mock_app_ctx

                    # Mock extraction
                    mock_extract.return_value = {
                        "success": True,
                        "code_examples": [
                            {
                                "name": "test_function",
                                "full_name": "test_module.test_function",
                                "code_type": "function",
                                "code_text": "def test_function(): pass",
                                "embedding_text": "function test_function",
                                "metadata": {"language": "python"},
                            }
                        ],
                        "extraction_summary": {
                            "classes": 10,
                            "methods": 50,
                            "functions": 20,
                        },
                    }

                    # Mock embeddings
                    mock_embeddings.return_value = [[0.1] * 1536]

                    # Register and get function
                    mcp_instance = MagicMock()
                    registered_funcs = {}

                    def mock_tool_decorator():
                        def decorator(func):
                            registered_funcs[func.__name__] = func
                            return func

                        return decorator

                    mcp_instance.tool = mock_tool_decorator

                    with patch(
                        "src.tools.validation.track_request", lambda x: lambda f: f
                    ):
                        register_validation_tools(mcp_instance)

                    extract_func = registered_funcs[
                        "extract_and_index_repository_code"
                    ]

                    result = await extract_func(
                        ctx=mock_context,
                        repo_name="test-repo",
                    )

                    result_data = json.loads(result)
                    assert result_data["success"] is True
                    assert result_data["indexed_count"] == 1

    @pytest.mark.asyncio
    async def test_extract_and_index_no_app_context(self, mock_mcp, mock_context):
        """Test extract_and_index when app context is not available."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
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

            with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                register_validation_tools(mcp_instance)

            extract_func = registered_funcs["extract_and_index_repository_code"]

            result = await extract_func(
                ctx=mock_context,
                repo_name="test-repo",
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_extract_and_index_no_code_examples(self, mock_mcp, mock_context):
        """Test extract_and_index when no code examples are found."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.validation.extract_repository_code"
            ) as mock_extract:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.repo_extractor = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_app_ctx.database_client.delete_repository_code_examples = (
                    AsyncMock()
                )
                mock_get_ctx.return_value = mock_app_ctx

                # Mock extraction with no results
                mock_extract.return_value = {
                    "success": True,
                    "code_examples": [],
                    "extraction_summary": {
                        "classes": 0,
                        "methods": 0,
                        "functions": 0,
                    },
                }

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                    register_validation_tools(mcp_instance)

                extract_func = registered_funcs["extract_and_index_repository_code"]

                result = await extract_func(
                    ctx=mock_context,
                    repo_name="test-repo",
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert result_data["indexed_count"] == 0

    @pytest.mark.asyncio
    async def test_extract_and_index_database_error(self, mock_mcp, mock_context):
        """Test extract_and_index with database error."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.validation.extract_repository_code"
            ) as mock_extract:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.repo_extractor = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_app_ctx.database_client.delete_repository_code_examples = (
                    AsyncMock()
                )
                mock_get_ctx.return_value = mock_app_ctx

                # Mock extraction error
                mock_extract.side_effect = DatabaseError("Database connection failed")

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                    register_validation_tools(mcp_instance)

                extract_func = registered_funcs["extract_and_index_repository_code"]

                result = await extract_func(
                    ctx=mock_context,
                    repo_name="test-repo",
                )

                result_data = json.loads(result)
                assert result_data["success"] is False
                assert "Database error" in result_data["error"]


class TestSmartCodeSearchTool:
    """Tests for the smart_code_search tool."""

    @pytest.mark.asyncio
    async def test_smart_code_search_success(self, mock_mcp, mock_context):
        """Test smart_code_search tool with successful search."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.validation.ValidatedCodeSearchService"
            ) as mock_service_class:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_app_ctx.repo_extractor = MagicMock()
                mock_app_ctx.repo_extractor.driver = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock search service
                mock_service = MagicMock()
                mock_service.search_and_validate_code = AsyncMock(
                    return_value={
                        "success": True,
                        "query": "authentication",
                        "results": [
                            {
                                "code": "def authenticate(): pass",
                                "confidence": 0.95,
                                "validated": True,
                            }
                        ],
                    }
                )
                mock_service_class.return_value = mock_service

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                    register_validation_tools(mcp_instance)

                search_func = registered_funcs["smart_code_search"]

                result = await search_func(
                    ctx=mock_context,
                    query="authentication",
                    match_count=5,
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert len(result_data["results"]) == 1

    @pytest.mark.asyncio
    async def test_smart_code_search_with_validation_modes(
        self, mock_mcp, mock_context
    ):
        """Test smart_code_search with different validation modes."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.validation.ValidatedCodeSearchService"
            ) as mock_service_class:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_app_ctx.repo_extractor = MagicMock()
                mock_app_ctx.repo_extractor.driver = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock search service
                mock_service = MagicMock()
                mock_service.search_and_validate_code = AsyncMock(
                    return_value={"success": True, "results": []}
                )
                mock_service_class.return_value = mock_service

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                    register_validation_tools(mcp_instance)

                search_func = registered_funcs["smart_code_search"]

                # Test fast mode
                await search_func(
                    ctx=mock_context,
                    query="test",
                    validation_mode="fast",
                )

                # Test thorough mode
                await search_func(
                    ctx=mock_context,
                    query="test",
                    validation_mode="thorough",
                )

                # Test balanced mode (default)
                await search_func(
                    ctx=mock_context,
                    query="test",
                    validation_mode="balanced",
                )

                # Verify service was called
                assert mock_service.search_and_validate_code.call_count == 3

    @pytest.mark.asyncio
    async def test_smart_code_search_no_database_client(self, mock_mcp, mock_context):
        """Test smart_code_search when database client is not available."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
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

            with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                register_validation_tools(mcp_instance)

            search_func = registered_funcs["smart_code_search"]

            result = await search_func(
                ctx=mock_context,
                query="test",
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_smart_code_search_database_error(self, mock_mcp, mock_context):
        """Test smart_code_search with database error."""
        with patch("src.tools.validation.get_app_context") as mock_get_ctx:
            with patch(
                "src.tools.validation.ValidatedCodeSearchService"
            ) as mock_service_class:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.database_client = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock search service with error
                mock_service = MagicMock()
                mock_service.search_and_validate_code = AsyncMock(
                    side_effect=DatabaseError("Search failed")
                )
                mock_service_class.return_value = mock_service

                # Register and get function
                mcp_instance = MagicMock()
                registered_funcs = {}

                def mock_tool_decorator():
                    def decorator(func):
                        registered_funcs[func.__name__] = func
                        return func

                    return decorator

                mcp_instance.tool = mock_tool_decorator

                with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                    register_validation_tools(mcp_instance)

                search_func = registered_funcs["smart_code_search"]

                result = await search_func(
                    ctx=mock_context,
                    query="test",
                )

                result_data = json.loads(result)
                assert result_data["success"] is False
                assert "Database error" in result_data["error"]


class TestCheckAiScriptHallucinationsEnhancedTool:
    """Tests for the check_ai_script_hallucinations_enhanced tool."""

    @pytest.mark.asyncio
    async def test_check_hallucinations_success(self, mock_mcp, mock_context):
        """Test check_ai_script_hallucinations_enhanced with successful analysis."""
        with patch("src.tools.validation.validate_script_path") as mock_validate:
            with patch("src.tools.validation.get_app_context") as mock_get_ctx:
                with patch(
                    "src.tools.validation.check_ai_script_hallucinations_enhanced_impl"
                ) as mock_check:
                    # Mock validation
                    mock_validate.return_value = {
                        "valid": True,
                        "container_path": "/app/analysis_scripts/test.py",
                    }

                    # Mock app context
                    mock_app_ctx = MagicMock()
                    mock_app_ctx.database_client = MagicMock()
                    mock_app_ctx.repo_extractor = MagicMock()
                    mock_app_ctx.repo_extractor.driver = MagicMock()
                    mock_get_ctx.return_value = mock_app_ctx

                    # Mock hallucination check
                    mock_check.return_value = json.dumps(
                        {
                            "success": True,
                            "script_path": "/app/analysis_scripts/test.py",
                            "hallucination_score": 0.05,
                            "issues": [],
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

                    with patch(
                        "src.tools.validation.track_request", lambda x: lambda f: f
                    ):
                        register_validation_tools(mcp_instance)

                    check_func = registered_funcs[
                        "check_ai_script_hallucinations_enhanced"
                    ]

                    result = await check_func(
                        ctx=mock_context,
                        script_path="/app/analysis_scripts/test.py",
                    )

                    result_data = json.loads(result)
                    assert result_data["success"] is True
                    assert "hallucination_score" in result_data

    @pytest.mark.asyncio
    async def test_check_hallucinations_invalid_path(self, mock_mcp, mock_context):
        """Test check_ai_script_hallucinations_enhanced with invalid script path."""
        with patch("src.tools.validation.validate_script_path") as mock_validate:
            # Mock validation failure
            mock_validate.return_value = {
                "valid": False,
                "error": "Script path does not exist",
            }

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                register_validation_tools(mcp_instance)

            check_func = registered_funcs["check_ai_script_hallucinations_enhanced"]

            result = await check_func(
                ctx=mock_context,
                script_path="/invalid/path/test.py",
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "error" in result_data

    @pytest.mark.asyncio
    async def test_check_hallucinations_with_options(self, mock_mcp, mock_context):
        """Test check_ai_script_hallucinations_enhanced with options."""
        with patch("src.tools.validation.validate_script_path") as mock_validate:
            with patch("src.tools.validation.get_app_context") as mock_get_ctx:
                with patch(
                    "src.tools.validation.check_ai_script_hallucinations_enhanced_impl"
                ) as mock_check:
                    # Mock validation
                    mock_validate.return_value = {
                        "valid": True,
                        "container_path": "/app/analysis_scripts/test.py",
                    }

                    # Mock app context
                    mock_app_ctx = MagicMock()
                    mock_app_ctx.database_client = MagicMock()
                    mock_app_ctx.repo_extractor = MagicMock()
                    mock_app_ctx.repo_extractor.driver = MagicMock()
                    mock_get_ctx.return_value = mock_app_ctx

                    # Mock hallucination check
                    mock_check.return_value = json.dumps(
                        {"success": True, "hallucination_score": 0.1}
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

                    with patch(
                        "src.tools.validation.track_request", lambda x: lambda f: f
                    ):
                        register_validation_tools(mcp_instance)

                    check_func = registered_funcs[
                        "check_ai_script_hallucinations_enhanced"
                    ]

                    await check_func(
                        ctx=mock_context,
                        script_path="/app/analysis_scripts/test.py",
                        include_code_suggestions=True,
                        detailed_analysis=True,
                    )

                    # Verify function was called (options are passed to implementation)
                    mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_hallucinations_error_handling(self, mock_mcp, mock_context):
        """Test check_ai_script_hallucinations_enhanced error handling."""
        with patch("src.tools.validation.validate_script_path") as mock_validate:
            with patch("src.tools.validation.get_app_context") as mock_get_ctx:
                with patch(
                    "src.tools.validation.check_ai_script_hallucinations_enhanced_impl"
                ) as mock_check:
                    # Mock validation
                    mock_validate.return_value = {
                        "valid": True,
                        "container_path": "/app/analysis_scripts/test.py",
                    }

                    # Mock app context
                    mock_app_ctx = MagicMock()
                    mock_app_ctx.database_client = MagicMock()
                    mock_get_ctx.return_value = mock_app_ctx

                    # Mock error
                    mock_check.side_effect = ValidationError("Analysis failed")

                    # Register and get function
                    mcp_instance = MagicMock()
                    registered_funcs = {}

                    def mock_tool_decorator():
                        def decorator(func):
                            registered_funcs[func.__name__] = func
                            return func

                        return decorator

                    mcp_instance.tool = mock_tool_decorator

                    with patch(
                        "src.tools.validation.track_request", lambda x: lambda f: f
                    ):
                        register_validation_tools(mcp_instance)

                    check_func = registered_funcs[
                        "check_ai_script_hallucinations_enhanced"
                    ]

                    with pytest.raises(MCPToolError) as exc_info:
                        await check_func(
                            ctx=mock_context,
                            script_path="/app/analysis_scripts/test.py",
                        )

                    assert "Enhanced hallucination check failed" in str(exc_info.value)


class TestGetScriptAnalysisInfoTool:
    """Tests for the get_script_analysis_info tool."""

    @pytest.mark.asyncio
    async def test_get_script_analysis_info(self, mock_mcp, mock_context):
        """Test get_script_analysis_info tool."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                register_validation_tools(mcp_instance)

            info_func = registered_funcs["get_script_analysis_info"]

            result = await info_func(ctx=mock_context)

            result_data = json.loads(result)
            assert "accessible_paths" in result_data
            assert "usage_examples" in result_data
            assert "instructions" in result_data
            assert "available_tools" in result_data

    @pytest.mark.asyncio
    async def test_get_script_analysis_info_directory_checks(
        self, mock_mcp, mock_context
    ):
        """Test get_script_analysis_info with directory existence checks."""
        with patch("os.path.exists") as mock_exists:
            # Some directories exist, some don't
            def exists_side_effect(path):
                if "user_scripts" in path:
                    return True
                return False

            mock_exists.side_effect = exists_side_effect

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.validation.track_request", lambda x: lambda f: f):
                register_validation_tools(mcp_instance)

            info_func = registered_funcs["get_script_analysis_info"]

            result = await info_func(ctx=mock_context)

            result_data = json.loads(result)
            assert "accessible_paths" in result_data
            # Check that paths were annotated with existence status
            # Some paths should have ✓ (exists) markers based on our mock
