"""
Unit tests for knowledge graph MCP tools.

Tests the following tools:
- query_knowledge_graph: Query and explore the knowledge graph
- parse_github_repository: Parse GitHub repos into Neo4j
- parse_repository_branch: Parse specific branches
- get_repository_info: Get repo metadata
- update_parsed_repository: Update already parsed repos
- parse_local_repository: Parse local Git repositories
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from src.core import MCPToolError
from src.core.exceptions import DatabaseError, KnowledgeGraphError, ValidationError
from src.tools.knowledge_graph import register_knowledge_graph_tools


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


class TestQueryKnowledgeGraphTool:
    """Tests for the query_knowledge_graph tool."""

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_repos_command(self, mock_mcp, mock_context):
        """Test query_knowledge_graph tool with repos command."""
        with patch(
            "src.tools.knowledge_graph.query_knowledge_graph"
        ) as mock_query_func:
            mock_query_func.return_value = json.dumps(
                {
                    "success": True,
                    "command": "repos",
                    "repositories": ["repo1", "repo2"],
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

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            query_func = registered_funcs["query_knowledge_graph"]

            result = await query_func(
                ctx=mock_context,
                command="repos",
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["command"] == "repos"
            assert len(result_data["repositories"]) == 2

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_explore_command(
        self, mock_mcp, mock_context
    ):
        """Test query_knowledge_graph tool with explore command."""
        with patch(
            "src.tools.knowledge_graph.query_knowledge_graph"
        ) as mock_query_func:
            mock_query_func.return_value = json.dumps(
                {
                    "success": True,
                    "command": "explore test-repo",
                    "repository": "test-repo",
                    "statistics": {
                        "files": 100,
                        "classes": 50,
                        "methods": 200,
                    },
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

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            query_func = registered_funcs["query_knowledge_graph"]

            result = await query_func(
                ctx=mock_context,
                command="explore test-repo",
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert "statistics" in result_data

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_error_handling(self, mock_mcp, mock_context):
        """Test query_knowledge_graph error handling."""
        with patch(
            "src.tools.knowledge_graph.query_knowledge_graph"
        ) as mock_query_func:
            mock_query_func.side_effect = KnowledgeGraphError("Neo4j connection failed")

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            query_func = registered_funcs["query_knowledge_graph"]

            with pytest.raises(MCPToolError) as exc_info:
                await query_func(
                    ctx=mock_context,
                    command="repos",
                )

            assert "Knowledge graph query failed" in str(exc_info.value)


class TestParseGithubRepositoryTool:
    """Tests for the parse_github_repository tool."""

    @pytest.mark.asyncio
    async def test_parse_github_repository_success(self, mock_mcp, mock_context):
        """Test parse_github_repository tool with successful parse."""
        with patch("src.tools.knowledge_graph.validate_github_url") as mock_validate:
            with patch(
                "src.tools.knowledge_graph.parse_github_repository_impl"
            ) as mock_parse:
                mock_validate.return_value = {"valid": True}
                mock_parse.return_value = json.dumps(
                    {
                        "success": True,
                        "repository": "test-repo",
                        "statistics": {
                            "files_processed": 100,
                            "classes_created": 50,
                            "methods_created": 200,
                        },
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
                    "src.tools.knowledge_graph.track_request", lambda x: lambda f: f
                ):
                    register_knowledge_graph_tools(mcp_instance)

                parse_func = registered_funcs["parse_github_repository"]

                result = await parse_func(
                    ctx=mock_context,
                    repo_url="https://github.com/user/test-repo.git",
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert "statistics" in result_data

    @pytest.mark.asyncio
    async def test_parse_github_repository_invalid_url(self, mock_mcp, mock_context):
        """Test parse_github_repository with invalid URL."""
        with patch("src.tools.knowledge_graph.validate_github_url") as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "error": "Invalid GitHub URL format",
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

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            parse_func = registered_funcs["parse_github_repository"]

            with pytest.raises(MCPToolError) as exc_info:
                await parse_func(
                    ctx=mock_context,
                    repo_url="not-a-url",
                )

            assert "Invalid GitHub URL" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_github_repository_error_handling(
        self, mock_mcp, mock_context
    ):
        """Test parse_github_repository error handling."""
        with patch("src.tools.knowledge_graph.validate_github_url") as mock_validate:
            with patch(
                "src.tools.knowledge_graph.parse_github_repository_impl"
            ) as mock_parse:
                mock_validate.return_value = {"valid": True}
                mock_parse.side_effect = KnowledgeGraphError("Parsing failed")

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
                    "src.tools.knowledge_graph.track_request", lambda x: lambda f: f
                ):
                    register_knowledge_graph_tools(mcp_instance)

                parse_func = registered_funcs["parse_github_repository"]

                with pytest.raises(MCPToolError) as exc_info:
                    await parse_func(
                        ctx=mock_context,
                        repo_url="https://github.com/user/test-repo.git",
                    )

                assert "Repository parsing failed" in str(exc_info.value)


class TestParseRepositoryBranchTool:
    """Tests for the parse_repository_branch tool."""

    @pytest.mark.asyncio
    async def test_parse_repository_branch_success(self, mock_mcp, mock_context):
        """Test parse_repository_branch tool with successful parse."""
        with patch("src.tools.knowledge_graph.validate_github_url") as mock_validate:
            with patch(
                "src.tools.knowledge_graph.parse_github_repository_with_branch"
            ) as mock_parse:
                mock_validate.return_value = {"valid": True}
                mock_parse.return_value = json.dumps(
                    {
                        "success": True,
                        "repository": "test-repo",
                        "branch": "feature-branch",
                        "statistics": {
                            "files_processed": 50,
                            "classes_created": 25,
                        },
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
                    "src.tools.knowledge_graph.track_request", lambda x: lambda f: f
                ):
                    register_knowledge_graph_tools(mcp_instance)

                parse_branch_func = registered_funcs["parse_repository_branch"]

                result = await parse_branch_func(
                    ctx=mock_context,
                    repo_url="https://github.com/user/test-repo.git",
                    branch="feature-branch",
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert result_data["branch"] == "feature-branch"

    @pytest.mark.asyncio
    async def test_parse_repository_branch_invalid_url(self, mock_mcp, mock_context):
        """Test parse_repository_branch with invalid URL."""
        with patch("src.tools.knowledge_graph.validate_github_url") as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "error": "Invalid GitHub URL",
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

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            parse_branch_func = registered_funcs["parse_repository_branch"]

            with pytest.raises(MCPToolError) as exc_info:
                await parse_branch_func(
                    ctx=mock_context,
                    repo_url="invalid-url",
                    branch="main",
                )

            assert "Invalid GitHub URL" in str(exc_info.value)


class TestGetRepositoryInfoTool:
    """Tests for the get_repository_info tool."""

    @pytest.mark.asyncio
    async def test_get_repository_info_success(self, mock_mcp, mock_context):
        """Test get_repository_info tool with successful retrieval."""
        with patch(
            "src.tools.knowledge_graph.get_repository_metadata_from_neo4j"
        ) as mock_get_info:
            mock_get_info.return_value = json.dumps(
                {
                    "success": True,
                    "repository": "test-repo",
                    "metadata": {
                        "files": 100,
                        "classes": 50,
                        "branches": ["main", "develop"],
                    },
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

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            get_info_func = registered_funcs["get_repository_info"]

            result = await get_info_func(
                ctx=mock_context,
                repo_name="test-repo",
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["repository"] == "test-repo"

    @pytest.mark.asyncio
    async def test_get_repository_info_error_handling(self, mock_mcp, mock_context):
        """Test get_repository_info error handling."""
        with patch(
            "src.tools.knowledge_graph.get_repository_metadata_from_neo4j"
        ) as mock_get_info:
            mock_get_info.side_effect = KnowledgeGraphError("Repository not found")

            # Register and get function
            mcp_instance = MagicMock()
            registered_funcs = {}

            def mock_tool_decorator():
                def decorator(func):
                    registered_funcs[func.__name__] = func
                    return func

                return decorator

            mcp_instance.tool = mock_tool_decorator

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            get_info_func = registered_funcs["get_repository_info"]

            with pytest.raises(MCPToolError) as exc_info:
                await get_info_func(
                    ctx=mock_context,
                    repo_name="nonexistent-repo",
                )

            assert "Failed to get repository info" in str(exc_info.value)


class TestUpdateParsedRepositoryTool:
    """Tests for the update_parsed_repository tool."""

    @pytest.mark.asyncio
    async def test_update_parsed_repository_success(self, mock_mcp, mock_context):
        """Test update_parsed_repository tool with successful update."""
        with patch("src.tools.knowledge_graph.validate_github_url") as mock_validate:
            with patch(
                "src.tools.knowledge_graph.update_repository_in_neo4j"
            ) as mock_update:
                mock_validate.return_value = {"valid": True}
                mock_update.return_value = json.dumps(
                    {
                        "success": True,
                        "repository": "test-repo",
                        "files_updated": 10,
                        "changes_detected": True,
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
                    "src.tools.knowledge_graph.track_request", lambda x: lambda f: f
                ):
                    register_knowledge_graph_tools(mcp_instance)

                update_func = registered_funcs["update_parsed_repository"]

                result = await update_func(
                    ctx=mock_context,
                    repo_url="https://github.com/user/test-repo.git",
                )

                result_data = json.loads(result)
                assert result_data["success"] is True
                assert result_data["changes_detected"] is True

    @pytest.mark.asyncio
    async def test_update_parsed_repository_invalid_url(self, mock_mcp, mock_context):
        """Test update_parsed_repository with invalid URL."""
        with patch("src.tools.knowledge_graph.validate_github_url") as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "error": "Invalid GitHub URL",
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

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            update_func = registered_funcs["update_parsed_repository"]

            with pytest.raises(MCPToolError) as exc_info:
                await update_func(
                    ctx=mock_context,
                    repo_url="invalid-url",
                )

            assert "Invalid GitHub URL" in str(exc_info.value)


class TestParseLocalRepositoryTool:
    """Tests for the parse_local_repository tool."""

    @pytest.mark.asyncio
    async def test_parse_local_repository_success(self, mock_mcp, mock_context):
        """Test parse_local_repository tool with successful parse."""
        with patch("src.tools.knowledge_graph.get_app_context") as mock_get_ctx:
            with patch("os.path.exists") as mock_exists:
                with patch("os.path.isdir") as mock_isdir:
                    with patch("os.path.abspath") as mock_abspath:
                        # Mock app context
                        mock_app_ctx = MagicMock()
                        mock_repo_extractor = MagicMock()
                        mock_repo_extractor.driver = MagicMock()
                        mock_repo_extractor.analyze_local_repository = AsyncMock()

                        # Mock Neo4j session
                        mock_session = MagicMock()
                        mock_result = MagicMock()
                        mock_stats = {
                            "file_count": 100,
                            "class_count": 50,
                            "method_count": 200,
                            "function_count": 80,
                        }
                        mock_result.single.return_value = mock_stats

                        async def mock_run(*args, **kwargs):
                            return mock_result

                        mock_session.run = mock_run
                        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                        mock_session.__aexit__ = AsyncMock(return_value=None)
                        mock_repo_extractor.driver.session.return_value = mock_session

                        mock_app_ctx.repo_extractor = mock_repo_extractor
                        mock_get_ctx.return_value = mock_app_ctx

                        # Mock path operations
                        mock_abspath.return_value = "/home/user/test-repo"
                        mock_exists.side_effect = lambda path: True
                        mock_isdir.return_value = True

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
                            "src.tools.knowledge_graph.track_request",
                            lambda x: lambda f: f,
                        ):
                            register_knowledge_graph_tools(mcp_instance)

                        parse_local_func = registered_funcs["parse_local_repository"]

                        result = await parse_local_func(
                            ctx=mock_context,
                            local_path="~/test-repo",
                        )

                        result_data = json.loads(result)
                        assert result_data["success"] is True
                        assert "statistics" in result_data

    @pytest.mark.asyncio
    async def test_parse_local_repository_path_not_allowed(
        self, mock_mcp, mock_context
    ):
        """Test parse_local_repository with path outside allowed directories."""
        with patch("src.tools.knowledge_graph.get_app_context") as mock_get_ctx:
            with patch("os.path.abspath") as mock_abspath:
                # Mock app context
                mock_app_ctx = MagicMock()
                mock_app_ctx.repo_extractor = MagicMock()
                mock_get_ctx.return_value = mock_app_ctx

                # Mock path to be outside allowed directories
                mock_abspath.return_value = "/root/secret-repo"

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
                    "src.tools.knowledge_graph.track_request", lambda x: lambda f: f
                ):
                    register_knowledge_graph_tools(mcp_instance)

                parse_local_func = registered_funcs["parse_local_repository"]

                result = await parse_local_func(
                    ctx=mock_context,
                    local_path="/root/secret-repo",
                )

                result_data = json.loads(result)
                assert result_data["success"] is False
                assert "not within allowed directories" in result_data["error"]

    @pytest.mark.asyncio
    async def test_parse_local_repository_not_git_repo(self, mock_mcp, mock_context):
        """Test parse_local_repository with non-git directory."""
        with patch("src.tools.knowledge_graph.get_app_context") as mock_get_ctx:
            with patch("os.path.exists") as mock_exists:
                with patch("os.path.isdir") as mock_isdir:
                    with patch("os.path.abspath") as mock_abspath:
                        # Mock app context
                        mock_app_ctx = MagicMock()
                        mock_app_ctx.repo_extractor = MagicMock()
                        mock_get_ctx.return_value = mock_app_ctx

                        # Mock path operations
                        mock_abspath.return_value = "/home/user/not-a-repo"
                        mock_isdir.return_value = True

                        # Directory exists but .git does not
                        def exists_side_effect(path):
                            if path.endswith(".git"):
                                return False
                            return True

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

                        with patch(
                            "src.tools.knowledge_graph.track_request",
                            lambda x: lambda f: f,
                        ):
                            register_knowledge_graph_tools(mcp_instance)

                        parse_local_func = registered_funcs["parse_local_repository"]

                        result = await parse_local_func(
                            ctx=mock_context,
                            local_path="~/not-a-repo",
                        )

                        result_data = json.loads(result)
                        assert result_data["success"] is False
                        assert "Not a Git repository" in result_data["error"]

    @pytest.mark.asyncio
    async def test_parse_local_repository_no_repo_extractor(
        self, mock_mcp, mock_context
    ):
        """Test parse_local_repository when repo_extractor is not available."""
        with patch("src.tools.knowledge_graph.get_app_context") as mock_get_ctx:
            # Mock app context without repo_extractor
            mock_app_ctx = MagicMock()
            mock_app_ctx.repo_extractor = None
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

            with patch("src.tools.knowledge_graph.track_request", lambda x: lambda f: f):
                register_knowledge_graph_tools(mcp_instance)

            parse_local_func = registered_funcs["parse_local_repository"]

            result = await parse_local_func(
                ctx=mock_context,
                local_path="~/test-repo",
            )

            result_data = json.loads(result)
            assert result_data["success"] is False
            assert "not available" in result_data["error"]
