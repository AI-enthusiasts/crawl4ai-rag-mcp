"""
Knowledge graph tools for MCP server.

This module contains Neo4j knowledge graph MCP tools including:
- query_knowledge_graph: Query and explore the knowledge graph
- parse_github_repository: Parse GitHub repos into Neo4j
- parse_repository_branch: Parse specific branches
- get_repository_info: Get repo metadata
- update_parsed_repository: Update already parsed repos
- parse_local_repository: Parse local Git repositories
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP

from src.core import MCPToolError, track_request
from src.core.context import get_app_context
from src.core.exceptions import (
    DatabaseError,
    KnowledgeGraphError,
    ValidationError,
)
from src.knowledge_graph import (
    query_knowledge_graph,
)
from src.knowledge_graph.repository import (
    parse_github_repository as parse_github_repository_impl,
)
from src.utils.validation import validate_github_url

logger = logging.getLogger(__name__)


async def parse_github_repository_wrapper(ctx: Context, repo_url: str) -> str:
    """Wrapper function to properly extract repo_extractor from context and call the implementation."""
    # Get the app context that was stored during lifespan
    app_ctx = get_app_context()

    if (
        not app_ctx
        or not hasattr(app_ctx, "repo_extractor")
        or not app_ctx.repo_extractor
    ):
        # Return a proper error message
        return json.dumps(
            {
                "success": False,
                "error": "Repository extractor not available. Neo4j may not be configured or the USE_KNOWLEDGE_GRAPH environment variable may be set to false.",
            },
            indent=2,
        )

    return await parse_github_repository_impl(app_ctx.repo_extractor, repo_url)  # type: ignore[no-any-return]


async def query_knowledge_graph_wrapper(ctx: Context, command: str) -> str:
    """Wrapper function to call the knowledge graph query implementation."""
    # The query_knowledge_graph function doesn't need a context parameter
    # It creates its own Neo4j connection from environment variables
    return await query_knowledge_graph(command)  # type: ignore[no-any-return]


def register_knowledge_graph_tools(mcp: "FastMCP") -> None:
    """
    Register knowledge graph MCP tools.

    Args:
        mcp: FastMCP instance to register tools with
    """

    @mcp.tool()
    @track_request("query_knowledge_graph")
    async def query_knowledge_graph(
        ctx: Context,
        command: str,
    ) -> str:
        """
        Query and explore the Neo4j knowledge graph containing repository data.

        This tool provides comprehensive access to the knowledge graph for exploring repositories,
        classes, methods, functions, and their relationships. Perfect for understanding what data
        is available for hallucination detection and debugging validation results.

        **⚠️ IMPORTANT: Always start with the `repos` command first!**
        Before using any other commands, run `repos` to see what repositories are available
        in your knowledge graph. This will help you understand what data you can explore.

        ## Available Commands:

        **Repository Commands:**
        - `repos` - **START HERE!** List all repositories in the knowledge graph
        - `explore <repo_name>` - Get detailed overview of a specific repository

        **Class Commands:**
        - `classes` - List all classes across all repositories (limited to 20)
        - `classes <repo_name>` - List classes in a specific repository
        - `class <class_name>` - Get detailed information about a specific class including methods and attributes

        **Method Commands:**
        - `method <method_name>` - Search for methods by name across all classes
        - `method <method_name> <class_name>` - Search for a method within a specific class

        **Custom Query:**
        - `query <cypher_query>` - Execute a custom Cypher query (results limited to 20 records)

        ## Knowledge Graph Schema:

        **Node Types:**
        - Repository: `(r:Repository {name: string})`
        - File: `(f:File {path: string, module_name: string})`
        - Class: `(c:Class {name: string, full_name: string})`
        - Method: `(m:Method {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
        - Function: `(func:Function {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
        - Attribute: `(a:Attribute {name: string, type: string})`

        **Relationships:**
        - `(r:Repository)-[:CONTAINS]->(f:File)`
        - `(f:File)-[:DEFINES]->(c:Class)`
        - `(c:Class)-[:HAS_METHOD]->(m:Method)`
        - `(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)`
        - `(f:File)-[:DEFINES]->(func:Function)`

        ## Example Workflow:
        ```
        1. repos                                    # See what repositories are available
        2. explore pydantic-ai                      # Explore a specific repository
        3. classes pydantic-ai                      # List classes in that repository
        4. class Agent                              # Explore the Agent class
        5. method run_stream                        # Search for run_stream method
        6. method __init__ Agent                    # Find Agent constructor
        7. query "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) WHERE m.name = 'run' RETURN c.name, m.name LIMIT 5"
        ```

        Args:
            command: Command string to execute (see available commands above)

        Returns:
            JSON string with query results, statistics, and metadata
        """
        try:
            return await query_knowledge_graph_wrapper(ctx, command)
        except KnowledgeGraphError as e:
            logger.error(f"Knowledge graph error: {e}")
            msg = f"Knowledge graph query failed: {e!s}"
            raise MCPToolError(msg) from e
        except DatabaseError as e:
            logger.error(f"Database error in knowledge graph: {e}")
            msg = f"Knowledge graph query failed: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception(f"Unexpected error in query_knowledge_graph tool: {e}")
            msg = f"Knowledge graph query failed: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("parse_github_repository")
    async def parse_github_repository(
        ctx: Context,
        repo_url: str,
    ) -> str:
        """
        Parse a GitHub repository into the Neo4j knowledge graph.

        This tool clones a GitHub repository, analyzes its Python files, and stores
        the code structure (classes, methods, functions, imports) in Neo4j for use
        in hallucination detection. The tool:

        - Clones the repository to a temporary location
        - Analyzes Python files to extract code structure
        - Stores classes, methods, functions, and imports in Neo4j
        - Provides detailed statistics about the parsing results
        - Automatically handles module name detection for imports

        Args:
            repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')

        Returns:
            JSON string with parsing results, statistics, and repository information
        """
        try:
            # Validate GitHub URL
            validation_result = validate_github_url(repo_url)
            if not validation_result["valid"]:
                raise MCPToolError(validation_result.get("error", "Invalid GitHub URL"))

            return await parse_github_repository_wrapper(ctx, repo_url)
        except MCPToolError:
            raise
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            msg = f"Repository parsing failed: {e!s}"
            raise MCPToolError(msg) from e
        except KnowledgeGraphError as e:
            logger.error(f"Knowledge graph error: {e}")
            msg = f"Repository parsing failed: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception(f"Unexpected error in parse_github_repository tool: {e}")
            msg = f"Repository parsing failed: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("parse_repository_branch")
    async def parse_repository_branch(
        ctx: Context,
        repo_url: str,
        branch: str,
    ) -> str:
        """
        Parse a specific branch of a GitHub repository into the Neo4j knowledge graph.

        This enhanced tool allows parsing specific branches of a repository, useful for:
        - Analyzing feature branches before merging
        - Comparing different versions of code
        - Tracking code evolution across branches

        The tool extracts:
        - Code structure (classes, methods, functions, imports)
        - Git metadata (branches, tags, recent commits)
        - Repository statistics (contributors, file count, size)

        Args:
            repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
            branch: Branch name to parse (e.g., 'main', 'develop', 'feature/new-feature')

        Returns:
            JSON string with parsing results, statistics, and branch information
        """
        try:
            # Validate GitHub URL
            validation_result = validate_github_url(repo_url)
            if not validation_result["valid"]:
                raise MCPToolError(validation_result.get("error", "Invalid GitHub URL"))

            # Parse repository with branch support
            from src.knowledge_graph.repository import (
                parse_github_repository_with_branch,
            )

            return await parse_github_repository_with_branch(ctx, repo_url, branch)
        except MCPToolError:
            raise
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            msg = f"Repository branch parsing failed: {e!s}"
            raise MCPToolError(msg) from e
        except KnowledgeGraphError as e:
            logger.error(f"Knowledge graph error: {e}")
            msg = f"Repository branch parsing failed: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception(f"Unexpected error in parse_repository_branch tool: {e}")
            msg = f"Repository branch parsing failed: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("get_repository_info")
    async def get_repository_info(
        ctx: Context,
        repo_name: str,
    ) -> str:
        """
        Get detailed information about a parsed repository from the knowledge graph.

        This tool retrieves comprehensive metadata about a repository including:
        - Repository statistics (file count, contributors, size)
        - Branch information with recent commits
        - Tag information
        - Code structure summary (classes, methods, functions)
        - Git history insights

        Use this after parsing a repository to understand its structure and history.

        Args:
            repo_name: Name of the repository in the knowledge graph (without .git extension)

        Returns:
            JSON string with comprehensive repository information
        """
        try:
            from src.knowledge_graph.repository import (
                get_repository_metadata_from_neo4j,
            )

            return await get_repository_metadata_from_neo4j(ctx, repo_name)
        except KnowledgeGraphError as e:
            logger.error(f"Knowledge graph error: {e}")
            msg = f"Failed to get repository info: {e!s}"
            raise MCPToolError(msg) from e
        except DatabaseError as e:
            logger.error(f"Database error: {e}")
            msg = f"Failed to get repository info: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception(f"Unexpected error in get_repository_info tool: {e}")
            msg = f"Failed to get repository info: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("update_parsed_repository")
    async def update_parsed_repository(
        ctx: Context,
        repo_url: str,
    ) -> str:
        """
        Update an already parsed repository with latest changes.

        This tool performs an incremental update of a repository:
        - Pulls latest changes from the remote repository
        - Identifies modified files since last parse
        - Updates only the changed components in Neo4j
        - Preserves existing relationships and metadata

        More efficient than re-parsing the entire repository for large codebases.

        Args:
            repo_url: GitHub repository URL to update

        Returns:
            JSON string with update results and changed files
        """
        try:
            # Validate GitHub URL
            validation_result = validate_github_url(repo_url)
            if not validation_result["valid"]:
                raise MCPToolError(validation_result.get("error", "Invalid GitHub URL"))

            from src.knowledge_graph.repository import update_repository_in_neo4j

            return await update_repository_in_neo4j(ctx, repo_url)
        except MCPToolError:
            raise
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            msg = f"Repository update failed: {e!s}"
            raise MCPToolError(msg) from e
        except KnowledgeGraphError as e:
            logger.error(f"Knowledge graph error: {e}")
            msg = f"Repository update failed: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception(f"Unexpected error in update_parsed_repository tool: {e}")
            msg = f"Repository update failed: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("parse_local_repository")
    async def parse_local_repository(
        ctx: Context,
        local_path: str,
    ) -> str:
        """
        Parse a local Git repository into the Neo4j knowledge graph.

        This tool parses a local Git repository directly without cloning, useful for:
        - Analyzing repositories already present on the system
        - Parsing private repositories not accessible via URL
        - Working with repositories that have been modified locally
        - Faster parsing of repositories you already have locally

        The tool analyzes multiple programming languages including:
        - Python (.py files)
        - JavaScript/TypeScript (.js, .ts, .jsx, .tsx files)
        - Go (.go files)
        - And more based on the multi-language analyzer factory

        Args:
            local_path: Absolute path to the local Git repository directory

        Returns:
            JSON string with parsing results, statistics, and repository information
        """
        try:
            # Get the app context
            app_ctx = get_app_context()

            if not app_ctx:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Application context not available",
                    },
                    indent=2,
                )

            # Check if repository extractor is available
            repo_extractor = getattr(app_ctx, "repo_extractor", None)
            if not repo_extractor:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Repository extractor not available. Neo4j may not be configured or USE_KNOWLEDGE_GRAPH may be false.",
                    },
                    indent=2,
                )

            # Security: Validate and sanitize local path
            local_path = os.path.abspath(os.path.expanduser(local_path))

            # Define allowed directories for repository parsing (configurable)
            allowed_prefixes = [
                os.path.expanduser("~/"),  # User home directory
                "/tmp/",                    # Temporary directory
                "/var/tmp/",               # Var temporary
                "/workspace/",             # Common workspace directory
            ]

            # Check if path is within allowed directories
            path_allowed = any(local_path.startswith(os.path.abspath(prefix)) for prefix in allowed_prefixes)

            if not path_allowed:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Path is not within allowed directories. Repository must be in home directory or temporary directories.",
                    },
                    indent=2,
                )

            # Validate local path exists
            if not Path(local_path).exists():
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Local path does not exist: {local_path}",
                    },
                    indent=2,
                )

            if not os.path.isdir(local_path):
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Path is not a directory: {local_path}",
                    },
                    indent=2,
                )

            # Check if it's a Git repository
            git_dir = Path(local_path) / ".git"
            if not git_dir.exists():
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Not a Git repository (no .git directory found): {local_path}",
                    },
                    indent=2,
                )

            # Extract repository name from path
            repo_name = os.path.basename(os.path.abspath(local_path))

            logger.info(f"Parsing local repository: {repo_name} at {local_path}")

            # Use a custom method to analyze local repository
            await repo_extractor.analyze_local_repository(local_path, repo_name)

            # Query Neo4j to get statistics about what was stored
            stats_query = """
            MATCH (r:Repository {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            WITH r,
                 COLLECT(DISTINCT f) as files,
                 COLLECT(DISTINCT c) as classes,
                 COLLECT(DISTINCT m) as methods,
                 COLLECT(DISTINCT func) as functions
            RETURN
                SIZE([f IN files WHERE f IS NOT NULL]) as file_count,
                SIZE([c IN classes WHERE c IS NOT NULL]) as class_count,
                SIZE([m IN methods WHERE m IS NOT NULL]) as method_count,
                SIZE([func IN functions WHERE func IS NOT NULL]) as function_count
            """

            async with repo_extractor.driver.session() as session:
                stats_result = await session.run(stats_query, repo_name=repo_name)
                stats = await stats_result.single()

            return json.dumps(
                {
                    "success": True,
                    "local_path": local_path,
                    "repository_name": repo_name,
                    "statistics": {
                        "files_processed": stats["file_count"] if stats else 0,
                        "classes_created": stats["class_count"] if stats else 0,
                        "methods_created": stats["method_count"] if stats else 0,
                        "functions_created": stats["function_count"] if stats else 0,
                    },
                    "message": f"Successfully parsed local repository '{repo_name}' into the knowledge graph",
                    "next_steps": [
                        "Use 'query_knowledge_graph' tool with 'explore <repo_name>' to see detailed statistics",
                        "Use 'check_ai_script_hallucinations' tool to validate AI-generated code against this repository",
                    ],
                },
                indent=2,
            )

        except ValidationError as e:
            logger.error(f"Validation error parsing local repository: {e}")
            return json.dumps(
                {
                    "success": False,
                    "local_path": local_path,
                    "error": f"Validation failed: {e!s}",
                },
                indent=2,
            )
        except KnowledgeGraphError as e:
            logger.error(f"Knowledge graph error parsing local repository: {e}")
            return json.dumps(
                {
                    "success": False,
                    "local_path": local_path,
                    "error": f"Knowledge graph parsing failed: {e!s}",
                },
                indent=2,
            )
        except Exception as e:
            logger.exception(f"Unexpected error parsing local repository {local_path}: {e}")
            return json.dumps(
                {
                    "success": False,
                    "local_path": local_path,
                    "error": f"Local repository parsing failed: {e!s}",
                },
                indent=2,
            )
