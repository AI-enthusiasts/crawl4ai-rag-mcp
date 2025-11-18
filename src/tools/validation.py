"""
Code validation tools for MCP server.

This module contains code validation and analysis MCP tools including:
- extract_and_index_repository_code: Index code from Neo4j to Qdrant
- smart_code_search: Validated semantic code search
- check_ai_script_hallucinations_enhanced: Enhanced hallucination detection
- get_script_analysis_info: Helper for script analysis setup
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP

from src.core import MCPToolError, track_request
from src.core.context import get_app_context
from src.core.exceptions import DatabaseError, KnowledgeGraphError, ValidationError
from src.knowledge_graph.code_extractor import extract_repository_code
from src.knowledge_graph.enhanced_validation import (
    check_ai_script_hallucinations_enhanced as check_hallucinations_enhanced_impl,
)
from src.services.validated_search import ValidatedCodeSearchService
from src.utils import create_embeddings_batch
from src.utils.validation import validate_script_path

logger = logging.getLogger(__name__)


def register_validation_tools(mcp: "FastMCP") -> None:
    """
    Register validation-related MCP tools.

    Args:
        mcp: FastMCP instance to register tools with
    """

    @mcp.tool()
    @track_request("extract_and_index_repository_code")
    async def extract_and_index_repository_code(
        _ctx: Context,
        repo_name: str,
    ) -> str:
        """
        Extract code examples from Neo4j knowledge graph and index them in Qdrant.

        This tool creates a bridge between Neo4j (knowledge graph) and Qdrant
        (vector database) for code search and validation. It:
        - Extracts structured code examples from Neo4j
        - Generates embeddings for semantic search
        - Stores code with rich metadata in Qdrant
        - Enables AI hallucination detection and code validation

        Args:
            _ctx: MCP context (unused)
            repo_name: Name of the repository in Neo4j to extract code from

        Returns:
            JSON string with indexing results and statistics
        """
        try:
            # Get the app context that was stored during lifespan
            app_ctx = get_app_context()

            if not app_ctx:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Application context not available",
                    },
                    indent=2,
                )

            # Check Neo4j availability
            if not hasattr(app_ctx, "repo_extractor") or not app_ctx.repo_extractor:
                error_msg = (
                    "Repository extractor not available. "
                    "Neo4j may not be configured or USE_KNOWLEDGE_GRAPH may be false."
                )
                return json.dumps(
                    {
                        "success": False,
                        "error": error_msg,
                    },
                    indent=2,
                )

            # Check database availability
            if not hasattr(app_ctx, "database_client") or not app_ctx.database_client:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Database client not available",
                    },
                    indent=2,
                )

            # Clean up any existing code examples for this repository
            logger.info(
                "Cleaning up existing code examples for repository: %s",
                repo_name,
            )
            try:
                # Method exists in QdrantAdapter, added to Protocol
                await app_ctx.database_client.delete_repository_code_examples(repo_name)
            except DatabaseError as cleanup_error:
                logger.warning("Database error during cleanup: %s", cleanup_error)
            except Exception as cleanup_error:
                logger.warning("Unexpected error during cleanup: %s", cleanup_error)

            # Extract code examples from Neo4j
            extraction_result = await extract_repository_code(
                app_ctx.repo_extractor, repo_name,
            )

            if not extraction_result["success"]:
                return json.dumps(extraction_result, indent=2)

            code_examples = extraction_result["code_examples"]

            if not code_examples:
                return json.dumps(
                    {
                        "success": True,
                        "repository_name": repo_name,
                        "message": "No code examples found to index",
                        "indexed_count": 0,
                    },
                    indent=2,
                )

            # Generate embeddings for code examples
            embedding_texts = [example["embedding_text"] for example in code_examples]
            logger.info(
                "Generating embeddings for %d code examples",
                len(embedding_texts),
            )

            embeddings = create_embeddings_batch(embedding_texts)

            if len(embeddings) != len(code_examples):
                error_msg = (
                    f"Embedding count mismatch: got {len(embeddings)}, "
                    f"expected {len(code_examples)}"
                )
                return json.dumps(
                    {
                        "success": False,
                        "error": error_msg,
                    },
                    indent=2,
                )

            # Prepare data for Qdrant storage
            urls = []
            chunk_numbers = []
            code_texts = []
            summaries = []
            metadatas = []
            source_ids = []

            for i, example in enumerate(code_examples):
                # Create a pseudo-URL for the code example
                pseudo_url = f"neo4j://repository/{repo_name}/{example['code_type']}/{example['name']}"
                urls.append(pseudo_url)
                chunk_numbers.append(i)
                code_texts.append(example["code_text"])
                summaries.append(
                    f"{example['code_type'].title()}: {example['full_name']}",
                )
                metadatas.append(example["metadata"])
                source_ids.append(repo_name)

            # Store in Qdrant
            logger.info("Storing %d code examples in Qdrant", len(code_examples))

            await app_ctx.database_client.add_code_examples(
                urls=urls,
                chunk_numbers=chunk_numbers,
                code_examples=code_texts,
                summaries=summaries,
                metadatas=metadatas,
                embeddings=embeddings,
                source_ids=source_ids,
            )

            # Update source information
            summary = (
                f"Code repository with "
                f"{extraction_result['extraction_summary']['classes']} classes, "
                f"{extraction_result['extraction_summary']['methods']} methods, "
                f"{extraction_result['extraction_summary']['functions']} functions"
            )
            await app_ctx.database_client.update_source_info(
                source_id=repo_name,
                summary=summary,
                word_count=sum(
                    len(example["code_text"].split()) for example in code_examples
                ),
            )

            success_msg = (
                f"Successfully indexed {len(code_examples)} "
                f"code examples from {repo_name}"
            )
            return json.dumps(
                {
                    "success": True,
                    "repository_name": repo_name,
                    "indexed_count": len(code_examples),
                    "extraction_summary": extraction_result["extraction_summary"],
                    "storage_summary": {
                        "embeddings_generated": len(embeddings),
                        "examples_stored": len(code_examples),
                        "total_code_words": sum(
                            len(example["code_text"].split())
                            for example in code_examples
                        ),
                    },
                    "message": success_msg,
                },
                indent=2,
            )

        except DatabaseError as e:
            logger.exception("Database error in extract_and_index_repository_code")
            return json.dumps(
                {
                    "success": False,
                    "repository_name": repo_name,
                    "error": f"Database error: {e!s}",
                },
                indent=2,
            )
        except KnowledgeGraphError as e:
            logger.exception(
                "Knowledge graph error in extract_and_index_repository_code",
            )
            return json.dumps(
                {
                    "success": False,
                    "repository_name": repo_name,
                    "error": f"Knowledge graph error: {e!s}",
                },
                indent=2,
            )
        except Exception as e:
            logger.exception(
                "Unexpected error in extract_and_index_repository_code tool",
            )
            return json.dumps(
                {
                    "success": False,
                    "repository_name": repo_name,
                    "error": str(e),
                },
                indent=2,
            )

    @mcp.tool()
    @track_request("smart_code_search")
    async def smart_code_search(
        _ctx: Context,
        query: str,
        match_count: int = 5,
        source_filter: str | None = None,
        min_confidence: float = 0.6,
        validation_mode: str = "balanced",
        *,
        include_suggestions: bool = True,
    ) -> str:
        """Smart code search combining Qdrant semantic search with Neo4j validation.

        This tool provides high-confidence code search results by:
        - Performing semantic search in Qdrant for relevant code examples
        - Validating each result against Neo4j knowledge graph structure
        - Adding confidence scores and validation metadata
        - Providing intelligent fallback when one system is unavailable
        - Options to control validation for speed vs accuracy trade-offs

        Args:
            _ctx: MCP context (unused)
            query: Search query for semantic matching
            match_count: Maximum number of results to return (default: 5)
            source_filter: Optional source repository filter (e.g., 'repo-name')
            min_confidence: Minimum confidence threshold 0.0-1.0 (default: 0.6)
            validation_mode: Validation approach - "fast", "balanced", "thorough"
            include_suggestions: Include correction suggestions (default: True)

        Returns:
            JSON string with validated search results, confidence scores, and metadata
        """
        try:
            # Get the app context
            app_ctx = get_app_context()

            if (
                not app_ctx
                or not hasattr(app_ctx, "database_client")
                or not app_ctx.database_client
            ):
                return json.dumps(
                    {
                        "success": False,
                        "error": "Database client not available",
                    },
                    indent=2,
                )

            # Initialize validated search service
            neo4j_driver = None
            if hasattr(app_ctx, "repo_extractor") and app_ctx.repo_extractor:
                # Extract Neo4j driver if available
                neo4j_driver = getattr(app_ctx.repo_extractor, "driver", None)

            validated_search = ValidatedCodeSearchService(
                app_ctx.database_client, neo4j_driver,
            )

            # Configure validation based on mode
            parallel_validation = True
            if validation_mode == "fast":
                parallel_validation = True
                min_confidence = max(min_confidence, 0.4)  # Lower threshold for speed
            elif validation_mode == "thorough":
                parallel_validation = False  # Sequential for thoroughness
                min_confidence = max(
                    min_confidence, 0.7,
                )  # Higher threshold for accuracy
            # balanced mode uses defaults

            # Perform validated search
            result = await validated_search.search_and_validate_code(
                query=query,
                match_count=match_count,
                source_filter=source_filter,
                min_confidence=min_confidence,
                include_suggestions=include_suggestions,
                parallel_validation=parallel_validation,
            )

            return json.dumps(result, indent=2)

        except DatabaseError as e:
            logger.exception("Database error in smart_code_search")
            return json.dumps(
                {
                    "success": False,
                    "query": query,
                    "error": f"Database error: {e!s}",
                },
                indent=2,
            )
        except ValidationError as e:
            logger.exception("Validation error in smart_code_search")
            return json.dumps(
                {
                    "success": False,
                    "query": query,
                    "error": f"Validation error: {e!s}",
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Unexpected error in smart_code_search tool")
            return json.dumps(
                {
                    "success": False,
                    "query": query,
                    "error": str(e),
                },
                indent=2,
            )

    @mcp.tool()
    @track_request("check_ai_script_hallucinations_enhanced")
    async def check_ai_script_hallucinations_enhanced(
        _ctx: Context,
        script_path: str,
        *,
        _include_code_suggestions: bool = True,
        _detailed_analysis: bool = True,
    ) -> str:
        """
        Enhanced AI script hallucination detection using both Neo4j and Qdrant.

        This tool provides comprehensive hallucination detection by:
        - Analyzing script structure and extracting code elements
        - Validating against Neo4j knowledge graph for structural correctness
        - Finding similar code examples in Qdrant for semantic validation
        - Providing detailed confidence scores and suggested corrections
        - Combining both validation approaches for maximum accuracy

        Improvements over basic hallucination detection:
        - Uses semantic search to find real code examples
        - Provides code suggestions from actual repositories
        - Combines structural and semantic validation
        - Better confidence scoring with multiple validation methods
        - Parallel validation for improved performance

        Args:
            _ctx: MCP context (unused)
            script_path: Absolute path to the Python script to analyze
            _include_code_suggestions: Code suggestions from real examples
            _detailed_analysis: Include detailed validation results

        Returns:
            JSON string with comprehensive hallucination detection results,
            confidence scores, and recommendations
        """
        try:
            # Validate script path
            validation_result = validate_script_path(script_path)
            if isinstance(validation_result, dict) and not validation_result.get(
                "valid", False,
            ):
                return json.dumps(
                    {
                        "success": False,
                        "error": validation_result.get(
                            "error", "Script validation failed",
                        ),
                    },
                    indent=2,
                )

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

            # Get database client (required)
            database_client = getattr(app_ctx, "database_client", None)

            # Get Neo4j driver (optional)
            neo4j_driver = None
            if hasattr(app_ctx, "repo_extractor") and app_ctx.repo_extractor:
                neo4j_driver = getattr(app_ctx.repo_extractor, "driver", None)

            # Use the container path if available from validation
            actual_path = validation_result.get("container_path", script_path)
            return await check_hallucinations_enhanced_impl(
                database_client=database_client,
                neo4j_driver=neo4j_driver,
                script_path=actual_path,
            )

        except ValidationError as e:
            logger.exception("Validation error in hallucination detection")
            msg = f"Enhanced hallucination check failed: {e!s}"
            raise MCPToolError(msg) from e
        except DatabaseError as e:
            logger.exception("Database error in hallucination detection")
            msg = f"Enhanced hallucination check failed: {e!s}"
            raise MCPToolError(msg) from e
        except KnowledgeGraphError as e:
            logger.exception("Knowledge graph error in hallucination detection")
            msg = f"Enhanced hallucination check failed: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception(
                "Unexpected error in enhanced hallucination detection tool",
            )
            msg = f"Enhanced hallucination check failed: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("get_script_analysis_info")
    async def get_script_analysis_info(_ctx: Context) -> str:
        """
        Get information about script analysis setup and paths.

        This helper tool provides information about:
        - Available script directories
        - How to use the hallucination detection tools
        - Path mapping between host and container

        Returns:
            JSON string with setup information and usage examples
        """
        info = {
            "accessible_paths": {
                "user_scripts": "./analysis_scripts/user_scripts/",
                "test_scripts": "./analysis_scripts/test_scripts/",
                "validation_results": "./analysis_scripts/validation_results/",
                "temp_scripts": "/tmp/ (maps to /app/tmp_scripts/ in container)",
            },
            "usage_examples": [
                {
                    "description": "Analyze a script in user_scripts directory",
                    "host_path": "./analysis_scripts/user_scripts/my_script.py",
                    "tool_call": (
                        "check_ai_script_hallucinations("
                        "script_path='analysis_scripts/user_scripts/my_script.py')"
                    ),
                },
                {
                    "description": "Analyze a script from /tmp",
                    "host_path": "/tmp/test.py",
                    "tool_call": (
                        "check_ai_script_hallucinations(script_path='/tmp/test.py')"
                    ),
                },
                {
                    "description": "Analyze with just filename (defaults to user_scripts)",
                    "host_path": "./analysis_scripts/user_scripts/script.py",
                    "tool_call": (
                        "check_ai_script_hallucinations(script_path='script.py')"
                    ),
                },
            ],
            "instructions": [
                (
                    "1. Place your Python scripts in "
                    "./analysis_scripts/user_scripts/ on your host machine"
                ),
                (
                    "2. Call the hallucination detection tools with the relative path"
                ),
                (
                    "3. Results will be saved to ./analysis_scripts/validation_results/"
                ),
                (
                    "4. The path translation is automatic - you can use convenient paths"
                ),
            ],
            "container_mappings": {
                "./analysis_scripts/": "/app/analysis_scripts/",
                "/tmp/": "/app/tmp_scripts/",
            },
            "available_tools": [
                "check_ai_script_hallucinations - Basic hallucination detection",
                (
                    "check_ai_script_hallucinations_enhanced - "
                    "Enhanced detection with code suggestions"
                ),
            ],
        }

        # Check which directories actually exist
        accessible_paths = cast("dict[str, str]", info["accessible_paths"])
        for key, path in accessible_paths.items():
            if "(" not in path:  # Skip paths with descriptions
                container_path = f"/app/analysis_scripts/{key.replace('_', '_')}/"
                if Path(container_path).exists():
                    accessible_paths[key] += " ✓ (exists)"
                else:
                    accessible_paths[key] += " ✗ (not found)"

        return json.dumps(info, indent=2)
