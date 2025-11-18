"""
Search tools for MCP server.

This module contains search-related MCP tools including:
- search: Basic web search with SearXNG integration
- agentic_search: Advanced autonomous search with iterative refinement
- analyze_code_cross_language: Cross-language code analysis
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP

from src.core import MCPToolError, track_request
from src.core.context import get_app_context
from src.core.exceptions import DatabaseError, SearchError
from src.database import (
    get_available_sources,
    perform_rag_query,
)
from src.services import search_and_process
from src.services.agentic_search import agentic_search_impl

logger = logging.getLogger(__name__)


def register_search_tools(mcp: "FastMCP") -> None:
    """
    Register search-related MCP tools.

    Args:
        mcp: FastMCP instance to register tools with
    """

    @mcp.tool()
    @track_request("search")
    async def search(
        ctx: Context,
        query: str,
        *,
        return_raw_markdown: bool = False,
        num_results: int = 6,
        batch_size: int = 20,
    ) -> str:
        """
        Comprehensive search tool integrating SearXNG, scraping, and RAG.

        Optionally, use `return_raw_markdown=true` to return raw markdown for
        more detailed analysis. This tool performs a complete search, scrape,
        and RAG workflow:
        1. Searches SearXNG with the provided query, obtaining `num_results`
        2. Extracts markdown from URLs, chunks into embedding data
        3. Scrapes all returned URLs using existing functionality
        4. Returns organized results with comprehensive metadata

        Args:
            ctx: The MCP context for execution
            query: The search query for SearXNG
            return_raw_markdown: Skip embedding/RAG, return raw markdown
            num_results: Number of search results to return (default: 6)
            batch_size: Batch size for database operations (default: 20)

        Returns:
            JSON string with search results, or raw markdown if enabled.
        """
        try:
            return await search_and_process(
                ctx=ctx,
                query=query,
                return_raw_markdown=return_raw_markdown,
                num_results=num_results,
                batch_size=batch_size,
            )
        except SearchError as e:
            logger.exception("Search error")
            msg = f"Search failed: {e!s}"
            raise MCPToolError(msg) from e
        except DatabaseError as e:
            logger.exception("Database error during search")
            msg = f"Search failed: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception("Unexpected error in search tool")
            msg = f"Search failed: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("agentic_search")
    async def agentic_search(
        ctx: Context,
        query: str,
        *,
        completeness_threshold: float | None = None,
        max_iterations: int | None = None,
        max_urls_per_iteration: int | None = None,
        url_score_threshold: float | None = None,
        use_search_hints: bool | None = None,
        deep_research: bool = True,
    ) -> str:
        """
        Intelligent search finding comprehensive answers from local or web.

        Operates in DEEP RESEARCH MODE (default):
        1. Decomposes query into essential topics (Definition, Examples, etc.)
        2. Generates multiple query variations per topic for better recall
        3. Evaluates completeness per topic, not just overall score
        4. Gap-driven iteration: searches only for uncovered topics
        5. Uses Reciprocal Rank Fusion to combine multi-query results

        Standard mode (deep_research=False):
        1. Checks Qdrant vector database for existing knowledge
        2. LLM evaluates answer completeness (0.0-1.0)
        3. If complete, returns from local storage immediately
        4. If incomplete, searches web and performs iterative cycles

        Args:
            ctx: The MCP context for execution
            query: Search question (required)
            completeness_threshold: Quality score 0.0-1.0, default 0.95
            max_iterations: Max search-crawl cycles, default 3 (1-10)
            max_urls_per_iteration: Max URLs per cycle, default 3 (1-20)
            url_score_threshold: Min URL relevance 0.0-1.0, default 0.7
            use_search_hints: Enable smart query refinement (default: false)
            deep_research: Use deep research with topic decomposition (default: true)

        Returns:
            JSON with results, completeness, iterations, search history.

        Raises:
            MCPToolError: If disabled or search fails critically.
        """
        try:
            return await agentic_search_impl(
                ctx=ctx,
                query=query,
                completeness_threshold=completeness_threshold,
                max_iterations=max_iterations,
                max_urls_per_iteration=max_urls_per_iteration,
                url_score_threshold=url_score_threshold,
                use_search_hints=use_search_hints,
                deep_research=deep_research,
            )
        except SearchError as e:
            logger.exception("Search error in agentic search")
            msg = f"Agentic search failed: {e!s}"
            raise MCPToolError(msg) from e
        except DatabaseError as e:
            logger.exception("Database error in agentic search")
            msg = f"Agentic search failed: {e!s}"
            raise MCPToolError(msg) from e
        except Exception as e:
            logger.exception("Unexpected error in agentic_search tool")
            msg = f"Agentic search failed: {e!s}"
            raise MCPToolError(msg) from e

    @mcp.tool()
    @track_request("analyze_code_cross_language")
    async def analyze_code_cross_language(
        _ctx: Context,
        query: str,
        *,
        languages: list[str] | str | None = None,
        match_count: int = 10,
        source_filter: str | None = None,
        include_file_context: bool = True,
    ) -> str:
        """
        Cross-language code analysis using semantic search.

        Searches across multiple programming languages simultaneously to enable:
        - Finding patterns across languages (auth in Python, JS, Go)
        - Comparing implementation approaches
        - Discovering code reuse opportunities
        - Understanding stack-wide concepts

        Supports Python, JavaScript, TypeScript, Go, and more based on
        parsed repositories in the knowledge graph.

        Args:
            ctx: The MCP context for execution
            query: Search query for code patterns across languages
            languages: Optional language list, e.g. ['python', 'js'].
                None searches all languages
            match_count: Max results per language (default: 10)
            source_filter: Optional repository filter (e.g., 'repo-name')
            include_file_context: Include file and language info (default: True)

        Returns:
            JSON with cross-language results by language and confidence.
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

            # Check database client availability
            database_client = getattr(app_ctx, "database_client", None)
            if not database_client:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Database client not available",
                    },
                    indent=2,
                )

            # Handle languages parameter (from JSON if needed)
            parsed_languages = None
            if languages is not None:
                if isinstance(languages, str):
                    stripped = languages.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        try:
                            parsed_languages = json.loads(languages)
                        except json.JSONDecodeError:
                            parsed_languages = [languages]
                    else:
                        parsed_languages = [languages]
                else:
                    parsed_languages = languages

            logger.info("Performing cross-language code analysis for query: %s", query)
            if parsed_languages:
                logger.info("Filtering by languages: %s", parsed_languages)

            # Get all available sources first to understand what repositories we have
            sources_result = await get_available_sources(database_client)
            sources_data = json.loads(sources_result)

            if not sources_data.get("success", False):
                return json.dumps(
                    {
                        "success": False,
                        "error": "Could not retrieve available sources for analysis",
                        "details": sources_data,
                    },
                    indent=2,
                )

            # Perform semantic search
            rag_result = await perform_rag_query(
                database_client,
                query=query,
                source=source_filter,
                match_count=match_count * 3,  # Get more results to filter by language
            )

            rag_data = json.loads(rag_result)

            if not rag_data.get("success", False):
                return json.dumps(
                    {
                        "success": False,
                        "error": "Semantic search failed",
                        "details": rag_data,
                    },
                    indent=2,
                )

            # Organize results by language
            results_by_language: dict[str, list[dict[str, Any]]] = {}

            for result in rag_data.get("results", []):
                # Extract language information from metadata or URL
                language = "unknown"
                metadata = result.get("metadata", {})
                url = result.get("url", "")

                # Try to determine language from metadata
                if "language" in metadata:
                    language = metadata["language"].lower()
                elif "file_extension" in metadata:
                    ext = metadata["file_extension"].lower()
                    language_map = {
                        "py": "python",
                        "js": "javascript",
                        "ts": "typescript",
                        "jsx": "javascript",
                        "tsx": "typescript",
                        "go": "go",
                        "java": "java",
                        "cpp": "c++",
                        "c": "c",
                        "rs": "rust",
                        "php": "php",
                        "rb": "ruby",
                        "swift": "swift",
                        "kt": "kotlin",
                        "cs": "csharp",
                    }
                    language = language_map.get(ext, ext)
                elif url:
                    # Try to extract from URL/filename
                    for ext, lang in {
                        ".py": "python",
                        ".js": "javascript",
                        ".ts": "typescript",
                        ".jsx": "javascript",
                        ".tsx": "typescript",
                        ".go": "go",
                        ".java": "java",
                        ".cpp": "c++",
                        ".c": "c",
                        ".rs": "rust",
                        ".php": "php",
                        ".rb": "ruby",
                        ".swift": "swift",
                        ".kt": "kotlin",
                        ".cs": "csharp",
                    }.items():
                        if ext in url.lower():
                            language = lang
                            break

                # Filter by languages if specified
                if parsed_languages:
                    lower_langs = [lang.lower() for lang in parsed_languages]
                    if language not in lower_langs:
                        continue

                # Initialize language group if needed
                if language not in results_by_language:
                    results_by_language[language] = []

                # Add file context if requested
                result_item = {
                    "content": result.get("content", ""),
                    "similarity_score": result.get("similarity_score", 0),
                    "source": result.get("source", "unknown"),
                }

                if include_file_context:
                    result_item["file_context"] = {
                        "url": url,
                        "metadata": metadata,
                        "language": language,
                    }

                results_by_language[language].append(result_item)

            # Limit results per language and sort by similarity
            for language, results in results_by_language.items():
                sorted_results = sorted(
                    results,
                    key=lambda x: x.get("similarity_score", 0),
                    reverse=True,
                )
                results_by_language[language] = sorted_results[:match_count]

            # Calculate summary statistics
            total_results = sum(
                len(results) for results in results_by_language.values()
            )
            languages_found = list(results_by_language.keys())

            # Calculate most relevant language
            most_relevant = None
            if results_by_language:
                most_relevant = max(
                    results_by_language.keys(),
                    key=lambda k: len(results_by_language[k]),
                )

            # Calculate average similarity per language
            avg_similarity = {}
            for lang, results in results_by_language.items():
                if results:
                    scores = [r.get("similarity_score", 0) for r in results]
                    avg_similarity[lang] = round(sum(scores) / len(results), 3)
                else:
                    avg_similarity[lang] = 0

            message = (
                f"Found {total_results} code examples across "
                f"{len(languages_found)} languages"
            )

            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "languages_requested": parsed_languages or "all",
                    "languages_found": languages_found,
                    "total_results": total_results,
                    "results_by_language": results_by_language,
                    "analysis_summary": {
                        "most_relevant_language": most_relevant,
                        "coverage": f"{len(languages_found)} languages analyzed",
                        "avg_similarity_per_language": avg_similarity,
                    },
                    "message": message,
                },
                indent=2,
            )

        except DatabaseError as e:
            logger.exception("Database error in cross-language code analysis")
            return json.dumps(
                {
                    "success": False,
                    "query": query,
                    "error": f"Database error: {e!s}",
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Unexpected error in cross-language code analysis")
            return json.dumps(
                {
                    "success": False,
                    "query": query,
                    "error": f"Cross-language analysis failed: {e!s}",
                },
                indent=2,
            )
