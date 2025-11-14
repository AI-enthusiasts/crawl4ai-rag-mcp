"""
Search tools for MCP server.

This module contains search-related MCP tools including:
- search: Basic web search with SearXNG integration
- agentic_search: Advanced autonomous search with iterative refinement
- analyze_code_cross_language: Cross-language code analysis
"""

import json
import logging
from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP

from core import MCPToolError, track_request
from database import (
    get_available_sources,
    perform_rag_query,
)
from services import search_and_process
from services.agentic_search import agentic_search_impl

logger = logging.getLogger(__name__)


def register_search_tools(mcp: "FastMCP") -> None:
    """
    Register search-related MCP tools.

    Args:
        mcp: FastMCP instance to register tools with
    """

    @mcp.tool()  # type: ignore[misc]
    @track_request("search")
    async def search(
        ctx: Context,
        query: str,
        return_raw_markdown: bool = False,
        num_results: int = 6,
        batch_size: int = 20,
    ) -> str:
        """
        Comprehensive search tool that integrates SearXNG search with scraping and RAG functionality.
        Optionally, use `return_raw_markdown=true` to return raw markdown for more detailed analysis.

        This tool performs a complete search, scrape, and RAG workflow:
        1. Searches SearXNG with the provided query, obtaining `num_results` URLs
        2. Extracts markdown from URLs, chunks embedding data into Supabase
        3. Scrapes all returned URLs using existing scraping functionality
        4. Returns organized results with comprehensive metadata

        Args:
            query: The search query for SearXNG
            return_raw_markdown: If True, skip embedding/RAG and return raw markdown content (default: False)
            num_results: Number of search results to return from SearXNG (default: 6)
            batch_size: Batch size for database operations (default: 20)

        Returns:
            JSON string with search results, or raw markdown of each URL if `return_raw_markdown=true`
        """
        try:
            return await search_and_process(
                ctx=ctx,
                query=query,
                return_raw_markdown=return_raw_markdown,
                num_results=num_results,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.exception(f"Error in search tool: {e}")
            msg = f"Search failed: {e!s}"
            raise MCPToolError(msg)

    @mcp.tool()  # type: ignore[misc]
    @track_request("agentic_search")
    async def agentic_search(
        ctx: Context,
        query: str,
        completeness_threshold: float | None = None,
        max_iterations: int | None = None,
        max_urls_per_iteration: int | None = None,
        url_score_threshold: float | None = None,
        use_search_hints: bool | None = None,
    ) -> str:
        """
        Search tool that automatically finds comprehensive answers from local knowledge base or web sources.

        FAST MODE: If knowledge exists in local database, returns answer immediately from Qdrant vector storage.
        SMART MODE: If local knowledge is missing or incomplete, automatically searches web (SearXNG),
        intelligently selects and crawls relevant URLs, indexes new content, and returns comprehensive answer.

        Uses LLM to evaluate answer completeness (0.0-1.0 score, default threshold: 0.95) and automatically
        decides whether to return local results or search web. Performs iterative search-crawl-evaluate cycles
        until answer quality meets threshold or max iterations reached (default: 3).

        Best for: comprehensive research questions, technical documentation queries, finding up-to-date information,
        discovering content not yet in local database.

        Args:
            query: Search question (required)
            completeness_threshold: Minimum answer quality score 0.0-1.0 (default: 0.95, higher = stricter)
            max_iterations: Maximum search-crawl cycles (default: 3, range: 1-10)
            max_urls_per_iteration: Maximum URLs to crawl per cycle (default: 3, range: 1-20)
            url_score_threshold: Minimum URL relevance score to crawl 0.0-1.0 (default: 0.7, higher = more selective)
            use_search_hints: Enable experimental smart query refinement from metadata (default: false)

        Returns:
            JSON with search results, completeness score, iterations performed, and detailed search history

        Raises:
            MCPToolError: If agentic search is disabled in configuration or search fails critically
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
            )
        except Exception as e:
            logger.exception(f"Error in agentic_search tool: {e}")
            msg = f"Agentic search failed: {e!s}"
            raise MCPToolError(msg)

    @mcp.tool()  # type: ignore[misc]
    @track_request("analyze_code_cross_language")
    async def analyze_code_cross_language(
        ctx: Context,
        query: str,
        languages: list[str] | None = None,
        match_count: int = 10,
        source_filter: str | None = None,
        include_file_context: bool = True,
    ) -> str:
        """
        Cross-language code analysis using semantic search across multiple programming languages.

        This tool performs advanced code analysis by searching across multiple programming languages
        simultaneously, enabling developers to:
        - Find similar patterns across different languages (e.g., authentication logic in Python, JS, Go)
        - Compare implementation approaches between languages
        - Discover code reuse opportunities
        - Understand how concepts are implemented across your stack

        Supported languages include Python, JavaScript, TypeScript, Go, and more based on
        the parsed repositories in your knowledge graph.

        Args:
            query: Search query for finding code patterns across languages
            languages: Optional list of languages to search (e.g., ['python', 'javascript', 'go']). If None, searches all languages
            match_count: Maximum number of results to return per language (default: 10)
            source_filter: Optional repository filter (e.g., 'repo-name')
            include_file_context: Whether to include file path and language context (default: True)

        Returns:
            JSON string with cross-language search results, organized by language and confidence scores
        """
        import json

        try:
            # Get the app context
            from core.context import get_app_context

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
                    if languages.strip().startswith("[") and languages.strip().endswith("]"):
                        try:
                            parsed_languages = json.loads(languages)
                        except json.JSONDecodeError:
                            parsed_languages = [languages]
                    else:
                        parsed_languages = [languages]
                else:
                    parsed_languages = languages

            logger.info(f"Performing cross-language code analysis for query: {query}")
            if parsed_languages:
                logger.info(f"Filtering by languages: {parsed_languages}")

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
            results_by_language = {}

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
                if parsed_languages and language not in [lang.lower() for lang in parsed_languages]:
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
            for language in results_by_language:
                results_by_language[language] = sorted(
                    results_by_language[language],
                    key=lambda x: x.get("similarity_score", 0),
                    reverse=True,
                )[:match_count]

            # Calculate summary statistics
            total_results = sum(len(results) for results in results_by_language.values())
            languages_found = list(results_by_language.keys())

            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "languages_requested": parsed_languages or "all",
                    "languages_found": languages_found,
                    "total_results": total_results,
                    "results_by_language": results_by_language,
                    "analysis_summary": {
                        "most_relevant_language": max(results_by_language.keys(), key=lambda k: len(results_by_language[k])) if results_by_language else None,
                        "coverage": f"{len(languages_found)} languages analyzed",
                        "avg_similarity_per_language": {
                            lang: round(
                                sum(r.get("similarity_score", 0) for r in results) / len(results), 3,
                            ) if results else 0
                            for lang, results in results_by_language.items()
                        },
                    },
                    "message": f"Found {total_results} code examples across {len(languages_found)} languages",
                },
                indent=2,
            )

        except Exception as e:
            logger.exception(f"Error in cross-language code analysis: {e}")
            return json.dumps(
                {
                    "success": False,
                    "query": query,
                    "error": f"Cross-language analysis failed: {e!s}",
                },
                indent=2,
            )
