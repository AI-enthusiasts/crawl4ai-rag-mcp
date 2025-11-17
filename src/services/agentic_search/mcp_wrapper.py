"""MCP tool wrapper for agentic search.

This module provides the MCP tool entry point for agentic search,
maintaining backward compatibility with existing MCP tool definitions.
"""

import logging

from fastmcp import Context

from src.config import get_settings
from src.core import MCPToolError

# Import singleton from parent module (avoid duplication)
# Per Pydantic AI docs: Reuse Agent instances to benefit from connection pooling
from . import get_agentic_search_service

logger = logging.getLogger(__name__)
settings = get_settings()


async def agentic_search_impl(
    ctx: Context,
    query: str,
    completeness_threshold: float | None = None,
    max_iterations: int | None = None,
    max_urls_per_iteration: int | None = None,
    url_score_threshold: float | None = None,
    use_search_hints: bool | None = None,
) -> str:
    """Execute agentic search and return JSON result.

    This is the main entry point called by the MCP tool.

    Args:
        ctx: FastMCP context
        query: User's search query
        completeness_threshold: Override default completeness threshold (0-1)
        max_iterations: Override default max iterations (1-10)
        max_urls_per_iteration: Override default max URLs per iteration (1-20)
        url_score_threshold: Override default URL score threshold (0-1)
        use_search_hints: Override default search hints setting

    Returns:
        JSON string with complete search results

    Raises:
        MCPToolError: If search fails
    """
    if not settings.agentic_search_enabled:
        msg = "Agentic search is not enabled. Set AGENTIC_SEARCH_ENABLED=true in your environment."
        raise MCPToolError(msg)

    try:
        # Get singleton service instance (connection pooling optimization)
        service = get_agentic_search_service()
        result = await service.execute_search(
            ctx=ctx,
            query=query,
            completeness_threshold=completeness_threshold,
            max_iterations=max_iterations,
            max_urls_per_iteration=max_urls_per_iteration,
            url_score_threshold=url_score_threshold,
            use_search_hints=use_search_hints,
        )
        return result.model_dump_json()

    except Exception as e:
        logger.exception("Agentic search implementation failed")
        raise MCPToolError(f"Agentic search failed: {e}") from e
