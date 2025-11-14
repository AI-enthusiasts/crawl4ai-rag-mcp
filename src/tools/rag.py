"""
RAG (Retrieval Augmented Generation) tools for MCP server.

This module contains RAG-related MCP tools including:
- get_available_sources: List all indexed sources
- perform_rag_query: Semantic search over indexed content
- search_code_examples: Search for code examples in vector database
"""

import json
import logging
from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP

from src.core import MCPToolError, track_request
from src.core.context import get_app_context
from src.database import (
    get_available_sources,
    perform_rag_query,
)
from src.database import (
    search_code_examples as search_code_examples_db,
)

logger = logging.getLogger(__name__)


async def get_available_sources_wrapper(ctx: Context) -> str:
    """
    Wrapper function to properly extract database_client from context and call the implementation.
    """
    import json

    # Get the app context that was stored during lifespan
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

    return await get_available_sources(app_ctx.database_client)


async def perform_rag_query_wrapper(
    ctx: Context,
    query: str,
    source: str | None = None,
    match_count: int = 5,
) -> str:
    """
    Wrapper function to properly extract database_client from context and call the implementation.
    """
    import json

    # Get the app context that was stored during lifespan
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

    return await perform_rag_query(
        app_ctx.database_client,
        query=query,
        source=source,
        match_count=match_count,
    )


async def search_code_examples_wrapper(
    ctx: Context,
    query: str,
    source_id: str | None = None,
    match_count: int = 5,
) -> str:
    """
    Wrapper function to properly extract database_client from context and call the implementation.
    """
    import json

    # Get the app context that was stored during lifespan
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

    return await search_code_examples_db(
        app_ctx.database_client,
        query=query,
        source_id=source_id,
        match_count=match_count,
    )


def register_rag_tools(mcp: "FastMCP") -> None:
    """
    Register RAG-related MCP tools.

    Args:
        mcp: FastMCP instance to register tools with
    """

    @mcp.tool()
    @track_request("get_available_sources")
    async def get_available_sources(ctx: Context) -> str:
        """
        Get all available sources from the sources table.

        This tool returns a list of all unique sources (domains) that have been crawled and stored
        in the database, along with their summaries and statistics. This is useful for discovering
        what content is available for querying.

        Always use this tool before calling the RAG query or code example query tool
        with a specific source filter!

        Args:
            NONE

        Returns:
            JSON string with the list of available sources and their details
        """
        try:
            return await get_available_sources_wrapper(ctx)
        except Exception as e:
            logger.exception(f"Error in get_available_sources tool: {e}")
            msg = f"Failed to get sources: {e!s}"
            raise MCPToolError(msg)

    @mcp.tool()
    @track_request("perform_rag_query")
    async def perform_rag_query(
        ctx: Context,
        query: str,
        source: str | None = None,
        match_count: int = 5,
    ) -> str:
        """
        Perform a RAG (Retrieval Augmented Generation) query on the stored content.

        This tool searches the vector database for content relevant to the query and returns
        the matching documents. Optionally filter by source domain.
        Get the source by using the get_available_sources tool before calling this search!

        Args:
            query: The search query
            source: Optional source domain to filter results (e.g., 'example.com')
            match_count: Maximum number of results to return (default: 5)

        Returns:
            JSON string with the search results
        """
        try:
            return await perform_rag_query_wrapper(
                ctx=ctx,
                query=query,
                source=source,
                match_count=match_count,
            )
        except Exception as e:
            logger.exception(f"Error in perform_rag_query tool: {e}")
            msg = f"RAG query failed: {e!s}"
            raise MCPToolError(msg)

    @mcp.tool()
    @track_request("search_code_examples")
    async def search_code_examples(
        ctx: Context,
        query: str,
        source_id: str | None = None,
        match_count: int = 5,
    ) -> str:
        """
        Search for code examples relevant to the query.

        This tool searches the vector database for code examples relevant to the query and returns
        the matching examples with their summaries. Optionally filter by source_id.
        Get the source_id by using the get_available_sources tool before calling this search!

        Use the get_available_sources tool first to see what sources are available for filtering.

        Args:
            query: The search query
            source_id: Optional source ID to filter results (e.g., 'example.com')
            match_count: Maximum number of results to return (default: 5)

        Returns:
            JSON string with the search results
        """
        try:
            return await search_code_examples_wrapper(
                ctx=ctx,
                query=query,
                source_id=source_id,
                match_count=match_count,
            )
        except Exception as e:
            logger.exception(f"Error in search_code_examples tool: {e}")
            msg = f"Code example search failed: {e!s}"
            raise MCPToolError(msg)
