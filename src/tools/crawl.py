"""
Crawling tools for MCP server.

This module contains web crawling and scraping MCP tools including:
- scrape_urls: Scrape one or more URLs and store content
- smart_crawl_url: Intelligently crawl URLs with type detection
"""

import json
import logging
from typing import TYPE_CHECKING

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP

from src.core import MCPToolError, track_request
from src.services import process_urls_for_mcp
from src.services import (
    smart_crawl_url as smart_crawl_url_service_impl,
)
from src.utils.url_helpers import clean_url

logger = logging.getLogger(__name__)


def register_crawl_tools(mcp: "FastMCP") -> None:
    """
    Register crawling-related MCP tools.

    Args:
        mcp: FastMCP instance to register tools with
    """

    @mcp.tool()
    @track_request("scrape_urls")
    async def scrape_urls(
        ctx: Context,
        url: str | list[str],
        batch_size: int = 20,
        *,
        return_raw_markdown: bool = False,
    ) -> str:
        """Scrape one or more URLs and store as embedding chunks.

        Scrape one or more URLs and store as embedding chunks in Supabase.
        Optionally return raw markdown instead of storing.

        Content is scraped and stored in Supabase for later retrieval. If
        `return_raw_markdown=True` is specified, raw markdown is returned directly.

        Args:
            ctx: MCP context
            url: URL or list of URLs for batch processing
            batch_size: Batch size for database operations (default: 20)
            return_raw_markdown: If True, return raw markdown (default: False)

        Returns:
            Summary of scraping operation or raw markdown if requested
        """
        # Security: Add input size limit to prevent JSON bomb attacks
        max_input_size = 50000  # 50KB limit for safety

        # Handle URL parameter which can be:
        # 1. Single URL string
        # 2. JSON string representation of a list (from MCP protocol)
        # 3. Actual Python list

        # Enhanced debug logging
        logger.debug(
            "scrape_urls received url parameter (type: %s)",
            type(url).__name__,
        )

        urls = []
        if isinstance(url, str):
            # Security check: Limit input size
            if len(url) > max_input_size:
                msg = f"Input too large: {len(url)} bytes (max: {max_input_size})"
                raise MCPToolError(msg)
            # Clean whitespace and normalize the string
            cleaned_url = url.strip()
            logger.debug("Processing string URL, cleaned: %r", cleaned_url)

            # Check if it's a JSON string representation of a list
            # Must start with [ and end with ] and likely contain quotes
            if (
                cleaned_url.startswith("[")
                and cleaned_url.endswith("]")
                and ('"' in cleaned_url or "'" in cleaned_url)
            ):
                logger.debug("Detected JSON array format, attempting to parse...")
                try:
                    # Handle common JSON escaping issues
                    # First, try to parse as-is
                    parsed = json.loads(cleaned_url)
                    if isinstance(parsed, list):
                        urls = parsed
                        logger.debug(
                            "Successfully parsed JSON array with %d URLs",
                            len(urls),
                        )
                    else:
                        urls = [
                            cleaned_url,
                        ]  # Single URL that looks like JSON but isn't a list
                        logger.debug(
                            "JSON parsed but not a list, treating as single",
                        )
                except json.JSONDecodeError as json_err:
                    logger.debug(
                        "JSON parsing failed (%s), treating as single URL",
                        json_err,
                    )
                    # Don't split by comma - URLs can have commas in parameters
                    urls = [cleaned_url]  # Treat as single URL
            else:
                urls = [cleaned_url]  # Single URL
                logger.debug("Single URL string detected")
        elif isinstance(url, list):
            urls = url  # Assume it's already a list
            logger.debug("List parameter received with %d URLs", len(urls))
        else:
            # Handle other types by converting to string (defensive programming)
            logger.warning(  # type: ignore[unreachable]
                "Unexpected URL parameter type %s, converting to string",
                type(url),
            )
            urls = [str(url)]

        try:
            # Clean and validate each URL in the final list
            cleaned_urls = []
            invalid_urls = []

            for i, raw_url in enumerate(urls):
                try:
                    # Convert to string if not already
                    url_str = str(raw_url).strip()
                    logger.debug("Processing URL %d/%d: %r", i + 1, len(urls), url_str)

                    if not url_str:
                        logger.warning("Empty URL at position %d, skipping", i + 1)
                        continue

                    # Clean the URL using utility function
                    cleaned_url = clean_url(url_str)
                    if cleaned_url:
                        cleaned_urls.append(cleaned_url)
                        logger.debug("URL %d cleaned: %s", i + 1, cleaned_url)
                    else:
                        invalid_urls.append(url_str)
                        logger.warning("URL %d failed cleaning: %s", i + 1, url_str)

                except Exception:
                    logger.exception(
                        "Error processing URL %d (%r)",
                        i + 1,
                        raw_url,
                    )
                    invalid_urls.append(str(raw_url))

            # Log final results
            logger.info(
                "URL processing complete: %d valid URLs, %d invalid URLs",
                len(cleaned_urls),
                len(invalid_urls),
            )
            if invalid_urls:
                logger.warning("Invalid URLs that were skipped: %s", invalid_urls)

            if cleaned_urls:
                # Use cleaned URLs for processing
                return await process_urls_for_mcp(
                    ctx=ctx,
                    urls=cleaned_urls,
                    batch_size=batch_size,
                    return_raw_markdown=return_raw_markdown,
                )
        except Exception as e:
            logger.exception("Error in scrape_urls tool")
            msg = f"Scraping failed: {e!s}"
            raise MCPToolError(msg) from e

        msg = "No valid URLs found after processing and cleaning"
        logger.error(msg)
        raise MCPToolError(msg)

    @mcp.tool()
    @track_request("smart_crawl_url")
    async def smart_crawl_url(
        ctx: Context,
        url: str,
        max_depth: int = 3,
        chunk_size: int = 5000,
        *,
        return_raw_markdown: bool = False,
        query: list[str] | str | None = None,
    ) -> str:
        """Intelligently crawl a URL and store content in Supabase.

        Automatically detects URL type and applies the appropriate crawling method:
        - For sitemaps: Extracts and crawls all URLs in parallel
        - For text files: Directly retrieves the content
        - For webpages: Recursively crawls internal links

        Args:
            ctx: MCP context
            url: URL to crawl (webpage, sitemap.xml, or .txt file)
            max_depth: Maximum recursion depth (default: 3)
            chunk_size: Maximum chunk size in characters (default: 5000)
            return_raw_markdown: If True, return raw markdown (default: False)
            query: List of queries for RAG search (default: None)

        Returns:
            Crawl summary, raw markdown, or RAG query results
        """
        try:
            # Handle query parameter which can be:
            # 1. None
            # 2. JSON string representation of a list (from MCP protocol)
            # 3. Actual Python list
            parsed_query = None
            if query is not None:
                if isinstance(query, str):
                    # Check if it's a JSON string representation of a list
                    if query.strip().startswith("[") and query.strip().endswith("]"):
                        try:
                            parsed = json.loads(query)
                            if isinstance(parsed, list):
                                parsed_query = parsed
                            else:
                                parsed_query = [query]  # Single query
                        except json.JSONDecodeError:
                            parsed_query = [query]  # Single query, JSON parsing failed
                    else:
                        parsed_query = [query]  # Single query
                else:
                    parsed_query = query  # Assume it's already a list

            # Call the implementation function with the correct aliased name
            return await smart_crawl_url_service_impl(
                ctx=ctx,
                url=url,
                max_depth=max_depth,
                chunk_size=chunk_size,
                return_raw_markdown=return_raw_markdown,
                query=parsed_query,
            )
        except Exception as e:
            logger.exception("Error in smart_crawl_url tool")
            msg = f"Smart crawl failed: {e!s}"
            raise MCPToolError(msg) from e
