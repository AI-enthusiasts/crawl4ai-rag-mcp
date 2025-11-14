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
        return_raw_markdown: bool = False,
    ) -> str:
        """
        Scrape **one or more URLs** and store their contents as embedding chunks in Supabase.
        Optionally, use `return_raw_markdown=true` to return raw markdown content without storing.

        The content is scraped and stored in Supabase for later retrieval and querying via perform_rag_query tool, unless
        `return_raw_markdown=True` is specified, in which case raw markdown is returned directly.

        Args:
            url: URL to scrape, or list of URLs for batch processing
            batch_size: Size of batches for database operations (default: 20)
            return_raw_markdown: If True, skip database storage and return raw markdown content (default: False)

        Returns:
            Summary of the scraping operation and storage in Supabase, or raw markdown content if requested
        """
        try:
            # Security: Add input size limit to prevent JSON bomb attacks
            MAX_INPUT_SIZE = 50000  # 50KB limit for safety

            # Handle URL parameter which can be:
            # 1. Single URL string
            # 2. JSON string representation of a list (from MCP protocol)
            # 3. Actual Python list

            # Enhanced debug logging
            logger.debug(
                f"scrape_urls received url parameter (type: {type(url).__name__})",
            )

            urls = []
            if isinstance(url, str):
                # Security check: Limit input size
                if len(url) > MAX_INPUT_SIZE:
                    msg = f"Input too large: {len(url)} bytes (max: {MAX_INPUT_SIZE})"
                    raise ValueError(
                        msg,
                    )
                # Clean whitespace and normalize the string
                cleaned_url = url.strip()
                logger.debug(f"Processing string URL, cleaned: {cleaned_url!r}")

                # Check if it's a JSON string representation of a list
                # Be more precise: must start with [ and end with ] and likely contain quotes
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
                                f"Successfully parsed JSON array with {len(urls)} URLs",
                            )
                        else:
                            urls = [
                                cleaned_url,
                            ]  # Single URL that looks like JSON but isn't a list
                            logger.debug(
                                "JSON parsed but result is not a list, treating as single URL",
                            )
                    except json.JSONDecodeError as json_err:
                        logger.debug(
                            f"JSON parsing failed ({json_err}), treating as single URL",
                        )
                        # Don't attempt fallback parsing with comma split as it can break valid URLs
                        # URLs can contain commas in query parameters
                        urls = [cleaned_url]  # Treat as single URL
                else:
                    urls = [cleaned_url]  # Single URL
                    logger.debug("Single URL string detected")
            elif isinstance(url, list):
                urls = url  # Assume it's already a list
                logger.debug(f"List parameter received with {len(urls)} URLs")
            else:
                # Handle other types by converting to string (defensive programming)
                logger.warning(  # type: ignore[unreachable]
                    f"Unexpected URL parameter type {type(url)}, converting to string",
                )
                urls = [str(url)]

            # Clean and validate each URL in the final list
            from src.utils.url_helpers import clean_url

            cleaned_urls = []
            invalid_urls = []

            for i, raw_url in enumerate(urls):
                try:
                    # Convert to string if not already
                    url_str = str(raw_url).strip()
                    logger.debug(f"Processing URL {i + 1}/{len(urls)}: {url_str!r}")

                    if not url_str:
                        logger.warning(f"Empty URL at position {i + 1}, skipping")
                        continue

                    # Clean the URL using utility function
                    cleaned_url = clean_url(url_str)
                    if cleaned_url:
                        cleaned_urls.append(cleaned_url)
                        logger.debug(f"URL {i + 1} cleaned successfully: {cleaned_url}")
                    else:
                        invalid_urls.append(url_str)
                        logger.warning(f"URL {i + 1} failed cleaning: {url_str}")

                except Exception as url_err:
                    logger.exception(
                        f"Error processing URL {i + 1} ({raw_url!r}): {url_err}",
                    )
                    invalid_urls.append(str(raw_url))

            # Log final results
            logger.info(
                f"URL processing complete: {len(cleaned_urls)} valid URLs, {len(invalid_urls)} invalid URLs",
            )
            if invalid_urls:
                logger.warning(f"Invalid URLs that were skipped: {invalid_urls}")

            if not cleaned_urls:
                error_msg = "No valid URLs found after processing and cleaning"
                logger.error(error_msg)
                raise MCPToolError(error_msg)

            # Use cleaned URLs for processing
            return await process_urls_for_mcp(
                ctx=ctx,
                urls=cleaned_urls,
                batch_size=batch_size,
                return_raw_markdown=return_raw_markdown,
            )
        except Exception as e:
            logger.exception(f"Error in scrape_urls tool: {e}")
            msg = f"Scraping failed: {e!s}"
            raise MCPToolError(msg)

    @mcp.tool()
    @track_request("smart_crawl_url")
    async def smart_crawl_url(
        ctx: Context,
        url: str,
        max_depth: int = 3,
        chunk_size: int = 5000,
        return_raw_markdown: bool = False,
        query: list[str] | str | None = None,
    ) -> str:
        """
        Intelligently crawl a URL based on its type and store content in Supabase.
        Enhanced with raw markdown return and RAG query capabilities.

        This tool automatically detects the URL type and applies the appropriate crawling method:
        - For sitemaps: Extracts and crawls all URLs in parallel
        - For text files (llms.txt): Directly retrieves the content
        - For regular webpages: Recursively crawls internal links up to the specified depth

        All crawled content is chunked and stored in Supabase for later retrieval and querying.

        Args:
            url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
            max_depth: Maximum recursion depth for regular URLs (default: 3)
            chunk_size: Maximum size of each content chunk in characters (default: 5000)
            return_raw_markdown: If True, return raw markdown content instead of just storing (default: False)
            query: List of queries to perform RAG search on crawled content (default: None)

        Returns:
            JSON string with crawl summary, raw markdown (if requested), or RAG query results
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
            logger.exception(f"Error in smart_crawl_url tool: {e}")
            msg = f"Smart crawl failed: {e!s}"
            raise MCPToolError(msg)
