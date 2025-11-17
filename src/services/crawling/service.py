"""Main crawling service orchestration and MCP entry points.

This module provides the main entry points for crawling operations,
including URL processing, database storage, and agentic search crawling.
"""

import asyncio
import json
import logging
import re
from typing import Any

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from fastmcp import Context

from src.core.constants import MAX_VISITED_URLS_LIMIT, URL_FILTER_PATTERNS
from src.core.context import get_app_context
from src.core.exceptions import DatabaseError
from src.core.stdout_utils import SuppressStdout
from src.utils import add_documents_to_database
from src.utils.text_processing import smart_chunk_markdown
from src.utils.url_helpers import extract_domain_from_url, normalize_url

from .batch import crawl_batch
from .memory import track_memory

logger = logging.getLogger(__name__)


async def process_urls_for_mcp(
    ctx: Context,
    urls: list[str],
    batch_size: int = 20,
    return_raw_markdown: bool = False,
) -> str:
    """Process URLs for MCP tools with context extraction and database storage.

    This is a bridge function that:
    1. Extracts the crawler from the MCP context
    2. Calls the low-level crawl_batch function
    3. Handles database storage and response formatting
    4. Supports return_raw_markdown option for direct content return

    Args:
        ctx: FastMCP context containing Crawl4AIContext
        urls: List of URLs to crawl
        batch_size: Batch size for database operations
        return_raw_markdown: If True, return raw markdown instead of storing

    Returns:
        JSON string with results
    """
    try:
        # Extract the Crawl4AI context from the FastMCP context
        if not hasattr(ctx, "crawl4ai_context") or not ctx.crawl4ai_context:
            # Get from global app context if available
            app_ctx = get_app_context()
            if not app_ctx:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Application context not available",
                    },
                )
            crawl4ai_ctx = app_ctx
        else:
            crawl4ai_ctx = ctx.crawl4ai_context

        # Validate that context has required attributes instead of strict type checking
        if not (
            hasattr(crawl4ai_ctx, "browser_config")
            and hasattr(crawl4ai_ctx, "database_client")
            and hasattr(crawl4ai_ctx, "dispatcher")
        ):
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid Crawl4AI context: missing required attributes (browser_config, database_client, dispatcher)",
                },
            )

        if not crawl4ai_ctx.browser_config or not crawl4ai_ctx.database_client or not crawl4ai_ctx.dispatcher:
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid Crawl4AI context: browser_config, database_client, or dispatcher is None",
                },
            )

        # Call the low-level crawl_batch function
        crawl_results = await crawl_batch(
            browser_config=crawl4ai_ctx.browser_config,
            urls=urls,
            dispatcher=crawl4ai_ctx.dispatcher,
        )

        if return_raw_markdown:
            # Return raw markdown content directly
            return json.dumps(
                {
                    "success": True,
                    "total_urls": len(urls),
                    "results": [
                        {
                            "url": result["url"],
                            "markdown": result["markdown"],
                            "success": True,
                        }
                        for result in crawl_results
                    ],
                },
            )

        # Store results in database
        stored_results = []
        for result in crawl_results:
            try:
                # Chunk the markdown content
                chunks = smart_chunk_markdown(result["markdown"], chunk_size=2000)

                if not chunks:
                    stored_results.append(
                        {
                            "url": result["url"],
                            "success": False,
                            "error": "No content to store",
                            "chunks_stored": 0,
                        },
                    )
                    continue

                # Extract source from URL for proper metadata storage
                source_id = extract_domain_from_url(result["url"])

                # Prepare data for database storage
                urls_list = [result["url"]] * len(chunks)
                chunk_numbers = list(range(len(chunks)))
                contents = chunks
                metadatas = [
                    {"url": result["url"], "chunk": i} for i in range(len(chunks))
                ]
                url_to_full_document = {result["url"]: result["markdown"]}
                source_ids = [source_id] * len(chunks) if source_id else None

                # Store in database
                await add_documents_to_database(
                    database=crawl4ai_ctx.database_client,
                    urls=urls_list,
                    chunk_numbers=chunk_numbers,
                    contents=contents,
                    metadatas=metadatas,
                    url_to_full_document=url_to_full_document,
                    batch_size=batch_size,
                    source_ids=source_ids,
                )

                stored_results.append(
                    {
                        "url": result["url"],
                        "success": True,
                        "chunks_stored": len(chunks),
                        "source_id": source_id,
                    },
                )
            except DatabaseError as e:
                logger.error(f"Database error storing {result['url']}: {e}")
                stored_results.append(
                    {
                        "url": result["url"],
                        "success": False,
                        "error": str(e),
                        "chunks_stored": 0,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to store {result['url']}: {e}")
                stored_results.append(
                    {
                        "url": result["url"],
                        "success": False,
                        "error": str(e),
                        "chunks_stored": 0,
                    },
                )

        return json.dumps(
            {
                "success": True,
                "total_urls": len(urls),
                "results": stored_results,
            },
        )

    except Exception as e:
        logger.error(f"Error in process_urls_for_mcp: {e}")
        return json.dumps(
            {
                "success": False,
                "error": str(e),
            },
        )


def should_filter_url(url: str, enable_filtering: bool = True) -> bool:
    """Check if URL should be filtered out using smart patterns.

    Filters out URLs that are likely to cause infinite crawling or contain
    duplicate/low-value content (GitHub commits, pagination, archives, etc.).

    Args:
        url: URL to check
        enable_filtering: Whether filtering is enabled (from settings)

    Returns:
        True if URL should be filtered (skipped), False otherwise
    """
    if not enable_filtering:
        return False

    # Check against all filter patterns
    for pattern in URL_FILTER_PATTERNS:
        if re.search(pattern, url):
            logger.debug(f"Filtering URL (matched pattern {pattern}): {url}")
            return True

    return False


async def crawl_urls_for_agentic_search(
    ctx: Context,
    urls: list[str],
    max_pages: int = 50,
    enable_url_filtering: bool = True,
) -> dict[str, Any]:
    """Crawl URLs recursively for agentic search with smart limits and filtering.

    This function is specifically designed for agentic search and provides:
    1. Recursive crawling of internal links (no depth limit)
    2. Smart URL filtering to avoid GitHub commits, pagination, etc.
    3. Page limit per iteration (prevents excessive crawling)
    4. Visited URL tracking (prevents re-crawling same pages)
    5. Full Qdrant indexing of all crawled pages

    Args:
        ctx: FastMCP context containing Crawl4AIContext
        urls: Starting URLs to crawl (typically 5 from agentic search)
        max_pages: Maximum pages to crawl across all URLs (default: 50)
        enable_url_filtering: Enable smart URL filtering (default: True)

    Returns:
        Dict with:
        - success: bool
        - urls_crawled: int (number of pages successfully crawled)
        - urls_stored: int (number of pages stored in Qdrant)
        - chunks_stored: int (total chunks stored)
        - urls_filtered: int (number of URLs filtered out)
        - error: str (optional, if failed)
    """
    try:
        # Extract the Crawl4AI context from the FastMCP context
        app_ctx = get_app_context()
        if not app_ctx:
            return {
                "success": False,
                "error": "Application context not available",
                "urls_crawled": 0,
                "urls_stored": 0,
                "chunks_stored": 0,
                "urls_filtered": 0,
            }

        if not (
            hasattr(app_ctx, "browser_config")
            and hasattr(app_ctx, "database_client")
            and hasattr(app_ctx, "dispatcher")
        ):
            return {
                "success": False,
                "error": "Invalid context: missing required attributes",
                "urls_crawled": 0,
                "urls_stored": 0,
                "chunks_stored": 0,
                "urls_filtered": 0,
            }

        browser_config = app_ctx.browser_config
        database_client = app_ctx.database_client
        dispatcher = app_ctx.dispatcher

        # Track visited URLs globally (across all starting URLs)
        visited: set[str] = set()
        # Protect visited set from race conditions (per asyncio.Lock docs)
        visited_lock = asyncio.Lock()
        urls_filtered = 0
        pages_crawled = 0
        total_urls_stored = 0
        total_chunks_stored = 0

        # Normalize starting URLs
        current_urls = {normalize_url(u) for u in urls}

        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        logger.info(
            f"Starting recursive crawl: {len(current_urls)} URLs, "
            f"max_pages={max_pages}, filtering={'enabled' if enable_url_filtering else 'disabled'}",
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            depth = 0
            while current_urls and pages_crawled < max_pages:
                depth += 1
                # Filter out already visited URLs (protected by lock per asyncio docs)
                async with visited_lock:
                    urls_to_crawl = [
                        url for url in current_urls if normalize_url(url) not in visited
                    ]

                if not urls_to_crawl:
                    logger.info(f"No more URLs to crawl at depth {depth}")
                    break

                # Limit to remaining page budget
                remaining_budget = max_pages - pages_crawled
                urls_to_crawl = urls_to_crawl[:remaining_budget]

                logger.info(
                    f"Depth {depth}: Crawling {len(urls_to_crawl)} URLs "
                    f"({pages_crawled}/{max_pages} pages so far)",
                )

                # Crawl batch
                async with track_memory(
                    f"agentic_crawl(depth={depth}, urls={len(urls_to_crawl)})",
                ) as mem_ctx:
                    with SuppressStdout():
                        results = await crawler.arun_many(
                            urls=urls_to_crawl,
                            config=run_config,
                            dispatcher=dispatcher,
                        )
                    mem_ctx["results"] = results

                next_level_urls = set()

                # Process results
                for result in results:
                    norm_url = normalize_url(result.url)

                    # Memory protection: Check visited set size limit (atomic with lock)
                    async with visited_lock:
                        if len(visited) >= MAX_VISITED_URLS_LIMIT:
                            logger.warning(
                                f"Visited URLs limit reached ({MAX_VISITED_URLS_LIMIT}), "
                                f"stopping recursive crawl to prevent memory exhaustion",
                            )
                            # Return early with current results
                            return {
                                "success": True,
                                "urls_crawled": pages_crawled,
                                "urls_stored": total_urls_stored,
                                "chunks_stored": total_chunks_stored,
                                "urls_filtered": urls_filtered,
                                "warning": f"Stopped early: visited URL limit ({MAX_VISITED_URLS_LIMIT}) reached",
                            }

                        visited.add(norm_url)

                    pages_crawled += 1

                    if result.success and result.markdown:
                        # Store in Qdrant
                        try:
                            chunks = smart_chunk_markdown(result.markdown, chunk_size=2000)
                            if chunks:
                                source_id = extract_domain_from_url(result.url)
                                urls_list = [result.url] * len(chunks)
                                chunk_numbers = list(range(len(chunks)))
                                metadatas = [
                                    {"url": result.url, "chunk": i}
                                    for i in range(len(chunks))
                                ]
                                url_to_full_document = {result.url: result.markdown}
                                source_ids = [source_id] * len(chunks) if source_id else None

                                await add_documents_to_database(
                                    database=database_client,
                                    urls=urls_list,
                                    chunk_numbers=chunk_numbers,
                                    contents=chunks,
                                    metadatas=metadatas,
                                    url_to_full_document=url_to_full_document,
                                    batch_size=20,
                                    source_ids=source_ids,
                                )

                                total_urls_stored += 1
                                total_chunks_stored += len(chunks)
                                logger.debug(
                                    f"Stored {len(chunks)} chunks from {result.url}",
                                )
                        except DatabaseError as e:
                            logger.error(f"Database error storing {result.url}: {e}")
                        except Exception as e:
                            logger.error(f"Failed to store {result.url}: {e}")

                        # Extract internal links for next level
                        if pages_crawled < max_pages:  # Only if budget remains
                            for link in result.links.get("internal", []):
                                next_url = normalize_url(link["href"])
                                # Check visited set with lock protection
                                async with visited_lock:
                                    is_visited = next_url in visited

                                if not is_visited:
                                    # Apply smart filtering
                                    if should_filter_url(next_url, enable_url_filtering):
                                        urls_filtered += 1
                                        continue
                                    next_level_urls.add(next_url)

                # Update URLs for next iteration
                current_urls = next_level_urls

        logger.info(
            f"Recursive crawl completed: {pages_crawled} pages crawled, "
            f"{total_urls_stored} stored, {total_chunks_stored} chunks, "
            f"{urls_filtered} URLs filtered",
        )

        return {
            "success": True,
            "urls_crawled": pages_crawled,
            "urls_stored": total_urls_stored,
            "chunks_stored": total_chunks_stored,
            "urls_filtered": urls_filtered,
        }

    except Exception as e:
        logger.exception(f"Error in crawl_urls_for_agentic_search: {e}")
        return {
            "success": False,
            "error": str(e),
            "urls_crawled": 0,
            "urls_stored": 0,
            "chunks_stored": 0,
            "urls_filtered": 0,
        }
