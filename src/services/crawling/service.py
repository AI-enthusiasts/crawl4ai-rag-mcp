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
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from fastmcp import Context

from src.core.constants import MAX_VISITED_URLS_LIMIT, URL_FILTER_PATTERNS
from src.core.context import get_app_context
from src.core.exceptions import DatabaseError
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
    *,
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
                    "error": (
                        "Invalid Crawl4AI context: missing required attributes "
                        "(browser_config, database_client, dispatcher)"
                    ),
                },
            )

        if not (
            crawl4ai_ctx.browser_config
            and crawl4ai_ctx.database_client
            and crawl4ai_ctx.dispatcher
        ):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Invalid Crawl4AI context: browser_config, "
                        "database_client, or dispatcher is None"
                    ),
                },
            )

        crawl_results = await crawl_batch(
            browser_config=crawl4ai_ctx.browser_config,
            urls=urls,
            dispatcher=crawl4ai_ctx.dispatcher,
            crawl4ai_logger=getattr(crawl4ai_ctx, "crawl4ai_logger", None),
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
                chunks = smart_chunk_markdown(result["markdown"], chunk_size=4000)

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
                logger.exception("Database error storing %s: %s", result["url"], e)
                stored_results.append(
                    {
                        "url": result["url"],
                        "success": False,
                        "error": str(e),
                        "chunks_stored": 0,
                    },
                )
            except Exception as e:
                logger.exception("Failed to store %s: %s", result["url"], e)
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
        logger.exception("Error in process_urls_for_mcp: %s", e)
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
            logger.debug("Filtering URL (matched pattern %s): %s", pattern, url)
            return True

    return False


async def crawl_urls_for_agentic_search(
    ctx: Context,
    urls: list[str],
    max_pages: int = 15,
    max_depth: int = 2,
    enable_url_filtering: bool = True,
) -> dict[str, Any]:
    """Crawl URLs recursively for agentic search with smart limits and filtering.

    This function is specifically designed for agentic search and provides:
    1. Recursive crawling of internal links (limited by max_depth)
    2. Smart URL filtering to avoid GitHub commits, pagination, etc.
    3. Page limit per iteration (prevents excessive crawling)
    4. Visited URL tracking (prevents re-crawling same pages)
    5. Full Qdrant indexing of all crawled pages

    Args:
        ctx: FastMCP context containing Crawl4AIContext
        urls: Starting URLs to crawl (typically 5 from agentic search)
        max_pages: Maximum pages to crawl across all URLs (default: 15)
        max_depth: Maximum crawl depth (1=only starting URLs, 2=+1 level, default: 2)
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

        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            # Performance optimizations for agentic search
            wait_until="domcontentloaded",  # Don't wait for all resources
            exclude_all_images=True,  # Skip image loading entirely
            process_iframes=False,  # Skip iframe processing
            excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.4,
                    threshold_type="fixed",
                    min_word_threshold=20,
                ),
            ),
        )

        logger.info(
            "Starting recursive crawl: %s URLs, max_pages=%s, max_depth=%s, filtering=%s",
            len(current_urls),
            max_pages,
            max_depth,
            "enabled" if enable_url_filtering else "disabled",
        )

        crawl4ai_logger = getattr(app_ctx, "crawl4ai_logger", None)
        crawler_kwargs: dict[str, Any] = {"config": browser_config}
        if crawl4ai_logger is not None:
            crawler_kwargs["logger"] = crawl4ai_logger

        async with AsyncWebCrawler(**crawler_kwargs) as crawler:
            depth = 0
            while current_urls and pages_crawled < max_pages and depth < max_depth:
                depth += 1
                # Filter out already visited URLs (protected by lock per asyncio docs)
                async with visited_lock:
                    urls_to_crawl = [
                        url for url in current_urls if normalize_url(url) not in visited
                    ]

                if not urls_to_crawl:
                    logger.info("No more URLs to crawl at depth %s", depth)
                    break

                # Limit to remaining page budget
                remaining_budget = max_pages - pages_crawled
                urls_to_crawl = urls_to_crawl[:remaining_budget]

                logger.info(
                    "Depth %s: Crawling %s URLs (%s/%s pages so far)",
                    depth,
                    len(urls_to_crawl),
                    pages_crawled,
                    max_pages,
                )

                # Crawl batch
                async with track_memory(
                    f"agentic_crawl(depth={depth}, urls={len(urls_to_crawl)})",
                ) as mem_ctx:
                    result_container = await crawler.arun_many(
                        urls=urls_to_crawl,
                        config=run_config,
                        dispatcher=dispatcher,
                    )
                    assert isinstance(result_container, list), (
                        "Expected list in batch mode"
                    )
                    results = result_container
                    mem_ctx["results"] = results

                next_level_urls = set()

                # Process results
                for result in results:
                    norm_url = normalize_url(result.url)

                    # Memory protection: Check visited set size limit (atomic with lock)
                    async with visited_lock:
                        if len(visited) >= MAX_VISITED_URLS_LIMIT:
                            logger.warning(
                                "Visited URLs limit reached (%s), stopping recursive crawl to prevent memory exhaustion",
                                MAX_VISITED_URLS_LIMIT,
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
                        # Use fit_markdown (filtered) if available, fallback to raw
                        content = (
                            result.markdown.fit_markdown
                            if result.markdown.fit_markdown
                            else result.markdown.raw_markdown
                        )
                        if not content:
                            continue

                        # Store in Qdrant
                        try:
                            chunks = smart_chunk_markdown(content, chunk_size=4000)
                            if chunks:
                                source_id = extract_domain_from_url(result.url)
                                urls_list = [result.url] * len(chunks)
                                chunk_numbers = list(range(len(chunks)))
                                metadatas = [
                                    {"url": result.url, "chunk": i}
                                    for i in range(len(chunks))
                                ]
                                url_to_full_document = {result.url: content}
                                source_ids = (
                                    [source_id] * len(chunks) if source_id else None
                                )

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
                                    "Stored %s chunks from %s",
                                    len(chunks),
                                    result.url,
                                )
                        except DatabaseError as e:
                            logger.exception(
                                "Database error storing %s: %s", result.url, e
                            )
                        except Exception as e:
                            logger.exception("Failed to store %s: %s", result.url, e)

                        # Extract internal links for next level
                        if pages_crawled < max_pages:  # Only if budget remains
                            for link in result.links.get("internal", []):
                                next_url = normalize_url(link["href"])
                                # Check visited set with lock protection
                                async with visited_lock:
                                    is_visited = next_url in visited

                                if not is_visited:
                                    # Apply smart filtering
                                    if should_filter_url(
                                        next_url, enable_url_filtering
                                    ):
                                        urls_filtered += 1
                                        continue
                                    next_level_urls.add(next_url)

                # Update URLs for next iteration
                current_urls = next_level_urls

        logger.info(
            "Recursive crawl completed: %s pages crawled, %s stored, %s chunks, %s URLs filtered",
            pages_crawled,
            total_urls_stored,
            total_chunks_stored,
            urls_filtered,
        )

        return {
            "success": True,
            "urls_crawled": pages_crawled,
            "urls_stored": total_urls_stored,
            "chunks_stored": total_chunks_stored,
            "urls_filtered": urls_filtered,
        }

    except Exception as e:
        logger.exception("Error in crawl_urls_for_agentic_search: %s", e)
        return {
            "success": False,
            "error": str(e),
            "urls_crawled": 0,
            "urls_stored": 0,
            "chunks_stored": 0,
            "urls_filtered": 0,
        }
