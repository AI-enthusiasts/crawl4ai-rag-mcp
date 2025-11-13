"""Crawling services for the Crawl4AI MCP server."""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)
from crawl4ai.utils import get_memory_stats
from fastmcp import Context

from core.logging import logger
from core.stdout_utils import SuppressStdout

# Import add_documents_to_database from utils package
from utils import add_documents_to_database
from utils.text_processing import smart_chunk_markdown
from utils.url_helpers import extract_domain_from_url, normalize_url


@asynccontextmanager
async def track_memory(operation_name: str):
    """
    Context manager to track memory usage before and after an operation.

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        dict: Dictionary to store results for memory analysis
    """
    start_memory_percent, start_available_gb, total_gb = get_memory_stats()
    logger.info(
        f"[{operation_name}] Memory before: {start_memory_percent:.1f}% used, "
        f"{start_available_gb:.2f}/{total_gb:.2f} GB available",
    )

    # Yield a dict to collect results
    context = {"results": None}

    try:
        yield context
    finally:
        end_memory_percent, end_available_gb, _ = get_memory_stats()
        memory_delta = end_memory_percent - start_memory_percent

        logger.info(
            f"[{operation_name}] Memory after: {end_memory_percent:.1f}% used "
            f"(Δ {memory_delta:+.1f}%), {end_available_gb:.2f} GB available",
        )

        # Log dispatch stats if results are available
        if context["results"]:
            dispatch_stats = []
            for r in context["results"]:
                if hasattr(r, "dispatch_result") and r.dispatch_result:
                    dispatch_stats.append(
                        {
                            "memory_usage": r.dispatch_result.memory_usage,
                            "peak_memory": r.dispatch_result.peak_memory,
                        },
                    )

            if dispatch_stats:
                avg_memory = sum(s["memory_usage"] for s in dispatch_stats) / len(
                    dispatch_stats,
                )
                peak_memory = max(s["peak_memory"] for s in dispatch_stats)
                logger.info(
                    f"[{operation_name}] Dispatch stats: "
                    f"avg {avg_memory:.1f} MB, peak {peak_memory:.1f} MB",
                )


async def crawl_markdown_file(
    browser_config: BrowserConfig,
    url: str,
) -> list[dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        browser_config: BrowserConfig for creating crawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    # Create crawler with context manager for automatic cleanup
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Run in executor to avoid blocking event loop
        with SuppressStdout():
            result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{"url": url, "markdown": result.markdown}]
        logger.error(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_batch(
    browser_config: BrowserConfig,
    urls: list[str],
    dispatcher: MemoryAdaptiveDispatcher,
) -> list[dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        dispatcher: Shared MemoryAdaptiveDispatcher for global concurrency control

    Returns:
        List of dictionaries with URL and markdown content

    Raises:
        ValueError: If URLs are invalid for crawl4ai
    """
    # Import validation functions
    from utils.validation import validate_urls_for_crawling

    # Enhanced debug logging - only log details in debug mode to avoid exposing sensitive data
    logger.info(f"crawl_batch received {len(urls)} URLs for processing")

    # Only log sensitive URL details in debug mode
    if logger.isEnabledFor(logging.DEBUG):
        # Don't log full URLs directly as they may contain auth tokens
        logger.debug(f"URL types: {[type(url).__name__ for url in urls]}")

    # Log details about each URL before validation
    for i, url in enumerate(urls):
        logger.debug(
            f"URL {i + 1}/{len(urls)}: {url!r} (type: {type(url).__name__}, length: {len(str(url))})",
        )
        if isinstance(url, str):
            logger.debug(f"  - Stripped: {url.strip()!r}")
            logger.debug(f"  - Contains whitespace: {url != url.strip()}")
            logger.debug(
                f"  - Starts with http: {url.strip().startswith(('http://', 'https://'))}",
            )

    # Validate URLs before passing to crawl4ai
    logger.debug("Starting URL validation...")
    validation_result = validate_urls_for_crawling(urls)
    logger.debug(f"URL validation completed. Result: {validation_result}")

    if not validation_result["valid"]:
        error_msg = f"URL validation failed: {validation_result['error']}"
        logger.error(error_msg)

        # Provide comprehensive debugging context
        if validation_result.get("invalid_urls"):
            logger.error(
                f"Invalid URLs that were rejected: {validation_result['invalid_urls']}",
            )
            # Log details about each invalid URL
            for invalid_url in validation_result["invalid_urls"]:
                logger.error(f"  - Invalid URL: {invalid_url!r}")

        if validation_result.get("valid_urls"):
            logger.info(
                f"Valid URLs that passed validation: {validation_result['valid_urls']}",
            )

        # Log the validation result structure for debugging
        logger.debug(f"Full validation result: {validation_result}")

        msg = f"Invalid URLs for crawling: {validation_result['error']}"
        raise ValueError(msg)

    # Use validated and normalized URLs
    validated_urls = validation_result["urls"]
    logger.info(
        f"URL validation successful! {len(validated_urls)} URLs ready for crawling",
    )

    # Log info about any URLs that were auto-fixed during validation
    if len(validated_urls) != len(urls):
        logger.info(
            f"URL count changed during validation: {len(urls)} -> {len(validated_urls)}",
        )
        logger.info(f"Original URLs: {urls}")
        logger.info(f"Validated URLs: {validated_urls}")

        # Log individual transformations
        for i, (orig, valid) in enumerate(zip(urls, validated_urls, strict=False)):
            if orig != valid:
                logger.info(f"URL {i + 1} transformed: {orig!r} -> {valid!r}")
    else:
        logger.debug("No URL transformations were needed during validation")

    logger.info(
        f"Starting crawl of {len(validated_urls)} validated URLs",
    )
    logger.debug(f"Final URLs for crawling: {validated_urls}")

    # Initialize crawler configuration with timeout to prevent 504 errors
    # Set page timeout to 45s to ensure pages don't hang indefinitely
    # NOTE: We intentionally do NOT use session_id here
    # Per crawl4ai docs: "When no session_id is provided, pages are automatically closed"
    # This prevents browser page leaks in batch operations
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        page_timeout=45000,  # 45 seconds in milliseconds
        # session_id=None - explicitly no session for automatic page cleanup
    )

    # Use shared dispatcher from context for global concurrency control
    # This ensures max_session_permit applies across ALL tool calls, not per-call

    try:
        # Create crawler with context manager for automatic cleanup
        # This ensures browser pages are properly closed after crawling
        async with AsyncWebCrawler(config=browser_config) as crawler:
            async with track_memory(f"crawl_batch({len(validated_urls)} URLs)") as mem_ctx:
                logger.info(
                    f"Starting arun_many for {len(validated_urls)} URLs "
                    f"(page_timeout={crawl_config.page_timeout}ms)",
                )

                # Crawler will automatically close all pages when exiting context manager
                with SuppressStdout():
                    results = await crawler.arun_many(
                        urls=validated_urls,
                        config=crawl_config,
                        dispatcher=dispatcher,
                    )

                # Store results for memory tracking
                mem_ctx["results"] = results

                logger.info(
                    f"arun_many completed: {len(results)} results returned",
                )
            # Crawler automatically closed here - all pages cleaned up

        # Log crawling results summary
        successful_results = [
            {"url": r.url, "markdown": r.markdown, "links": r.links}
            for r in results
            if r.success and r.markdown
        ]

        failed_results = [r for r in results if not r.success or not r.markdown]

        logger.info(
            f"Crawling complete: {len(successful_results)} successful, "
            f"{len(failed_results)} failed, "
            f"total_processed={len(results)}",
        )

        if successful_results:
            logger.debug(f"Successful URLs: {[r['url'] for r in successful_results]}")

        if failed_results:
            logger.warning("Failed URLs and reasons:")
            for failed_result in failed_results:
                logger.warning(
                    f"  - {failed_result.url}: success={failed_result.success}, "
                    f"has_markdown={bool(failed_result.markdown)}",
                )

        return successful_results

    except Exception as e:
        logger.error(
            f"Crawl4AI error during batch crawl: {type(e).__name__}: {e}",
            exc_info=True,
        )
        logger.error(
            f"Failed URLs: {validated_urls}",
        )
        logger.error(
            f"Crawler config: cache_mode={crawl_config.cache_mode}, "
            f"stream={crawl_config.stream}, "
            f"page_timeout={crawl_config.page_timeout}ms, "
            f"session_id={crawl_config.session_id}",
        )
        logger.error(
            f"Dispatcher config: memory_threshold={dispatcher.memory_threshold_percent}%, "
            f"max_sessions={dispatcher.max_session_permit}, "
            f"check_interval={dispatcher.check_interval}s",
        )

        # Re-raise with more context
        msg = f"Crawling failed for {len(validated_urls)} URLs: {e}"
        raise ValueError(msg) from e


async def crawl_recursive_internal_links(
    browser_config: BrowserConfig,
    start_urls: list[str],
    dispatcher: MemoryAdaptiveDispatcher,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        browser_config: BrowserConfig for creating crawler instance
        start_urls: List of starting URLs
        dispatcher: Shared MemoryAdaptiveDispatcher for global concurrency control
        max_depth: Maximum recursion depth

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    # Use shared dispatcher from context for global concurrency control

    visited = set()
    current_urls = {normalize_url(u) for u in start_urls}
    results_all = []

    # Create crawler with context manager for automatic cleanup
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            urls_to_crawl = [
                normalize_url(url)
                for url in current_urls
                if normalize_url(url) not in visited
            ]
            if not urls_to_crawl:
                break

            async with track_memory(
                f"recursive_crawl(depth={depth}, urls={len(urls_to_crawl)})",
            ) as mem_ctx:
                # Run in executor to avoid blocking event loop
                with SuppressStdout():
                    results = await crawler.arun_many(
                        urls=urls_to_crawl,
                        config=run_config,
                        dispatcher=dispatcher,
                    )
                mem_ctx["results"] = results

            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({"url": result.url, "markdown": result.markdown})
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)

            current_urls = next_level_urls

    return results_all


async def process_urls_for_mcp(
    ctx: Context,
    urls: list[str],
    batch_size: int = 20,
    return_raw_markdown: bool = False,
) -> str:
    """
    Process URLs for MCP tools with context extraction and database storage.

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
            from core.context import get_app_context

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
                urls = [result["url"]] * len(chunks)
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
                    urls=urls,
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
