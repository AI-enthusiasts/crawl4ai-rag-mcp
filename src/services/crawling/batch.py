"""Batch processing and URL filtering for crawling operations.

This module provides utilities for batch crawling multiple URLs in parallel
with efficient memory management and intelligent URL filtering to prevent
infinite crawling or duplicate content.

Key features:
- Batch crawling with memory-adaptive concurrency control
- Smart URL filtering to skip GitHub commits, pagination, archives, etc.
- URL validation and normalization
- Comprehensive logging and error handling
"""

import logging
import re
from typing import Any

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)
from crawl4ai.async_logger import AsyncLoggerBase
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from src.core.constants import URL_FILTER_PATTERNS
from src.core.exceptions import CrawlError
from src.core.logging import logger
from src.services.crawling.memory import track_memory


async def crawl_batch(
    browser_config: BrowserConfig,
    urls: list[str],
    dispatcher: MemoryAdaptiveDispatcher,
    crawl4ai_logger: AsyncLoggerBase | None = None,
) -> list[dict[str, Any]]:
    """Batch crawl multiple URLs in parallel.

    Args:
        browser_config: BrowserConfig for creating crawler instance
        urls: List of URLs to crawl
        dispatcher: Shared MemoryAdaptiveDispatcher for global concurrency control
        crawl4ai_logger: Optional stderr-based logger for MCP stdio compliance

    Returns:
        List of dictionaries with URL and markdown content

    Raises:
        ValueError: If URLs are invalid for crawl4ai
    """
    # Import validation functions
    from src.utils.validation import validate_urls_for_crawling

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
        page_timeout=45000,
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.4,
                threshold_type="fixed",
                min_word_threshold=20,
            ),
        ),
    )

    # Use shared dispatcher from context for global concurrency control
    # This ensures max_session_permit applies across ALL tool calls, not per-call

    try:
        # Create crawler with context manager for automatic cleanup
        # This ensures browser pages are properly closed after crawling
        # Pass custom logger if provided (for MCP stdio compliance)
        crawler_kwargs: dict[str, Any] = {"config": browser_config}
        if crawl4ai_logger is not None:
            crawler_kwargs["logger"] = crawl4ai_logger

        async with AsyncWebCrawler(**crawler_kwargs) as crawler:
            async with track_memory(
                f"crawl_batch({len(validated_urls)} URLs)"
            ) as mem_ctx:
                logger.info(
                    f"Starting arun_many for {len(validated_urls)} URLs "
                    f"(page_timeout={crawl_config.page_timeout}ms)",
                )

                result_container = await crawler.arun_many(
                    urls=validated_urls,
                    config=crawl_config,
                    dispatcher=dispatcher,
                )
                # stream=False in config, so this is List[CrawlResult]
                assert isinstance(result_container, list), "Expected list in batch mode"
                results = result_container

                # Store results for memory tracking
                mem_ctx["results"] = results

                logger.info(
                    f"arun_many completed: {len(results)} results returned",
                )
            # Crawler automatically closed here - all pages cleaned up

        # Log crawling results summary
        successful_results = []
        for r in results:
            if not r.success or not r.markdown:
                continue
            content = (
                r.markdown.fit_markdown
                if r.markdown.fit_markdown
                else r.markdown.raw_markdown
            )
            if content:
                successful_results.append(
                    {
                        "url": r.url,
                        "markdown": content,
                        "links": r.links,
                    }
                )

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

    except CrawlError as e:
        logger.error(f"Crawl4AI error during batch crawl: {e}")
        logger.error(f"Failed URLs: {validated_urls}")
        # Re-raise with more context
        msg = f"Crawling failed for {len(validated_urls)} URLs: {e}"
        raise CrawlError(msg) from e
    except Exception as e:
        logger.error(
            f"Unexpected error during batch crawl: {type(e).__name__}: {e}",
            exc_info=True,
        )
        logger.error(f"Failed URLs: {validated_urls}")
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
        raise CrawlError(msg) from e


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
