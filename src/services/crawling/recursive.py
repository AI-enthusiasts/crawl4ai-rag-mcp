"""Recursive link crawling utilities.

This module provides recursive internal link crawling functionality
for discovering and crawling related pages within a website.
"""

import logging
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, MemoryAdaptiveDispatcher

from src.core.stdout_utils import SuppressStdout
from src.utils.url_helpers import normalize_url

from .memory import track_memory

logger = logging.getLogger(__name__)


async def crawl_recursive_internal_links(
    browser_config: BrowserConfig,
    start_urls: list[str],
    dispatcher: MemoryAdaptiveDispatcher,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Recursively crawl internal links from start URLs up to a maximum depth.

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
