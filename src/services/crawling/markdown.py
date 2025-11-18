"""Markdown file crawling utilities.

This module provides specialized crawling for .txt and markdown files.
"""

import logging
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from src.core.stdout_utils import SuppressStdout

logger = logging.getLogger(__name__)


async def crawl_markdown_file(
    browser_config: BrowserConfig,
    url: str,
) -> list[dict[str, Any]]:
    """Crawl a .txt or markdown file.

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
        logger.error("Failed to crawl %s: %s", url, result.error_message)
        return []
