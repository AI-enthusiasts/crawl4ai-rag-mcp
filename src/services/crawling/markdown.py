"""Markdown file crawling utilities.

This module provides specialized crawling for .txt and markdown files.
"""

import logging
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_logger import AsyncLoggerBase

logger = logging.getLogger(__name__)


async def crawl_markdown_file(
    browser_config: BrowserConfig,
    url: str,
    crawl4ai_logger: AsyncLoggerBase | None = None,
) -> list[dict[str, Any]]:
    """Crawl a .txt or markdown file.

    Args:
        browser_config: BrowserConfig for creating crawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    crawler_kwargs: dict[str, Any] = {"config": browser_config}
    if crawl4ai_logger is not None:
        crawler_kwargs["logger"] = crawl4ai_logger

    async with AsyncWebCrawler(**crawler_kwargs) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            content = result.markdown.raw_markdown or ""
            if content:
                return [{"url": url, "markdown": content}]
        logger.error("Failed to crawl %s: %s", url, result.error_message)
        return []
