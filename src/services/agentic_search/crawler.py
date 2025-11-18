"""Selective crawling for agentic search (Stage 3).

This module handles:
- Duplicate URL detection using Qdrant
- Recursive crawling with smart limits
- URL filtering and storage in vector database
"""

import logging

from fastmcp import Context

from src.core.context import get_app_context
from src.core.exceptions import DatabaseError
from src.services.agentic_models import ActionType, SearchIteration
from src.services.crawling import crawl_urls_for_agentic_search

from .config import AgenticSearchConfig

logger = logging.getLogger(__name__)


class SelectiveCrawler:
    """Performs selective crawling with deduplication and smart limits."""

    def __init__(self, config: AgenticSearchConfig) -> None:
        """Initialize crawler with shared configuration.

        Args:
            config: Shared agentic search configuration
        """
        self.config = config

    async def crawl_and_store(
        self,
        ctx: Context,
        urls: list[str],
        query: str,
        use_hints: bool,
        iteration: int,
        search_history: list[SearchIteration],
    ) -> int:
        """STAGE 3: Crawl promising URLs recursively with smart filtering and limits.

        Args:
            ctx: FastMCP context
            urls: URLs to crawl (starting points)
            query: Original query
            use_hints: Whether to use search hints
            iteration: Current iteration number
            search_history: History to append to

        Returns:
            Number of URLs successfully stored in Qdrant
        """
        logger.info("STAGE 3: Recursively crawling %d promising URLs", len(urls))

        # HIGH PRIORITY FIX #10: Duplicate detection - filter out already crawled URLs
        # Uses Qdrant count() for efficient existence check (per Qdrant docs)
        app_ctx = get_app_context()
        if not app_ctx:
            msg = "Application context not available"
            raise DatabaseError(msg)
        database_client = app_ctx.database_client
        urls_to_crawl = []
        urls_skipped = 0

        for url in urls:
            # Check if URL already exists in database (efficient existence check)
            try:
                exists = await database_client.url_exists(url)
                if exists:
                    logger.info("Skipping duplicate URL (already in database): %s", url)
                    urls_skipped += 1
                else:
                    urls_to_crawl.append(url)
            except DatabaseError as e:
                # On database error, include URL (fail open)
                logger.warning("Database error checking duplicate for %s: %s", url, e)
                urls_to_crawl.append(url)
            except Exception as e:
                # On unexpected error, include URL (fail open)
                logger.warning("Unexpected error checking duplicate for %s: %s", url, e)
                urls_to_crawl.append(url)

        if urls_skipped > 0:
            logger.info(
                "Filtered %d/%d duplicate URLs, crawling %d new URLs",
                urls_skipped,
                len(urls),
                len(urls_to_crawl),
            )

        if not urls_to_crawl:
            logger.info("All URLs already in database, skipping crawl")
            return 0

        # Crawl recursively with smart limits and filtering
        crawl_result = await crawl_urls_for_agentic_search(
            ctx=ctx,
            urls=urls_to_crawl,  # Use filtered URLs (duplicates removed)
            max_pages=self.config.max_pages_per_iteration,
            max_depth=self.config.max_crawl_depth,
            enable_url_filtering=self.config.enable_url_filtering,
        )

        # Extract results (cast to int for type safety)
        urls_crawled: int = crawl_result.get("urls_crawled", 0)
        urls_stored: int = crawl_result.get("urls_stored", 0)
        chunks_stored: int = crawl_result.get("chunks_stored", 0)
        urls_filtered: int = crawl_result.get("urls_filtered", 0)

        logger.info(
            "Crawled %d pages, stored %d URLs, %d chunks, filtered %d URLs",
            urls_crawled,
            urls_stored,
            chunks_stored,
            urls_filtered,
        )

        search_history.append(
            SearchIteration(
                iteration=iteration,
                query=query,
                action=ActionType.CRAWL,
                urls=urls,
                urls_stored=urls_stored,
                chunks_stored=chunks_stored,
            ),
        )

        # Note: Search hints feature requires Crawl4AI metadata capabilities
        # Currently not implemented - would generate optimized Qdrant queries from metadata
        if use_hints:
            logger.info("Search hints requested but not yet implemented")

        return urls_stored
