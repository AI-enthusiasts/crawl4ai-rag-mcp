"""
Crawler wrapper with periodic restart to prevent browser process accumulation.

This is a workaround for crawl4ai Issue #943:
https://github.com/unclecode/crawl4ai/issues/943

Per crawl4ai v0.7.3 documentation:
"Restart crawler periodically to clear accumulated processes"
"""

from typing import Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig

from .logging import logger


class CrawlerWrapper:
    """
    Wrapper around AsyncWebCrawler that automatically restarts after N operations.
    
    This prevents browser process accumulation in long-running applications.
    Recommended by crawl4ai documentation for Docker deployments.
    """

    def __init__(
        self,
        config: BrowserConfig,
        max_operations: int = 100,
    ):
        """
        Initialize crawler wrapper.
        
        Args:
            config: Browser configuration
            max_operations: Number of operations before automatic restart (default: 100)
        """
        self.config = config
        self.max_operations = max_operations
        self.operation_count = 0
        self._crawler: Optional[AsyncWebCrawler] = None

    async def initialize(self) -> None:
        """Initialize the crawler."""
        await self._create_crawler()

    async def _create_crawler(self) -> None:
        """Create a new crawler instance."""
        # Close old crawler if exists
        if self._crawler:
            try:
                await self._crawler.close()
                logger.info("✓ Old crawler closed for restart")
            except Exception as e:
                logger.warning(f"Error closing old crawler: {e}")

        # Create new crawler
        self._crawler = AsyncWebCrawler(config=self.config)
        await self._crawler.start()
        self.operation_count = 0
        logger.info(
            f"✓ Crawler initialized (will restart after {self.max_operations} operations)"
        )

    async def get_crawler(self) -> AsyncWebCrawler:
        """
        Get crawler instance, restarting if needed.
        
        Returns:
            AsyncWebCrawler instance
        """
        # Check if restart needed
        if self.operation_count >= self.max_operations:
            logger.info(
                f"Restarting crawler after {self.operation_count} operations "
                f"(workaround for crawl4ai Issue #943)"
            )
            await self._create_crawler()

        if not self._crawler:
            await self._create_crawler()

        return self._crawler

    def increment_operation_count(self) -> None:
        """Increment operation counter."""
        self.operation_count += 1

    async def close(self) -> None:
        """Close the crawler."""
        if self._crawler:
            await self._crawler.close()
            logger.info("✓ Crawler closed")
            self._crawler = None
