"""Tests for CrawlerWrapper with periodic restart."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCrawlerWrapper:
    """Test crawler wrapper with automatic restart."""

    @pytest.mark.asyncio
    async def test_wrapper_initialization(self):
        """Test that wrapper initializes correctly."""
        from crawl4ai import BrowserConfig
        from core.crawler_wrapper import CrawlerWrapper
        
        config = BrowserConfig(headless=True)
        wrapper = CrawlerWrapper(config=config, max_operations=10)
        
        assert wrapper.config == config
        assert wrapper.max_operations == 10
        assert wrapper.operation_count == 0
        assert wrapper._crawler is None

    @pytest.mark.asyncio
    async def test_wrapper_creates_crawler(self):
        """Test that wrapper creates crawler on first get."""
        from crawl4ai import BrowserConfig
        from core.crawler_wrapper import CrawlerWrapper
        
        with patch('core.crawler_wrapper.AsyncWebCrawler') as MockCrawler:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            MockCrawler.return_value = mock_instance
            
            config = BrowserConfig(headless=True)
            wrapper = CrawlerWrapper(config=config, max_operations=10)
            
            crawler = await wrapper.get_crawler()
            
            assert crawler is not None
            mock_instance.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_wrapper_restarts_after_max_operations(self):
        """Test that wrapper restarts crawler after max operations."""
        from crawl4ai import BrowserConfig
        from core.crawler_wrapper import CrawlerWrapper
        
        with patch('core.crawler_wrapper.AsyncWebCrawler') as MockCrawler:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.close = AsyncMock()
            MockCrawler.return_value = mock_instance
            
            config = BrowserConfig(headless=True)
            wrapper = CrawlerWrapper(config=config, max_operations=3)
            
            # First get - creates crawler
            await wrapper.get_crawler()
            assert wrapper.operation_count == 0
            
            # Simulate 3 operations
            wrapper.increment_operation_count()
            wrapper.increment_operation_count()
            wrapper.increment_operation_count()
            assert wrapper.operation_count == 3
            
            # Next get should restart
            await wrapper.get_crawler()
            
            # Should have closed old and started new
            mock_instance.close.assert_called_once()
            assert mock_instance.start.call_count == 2  # Initial + restart
            assert wrapper.operation_count == 0  # Reset after restart

    @pytest.mark.asyncio
    async def test_wrapper_close(self):
        """Test that wrapper properly closes crawler."""
        from crawl4ai import BrowserConfig
        from core.crawler_wrapper import CrawlerWrapper
        
        with patch('core.crawler_wrapper.AsyncWebCrawler') as MockCrawler:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.close = AsyncMock()
            MockCrawler.return_value = mock_instance
            
            config = BrowserConfig(headless=True)
            wrapper = CrawlerWrapper(config=config)
            
            await wrapper.get_crawler()
            await wrapper.close()
            
            mock_instance.close.assert_called_once()
            assert wrapper._crawler is None
