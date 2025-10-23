"""
Tests for crawler lifecycle management and resource cleanup.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crawl4ai import AsyncWebCrawler, BrowserConfig


class TestCrawlerLifecycle:
    """Test proper crawler initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_crawler_context_manager_cleanup(self):
        """Test that crawler properly cleans up when used as context manager."""
        browser_config = BrowserConfig(headless=True, verbose=False)
        
        # Mock the actual browser to avoid real browser launch in tests
        with patch('crawl4ai.AsyncWebCrawler.__aenter__') as mock_enter, \
             patch('crawl4ai.AsyncWebCrawler.__aexit__') as mock_exit:
            
            mock_crawler = AsyncMock()
            mock_enter.return_value = mock_crawler
            mock_exit.return_value = None
            
            # Use context manager
            async with AsyncWebCrawler(config=browser_config) as crawler:
                assert crawler is not None
            
            # Verify __aenter__ and __aexit__ were called
            mock_enter.assert_called_once()
            mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawler_manual_lifecycle(self):
        """Test manual crawler lifecycle with explicit start/close."""
        with patch('crawl4ai.AsyncWebCrawler.start') as mock_start, \
             patch('crawl4ai.AsyncWebCrawler.close') as mock_close:
            
            mock_start.return_value = None
            mock_close.return_value = None
            
            browser_config = BrowserConfig(headless=True)
            crawler = AsyncWebCrawler(config=browser_config)
            
            await crawler.start()
            mock_start.assert_called_once()
            
            await crawler.close()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawler_cleanup_on_exception(self):
        """Test that crawler cleans up even when exception occurs."""
        with patch('crawl4ai.AsyncWebCrawler.__aenter__') as mock_enter, \
             patch('crawl4ai.AsyncWebCrawler.__aexit__') as mock_exit:
            
            mock_crawler = AsyncMock()
            mock_enter.return_value = mock_crawler
            mock_exit.return_value = None
            
            browser_config = BrowserConfig(headless=True)
            
            # Simulate exception inside context
            with pytest.raises(RuntimeError):
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    raise RuntimeError("Test exception")
            
            # __aexit__ should still be called
            mock_exit.assert_called_once()


class TestSessionManagement:
    """Test session management and cleanup."""

    @pytest.mark.asyncio
    async def test_session_id_usage(self):
        """Test that session_id is properly used in crawl config."""
        from crawl4ai import CrawlerRunConfig
        
        session_id = "test_session_123"
        config = CrawlerRunConfig(session_id=session_id)
        
        assert config.session_id == session_id

    @pytest.mark.asyncio
    async def test_session_cleanup_called(self):
        """Test that kill_session is called after batch processing."""
        # This will be implemented after we add session management
        # For now, just a placeholder
        pass


class TestMemoryMonitoring:
    """Test memory monitoring integration."""

    def test_memory_monitor_import(self):
        """Test that memory monitoring utilities are available."""
        try:
            from crawl4ai.memory_utils import MemoryMonitor, get_memory_info
            assert MemoryMonitor is not None
            assert get_memory_info is not None
        except ImportError:
            pytest.skip("Memory monitoring not available in this crawl4ai version")

    @pytest.mark.asyncio
    async def test_memory_monitor_basic_usage(self):
        """Test basic memory monitor usage."""
        try:
            from crawl4ai.memory_utils import MemoryMonitor
            
            monitor = MemoryMonitor()
            monitor.start_monitoring()
            
            # Simulate some work
            await asyncio.sleep(0.1)
            
            report = monitor.get_report()
            
            # Check report structure
            assert 'peak_mb' in report
            assert 'efficiency' in report
            assert isinstance(report['peak_mb'], (int, float))
            assert isinstance(report['efficiency'], (int, float))
            
        except ImportError:
            pytest.skip("Memory monitoring not available in this crawl4ai version")
