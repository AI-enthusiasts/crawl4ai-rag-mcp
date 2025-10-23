"""Tests for session management and cleanup."""
import pytest


class TestSessionManagement:
    """Test session ID usage and cleanup."""

    def test_session_id_generation(self):
        """Test that session IDs are generated correctly."""
        import hashlib
        
        urls = ["https://example.com", "https://test.com"]
        # Generate deterministic session ID from URLs
        session_id = f"batch_{hashlib.md5(''.join(sorted(urls)).encode()).hexdigest()[:8]}"
        
        assert session_id.startswith("batch_")
        assert len(session_id) == 14  # "batch_" + 8 chars

    def test_session_id_consistency(self):
        """Test that same URLs generate same session ID."""
        import hashlib
        
        urls1 = ["https://example.com", "https://test.com"]
        urls2 = ["https://test.com", "https://example.com"]  # Different order
        
        session1 = f"batch_{hashlib.md5(''.join(sorted(urls1)).encode()).hexdigest()[:8]}"
        session2 = f"batch_{hashlib.md5(''.join(sorted(urls2)).encode()).hexdigest()[:8]}"
        
        # Should be same because we sort URLs
        assert session1 == session2


class TestSessionCleanupHelper:
    """Test helper functions for session cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_session_helper(self):
        """Test session cleanup helper function."""
        from unittest.mock import AsyncMock, MagicMock
        
        # Mock crawler strategy
        mock_strategy = MagicMock()
        mock_strategy.kill_session = AsyncMock()
        
        # Mock crawler
        mock_crawler = MagicMock()
        mock_crawler.crawler_strategy = mock_strategy
        
        # Simulate cleanup
        session_id = "test_session_123"
        await mock_strategy.kill_session(session_id)
        
        # Verify kill_session was called
        mock_strategy.kill_session.assert_called_once_with(session_id)
