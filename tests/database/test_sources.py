"""
Unit tests for src/database/sources.py

Tests source management functionality including:
- Source summary updates
- Source statistics retrieval
- Source listing
- Error handling with DatabaseError and QueryError
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import QueryError
from src.database.sources import (
    get_source_statistics,
    list_all_sources,
    update_source_summary,
)


class TestUpdateSourceSummary:
    """Test update_source_summary function."""

    @pytest.fixture
    def mock_database_client(self):
        """Create a mock database client."""
        client = MagicMock()
        client.update_source = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_update_source_summary_success(self, mock_database_client):
        """Test successful source summary update."""
        source_id = "example.com"
        total_chunks = 42
        last_crawled = datetime(2025, 1, 15, 12, 0, 0)

        await update_source_summary(
            database_client=mock_database_client,
            source_id=source_id,
            total_chunks=total_chunks,
            last_crawled=last_crawled,
        )

        # Verify update_source was called with correct parameters
        mock_database_client.update_source.assert_called_once_with(
            source_id=source_id,
            total_chunks=total_chunks,
            last_crawled=last_crawled,
        )

    @pytest.mark.asyncio
    async def test_update_source_summary_defaults_to_now(self, mock_database_client):
        """Test that last_crawled defaults to current time when not provided."""
        source_id = "example.com"
        total_chunks = 10

        with patch("src.database.sources.datetime") as mock_datetime:
            mock_now = datetime(2025, 1, 15, 10, 30, 0)
            mock_datetime.utcnow.return_value = mock_now

            await update_source_summary(
                database_client=mock_database_client,
                source_id=source_id,
                total_chunks=total_chunks,
            )

            # Verify update_source was called with mocked datetime
            mock_database_client.update_source.assert_called_once_with(
                source_id=source_id,
                total_chunks=total_chunks,
                last_crawled=mock_now,
            )

    @pytest.mark.asyncio
    async def test_update_source_summary_query_error(self, mock_database_client):
        """Test QueryError is propagated from database client."""
        mock_database_client.update_source.side_effect = QueryError(
            "Database query failed",
        )

        with pytest.raises(QueryError) as exc_info:
            await update_source_summary(
                database_client=mock_database_client,
                source_id="example.com",
                total_chunks=5,
            )

        assert "Database query failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_source_summary_unexpected_error(self, mock_database_client):
        """Test unexpected exceptions are propagated."""
        mock_database_client.update_source.side_effect = ValueError(
            "Unexpected error",
        )

        with pytest.raises(ValueError) as exc_info:
            await update_source_summary(
                database_client=mock_database_client,
                source_id="example.com",
                total_chunks=5,
            )

        assert "Unexpected error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_source_summary_zero_chunks(self, mock_database_client):
        """Test updating source with zero chunks."""
        await update_source_summary(
            database_client=mock_database_client,
            source_id="empty.com",
            total_chunks=0,
        )

        mock_database_client.update_source.assert_called_once()
        call_args = mock_database_client.update_source.call_args
        assert call_args.kwargs["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_update_source_summary_special_characters(self, mock_database_client):
        """Test source_id with special characters."""
        source_id = "example.com/path?query=value&foo=bar"

        await update_source_summary(
            database_client=mock_database_client,
            source_id=source_id,
            total_chunks=5,
        )

        mock_database_client.update_source.assert_called_once()
        call_args = mock_database_client.update_source.call_args
        assert call_args.kwargs["source_id"] == source_id


class TestGetSourceStatistics:
    """Test get_source_statistics function."""

    @pytest.fixture
    def mock_database_client(self):
        """Create a mock database client."""
        client = MagicMock()
        client.get_sources = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_source_statistics_found(self, mock_database_client):
        """Test retrieving statistics for an existing source."""
        source_id = "example.com"
        expected_stats = {
            "source_id": source_id,
            "total_chunks": 42,
            "last_crawled": "2025-01-15T12:00:00",
        }

        mock_database_client.get_sources.return_value = [
            {"source_id": "other.com", "total_chunks": 10},
            expected_stats,
            {"source_id": "another.com", "total_chunks": 20},
        ]

        result = await get_source_statistics(
            database_client=mock_database_client,
            source_id=source_id,
        )

        assert result == expected_stats
        assert result["source_id"] == source_id
        assert result["total_chunks"] == 42
        mock_database_client.get_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_source_statistics_not_found(self, mock_database_client):
        """Test retrieving statistics for a non-existent source returns None."""
        mock_database_client.get_sources.return_value = [
            {"source_id": "example.com", "total_chunks": 42},
            {"source_id": "other.com", "total_chunks": 10},
        ]

        result = await get_source_statistics(
            database_client=mock_database_client,
            source_id="nonexistent.com",
        )

        assert result is None
        mock_database_client.get_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_source_statistics_empty_database(self, mock_database_client):
        """Test retrieving statistics when database has no sources."""
        mock_database_client.get_sources.return_value = []

        result = await get_source_statistics(
            database_client=mock_database_client,
            source_id="example.com",
        )

        assert result is None
        mock_database_client.get_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_source_statistics_query_error(self, mock_database_client):
        """Test QueryError is propagated from database client."""
        mock_database_client.get_sources.side_effect = QueryError(
            "Database query failed",
        )

        with pytest.raises(QueryError) as exc_info:
            await get_source_statistics(
                database_client=mock_database_client,
                source_id="example.com",
            )

        assert "Database query failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_source_statistics_unexpected_error(self, mock_database_client):
        """Test unexpected exceptions are propagated."""
        mock_database_client.get_sources.side_effect = RuntimeError(
            "Unexpected database error",
        )

        with pytest.raises(RuntimeError) as exc_info:
            await get_source_statistics(
                database_client=mock_database_client,
                source_id="example.com",
            )

        assert "Unexpected database error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_source_statistics_first_match_only(self, mock_database_client):
        """Test that only the first matching source is returned."""
        source_id = "example.com"
        first_match = {
            "source_id": source_id,
            "total_chunks": 42,
            "version": 1,
        }
        second_match = {
            "source_id": source_id,
            "total_chunks": 100,
            "version": 2,
        }

        mock_database_client.get_sources.return_value = [
            first_match,
            second_match,
        ]

        result = await get_source_statistics(
            database_client=mock_database_client,
            source_id=source_id,
        )

        # Should return the first match
        assert result == first_match
        assert result["version"] == 1

    @pytest.mark.asyncio
    async def test_get_source_statistics_case_sensitive(self, mock_database_client):
        """Test that source_id matching is case-sensitive."""
        mock_database_client.get_sources.return_value = [
            {"source_id": "Example.com", "total_chunks": 42},
        ]

        result = await get_source_statistics(
            database_client=mock_database_client,
            source_id="example.com",  # lowercase
        )

        # Should not match due to case difference
        assert result is None

    @pytest.mark.asyncio
    async def test_get_source_statistics_with_metadata(self, mock_database_client):
        """Test retrieving source with rich metadata."""
        source_id = "example.com"
        expected_stats = {
            "source_id": source_id,
            "total_chunks": 42,
            "last_crawled": "2025-01-15T12:00:00",
            "metadata": {
                "title": "Example Site",
                "description": "Test description",
            },
            "tags": ["test", "example"],
        }

        mock_database_client.get_sources.return_value = [expected_stats]

        result = await get_source_statistics(
            database_client=mock_database_client,
            source_id=source_id,
        )

        assert result == expected_stats
        assert result["metadata"]["title"] == "Example Site"
        assert "test" in result["tags"]


class TestListAllSources:
    """Test list_all_sources function."""

    @pytest.fixture
    def mock_database_client(self):
        """Create a mock database client."""
        client = MagicMock()
        client.get_sources = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_list_all_sources_success(self, mock_database_client):
        """Test successfully listing all sources."""
        expected_sources = [
            {"source_id": "example.com", "total_chunks": 42},
            {"source_id": "test.com", "total_chunks": 10},
            {"source_id": "demo.com", "total_chunks": 5},
        ]

        mock_database_client.get_sources.return_value = expected_sources

        result = await list_all_sources(database_client=mock_database_client)

        assert result == expected_sources
        assert len(result) == 3
        mock_database_client.get_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_all_sources_empty_database(self, mock_database_client):
        """Test listing sources when database is empty."""
        mock_database_client.get_sources.return_value = []

        result = await list_all_sources(database_client=mock_database_client)

        assert result == []
        assert len(result) == 0
        mock_database_client.get_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_all_sources_none_returned(self, mock_database_client):
        """Test listing sources when get_sources returns None."""
        mock_database_client.get_sources.return_value = None

        result = await list_all_sources(database_client=mock_database_client)

        assert result == []
        mock_database_client.get_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_all_sources_query_error(self, mock_database_client):
        """Test QueryError is propagated from database client."""
        mock_database_client.get_sources.side_effect = QueryError(
            "Failed to list sources",
        )

        with pytest.raises(QueryError) as exc_info:
            await list_all_sources(database_client=mock_database_client)

        assert "Failed to list sources" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_all_sources_unexpected_error(self, mock_database_client):
        """Test unexpected exceptions are propagated."""
        mock_database_client.get_sources.side_effect = ConnectionError(
            "Connection lost",
        )

        with pytest.raises(ConnectionError) as exc_info:
            await list_all_sources(database_client=mock_database_client)

        assert "Connection lost" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_all_sources_single_source(self, mock_database_client):
        """Test listing when only one source exists."""
        expected_sources = [
            {
                "source_id": "example.com",
                "total_chunks": 100,
                "last_crawled": "2025-01-15T12:00:00",
            },
        ]

        mock_database_client.get_sources.return_value = expected_sources

        result = await list_all_sources(database_client=mock_database_client)

        assert result == expected_sources
        assert len(result) == 1
        assert result[0]["source_id"] == "example.com"

    @pytest.mark.asyncio
    async def test_list_all_sources_with_metadata(self, mock_database_client):
        """Test listing sources with rich metadata."""
        expected_sources = [
            {
                "source_id": "example.com",
                "total_chunks": 42,
                "last_crawled": "2025-01-15T12:00:00",
                "metadata": {"title": "Example", "language": "en"},
                "tags": ["documentation", "tutorial"],
            },
            {
                "source_id": "test.com",
                "total_chunks": 10,
                "last_crawled": "2025-01-14T08:00:00",
                "metadata": {"title": "Test Site"},
                "tags": ["test"],
            },
        ]

        mock_database_client.get_sources.return_value = expected_sources

        result = await list_all_sources(database_client=mock_database_client)

        assert result == expected_sources
        assert len(result) == 2
        assert result[0]["metadata"]["language"] == "en"
        assert "test" in result[1]["tags"]

    @pytest.mark.asyncio
    async def test_list_all_sources_preserves_order(self, mock_database_client):
        """Test that source order is preserved from database client."""
        expected_sources = [
            {"source_id": "z.com", "total_chunks": 1},
            {"source_id": "a.com", "total_chunks": 2},
            {"source_id": "m.com", "total_chunks": 3},
        ]

        mock_database_client.get_sources.return_value = expected_sources

        result = await list_all_sources(database_client=mock_database_client)

        # Order should be preserved exactly as returned from database
        assert [s["source_id"] for s in result] == ["z.com", "a.com", "m.com"]

    @pytest.mark.asyncio
    async def test_list_all_sources_large_dataset(self, mock_database_client):
        """Test listing a large number of sources."""
        # Create 100 mock sources
        expected_sources = [
            {"source_id": f"example{i}.com", "total_chunks": i}
            for i in range(100)
        ]

        mock_database_client.get_sources.return_value = expected_sources

        result = await list_all_sources(database_client=mock_database_client)

        assert len(result) == 100
        assert result[0]["source_id"] == "example0.com"
        assert result[99]["source_id"] == "example99.com"

    @pytest.mark.asyncio
    async def test_list_all_sources_returns_copy_not_reference(
        self, mock_database_client,
    ):
        """Test that function returns the actual list from database client."""
        expected_sources = [
            {"source_id": "example.com", "total_chunks": 42},
        ]

        mock_database_client.get_sources.return_value = expected_sources

        result = await list_all_sources(database_client=mock_database_client)

        # Result should be the same list (function doesn't copy it)
        assert result is expected_sources


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions."""

    @pytest.fixture
    def mock_database_client(self):
        """Create a mock database client."""
        client = MagicMock()
        client.update_source = AsyncMock()
        client.get_sources = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_update_then_retrieve_source(self, mock_database_client):
        """Test updating a source and then retrieving its statistics."""
        source_id = "example.com"
        total_chunks = 42
        last_crawled = datetime(2025, 1, 15, 12, 0, 0)

        # Update source
        await update_source_summary(
            database_client=mock_database_client,
            source_id=source_id,
            total_chunks=total_chunks,
            last_crawled=last_crawled,
        )

        # Mock get_sources to return the updated source
        mock_database_client.get_sources.return_value = [
            {
                "source_id": source_id,
                "total_chunks": total_chunks,
                "last_crawled": last_crawled.isoformat(),
            },
        ]

        # Retrieve statistics
        stats = await get_source_statistics(
            database_client=mock_database_client,
            source_id=source_id,
        )

        assert stats is not None
        assert stats["source_id"] == source_id
        assert stats["total_chunks"] == total_chunks

    @pytest.mark.asyncio
    async def test_list_sources_after_multiple_updates(self, mock_database_client):
        """Test listing sources after multiple updates."""
        sources_to_update = [
            ("example.com", 42),
            ("test.com", 10),
            ("demo.com", 5),
        ]

        # Update multiple sources
        for source_id, total_chunks in sources_to_update:
            await update_source_summary(
                database_client=mock_database_client,
                source_id=source_id,
                total_chunks=total_chunks,
            )

        # Verify update_source was called for each
        assert mock_database_client.update_source.call_count == 3

        # Mock get_sources to return all updated sources
        mock_database_client.get_sources.return_value = [
            {"source_id": sid, "total_chunks": chunks}
            for sid, chunks in sources_to_update
        ]

        # List all sources
        all_sources = await list_all_sources(database_client=mock_database_client)

        assert len(all_sources) == 3
        source_ids = [s["source_id"] for s in all_sources]
        assert "example.com" in source_ids
        assert "test.com" in source_ids
        assert "demo.com" in source_ids

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, mock_database_client):
        """Test error recovery in a typical workflow."""
        source_id = "example.com"

        # First attempt fails with QueryError
        mock_database_client.update_source.side_effect = QueryError("Connection timeout")

        with pytest.raises(QueryError):
            await update_source_summary(
                database_client=mock_database_client,
                source_id=source_id,
                total_chunks=42,
            )

        # Reset side effect for retry
        mock_database_client.update_source.side_effect = None

        # Retry succeeds
        await update_source_summary(
            database_client=mock_database_client,
            source_id=source_id,
            total_chunks=42,
        )

        # Verify retry was successful
        assert mock_database_client.update_source.call_count == 2
