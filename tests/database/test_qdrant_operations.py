"""
Comprehensive unit tests for Qdrant operations module.

Tests all CRUD operations with mocked Qdrant client (no actual DB calls).
Covers success paths, error handling, and edge cases.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import (
    PointIdsList,
    PointStruct,
)

from src.core.exceptions import QueryError, VectorStoreError
from src.database.qdrant import operations


class TestGeneratePointId:
    """Test point ID generation."""

    def test_generate_point_id_deterministic(self):
        """Test that point IDs are deterministic."""
        url = "https://example.com/page"
        chunk_number = 1

        # Generate ID twice - should be identical
        id1 = operations._generate_point_id(url, chunk_number)
        id2 = operations._generate_point_id(url, chunk_number)

        assert id1 == id2
        assert isinstance(id1, str)
        # Verify it's a valid UUID
        assert uuid.UUID(id1)

    def test_generate_point_id_unique_per_chunk(self):
        """Test that different chunks get different IDs."""
        url = "https://example.com/page"
        id1 = operations._generate_point_id(url, 1)
        id2 = operations._generate_point_id(url, 2)

        assert id1 != id2

    def test_generate_point_id_unique_per_url(self):
        """Test that different URLs get different IDs."""
        chunk_number = 1
        id1 = operations._generate_point_id("https://example.com/page1", chunk_number)
        id2 = operations._generate_point_id("https://example.com/page2", chunk_number)

        assert id1 != id2


class TestAddDocuments:
    """Test add_documents function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        client.upsert = AsyncMock()
        client.scroll = AsyncMock(return_value=([], None))
        client.delete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_add_documents_success(self, mock_client):
        """Test successful document addition."""
        urls = ["https://example.com/page1", "https://example.com/page2"]
        chunk_numbers = [1, 2]
        contents = ["Content 1", "Content 2"]
        metadatas = [{"key1": "value1"}, {"key2": "value2"}]
        embeddings = [[0.1] * 1536, [0.2] * 1536]

        await operations.add_documents(
            client=mock_client,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # Verify upsert was called
        assert mock_client.upsert.called
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == operations.CRAWLED_PAGES
        points = call_args.kwargs["points"]
        assert len(points) == 2

        # Verify point structure
        point = points[0]
        assert isinstance(point, PointStruct)
        assert point.payload["url"] == urls[0]
        assert point.payload["chunk_number"] == chunk_numbers[0]
        assert point.payload["content"] == contents[0]
        assert point.payload["metadata"] == metadatas[0]
        assert "source_id" in point.payload

    @pytest.mark.asyncio
    async def test_add_documents_with_source_ids(self, mock_client):
        """Test document addition with explicit source IDs."""
        urls = ["https://example.com/page1"]
        chunk_numbers = [1]
        contents = ["Content 1"]
        metadatas = [{"key": "value"}]
        embeddings = [[0.1] * 1536]
        source_ids = ["custom-source-id"]

        await operations.add_documents(
            client=mock_client,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
            source_ids=source_ids,
        )

        call_args = mock_client.upsert.call_args
        points = call_args.kwargs["points"]
        assert points[0].payload["source_id"] == "custom-source-id"

    @pytest.mark.asyncio
    async def test_add_documents_extracts_source_id_from_url(self, mock_client):
        """Test automatic source_id extraction from URL."""
        urls = ["https://www.example.com/page1"]
        chunk_numbers = [1]
        contents = ["Content"]
        metadatas = [{}]
        embeddings = [[0.1] * 1536]

        await operations.add_documents(
            client=mock_client,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        call_args = mock_client.upsert.call_args
        points = call_args.kwargs["points"]
        # Should strip "www." prefix
        assert points[0].payload["source_id"] == "example.com"

    @pytest.mark.asyncio
    async def test_add_documents_batching(self, mock_client):
        """Test that large document sets are batched."""
        # Create 250 documents (should be 3 batches with BATCH_SIZE=100)
        num_docs = 250
        urls = [f"https://example.com/page{i}" for i in range(num_docs)]
        chunk_numbers = list(range(num_docs))
        contents = [f"Content {i}" for i in range(num_docs)]
        metadatas = [{"idx": i} for i in range(num_docs)]
        embeddings = [[0.1] * 1536 for _ in range(num_docs)]

        await operations.add_documents(
            client=mock_client,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # Should have called upsert 3 times (100 + 100 + 50)
        assert mock_client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_add_documents_deletes_existing(self, mock_client):
        """Test that existing documents are deleted before adding new ones."""
        # Mock scroll to return existing documents
        existing_point = MagicMock()
        existing_point.id = "existing-id"
        mock_client.scroll = AsyncMock(return_value=([existing_point], None))

        urls = ["https://example.com/page1"]
        chunk_numbers = [1]
        contents = ["Content"]
        metadatas = [{}]
        embeddings = [[0.1] * 1536]

        await operations.add_documents(
            client=mock_client,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # Verify delete was called
        assert mock_client.delete.called

    @pytest.mark.asyncio
    async def test_add_documents_upsert_error(self, mock_client):
        """Test error handling during upsert."""
        mock_client.upsert = AsyncMock(side_effect=VectorStoreError("Upsert failed"))

        urls = ["https://example.com/page1"]
        chunk_numbers = [1]
        contents = ["Content"]
        metadatas = [{}]
        embeddings = [[0.1] * 1536]

        with pytest.raises(VectorStoreError):
            await operations.add_documents(
                client=mock_client,
                urls=urls,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    @pytest.mark.asyncio
    async def test_add_documents_unexpected_error(self, mock_client):
        """Test handling of unexpected errors during upsert."""
        mock_client.upsert = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        urls = ["https://example.com/page1"]
        chunk_numbers = [1]
        contents = ["Content"]
        metadatas = [{}]
        embeddings = [[0.1] * 1536]

        with pytest.raises(RuntimeError):
            await operations.add_documents(
                client=mock_client,
                urls=urls,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )


class TestUrlExists:
    """Test url_exists function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_url_exists_true(self, mock_client):
        """Test when URL exists in database."""
        count_result = MagicMock()
        count_result.count = 5
        mock_client.count = AsyncMock(return_value=count_result)

        result = await operations.url_exists(mock_client, "https://example.com/page")

        assert result is True
        assert mock_client.count.called
        call_args = mock_client.count.call_args
        assert call_args.kwargs["collection_name"] == operations.CRAWLED_PAGES
        assert call_args.kwargs["exact"] is False

    @pytest.mark.asyncio
    async def test_url_exists_false(self, mock_client):
        """Test when URL does not exist in database."""
        count_result = MagicMock()
        count_result.count = 0
        mock_client.count = AsyncMock(return_value=count_result)

        result = await operations.url_exists(mock_client, "https://example.com/page")

        assert result is False


class TestGetDocumentsByUrl:
    """Test get_documents_by_url function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_documents_by_url_success(self, mock_client):
        """Test successful retrieval of documents."""
        # Mock scroll results
        point1 = MagicMock()
        point1.id = "id1"
        point1.payload = {
            "url": "https://example.com/page",
            "chunk_number": 2,
            "content": "Content 2",
            "metadata": {},
        }

        point2 = MagicMock()
        point2.id = "id2"
        point2.payload = {
            "url": "https://example.com/page",
            "chunk_number": 1,
            "content": "Content 1",
            "metadata": {},
        }

        mock_client.scroll = AsyncMock(return_value=([point1, point2], None))

        result = await operations.get_documents_by_url(
            mock_client,
            "https://example.com/page",
        )

        assert len(result) == 2
        # Should be sorted by chunk_number
        assert result[0]["chunk_number"] == 1
        assert result[1]["chunk_number"] == 2
        assert result[0]["id"] == "id2"
        assert result[1]["id"] == "id1"

    @pytest.mark.asyncio
    async def test_get_documents_by_url_empty(self, mock_client):
        """Test retrieval when no documents exist."""
        mock_client.scroll = AsyncMock(return_value=([], None))

        result = await operations.get_documents_by_url(
            mock_client,
            "https://example.com/page",
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_documents_by_url_skips_none_payload(self, mock_client):
        """Test that points with None payload are skipped."""
        point1 = MagicMock()
        point1.id = "id1"
        point1.payload = None

        point2 = MagicMock()
        point2.id = "id2"
        point2.payload = {
            "url": "https://example.com/page",
            "chunk_number": 1,
            "content": "Content",
        }

        mock_client.scroll = AsyncMock(return_value=([point1, point2], None))

        result = await operations.get_documents_by_url(
            mock_client,
            "https://example.com/page",
        )

        assert len(result) == 1
        assert result[0]["id"] == "id2"


class TestDeleteDocumentsByUrl:
    """Test delete_documents_by_url function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_delete_documents_by_url_success(self, mock_client):
        """Test successful deletion of documents."""
        point1 = MagicMock()
        point1.id = "id1"
        point2 = MagicMock()
        point2.id = "id2"

        mock_client.scroll = AsyncMock(return_value=([point1, point2], None))
        mock_client.delete = AsyncMock()

        await operations.delete_documents_by_url(
            mock_client,
            ["https://example.com/page"],
        )

        # Verify delete was called with correct IDs
        assert mock_client.delete.called
        call_args = mock_client.delete.call_args
        assert call_args.kwargs["collection_name"] == operations.CRAWLED_PAGES
        points_selector = call_args.kwargs["points_selector"]
        assert isinstance(points_selector, PointIdsList)
        assert len(points_selector.points) == 2
        assert "id1" in points_selector.points
        assert "id2" in points_selector.points

    @pytest.mark.asyncio
    async def test_delete_documents_by_url_no_documents(self, mock_client):
        """Test deletion when no documents exist."""
        mock_client.scroll = AsyncMock(return_value=([], None))
        mock_client.delete = AsyncMock()

        await operations.delete_documents_by_url(
            mock_client,
            ["https://example.com/page"],
        )

        # Delete should not be called
        assert not mock_client.delete.called

    @pytest.mark.asyncio
    async def test_delete_documents_by_url_multiple_urls(self, mock_client):
        """Test deletion of multiple URLs."""
        point1 = MagicMock()
        point1.id = "id1"
        point2 = MagicMock()
        point2.id = "id2"

        # Mock scroll to return different points for different URLs
        async def scroll_side_effect(*args, **kwargs):
            filter_cond = kwargs.get("scroll_filter")
            if filter_cond:
                # Return different points based on filter
                return ([point1], None) if "page1" in str(kwargs) else ([point2], None)
            return ([], None)

        mock_client.scroll = AsyncMock(side_effect=scroll_side_effect)
        mock_client.delete = AsyncMock()

        await operations.delete_documents_by_url(
            mock_client,
            ["https://example.com/page1", "https://example.com/page2"],
        )

        # Delete should be called twice (once per URL)
        assert mock_client.delete.call_count == 2


class TestAddSource:
    """Test add_source function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        client.upsert = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_add_source_success(self, mock_client):
        """Test successful source addition."""
        await operations.add_source(
            client=mock_client,
            source_id="test-source",
            url="https://example.com",
            title="Test Source",
            description="Test Description",
            metadata={"key": "value"},
            embedding=[0.1] * 1536,
        )

        # Verify upsert was called
        assert mock_client.upsert.called
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == operations.SOURCES
        points = call_args.kwargs["points"]
        assert len(points) == 1

        point = points[0]
        assert isinstance(point, PointStruct)
        assert point.payload["source_id"] == "test-source"
        assert point.payload["url"] == "https://example.com"
        assert point.payload["title"] == "Test Source"
        assert point.payload["description"] == "Test Description"
        assert point.payload["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_add_source_deterministic_id(self, mock_client):
        """Test that source IDs are deterministic."""
        source_id = "test-source"

        # Add source twice
        for _ in range(2):
            await operations.add_source(
                client=mock_client,
                source_id=source_id,
                url="https://example.com",
                title="Test",
                description="Test",
                metadata={},
                embedding=[0.1] * 1536,
            )

        # Both calls should use same point ID
        call1_id = mock_client.upsert.call_args_list[0].kwargs["points"][0].id
        call2_id = mock_client.upsert.call_args_list[1].kwargs["points"][0].id
        assert call1_id == call2_id


class TestSearchSources:
    """Test search_sources function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_search_sources_success(self, mock_client):
        """Test successful source search."""
        # Mock search results
        result1 = MagicMock()
        result1.id = "id1"
        result1.score = 0.9
        result1.payload = {
            "source_id": "source1",
            "title": "Source 1",
            "url": "https://example.com/1",
        }

        result2 = MagicMock()
        result2.id = "id2"
        result2.score = 0.8
        result2.payload = {
            "source_id": "source2",
            "title": "Source 2",
            "url": "https://example.com/2",
        }

        mock_client.search = AsyncMock(return_value=[result1, result2])

        results = await operations.search_sources(
            client=mock_client,
            query_embedding=[0.1] * 1536,
            match_count=10,
        )

        assert len(results) == 2
        assert results[0]["source_id"] == "source1"
        assert results[0]["similarity"] == 0.9
        assert results[0]["id"] == "id1"
        assert results[1]["source_id"] == "source2"
        assert results[1]["similarity"] == 0.8

        # Verify search was called correctly
        call_args = mock_client.search.call_args
        assert call_args.kwargs["collection_name"] == operations.SOURCES
        assert call_args.kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_search_sources_empty(self, mock_client):
        """Test search with no results."""
        mock_client.search = AsyncMock(return_value=[])

        results = await operations.search_sources(
            client=mock_client,
            query_embedding=[0.1] * 1536,
            match_count=10,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_sources_skips_none_payload(self, mock_client):
        """Test that results with None payload are skipped."""
        result1 = MagicMock()
        result1.id = "id1"
        result1.score = 0.9
        result1.payload = None

        result2 = MagicMock()
        result2.id = "id2"
        result2.score = 0.8
        result2.payload = {"source_id": "source2"}

        mock_client.search = AsyncMock(return_value=[result1, result2])

        results = await operations.search_sources(
            client=mock_client,
            query_embedding=[0.1] * 1536,
            match_count=10,
        )

        assert len(results) == 1
        assert results[0]["source_id"] == "source2"


class TestUpdateSource:
    """Test update_source function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_update_source_success(self, mock_client):
        """Test successful source update."""
        # Mock existing source
        existing_point = MagicMock()
        existing_point.id = "existing-id"
        existing_point.payload = {
            "source_id": "test-source",
            "title": "Old Title",
            "url": "https://example.com",
        }

        mock_client.retrieve = AsyncMock(return_value=[existing_point])
        mock_client.set_payload = AsyncMock()

        await operations.update_source(
            client=mock_client,
            source_id="test-source",
            updates={"title": "New Title", "description": "New Description"},
        )

        # Verify set_payload was called
        assert mock_client.set_payload.called
        call_args = mock_client.set_payload.call_args
        assert call_args.kwargs["collection_name"] == operations.SOURCES
        payload = call_args.kwargs["payload"]
        assert payload["title"] == "New Title"
        assert payload["description"] == "New Description"
        # Old fields should be preserved
        assert payload["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_update_source_not_found(self, mock_client):
        """Test update when source doesn't exist."""
        mock_client.retrieve = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="Source .* not found"):
            await operations.update_source(
                client=mock_client,
                source_id="nonexistent-source",
                updates={"title": "New Title"},
            )

    @pytest.mark.asyncio
    async def test_update_source_none_payload(self, mock_client):
        """Test update when existing point has None payload."""
        existing_point = MagicMock()
        existing_point.id = "existing-id"
        existing_point.payload = None

        mock_client.retrieve = AsyncMock(return_value=[existing_point])
        mock_client.set_payload = AsyncMock()

        await operations.update_source(
            client=mock_client,
            source_id="test-source",
            updates={"title": "New Title"},
        )

        # Should still work with empty payload
        assert mock_client.set_payload.called
        call_args = mock_client.set_payload.call_args
        payload = call_args.kwargs["payload"]
        assert payload["title"] == "New Title"

    @pytest.mark.asyncio
    async def test_update_source_query_error(self, mock_client):
        """Test update with query error."""
        mock_client.retrieve = AsyncMock(side_effect=QueryError("Query failed"))

        with pytest.raises(QueryError):
            await operations.update_source(
                client=mock_client,
                source_id="test-source",
                updates={"title": "New Title"},
            )

    @pytest.mark.asyncio
    async def test_update_source_unexpected_error(self, mock_client):
        """Test update with unexpected error."""
        mock_client.retrieve = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with pytest.raises(RuntimeError):
            await operations.update_source(
                client=mock_client,
                source_id="test-source",
                updates={"title": "New Title"},
            )


class TestGetSources:
    """Test get_sources function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_sources_success(self, mock_client):
        """Test successful retrieval of all sources."""
        point1 = MagicMock()
        point1.id = "id1"
        point1.payload = {
            "source_id": "source1",
            "summary": "Summary 1",
            "total_word_count": 100,
            "created_at": "2024-01-01",
            "updated_at": "2024-01-02",
            "enabled": True,
            "url_count": 5,
        }

        point2 = MagicMock()
        point2.id = "id2"
        point2.payload = {
            "source_id": "source2",
            "summary": "Summary 2",
            "total_word_count": 200,
            "created_at": "2024-01-03",
            "updated_at": "2024-01-04",
            "enabled": False,
            "url_count": 3,
        }

        mock_client.scroll = AsyncMock(return_value=([point1, point2], None))

        result = await operations.get_sources(mock_client)

        assert len(result) == 2
        assert result[0]["source_id"] == "source1"
        assert result[0]["summary"] == "Summary 1"
        assert result[0]["total_word_count"] == 100
        assert result[0]["enabled"] is True
        assert result[1]["source_id"] == "source2"

    @pytest.mark.asyncio
    async def test_get_sources_pagination(self, mock_client):
        """Test pagination through multiple batches."""
        # First batch
        point1 = MagicMock()
        point1.id = "id1"
        point1.payload = {"source_id": "source1"}

        # Second batch
        point2 = MagicMock()
        point2.id = "id2"
        point2.payload = {"source_id": "source2"}

        # Mock scroll to return two batches
        mock_client.scroll = AsyncMock(
            side_effect=[
                ([point1], "offset1"),  # First call returns offset
                ([point2], None),  # Second call returns None
            ],
        )

        result = await operations.get_sources(mock_client)

        assert len(result) == 2
        assert mock_client.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_get_sources_empty(self, mock_client):
        """Test retrieval with no sources."""
        mock_client.scroll = AsyncMock(return_value=([], None))

        result = await operations.get_sources(mock_client)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_sources_skips_none_payload(self, mock_client):
        """Test that points with None payload are skipped."""
        point1 = MagicMock()
        point1.id = "id1"
        point1.payload = None

        point2 = MagicMock()
        point2.id = "id2"
        point2.payload = {"source_id": "source2"}

        mock_client.scroll = AsyncMock(return_value=([point1, point2], None))

        result = await operations.get_sources(mock_client)

        assert len(result) == 1
        assert result[0]["source_id"] == "source2"

    @pytest.mark.asyncio
    async def test_get_sources_defaults_missing_fields(self, mock_client):
        """Test that missing fields get default values."""
        point = MagicMock()
        point.id = "id1"
        point.payload = {"source_id": "source1"}  # Missing other fields

        mock_client.scroll = AsyncMock(return_value=([point], None))

        result = await operations.get_sources(mock_client)

        assert len(result) == 1
        assert result[0]["source_id"] == "source1"
        assert result[0]["summary"] == ""
        assert result[0]["total_word_count"] == 0
        assert result[0]["enabled"] is True
        assert result[0]["url_count"] == 0

    @pytest.mark.asyncio
    async def test_get_sources_query_error(self, mock_client):
        """Test handling of query errors."""
        mock_client.scroll = AsyncMock(side_effect=QueryError("Query failed"))

        result = await operations.get_sources(mock_client)

        # Should return empty list on error
        assert result == []

    @pytest.mark.asyncio
    async def test_get_sources_unexpected_error(self, mock_client):
        """Test handling of unexpected errors."""
        mock_client.scroll = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        result = await operations.get_sources(mock_client)

        # Should return empty list on error
        assert result == []

    @pytest.mark.asyncio
    async def test_get_sources_sorted(self, mock_client):
        """Test that sources are sorted by source_id."""
        point1 = MagicMock()
        point1.id = "id1"
        point1.payload = {"source_id": "zebra"}

        point2 = MagicMock()
        point2.id = "id2"
        point2.payload = {"source_id": "alpha"}

        mock_client.scroll = AsyncMock(return_value=([point1, point2], None))

        result = await operations.get_sources(mock_client)

        # Should be sorted alphabetically
        assert result[0]["source_id"] == "alpha"
        assert result[1]["source_id"] == "zebra"


class TestUpdateSourceInfo:
    """Test update_source_info function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_update_source_info_existing_source(self, mock_client):
        """Test updating existing source info."""
        # Mock existing source
        existing_point = MagicMock()
        existing_point.id = "existing-id"
        existing_point.payload = {
            "source_id": "test-source",
            "summary": "Old Summary",
            "total_word_count": 50,
            "created_at": "2024-01-01",
        }

        mock_client.retrieve = AsyncMock(return_value=[existing_point])
        mock_client.set_payload = AsyncMock()

        await operations.update_source_info(
            client=mock_client,
            source_id="test-source",
            summary="New Summary",
            word_count=100,
        )

        # Verify set_payload was called
        assert mock_client.set_payload.called
        call_args = mock_client.set_payload.call_args
        payload = call_args.kwargs["payload"]
        assert payload["summary"] == "New Summary"
        assert payload["total_word_count"] == 100
        assert "updated_at" in payload
        # created_at should be preserved
        assert payload["created_at"] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_update_source_info_new_source(self, mock_client):
        """Test creating new source when it doesn't exist."""
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.upsert = AsyncMock()

        await operations.update_source_info(
            client=mock_client,
            source_id="new-source",
            summary="New Summary",
            word_count=100,
        )

        # Verify upsert was called
        assert mock_client.upsert.called
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == operations.SOURCES
        points = call_args.kwargs["points"]
        assert len(points) == 1
        point = points[0]
        assert point.payload["source_id"] == "new-source"
        assert point.payload["summary"] == "New Summary"
        assert point.payload["total_word_count"] == 100
        assert point.payload["enabled"] is True

    @pytest.mark.asyncio
    async def test_update_source_info_retrieve_error(self, mock_client):
        """Test creating new source when retrieve fails."""
        mock_client.retrieve = AsyncMock(side_effect=QueryError("Query failed"))
        mock_client.upsert = AsyncMock()

        await operations.update_source_info(
            client=mock_client,
            source_id="test-source",
            summary="Summary",
            word_count=100,
        )

        # Should create new source on error
        assert mock_client.upsert.called

    @pytest.mark.asyncio
    async def test_update_source_info_query_error_on_upsert(self, mock_client):
        """Test error handling during upsert."""
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.upsert = AsyncMock(side_effect=QueryError("Upsert failed"))

        with pytest.raises(QueryError):
            await operations.update_source_info(
                client=mock_client,
                source_id="test-source",
                summary="Summary",
                word_count=100,
            )

    @pytest.mark.asyncio
    async def test_update_source_info_unexpected_error(self, mock_client):
        """Test handling of unexpected errors."""
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.upsert = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with pytest.raises(RuntimeError):
            await operations.update_source_info(
                client=mock_client,
                source_id="test-source",
                summary="Summary",
                word_count=100,
            )


class TestCreateNewSource:
    """Test _create_new_source helper function."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked AsyncQdrantClient."""
        client = AsyncMock()
        client.upsert = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_create_new_source_success(self, mock_client):
        """Test successful creation of new source."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test-source"))

        await operations._create_new_source(
            client=mock_client,
            source_id="test-source",
            summary="Test Summary",
            word_count=100,
            timestamp="2024-01-01T00:00:00",
            point_id=point_id,
        )

        # Verify upsert was called
        assert mock_client.upsert.called
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == operations.SOURCES
        points = call_args.kwargs["points"]
        assert len(points) == 1

        point = points[0]
        assert point.id == point_id
        assert len(point.vector) == 1536  # OpenAI embedding dimensions
        assert point.payload["source_id"] == "test-source"
        assert point.payload["summary"] == "Test Summary"
        assert point.payload["total_word_count"] == 100
        assert point.payload["created_at"] == "2024-01-01T00:00:00"
        assert point.payload["updated_at"] == "2024-01-01T00:00:00"
        assert point.payload["enabled"] is True
        assert point.payload["url_count"] == 1

    @pytest.mark.asyncio
    async def test_create_new_source_deterministic_embedding(self, mock_client):
        """Test that embeddings are deterministic for same source_id."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test-source"))

        # Create source twice
        for _ in range(2):
            await operations._create_new_source(
                client=mock_client,
                source_id="test-source",
                summary="Test",
                word_count=100,
                timestamp="2024-01-01",
                point_id=point_id,
            )

        # Extract embeddings from both calls
        embedding1 = mock_client.upsert.call_args_list[0].kwargs["points"][0].vector
        embedding2 = mock_client.upsert.call_args_list[1].kwargs["points"][0].vector

        # Should be identical
        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_create_new_source_vector_store_error(self, mock_client):
        """Test error handling during source creation."""
        mock_client.upsert = AsyncMock(side_effect=VectorStoreError("Upsert failed"))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test-source"))

        with pytest.raises(VectorStoreError):
            await operations._create_new_source(
                client=mock_client,
                source_id="test-source",
                summary="Test",
                word_count=100,
                timestamp="2024-01-01",
                point_id=point_id,
            )

    @pytest.mark.asyncio
    async def test_create_new_source_unexpected_error(self, mock_client):
        """Test handling of unexpected errors."""
        mock_client.upsert = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test-source"))

        with pytest.raises(RuntimeError):
            await operations._create_new_source(
                client=mock_client,
                source_id="test-source",
                summary="Test",
                word_count=100,
                timestamp="2024-01-01",
                point_id=point_id,
            )
