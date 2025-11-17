"""Integration tests for QdrantAdapter.

Tests real integration scenarios with actual Qdrant instance.
NO MOCKS - Real database operations only.

These tests require:
- Qdrant running on localhost:6333
- No authentication configured
"""

import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from src.database.qdrant_adapter import QdrantAdapter

# Constants
VECTOR_SIZE = 1536


@pytest.fixture
async def qdrant_client():
    """Create Qdrant client for integration tests."""
    client = AsyncQdrantClient(url="http://localhost:6333", timeout=10.0)

    # Verify Qdrant is available
    try:
        await client.get_collections()
        yield client
    except Exception as e:
        pytest.skip(f"Qdrant not available at localhost:6333: {e}")
    finally:
        await client.close()


@pytest.fixture
async def test_adapter(qdrant_client):
    """Create adapter and clean collection before/after test."""
    # Create adapter instance
    adapter = QdrantAdapter(url="http://localhost:6333")
    await adapter.initialize()

    # Clean crawled_pages collection before test
    try:
        await qdrant_client.delete_collection(adapter.CRAWLED_PAGES)
        # Recreate it
        await adapter._ensure_collections()
    except Exception:
        pass  # Collection may not exist

    yield adapter

    # Cleanup: clear collection after test
    try:
        await qdrant_client.delete_collection(adapter.CRAWLED_PAGES)
    except Exception:
        pass


class TestQdrantAdapterIntegrationEdgeCases:
    """Integration tests for edge cases with real Qdrant."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_results_from_search(self, test_adapter):
        """Test handling empty search results from real Qdrant.

        Real scenario: Searching in empty collection returns no results.
        """
        # Search in empty collection
        result = await test_adapter.search_documents(
            query_embedding=[0.5] * VECTOR_SIZE,
            match_count=10,
        )

        assert result == [], "Empty collection should return empty results"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_with_none_filter_parameters(self, test_adapter):
        """Test search with None filter parameters on real Qdrant.

        Real scenario: Filters should be optional, None should work.
        """
        # Add a test document
        await test_adapter.client.upsert(
            collection_name=test_adapter.CRAWLED_PAGES,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.1] * VECTOR_SIZE,
                    payload={"content": "test content", "url": "https://test.com"},
                )
            ],
        )

        # Search with None filters
        result = await test_adapter.search_documents(
            query_embedding=[0.1] * VECTOR_SIZE,
            match_count=10,
            filter_metadata=None,
            source_filter=None,
        )

        assert len(result) > 0, "Should find documents with None filters"
        assert result[0]["content"] == "test content"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_zero_match_count_validation_error(self, test_adapter):
        """Test search with zero match count on real Qdrant.

        Real scenario: Qdrant rejects match_count=0 with validation error.
        This test documents actual Qdrant behavior - limit must be >= 1.
        """
        # Add a test document
        await test_adapter.client.upsert(
            collection_name=test_adapter.CRAWLED_PAGES,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.2] * VECTOR_SIZE,
                    payload={"content": "test", "url": "https://test.com"},
                )
            ],
        )

        # Search with match_count=0 should raise validation error
        with pytest.raises(Exception) as exc_info:
            await test_adapter.search_documents(
                query_embedding=[0.2] * VECTOR_SIZE,
                match_count=0,
            )

        # Qdrant validates limit parameter
        assert "value 0 invalid" in str(exc_info.value) or "must be 1 or larger" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_very_large_match_count(self, test_adapter):
        """Test search with very large match count on real Qdrant.

        Real scenario: Large match_count should work, limited by actual data.
        """
        # Add 5 test documents
        points = []
        for i in range(5):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.3 + i * 0.01] * VECTOR_SIZE,
                    payload={
                        "content": f"test content {i}",
                        "url": f"https://test{i}.com",
                    },
                )
            )

        await test_adapter.client.upsert(
            collection_name=test_adapter.CRAWLED_PAGES,
            points=points,
        )

        # Search with very large match_count
        result = await test_adapter.search_documents(
            query_embedding=[0.3] * VECTOR_SIZE,
            match_count=10000,
        )

        # Should return all 5 documents (limited by actual data, not match_count)
        assert len(result) == 5, "Should return all available documents"
        assert all("content" in doc for doc in result)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_nonexistent_collection_raises_error(self, qdrant_client):
        """Test that searching non-existent collection raises DatabaseError.

        Real scenario: Database operation on non-existent collection should fail.
        """
        # Create adapter without initializing
        adapter = QdrantAdapter(url="http://localhost:6333")
        # Don't call initialize() - collections don't exist yet

        # Delete collection if it exists
        try:
            await qdrant_client.delete_collection(adapter.CRAWLED_PAGES)
        except Exception:
            pass

        # Search should raise error
        with pytest.raises(Exception) as exc_info:
            await adapter.search_documents(
                query_embedding=[0.5] * VECTOR_SIZE,
                match_count=10,
            )

        # Should be some kind of error from Qdrant
        assert exc_info.value is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_with_filters(self, test_adapter):
        """Test search with metadata filters on real Qdrant.

        Real scenario: Filtering should work correctly.
        """
        # Add documents with different metadata
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.4] * VECTOR_SIZE,
                payload={
                    "content": "python content",
                    "url": "https://python.com",
                    "metadata": {"language": "python", "category": "docs"},
                },
            ),
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.41] * VECTOR_SIZE,
                payload={
                    "content": "javascript content",
                    "url": "https://js.com",
                    "metadata": {"language": "javascript", "category": "docs"},
                },
            ),
        ]

        await test_adapter.client.upsert(
            collection_name=test_adapter.CRAWLED_PAGES,
            points=points,
        )

        # Search with language filter
        result = await test_adapter.search_documents(
            query_embedding=[0.4] * VECTOR_SIZE,
            match_count=10,
            filter_metadata={"language": "python"},
        )

        assert len(result) >= 1, "Should find python documents"
        assert all(
            doc.get("metadata", {}).get("language") == "python"
            for doc in result
        ), "All results should match filter"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_url_list_for_delete(self, test_adapter):
        """Test deleting with empty URL list on real Qdrant.

        Real scenario: Deleting nothing should not fail.
        """
        # Add a document first
        await test_adapter.client.upsert(
            collection_name=test_adapter.CRAWLED_PAGES,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.6] * VECTOR_SIZE,
                    payload={"content": "test", "url": "https://test.com"},
                )
            ],
        )

        # Delete with empty list should not fail
        await test_adapter.delete_documents_by_url([])

        # Document should still exist
        result = await test_adapter.search_documents(
            query_embedding=[0.6] * VECTOR_SIZE,
            match_count=1,
        )

        assert len(result) == 1, "Document should still exist after empty delete"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
