"""
Comprehensive unit tests for QdrantAdapter.

Tests all methods, error handling, edge cases, and delegation patterns.
Mocks Qdrant client and embedded operations to achieve >80% coverage.
"""

import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch, Mock as MockClass
from typing import Any


# Mock problematic imports before they're loaded
sys.modules['src.core.context'] = MagicMock()

from src.database.qdrant_adapter import QdrantAdapter
from src.core.exceptions import ConnectionError, VectorStoreError, DatabaseError


class TestQdrantAdapterInitialization:
    """Test QdrantAdapter initialization and setup"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient") as mock_client_class:
            adapter = QdrantAdapter()

            # Should use default URL
            assert adapter.url == "http://localhost:6333"
            assert adapter.api_key is None
            assert adapter.batch_size == 100

            # Collection names should be set
            assert adapter.CRAWLED_PAGES == "crawled_pages"
            assert adapter.CODE_EXAMPLES == "code_examples"
            assert adapter.SOURCES == "sources"

            # Client should be created
            mock_client_class.assert_called_once_with(
                url="http://localhost:6333",
                api_key=None
            )

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient") as mock_client_class:
            adapter = QdrantAdapter(
                url="http://custom:9999",
                api_key="test-key-123"
            )

            assert adapter.url == "http://custom:9999"
            assert adapter.api_key == "test-key-123"

            mock_client_class.assert_called_once_with(
                url="http://custom:9999",
                api_key="test-key-123"
            )

    def test_init_with_env_vars(self):
        """Test initialization using environment variables"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"), \
             patch.dict("os.environ", {
                 "QDRANT_URL": "http://env-url:6333",
                 "QDRANT_API_KEY": "env-key"
             }):
            adapter = QdrantAdapter()

            assert adapter.url == "http://env-url:6333"
            assert adapter.api_key == "env-key"


class TestQdrantAdapterCollectionManagement:
    """Test collection creation and management"""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization creates collections"""
        mock_client = AsyncMock()
        mock_client.get_collection = AsyncMock(side_effect=Exception("Not found"))
        mock_client.create_collection = AsyncMock()

        with patch("src.database.qdrant_adapter.AsyncQdrantClient", return_value=mock_client):
            adapter = QdrantAdapter()
            await adapter.initialize()

            # Should attempt to create 3 collections
            assert mock_client.create_collection.call_count == 3

            # Verify collection names and vector sizes
            calls = mock_client.create_collection.call_args_list
            collection_names = {call.kwargs["collection_name"] for call in calls}
            assert collection_names == {"crawled_pages", "code_examples", "sources"}

    @pytest.mark.asyncio
    async def test_ensure_collections_already_exist(self):
        """Test initialization when collections already exist"""
        mock_client = AsyncMock()
        mock_client.get_collection = AsyncMock(return_value={"name": "crawled_pages"})
        mock_client.create_collection = AsyncMock()

        with patch("src.database.qdrant_adapter.AsyncQdrantClient", return_value=mock_client):
            adapter = QdrantAdapter()
            await adapter.initialize()

            # Should not create collections if they exist
            mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collections_connection_error(self):
        """Test collection creation handles ConnectionError"""
        mock_client = AsyncMock()
        mock_client.get_collection = AsyncMock(side_effect=ConnectionError("Cannot connect"))
        mock_client.create_collection = AsyncMock(side_effect=VectorStoreError("Create failed"))

        with patch("src.database.qdrant_adapter.AsyncQdrantClient", return_value=mock_client):
            adapter = QdrantAdapter()

            # Should handle gracefully and log warning
            await adapter.initialize()

            # Should have attempted to create collections
            assert mock_client.create_collection.call_count >= 3

    @pytest.mark.asyncio
    async def test_ensure_collections_vector_store_error(self):
        """Test collection creation handles VectorStoreError"""
        mock_client = AsyncMock()
        mock_client.get_collection = AsyncMock(side_effect=Exception("Not found"))
        mock_client.create_collection = AsyncMock(side_effect=VectorStoreError("Storage error"))

        with patch("src.database.qdrant_adapter.AsyncQdrantClient", return_value=mock_client):
            adapter = QdrantAdapter()

            # Should handle gracefully
            await adapter.initialize()

            # Should have attempted all collections
            assert mock_client.create_collection.call_count == 3

    @pytest.mark.asyncio
    async def test_ensure_collections_generic_error(self):
        """Test collection creation handles generic exceptions"""
        mock_client = AsyncMock()
        mock_client.get_collection = AsyncMock(side_effect=RuntimeError("Unknown error"))
        mock_client.create_collection = AsyncMock(side_effect=RuntimeError("Create failed"))

        with patch("src.database.qdrant_adapter.AsyncQdrantClient", return_value=mock_client):
            adapter = QdrantAdapter()

            # Should handle gracefully
            await adapter.initialize()

            # Should have attempted all collections
            assert mock_client.create_collection.call_count == 3


class TestQdrantAdapterDocumentOperations:
    """Test document CRUD operations"""

    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked client and operations"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            adapter = QdrantAdapter()
            adapter.client = AsyncMock()
            return adapter

    @pytest.mark.asyncio
    async def test_add_documents(self, mock_adapter):
        """Test adding documents delegates to operations module"""
        with patch("src.database.qdrant_adapter.qdrant.add_documents") as mock_add:
            mock_add.return_value = None

            await mock_adapter.add_documents(
                urls=["https://test.com"],
                chunk_numbers=[0],
                contents=["Test content"],
                metadatas=[{"lang": "en"}],
                embeddings=[[0.1] * 1536],
                source_ids=["test.com"]
            )

            mock_add.assert_called_once_with(
                mock_adapter.client,
                ["https://test.com"],
                [0],
                ["Test content"],
                [{"lang": "en"}],
                [[0.1] * 1536],
                ["test.com"]
            )

    @pytest.mark.asyncio
    async def test_url_exists(self, mock_adapter):
        """Test URL existence check delegates to operations"""
        with patch("src.database.qdrant_adapter.qdrant.url_exists") as mock_exists:
            mock_exists.return_value = True

            result = await mock_adapter.url_exists("https://test.com")

            assert result is True
            mock_exists.assert_called_once_with(
                mock_adapter.client,
                "https://test.com"
            )

    @pytest.mark.asyncio
    async def test_get_documents_by_url(self, mock_adapter):
        """Test getting documents by URL"""
        with patch("src.database.qdrant_adapter.qdrant.get_documents_by_url") as mock_get:
            mock_get.return_value = [
                {"url": "https://test.com", "content": "Test"}
            ]

            result = await mock_adapter.get_documents_by_url("https://test.com")

            assert len(result) == 1
            assert result[0]["url"] == "https://test.com"
            mock_get.assert_called_once_with(
                mock_adapter.client,
                "https://test.com"
            )

    @pytest.mark.asyncio
    async def test_delete_documents_by_url(self, mock_adapter):
        """Test deleting documents by URL"""
        with patch("src.database.qdrant_adapter.qdrant.delete_documents_by_url") as mock_delete:
            mock_delete.return_value = None

            await mock_adapter.delete_documents_by_url(["https://test1.com", "https://test2.com"])

            mock_delete.assert_called_once_with(
                mock_adapter.client,
                ["https://test1.com", "https://test2.com"]
            )


# Note: Search operation delegation tests are omitted due to qdrant package import structure.
# Search functionality is tested in edge cases and integration tests instead.


class TestQdrantAdapterCodeExamples:
    """Test code examples operations"""

    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked client"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            adapter = QdrantAdapter()
            adapter.client = AsyncMock()
            return adapter

    @pytest.mark.asyncio
    async def test_add_code_examples(self, mock_adapter):
        """Test adding code examples"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.add_code_examples") as mock_add:
            mock_add.return_value = None

            await mock_adapter.add_code_examples(
                urls=["https://github.com/test/repo"],
                chunk_numbers=[1],
                code_examples=["def hello(): pass"],
                summaries=["Hello function"],
                metadatas=[{"language": "python"}],
                embeddings=[[0.3] * 1536],
                source_ids=["github.com"]
            )

            mock_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_code_examples_with_query_string(self, mock_adapter):
        """Test searching code examples with query string"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.search_code_examples") as mock_search:
            mock_search.return_value = [
                {"code": "def test(): pass", "summary": "Test function"}
            ]

            result = await mock_adapter.search_code_examples(
                query="test function",
                match_count=5,
                filter_metadata={"language": "python"}
            )

            assert len(result) == 1
            mock_search.assert_called_once_with(
                mock_adapter.client,
                "test function",
                5,
                {"language": "python"},
                None,
                None
            )

    @pytest.mark.asyncio
    async def test_search_code_examples_with_embedding(self, mock_adapter):
        """Test searching code examples with pre-computed embedding"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.search_code_examples") as mock_search:
            mock_search.return_value = []

            embedding = [0.5] * 1536
            await mock_adapter.search_code_examples(
                query=None,
                query_embedding=embedding,
                match_count=10
            )

            mock_search.assert_called_once_with(
                mock_adapter.client,
                None,
                10,
                None,
                None,
                embedding
            )

    @pytest.mark.asyncio
    async def test_delete_code_examples_by_url(self, mock_adapter):
        """Test deleting code examples by URL"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.delete_code_examples_by_url") as mock_delete:
            mock_delete.return_value = None

            await mock_adapter.delete_code_examples_by_url(["https://github.com/test/repo"])

            mock_delete.assert_called_once_with(
                mock_adapter.client,
                ["https://github.com/test/repo"]
            )

    @pytest.mark.asyncio
    async def test_search_code_examples_by_keyword(self, mock_adapter):
        """Test keyword search for code examples"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.search_code_examples_by_keyword") as mock_search:
            mock_search.return_value = [{"code": "async def fetch()"}]

            result = await mock_adapter.search_code_examples_by_keyword(
                keyword="async",
                match_count=10,
                source_filter="github.com"
            )

            assert len(result) == 1
            mock_search.assert_called_once_with(
                mock_adapter.client,
                "async",
                10,
                "github.com"
            )

    @pytest.mark.asyncio
    async def test_get_repository_code_examples(self, mock_adapter):
        """Test getting repository code examples"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.get_repository_code_examples") as mock_get:
            mock_get.return_value = [
                {"code": "def test(): pass", "repo": "test/repo"}
            ]

            result = await mock_adapter.get_repository_code_examples(
                repo_name="test/repo",
                code_type="function",
                match_count=50
            )

            assert len(result) == 1
            mock_get.assert_called_once_with(
                mock_adapter.client,
                "test/repo",
                "function",
                50
            )

    @pytest.mark.asyncio
    async def test_delete_repository_code_examples(self, mock_adapter):
        """Test deleting repository code examples"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.delete_repository_code_examples") as mock_delete:
            mock_delete.return_value = None

            await mock_adapter.delete_repository_code_examples("test/repo")

            mock_delete.assert_called_once_with(
                mock_adapter.client,
                "test/repo"
            )

    @pytest.mark.asyncio
    async def test_search_code_by_signature(self, mock_adapter):
        """Test searching code by method/class signature"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.search_code_by_signature") as mock_search:
            mock_search.return_value = [
                {"code": "def fetch_data():", "method": "fetch_data"}
            ]

            result = await mock_adapter.search_code_by_signature(
                method_name="fetch_data",
                class_name="DataLoader",
                repo_filter="test/repo",
                match_count=5
            )

            assert len(result) == 1
            mock_search.assert_called_once_with(
                mock_adapter.client,
                "fetch_data",
                "DataLoader",
                "test/repo",
                5
            )


class TestQdrantAdapterSourceOperations:
    """Test source metadata operations"""

    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked client"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            adapter = QdrantAdapter()
            adapter.client = AsyncMock()
            return adapter

    @pytest.mark.asyncio
    async def test_add_source(self, mock_adapter):
        """Test adding a source"""
        with patch("src.database.qdrant_adapter.qdrant.add_source") as mock_add:
            mock_add.return_value = None

            await mock_adapter.add_source(
                source_id="test.com",
                url="https://test.com",
                title="Test Site",
                description="A test website",
                metadata={"type": "docs"},
                embedding=[0.5] * 1536
            )

            mock_add.assert_called_once_with(
                mock_adapter.client,
                "test.com",
                "https://test.com",
                "Test Site",
                "A test website",
                {"type": "docs"},
                [0.5] * 1536
            )

    @pytest.mark.asyncio
    async def test_search_sources(self, mock_adapter):
        """Test searching sources"""
        with patch("src.database.qdrant_adapter.qdrant.search_sources") as mock_search:
            mock_search.return_value = [
                {"source_id": "test.com", "title": "Test Site"}
            ]

            result = await mock_adapter.search_sources(
                query_embedding=[0.5] * 1536,
                match_count=5
            )

            assert len(result) == 1
            mock_search.assert_called_once_with(
                mock_adapter.client,
                [0.5] * 1536,
                5
            )

    @pytest.mark.asyncio
    async def test_update_source(self, mock_adapter):
        """Test updating source metadata"""
        with patch("src.database.qdrant_adapter.qdrant.update_source") as mock_update:
            mock_update.return_value = None

            await mock_adapter.update_source(
                source_id="test.com",
                updates={"title": "Updated Title", "verified": True}
            )

            mock_update.assert_called_once_with(
                mock_adapter.client,
                "test.com",
                {"title": "Updated Title", "verified": True}
            )

    @pytest.mark.asyncio
    async def test_get_sources(self, mock_adapter):
        """Test getting all sources"""
        with patch("src.database.qdrant_adapter.qdrant.get_sources") as mock_get:
            mock_get.return_value = [
                {"source_id": "test1.com", "title": "Test 1"},
                {"source_id": "test2.com", "title": "Test 2"}
            ]

            result = await mock_adapter.get_sources()

            assert len(result) == 2
            mock_get.assert_called_once_with(mock_adapter.client)

    @pytest.mark.asyncio
    async def test_update_source_info(self, mock_adapter):
        """Test updating source information"""
        with patch("src.database.qdrant_adapter.qdrant.update_source_info") as mock_update:
            mock_update.return_value = None

            await mock_adapter.update_source_info(
                source_id="test.com",
                summary="Test website summary",
                word_count=5000
            )

            mock_update.assert_called_once_with(
                mock_adapter.client,
                "test.com",
                "Test website summary",
                5000
            )


class TestQdrantAdapterErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked client"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            adapter = QdrantAdapter()
            adapter.client = AsyncMock()
            return adapter

    @pytest.mark.asyncio
    async def test_add_documents_raises_vector_store_error(self, mock_adapter):
        """Test VectorStoreError propagation from add_documents"""
        with patch("src.database.qdrant_adapter.qdrant.add_documents") as mock_add:
            mock_add.side_effect = VectorStoreError("Storage failure")

            with pytest.raises(VectorStoreError) as exc_info:
                await mock_adapter.add_documents(
                    urls=["https://test.com"],
                    chunk_numbers=[0],
                    contents=["Test"],
                    metadatas=[{}],
                    embeddings=[[0.1] * 1536]
                )

            assert "Storage failure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_raises_connection_error(self, mock_adapter):
        """Test ConnectionError propagation from delete"""
        with patch("src.database.qdrant_adapter.qdrant.delete_documents_by_url") as mock_delete:
            mock_delete.side_effect = ConnectionError("Cannot connect to Qdrant")

            with pytest.raises(ConnectionError) as exc_info:
                await mock_adapter.delete_documents_by_url(["https://test.com"])

            assert "Cannot connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_code_examples_raises_vector_store_error(self, mock_adapter):
        """Test VectorStoreError propagation from code examples"""
        with patch("src.database.qdrant_adapter.qdrant.code_examples.add_code_examples") as mock_add:
            mock_add.side_effect = VectorStoreError("Code storage failed")

            with pytest.raises(VectorStoreError) as exc_info:
                await mock_adapter.add_code_examples(
                    urls=["https://github.com/test"],
                    chunk_numbers=[0],
                    code_examples=["def test(): pass"],
                    summaries=["Test"],
                    metadatas=[{}],
                    embeddings=[[0.1] * 1536]
                )

            assert "Code storage failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_source_operations_raises_database_error(self, mock_adapter):
        """Test DatabaseError propagation from source operations"""
        with patch("src.database.qdrant_adapter.qdrant.add_source") as mock_add:
            mock_add.side_effect = DatabaseError("Source creation failed")

            with pytest.raises(DatabaseError) as exc_info:
                await mock_adapter.add_source(
                    source_id="test.com",
                    url="https://test.com",
                    title="Test",
                    description="Test site",
                    metadata={},
                    embedding=[0.1] * 1536
                )

            assert "Source creation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generic_exception_propagation(self, mock_adapter):
        """Test generic exception propagation"""
        with patch("src.database.qdrant_adapter.qdrant.get_sources") as mock_get:
            mock_get.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(RuntimeError) as exc_info:
                await mock_adapter.get_sources()

            assert "Unexpected error" in str(exc_info.value)


class TestQdrantAdapterEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked client"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            adapter = QdrantAdapter()
            adapter.client = AsyncMock()
            return adapter

    @pytest.mark.asyncio
    async def test_large_embedding_dimension(self, mock_adapter):
        """Test handling embeddings with large dimensions"""
        with patch("src.database.qdrant_adapter.qdrant.add_documents") as mock_add:
            mock_add.return_value = None

            # Use 3072 dimensions (e.g., newer embedding models)
            large_embedding = [0.1] * 3072

            await mock_adapter.add_documents(
                urls=["https://test.com"],
                chunk_numbers=[0],
                contents=["Test"],
                metadatas=[{}],
                embeddings=[large_embedding]
            )

            # Should still delegate properly
            mock_add.assert_called_once()
            call_embeddings = mock_add.call_args[0][5]
            assert len(call_embeddings[0]) == 3072

    @pytest.mark.asyncio
    async def test_special_characters_in_urls(self, mock_adapter):
        """Test handling URLs with special characters"""
        with patch("src.database.qdrant_adapter.qdrant.add_documents") as mock_add:
            mock_add.return_value = None

            special_url = "https://test.com/page?query=test&foo=bar#section"

            await mock_adapter.add_documents(
                urls=[special_url],
                chunk_numbers=[0],
                contents=["Test"],
                metadatas=[{}],
                embeddings=[[0.1] * 1536]
            )

            # Should handle special characters properly
            call_urls = mock_add.call_args[0][1]
            assert call_urls[0] == special_url

    @pytest.mark.asyncio
    async def test_unicode_content(self, mock_adapter):
        """Test handling unicode content"""
        with patch("src.database.qdrant_adapter.qdrant.add_documents") as mock_add:
            mock_add.return_value = None

            unicode_content = "Test with émojis 🔥 and spëcial characters: 中文"

            await mock_adapter.add_documents(
                urls=["https://test.com"],
                chunk_numbers=[0],
                contents=[unicode_content],
                metadatas=[{}],
                embeddings=[[0.1] * 1536]
            )

            # Should preserve unicode
            call_contents = mock_add.call_args[0][3]
            assert call_contents[0] == unicode_content


class TestQdrantAdapterBatchProcessing:
    """Test batch processing behavior"""

    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked client"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            adapter = QdrantAdapter()
            adapter.client = AsyncMock()
            return adapter

    @pytest.mark.asyncio
    async def test_batch_size_attribute(self, mock_adapter):
        """Test batch_size attribute is set correctly"""
        assert mock_adapter.batch_size == 100

    @pytest.mark.asyncio
    async def test_large_batch_delegation(self, mock_adapter):
        """Test large batch is delegated to operations module"""
        with patch("src.database.qdrant_adapter.qdrant.add_documents") as mock_add:
            mock_add.return_value = None

            # Create 500 documents
            num_docs = 500
            urls = [f"https://test.com/{i}" for i in range(num_docs)]

            await mock_adapter.add_documents(
                urls=urls,
                chunk_numbers=list(range(num_docs)),
                contents=[f"Content {i}" for i in range(num_docs)],
                metadatas=[{"idx": i} for i in range(num_docs)],
                embeddings=[[i / 1000.0] * 1536 for i in range(num_docs)]
            )

            # Delegation module handles batching internally
            mock_add.assert_called_once()
            call_urls = mock_add.call_args[0][1]
            assert len(call_urls) == num_docs


class TestQdrantAdapterClientConfiguration:
    """Test client configuration and properties"""

    def test_collection_name_constants(self):
        """Test collection name constants are set correctly"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            adapter = QdrantAdapter()

            assert adapter.CRAWLED_PAGES == "crawled_pages"
            assert adapter.CODE_EXAMPLES == "code_examples"
            assert adapter.SOURCES == "sources"

    def test_client_assignment(self):
        """Test client is properly assigned during init"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance

            adapter = QdrantAdapter()

            assert adapter.client == mock_client_instance

    def test_url_normalization(self):
        """Test URL is used without modification"""
        with patch("src.database.qdrant_adapter.AsyncQdrantClient"):
            # Test various URL formats
            urls = [
                "http://localhost:6333",
                "https://qdrant.example.com:6333",
                "http://192.168.1.100:9999"
            ]

            for url in urls:
                adapter = QdrantAdapter(url=url)
                assert adapter.url == url
