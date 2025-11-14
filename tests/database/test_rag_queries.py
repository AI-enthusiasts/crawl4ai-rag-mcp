"""
Comprehensive unit tests for src/database/rag_queries.py.

Tests RAG query functions including vector search, hybrid search,
code example retrieval, and comprehensive error handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import QueryError, VectorStoreError


class TestGetAvailableSources:
    """Test get_available_sources function"""

    @pytest.mark.asyncio
    async def test_get_available_sources_success(self):
        """Test successful retrieval of available sources"""
        from src.database.rag_queries import get_available_sources

        # Mock database client
        mock_client = MagicMock()
        mock_client.get_sources = AsyncMock(
            return_value=[
                {
                    "source_id": "example.com",
                    "summary": "Example website",
                    "total_chunks": 100,
                    "first_crawled": "2024-01-01",
                    "last_crawled": "2024-01-15",
                },
                {
                    "source_id": "docs.python.org",
                    "summary": "Python documentation",
                    "total_chunks": 500,
                    "first_crawled": "2024-01-02",
                    "last_crawled": "2024-01-16",
                },
            ],
        )

        # Execute
        result = await get_available_sources(mock_client)

        # Verify
        assert mock_client.get_sources.called
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["count"] == 2
        assert len(result_data["sources"]) == 2
        assert result_data["sources"][0]["source_id"] == "example.com"
        assert result_data["sources"][1]["source_id"] == "docs.python.org"
        assert "Found 2 unique sources" in result_data["message"]

    @pytest.mark.asyncio
    async def test_get_available_sources_empty(self):
        """Test with no sources available"""
        from src.database.rag_queries import get_available_sources

        mock_client = MagicMock()
        mock_client.get_sources = AsyncMock(return_value=[])

        result = await get_available_sources(mock_client)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["count"] == 0
        assert len(result_data["sources"]) == 0
        assert "Found 0 unique sources" in result_data["message"]

    @pytest.mark.asyncio
    async def test_get_available_sources_none_returned(self):
        """Test when database returns None"""
        from src.database.rag_queries import get_available_sources

        mock_client = MagicMock()
        mock_client.get_sources = AsyncMock(return_value=None)

        result = await get_available_sources(mock_client)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["count"] == 0
        assert len(result_data["sources"]) == 0

    @pytest.mark.asyncio
    async def test_get_available_sources_query_error(self):
        """Test handling of QueryError"""
        from src.database.rag_queries import get_available_sources

        mock_client = MagicMock()
        mock_client.get_sources = AsyncMock(
            side_effect=QueryError("Database query failed"),
        )

        result = await get_available_sources(mock_client)

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Database query failed" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_available_sources_generic_error(self):
        """Test handling of generic Exception"""
        from src.database.rag_queries import get_available_sources

        mock_client = MagicMock()
        mock_client.get_sources = AsyncMock(
            side_effect=Exception("Unexpected error"),
        )

        result = await get_available_sources(mock_client)

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Unexpected error" in result_data["error"]


class TestPerformRagQuery:
    """Test perform_rag_query function"""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_vector_search_success(self):
        """Test successful vector search"""
        from src.database.rag_queries import perform_rag_query

        # Mock database client
        mock_client = MagicMock()
        mock_client.search = AsyncMock(
            return_value=[
                {
                    "content": "Test content 1",
                    "source_id": "example.com",
                    "url": "https://example.com/page1",
                    "title": "Test Page 1",
                    "chunk_index": 0,
                    "score": 0.95,
                },
                {
                    "content": "Test content 2",
                    "source_id": "example.com",
                    "url": "https://example.com/page2",
                    "title": "Test Page 2",
                    "chunk_index": 1,
                    "score": 0.85,
                },
            ],
        )

        # Execute
        result = await perform_rag_query(
            mock_client,
            query="test query",
            match_count=5,
        )

        # Verify
        assert mock_client.search.called
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["query"] == "test query"
        assert result_data["match_count"] == 2
        assert result_data["search_type"] == "vector"
        assert len(result_data["results"]) == 2
        assert result_data["results"][0]["similarity_score"] == 0.95
        assert result_data["results"][0]["source"] == "example.com"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "true"})
    async def test_perform_rag_query_hybrid_search_success(self):
        """Test successful hybrid search"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.hybrid_search = AsyncMock(
            return_value=[
                {
                    "content": "Hybrid result",
                    "source_id": "docs.python.org",
                    "url": "https://docs.python.org",
                    "title": "Python Docs",
                    "chunk_index": 5,
                    "score": 0.92,
                },
            ],
        )

        result = await perform_rag_query(
            mock_client,
            query="python tutorial",
            match_count=10,
        )

        # Verify hybrid search was called
        assert mock_client.hybrid_search.called
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["search_type"] == "hybrid"
        assert result_data["match_count"] == 1

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_with_source_filter(self):
        """Test query with source filter"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[])

        await perform_rag_query(
            mock_client,
            query="test",
            source="example.com",
            match_count=5,
        )

        # Verify source_filter was passed
        call_args = mock_client.search.call_args
        assert call_args.kwargs.get("source_filter") == "example.com"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_with_empty_source_filter(self):
        """Test query with empty source filter (should be None)"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[])

        # Test with empty string
        await perform_rag_query(
            mock_client,
            query="test",
            source="   ",  # whitespace only
            match_count=5,
        )

        call_args = mock_client.search.call_args
        assert call_args.kwargs.get("source_filter") is None

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_empty_results(self):
        """Test query with no results"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[])

        result = await perform_rag_query(mock_client, query="no results")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["match_count"] == 0
        assert len(result_data["results"]) == 0

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_missing_score(self):
        """Test handling of results without score field"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(
            return_value=[
                {
                    "content": "Content without score",
                    "source_id": "example.com",
                    "url": "https://example.com",
                    "title": "Test",
                    "chunk_index": 0,
                    # No score field
                },
            ],
        )

        result = await perform_rag_query(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is True
        # Default score should be 0
        assert result_data["results"][0]["similarity_score"] == 0

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_query_error(self):
        """Test handling of QueryError"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(
            side_effect=QueryError("Search query failed"),
        )

        result = await perform_rag_query(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert result_data["query"] == "test"
        assert "Search query failed" in result_data["error"]

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_vector_store_error(self):
        """Test handling of VectorStoreError"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(
            side_effect=VectorStoreError("Vector store unavailable"),
        )

        result = await perform_rag_query(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Vector store unavailable" in result_data["error"]

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "true"})
    async def test_perform_rag_query_hybrid_generic_error(self):
        """Test handling of generic Exception in hybrid search"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.hybrid_search = AsyncMock(
            side_effect=Exception("Unexpected hybrid error"),
        )

        result = await perform_rag_query(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Unexpected hybrid error" in result_data["error"]

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_default_match_count(self):
        """Test query with default match_count parameter"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[])

        await perform_rag_query(mock_client, query="test")

        # Verify default match_count is 5
        call_args = mock_client.search.call_args
        assert call_args.kwargs.get("match_count") == 5


class TestSearchCodeExamples:
    """Test search_code_examples function"""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "false"})
    async def test_search_code_examples_disabled(self):
        """Test when code example extraction is disabled"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()

        result = await search_code_examples(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Code example extraction is disabled" in result_data["error"]
        # Client should not be called
        assert not hasattr(mock_client, "search_code_examples") or not mock_client.search_code_examples.called

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_success(self):
        """Test successful code example search"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(
            return_value=[
                {
                    "code": "def hello():\n    print('Hello')",
                    "summary": "Simple hello function",
                    "source_id": "github.com/example",
                    "url": "https://github.com/example/repo",
                    "programming_language": "python",
                    "score": 0.88,
                },
                {
                    "code": "function greet() { console.log('Hi'); }",
                    "summary": "JavaScript greeting",
                    "source_id": "github.com/jsrepo",
                    "url": "https://github.com/jsrepo",
                    "programming_language": "javascript",
                    "score": 0.75,
                },
            ],
        )

        result = await search_code_examples(
            mock_client,
            query="greeting function",
            match_count=10,
        )

        # Verify
        assert mock_client.search_code_examples.called
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["query"] == "greeting function"
        assert result_data["match_count"] == 2
        assert result_data["search_type"] == "code_examples"
        assert len(result_data["results"]) == 2
        assert result_data["results"][0]["programming_language"] == "python"
        assert result_data["results"][1]["programming_language"] == "javascript"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_with_source_filter(self):
        """Test code search with source_id filter"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(return_value=[])

        await search_code_examples(
            mock_client,
            query="test",
            source_id="github.com/myrepo",
            match_count=5,
        )

        # Verify filter was constructed
        call_args = mock_client.search_code_examples.call_args
        filter_metadata = call_args.kwargs.get("filter_metadata")
        assert filter_metadata == {"source_id": "github.com/myrepo"}

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_with_empty_source_filter(self):
        """Test code search with empty source_id (no filter applied)"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(return_value=[])

        await search_code_examples(
            mock_client,
            query="test",
            source_id="   ",  # whitespace only
            match_count=5,
        )

        # Verify no filter was applied
        call_args = mock_client.search_code_examples.call_args
        filter_metadata = call_args.kwargs.get("filter_metadata")
        assert filter_metadata is None

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_empty_results(self):
        """Test code search with no results"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(return_value=[])

        result = await search_code_examples(mock_client, query="no results")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["match_count"] == 0
        assert len(result_data["results"]) == 0

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_missing_score(self):
        """Test handling of code results without score field"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(
            return_value=[
                {
                    "code": "print('test')",
                    "summary": "Test code",
                    "source_id": "example.com",
                    "url": "https://example.com",
                    "programming_language": "python",
                    # No score field
                },
            ],
        )

        result = await search_code_examples(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is True
        # Default score should be 0
        assert result_data["results"][0]["similarity_score"] == 0

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_query_error(self):
        """Test handling of QueryError in code search"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(
            side_effect=QueryError("Code search failed"),
        )

        result = await search_code_examples(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert result_data["query"] == "test"
        assert "Code search failed" in result_data["error"]

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_vector_store_error(self):
        """Test handling of VectorStoreError in code search"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(
            side_effect=VectorStoreError("Vector store error"),
        )

        result = await search_code_examples(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Vector store error" in result_data["error"]

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_generic_error(self):
        """Test handling of generic Exception in code search"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(
            side_effect=Exception("Unexpected code search error"),
        )

        result = await search_code_examples(mock_client, query="test")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Unexpected code search error" in result_data["error"]

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_default_match_count(self):
        """Test code search with default match_count parameter"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(return_value=[])

        await search_code_examples(mock_client, query="test")

        # Verify default match_count is 5
        call_args = mock_client.search_code_examples.call_args
        assert call_args.kwargs.get("match_count") == 5

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_none_source_id(self):
        """Test code search with None source_id"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(return_value=[])

        await search_code_examples(
            mock_client,
            query="test",
            source_id=None,
            match_count=5,
        )

        # Verify no filter was applied
        call_args = mock_client.search_code_examples.call_args
        filter_metadata = call_args.kwargs.get("filter_metadata")
        assert filter_metadata is None


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_all_fields_present(self):
        """Test that all expected fields are present in successful response"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(
            return_value=[
                {
                    "content": "Full content",
                    "source_id": "example.com",
                    "url": "https://example.com",
                    "title": "Example Title",
                    "chunk_index": 3,
                    "score": 0.91,
                },
            ],
        )

        result = await perform_rag_query(
            mock_client,
            query="test query",
            source="example.com",
            match_count=10,
        )

        result_data = json.loads(result)

        # Verify top-level fields
        assert "success" in result_data
        assert "query" in result_data
        assert "source_filter" in result_data
        assert "match_count" in result_data
        assert "results" in result_data
        assert "search_type" in result_data

        # Verify result fields
        result_item = result_data["results"][0]
        assert "content" in result_item
        assert "source" in result_item
        assert "url" in result_item
        assert "title" in result_item
        assert "chunk_index" in result_item
        assert "similarity_score" in result_item

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_all_fields_present(self):
        """Test that all expected fields are present in code search response"""
        from src.database.rag_queries import search_code_examples

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(
            return_value=[
                {
                    "code": "def test(): pass",
                    "summary": "Test function",
                    "source_id": "github.com",
                    "url": "https://github.com/test",
                    "programming_language": "python",
                    "score": 0.95,
                },
            ],
        )

        result = await search_code_examples(
            mock_client,
            query="test function",
            source_id="github.com",
            match_count=3,
        )

        result_data = json.loads(result)

        # Verify top-level fields
        assert "success" in result_data
        assert "query" in result_data
        assert "source_filter" in result_data
        assert "match_count" in result_data
        assert "results" in result_data
        assert "search_type" in result_data
        assert result_data["search_type"] == "code_examples"

        # Verify code result fields
        result_item = result_data["results"][0]
        assert "code" in result_item
        assert "summary" in result_item
        assert "source_id" in result_item
        assert "url" in result_item
        assert "programming_language" in result_item
        assert "similarity_score" in result_item

    @pytest.mark.asyncio
    async def test_get_available_sources_partial_data(self):
        """Test sources with missing optional fields"""
        from src.database.rag_queries import get_available_sources

        mock_client = MagicMock()
        mock_client.get_sources = AsyncMock(
            return_value=[
                {
                    "source_id": "example.com",
                    # Missing summary, total_chunks, etc.
                },
            ],
        )

        result = await get_available_sources(mock_client)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["count"] == 1
        # Should handle missing fields gracefully
        assert result_data["sources"][0]["source_id"] == "example.com"
        assert result_data["sources"][0].get("summary") is None

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_HYBRID_SEARCH": "false"})
    async def test_perform_rag_query_source_filter_variations(self):
        """Test various source filter input variations"""
        from src.database.rag_queries import perform_rag_query

        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[])

        # Test with leading/trailing whitespace
        await perform_rag_query(
            mock_client,
            query="test",
            source="  example.com  ",
        )

        call_args = mock_client.search.call_args
        # Should be trimmed
        assert call_args.kwargs.get("source_filter") == "example.com"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"})
    async def test_search_code_examples_large_code_block(self):
        """Test code search with large code blocks"""
        from src.database.rag_queries import search_code_examples

        large_code = "def function():\n" + "    pass\n" * 1000

        mock_client = MagicMock()
        mock_client.search_code_examples = AsyncMock(
            return_value=[
                {
                    "code": large_code,
                    "summary": "Large function",
                    "source_id": "example.com",
                    "url": "https://example.com",
                    "programming_language": "python",
                    "score": 0.8,
                },
            ],
        )

        result = await search_code_examples(mock_client, query="large function")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["results"][0]["code"]) > 1000
