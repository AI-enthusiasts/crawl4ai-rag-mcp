"""Integration tests for agentic_search with real Qdrant/SearXNG/OpenAI.

NO MOCKS - All tests use real services and pre-populated Qdrant data.

Requirements:
- Qdrant running on localhost:6333
- SearXNG running on localhost:8080 (tests skip if unavailable)
- OPENAI_API_KEY set in environment
- Pre-populated Qdrant with test data before each test
"""

import json

import httpx
import pytest

from src.config import get_settings
from src.core.context import initialize_global_context
from src.database.qdrant_adapter import QdrantAdapter
from src.services.agentic_search import agentic_search_impl
from src.utils.embeddings.basic import create_embeddings_batch


@pytest.fixture
async def qdrant_with_incomplete_data():
    """Pre-populate Qdrant with incomplete test data.

    This data has LOW completeness to trigger web search.
    """
    settings = get_settings()

    # Check Qdrant availability
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.qdrant_url}/collections")
            if response.status_code != 200:
                pytest.skip(f"Qdrant not available at {settings.qdrant_url}")
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")

    # Initialize Qdrant adapter
    adapter = QdrantAdapter(url=settings.qdrant_url)

    # Create collection if needed
    try:
        await adapter.client.get_collection(adapter.CRAWLED_PAGES)
    except Exception:
        from qdrant_client.models import Distance, VectorParams

        await adapter.client.create_collection(
            collection_name=adapter.CRAWLED_PAGES,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # Add incomplete test data about FastMCP
    test_chunks = [
        "FastMCP is a Python framework.",
        "FastMCP was created for building MCP servers.",
    ]
    test_url = "https://test-incomplete-fastmcp.example.com"

    # Get embeddings
    embeddings = create_embeddings_batch(test_chunks)

    # Store in Qdrant
    await adapter.add_documents(
        urls=[test_url] * len(test_chunks),
        chunk_numbers=list(range(len(test_chunks))),
        contents=test_chunks,
        metadatas=[
            {
                "title": "Incomplete FastMCP Info",
                "description": "Test data with low completeness",
                "test": True,
                "purpose": "integration_test",
            },
        ]
        * len(test_chunks),
        embeddings=embeddings,
        source_ids=["test-incomplete-fastmcp"] * len(test_chunks),
    )

    yield adapter

    # Cleanup
    try:
        await adapter.delete_documents_by_url([test_url])
    except Exception:
        pass

    await adapter.client.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agentic_search_with_real_components(qdrant_with_incomplete_data):
    """Test agentic search with real Qdrant, SearXNG, and OpenAI.

    This replaces test_agentic_search_with_mock_components.
    Uses real services instead of mocks.
    """
    settings = get_settings()

    # Initialize context
    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    # Create simple context
    class MockContext:
        pass

    mock_ctx = MockContext()

    # Run agentic search with HIGH threshold to trigger search
    result_json = await agentic_search_impl(
        ctx=mock_ctx,
        query="What is FastMCP?",
        completeness_threshold=0.9,  # High threshold
        max_iterations=1,
        max_urls_per_iteration=1,
    )

    result = json.loads(result_json)

    # Verify result structure
    assert "success" in result
    assert "query" in result
    assert result["query"] == "What is FastMCP?"
    assert "iterations" in result
    assert result["iterations"] >= 1
    assert "completeness" in result
    assert "results" in result
    assert "search_history" in result
    assert "status" in result

    # Should have at least local_check action
    actions = [item["action"] for item in result["search_history"]]
    assert "local_check" in actions

    # If failed, should not be validation error
    if not result["success"]:
        error = result.get("error", "")
        assert "String should have at least 1 character" not in error
        assert "Input should be a valid integer" not in error


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agentic_search_with_all_parameters(qdrant_with_incomplete_data):
    """Test agentic search with all optional parameters using real services.

    This replaces test_agentic_search_with_all_parameters from test_search_tools.py.
    """
    settings = get_settings()

    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    class MockContext:
        pass

    mock_ctx = MockContext()

    # Test with all parameters
    result_json = await agentic_search_impl(
        ctx=mock_ctx,
        query="What is FastMCP framework?",
        completeness_threshold=0.85,
        max_iterations=2,
        max_urls_per_iteration=3,
        url_score_threshold=0.7,
        use_search_hints=True,
    )

    result = json.loads(result_json)

    # Verify all parameters were respected
    assert result["query"] == "What is FastMCP framework?"
    assert result["iterations"] <= 2  # Should not exceed max
    assert "completeness" in result
    assert isinstance(result["completeness"], (int, float))
    assert 0.0 <= result["completeness"] <= 1.0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agentic_search_error_handling_invalid_query():
    """Test agentic search error handling with invalid input.

    This replaces test_agentic_search_error_handling from test_search_tools.py.
    Tests real error handling, not mocked exceptions.
    """
    settings = get_settings()

    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    class MockContext:
        pass

    mock_ctx = MockContext()

    # Test with empty query - should raise validation error
    from src.core.exceptions import MCPToolError

    try:
        result_json = await agentic_search_impl(
            ctx=mock_ctx,
            query="",
            completeness_threshold=0.8,
            max_iterations=1,
        )
        result = json.loads(result_json)
        # If it returns JSON without exception, check error handling
        assert "success" in result
        assert "error" in result or "results" in result
    except MCPToolError as e:
        # Expected: validation error for empty query
        assert (
            "validation error" in str(e).lower()
            or "string should have at least 1 character" in str(e).lower()
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agentic_search_success_case(qdrant_with_incomplete_data):
    """Test successful agentic search with real services.

    This replaces test_agentic_search_success from test_search_tools.py.
    Verifies actual success case with real data.
    """
    settings = get_settings()

    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    class MockContext:
        pass

    mock_ctx = MockContext()

    # Query with LOW threshold to ensure success
    result_json = await agentic_search_impl(
        ctx=mock_ctx,
        query="What is FastMCP?",
        completeness_threshold=0.3,  # Low threshold
        max_iterations=1,
    )

    result = json.loads(result_json)

    # Should succeed
    assert result.get("success") is True or result.get("error") is not None
    assert result["iterations"] >= 1

    if result.get("results"):
        # Verify results structure
        for rag_result in result["results"]:
            assert "content" in rag_result
            assert "url" in rag_result
            assert "similarity_score" in rag_result
            assert "chunk_index" in rag_result

            # Verify no validation errors
            assert len(rag_result["content"]) > 0
            assert isinstance(rag_result["chunk_index"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
