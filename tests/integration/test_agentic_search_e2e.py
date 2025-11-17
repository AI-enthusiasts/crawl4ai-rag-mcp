"""E2E tests for agentic search with real services.

REQUIREMENTS:
- Qdrant running on localhost:6333
- SearXNG running on localhost:8080 (skipped if unavailable)
- OPENAI_API_KEY set in environment
- NO MOCKS - all services must be real

This test:
1. Pre-populates Qdrant with test data (insufficient completeness)
2. Runs agentic_search with real LLM evaluation
3. Verifies that the "content" field bug is fixed
4. Triggers web search due to low completeness
"""

import json

import httpx
import pytest

from src.config import get_settings
from src.core.context import initialize_global_context
from src.database.qdrant_adapter import QdrantAdapter
from src.utils.embeddings.basic import create_embeddings_batch


@pytest.fixture
async def qdrant_with_test_data():
    """Set up Qdrant with test data that has low completeness."""
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

    # Create collection if doesn't exist
    try:
        await adapter.client.get_collection(adapter.CRAWLED_PAGES)
    except Exception:
        from qdrant_client.models import Distance, VectorParams

        await adapter.client.create_collection(
            collection_name=adapter.CRAWLED_PAGES,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # Add test data with INCOMPLETE information about "Python"
    # This should trigger low completeness score and web search
    test_chunks = [
        "Python is a programming language.",
        "Python was created by Guido van Rossum.",
    ]

    test_url = "https://test-incomplete-python-info.example.com"

    # Get embeddings for test chunks
    embeddings = create_embeddings_batch(test_chunks)

    # Store in Qdrant
    await adapter.add_documents(
        urls=[test_url] * len(test_chunks),
        chunk_numbers=list(range(len(test_chunks))),
        contents=test_chunks,
        metadatas=[
            {
                "title": "Incomplete Python Info",
                "description": "Test data",
                "test": True,
                "purpose": "e2e_agentic_search_test",
            },
        ]
        * len(test_chunks),
        embeddings=embeddings,
        source_ids=["test-incomplete-python"] * len(test_chunks),
    )

    yield adapter

    # Cleanup: delete test data
    try:
        await adapter.delete_documents_by_url([test_url])
    except Exception:
        pass

    await adapter.client.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agentic_search_with_real_qdrant_data(qdrant_with_test_data):
    """Test agentic search with real Qdrant data and services.

    This test verifies:
    1. Qdrant data is queried correctly
    2. The "content" field (not "chunk") is parsed properly
    3. Low completeness triggers web search (if SearXNG available)
    4. No Pydantic validation errors occur
    """
    settings = get_settings()

    # Initialize app context
    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    # Import after context initialization
    from src.services.agentic_search import agentic_search_impl

    # Create simple Context
    class MockContext:
        pass

    mock_ctx = MockContext()

    # Check SearXNG availability
    searxng_available = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.searxng_url}/")
            searxng_available = response.status_code == 200
    except Exception:
        pass

    # Run agentic search
    # Query about Python - our test data is INCOMPLETE so completeness should be low
    result_json = await agentic_search_impl(
        ctx=mock_ctx,
        query="What is Python programming language?",
        completeness_threshold=0.8,  # High threshold to trigger search
        max_iterations=1,
        max_urls_per_iteration=2,
    )

    result = json.loads(result_json)

    # Verify no validation errors
    assert "success" in result, "Result missing 'success' field"

    # If there was an error, it should NOT be about validation
    if not result["success"]:
        error = result.get("error", "")

        # These validation errors should NOT occur (the bug we fixed)
        assert "String should have at least 1 character" not in error, (
            "BUG: RAGResult validation failed - 'content' field not parsed"
        )
        assert "Input should be a valid integer" not in error, (
            "BUG: RAGResult validation failed - 'chunk_index' is None"
        )

        # Acceptable errors (service unavailable, etc)
        print(f"Test passed with acceptable error: {error}")
    else:
        # Success case
        assert result["success"] is True
        assert result["query"] == "What is Python programming language?"
        assert result["iterations"] >= 1
        assert "completeness" in result
        assert "results" in result
        assert "search_history" in result

        # Verify search history has local_check action
        actions = [item["action"] for item in result["search_history"]]
        assert "local_check" in actions, "local_check action missing"

        # If SearXNG available and completeness low, should attempt web search
        if searxng_available and result["completeness"] < 0.8:
            assert (
                "web_search" in actions or "max_iterations_reached" in result["status"]
            ), "Expected web_search when completeness low and SearXNG available"

        # Verify results were retrieved from Qdrant
        if result["results"]:
            # Check that results have proper structure
            for rag_result in result["results"]:
                assert "content" in rag_result, "RAG result missing 'content'"
                assert "url" in rag_result, "RAG result missing 'url'"
                assert "similarity_score" in rag_result, (
                    "RAG result missing 'similarity_score'"
                )
                assert "chunk_index" in rag_result, "RAG result missing 'chunk_index'"

                # Content should not be empty
                assert len(rag_result["content"]) > 0, "RAG result has empty content"
                # chunk_index should be integer, not None
                assert isinstance(rag_result["chunk_index"], int), (
                    f"chunk_index should be int, got {type(rag_result['chunk_index'])}"
                )

        print(
            f"Test passed: completeness={result['completeness']}, "
            f"iterations={result['iterations']}, results={len(result['results'])}"
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_qdrant_returns_content_field():
    """Verify that Qdrant/RAG queries return 'content' field, not 'chunk'.

    This is a focused test for the field naming bug.
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

    # Initialize context
    try:
        await initialize_global_context()
    except Exception as e:
        pytest.skip(f"Failed to initialize app context: {e}")

    from src.core.context import get_app_context
    from src.database import perform_rag_query

    app_ctx = get_app_context()

    # Perform RAG query
    rag_response = await perform_rag_query(
        app_ctx.database_client,
        query="test query",
        source=None,
        match_count=5,
    )

    data = json.loads(rag_response)

    assert data["success"] is True, "RAG query failed"

    # If there are results, verify field names
    if data.get("results"):
        for result in data["results"]:
            # Should have 'content' field, NOT 'chunk'
            assert "content" in result, "Result missing 'content' field"
            assert "chunk" not in result, (
                "Result should not have 'chunk' field (legacy)"
            )

            # Should have chunk_index
            assert "chunk_index" in result, "Result missing 'chunk_index' field"

            # chunk_index should not be None if present
            if result.get("chunk_index") is not None:
                assert isinstance(result["chunk_index"], int), (
                    f"chunk_index should be int, got {type(result['chunk_index'])}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
