"""
Test helpers for Qdrant integration tests.

Provides simple helper functions for testing Qdrant operations
without cluttering production code.
"""

from typing import Any

from src.utils.embeddings import (
    add_code_examples_to_database,
    add_documents_to_database,
    search_documents,
)


async def store_crawled_page(
    client: Any,
    url: str,
    content: str,
    metadata: dict[str, Any],
) -> None:
    """
    Store a single crawled page in the database.

    Test helper that wraps add_documents_to_database for single documents.

    Args:
        client: Database adapter instance (QdrantAdapter or SupabaseAdapter)
        url: URL of the page
        content: Page content
        metadata: Additional metadata
    """
    await add_documents_to_database(
        database=client,
        urls=[url],
        chunk_numbers=[0],
        contents=[content],
        metadatas=[metadata],
    )


async def search_crawled_pages(
    client: Any,
    query: str,
    source: str | None = None,
    limit: int = 10,
    filter_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Search for crawled pages using vector similarity.

    Test helper that wraps search_documents.

    Args:
        client: Database adapter instance
        query: Search query text
        source: Optional source filter
        limit: Maximum number of results
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents with similarity scores
    """
    # Build filter if source is provided
    if source and filter_metadata is None:
        filter_metadata = {"source": source}
    elif source and filter_metadata is not None:
        filter_metadata["source"] = source

    return await search_documents(
        database=client,
        query=query,
        match_count=limit,
        filter_metadata=filter_metadata,
    )


async def store_code_example(
    client: Any,
    url: str,
    code_example: str,
    summary: str,
    metadata: dict[str, Any],
) -> None:
    """
    Store a single code example in the database.

    Test helper that wraps add_code_examples_to_database.

    Args:
        client: Database adapter instance
        url: URL where the code was found
        code_example: The code example text
        summary: Summary of what the code does
        metadata: Additional metadata
    """
    await add_code_examples_to_database(
        database=client,
        urls=[url],
        chunk_numbers=[0],
        code_examples=[code_example],
        summaries=[summary],
        metadatas=[metadata],
    )
