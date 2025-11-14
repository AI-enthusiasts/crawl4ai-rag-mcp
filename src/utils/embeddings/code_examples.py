"""Code example management utilities for embeddings."""

from typing import Any

from anyio.to_thread import run_sync as run_in_thread

from src.core.logging import logger

from .basic import create_embedding, create_embeddings_batch


async def add_code_examples_to_database(
    database: Any,  # VectorDatabase instance
    urls: list[str],
    chunk_numbers: list[int],
    code_examples: list[str],
    summaries: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int = 20,
) -> None:
    """
    Add code examples to the database with embeddings.

    Args:
        database: VectorDatabase instance (the database adapter)
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code examples
        summaries: List of summaries for the code examples
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return  # Early return for empty lists

    # Generate embeddings for summaries in batches

    embeddings = []
    for i in range(0, len(summaries), batch_size):
        batch_texts = summaries[i : i + batch_size]
        # Run in thread to avoid blocking event loop
        batch_embeddings = await run_in_thread(create_embeddings_batch, batch_texts)
        embeddings.extend(batch_embeddings)

    # Store code examples with embeddings using the database adapter
    await database.add_code_examples(
        urls=urls,
        chunk_numbers=chunk_numbers,
        code_examples=code_examples,
        summaries=summaries,
        metadatas=metadatas,
        embeddings=embeddings,
    )


async def search_code_examples(
    database: Any,  # VectorDatabase instance
    query: str,
    match_count: int = 5,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search for code examples using vector similarity with enhanced query.

    Args:
        database: VectorDatabase instance (the database adapter)
        query: Search query text
        match_count: Maximum number of results to return
        source_filter: Optional source ID filter

    Returns:
        List of code examples with similarity scores
    """
    # Enhance the query for code search
    enhanced_query = f"Code example for {query}"

    # Generate embedding for the enhanced query
    # Run in thread to avoid blocking event loop
    query_embedding = await run_in_thread(create_embedding, enhanced_query)

    # Search using the database adapter
    return await database.search_code_examples(
        query_embedding=query_embedding,
        match_count=match_count,
        source_filter=source_filter,
    )
