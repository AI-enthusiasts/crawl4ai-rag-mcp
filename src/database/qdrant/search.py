"""
Qdrant search module.

Search operations for documents in Qdrant vector database.
All functions are standalone and accept QdrantClient as first parameter.
"""

from typing import Any, Sequence, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

# Constants
CRAWLED_PAGES = "crawled_pages"


async def search_documents(
    client: AsyncQdrantClient,
    query_embedding: list[float],
    match_count: int = 10,
    filter_metadata: dict[str, Any] | None = None,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar documents"""
    # Build filter conditions
    filter_conditions = []

    if filter_metadata:
        for key, value in filter_metadata.items():
            filter_conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                ),
            )

    if source_filter:
        filter_conditions.append(
            FieldCondition(
                key="source_id",  # Changed from metadata.source to source_id
                match=MatchValue(value=source_filter),
            ),
        )

    # Create filter if conditions exist
    search_filter = None
    if filter_conditions:
        search_filter = Filter(must=cast(Sequence[Condition], filter_conditions))

    # Perform search
    results = await client.search(
        collection_name=CRAWLED_PAGES,
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=match_count,
    )

    # Format results
    formatted_results = []
    for result in results:
        if result.payload is None:
            continue
        doc = result.payload.copy()
        doc["similarity"] = result.score  # Interface expects "similarity"
        doc["id"] = result.id
        formatted_results.append(doc)

    return formatted_results


async def search_documents_by_keyword(
    client: AsyncQdrantClient,
    keyword: str,
    match_count: int = 10,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search documents by keyword using scroll API"""
    filter_conditions = []

    # Add keyword filter - search in content
    filter_conditions.append(
        FieldCondition(
            key="content",
            match=MatchValue(value=keyword),
        ),
    )

    if source_filter:
        filter_conditions.append(
            FieldCondition(
                key="source_id",  # Changed from metadata.source to source_id
                match=MatchValue(value=source_filter),
            ),
        )

    search_filter = Filter(must=cast(Sequence[Condition], filter_conditions))

    # Use scroll to find matching documents
    points, _ = await client.scroll(
        collection_name=CRAWLED_PAGES,
        scroll_filter=search_filter,
        limit=match_count,
    )

    # Format results
    formatted_results = []
    for point in points[:match_count]:
        if point.payload is None:
            continue
        doc = point.payload.copy()
        doc["id"] = point.id
        formatted_results.append(doc)

    return formatted_results


async def search(
    client: AsyncQdrantClient,
    query: str,
    match_count: int = 10,
    filter_metadata: dict[str, Any] | None = None,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Generic search method that generates embeddings internally.

    Args:
        client: Qdrant client instance
        query: Search query string
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_filter: Optional source filter

    Returns:
        List of matching documents with similarity scores
    """
    # Generate embedding for the query
    from src.utils import create_embedding

    query_embedding = create_embedding(query)

    # Delegate to the existing search_documents method
    return await search_documents(
        client=client,
        query_embedding=query_embedding,
        match_count=match_count,
        filter_metadata=filter_metadata,
        source_filter=source_filter,
    )


async def hybrid_search(
    client: AsyncQdrantClient,
    query: str,
    match_count: int = 10,
    filter_metadata: dict[str, Any] | None = None,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid search combining vector similarity and keyword matching.

    Args:
        client: Qdrant client instance
        query: Search query string
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_filter: Optional source filter

    Returns:
        List of matching documents combining vector and keyword results
    """
    # Perform vector search
    vector_results = await search(
        client=client,
        query=query,
        match_count=match_count // 2 + 1,  # Get half from vector search
        filter_metadata=filter_metadata,
        source_filter=source_filter,
    )

    # Perform keyword search
    keyword_results = await search_documents_by_keyword(
        client=client,
        keyword=query,
        match_count=match_count // 2 + 1,  # Get half from keyword search
        source_filter=source_filter,
    )

    # Combine and deduplicate results
    combined_results = {}

    # Add vector results with their similarity scores
    for result in vector_results:
        doc_id = result.get("id", result.get("url", ""))
        if doc_id:
            result["search_type"] = "vector"
            result["combined_score"] = (
                result.get("similarity", 0.0) * 0.7
            )  # Weight vector search more
            combined_results[doc_id] = result

    # Add keyword results (give them a base similarity score)
    for result in keyword_results:
        doc_id = result.get("id", result.get("url", ""))
        if doc_id:
            if doc_id in combined_results:
                # Document found in both searches - boost the score
                combined_results[doc_id]["combined_score"] += (
                    0.3  # Boost for appearing in both
                )
                combined_results[doc_id]["search_type"] = "hybrid"
            else:
                # Document only found in keyword search
                result["search_type"] = "keyword"
                result["similarity"] = 0.5  # Base similarity for keyword matches
                result["combined_score"] = (
                    0.3  # Lower weight for keyword-only matches
                )
                combined_results[doc_id] = result

    # Sort by combined score and return top results
    final_results = list(combined_results.values())
    final_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

    # Update similarity to reflect combined score and limit results
    for result in final_results[:match_count]:
        result["similarity"] = result.get("combined_score", 0)
        # Remove the temporary combined_score field
        result.pop("combined_score", None)

    return final_results[:match_count]
