"""
Qdrant code examples module.

Operations for managing code examples in Qdrant vector database.
All functions are standalone and accept QdrantClient as first parameter.
"""

import uuid
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Condition,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
)

# Constants
CODE_EXAMPLES = "code_examples"
BATCH_SIZE = 100


async def add_code_examples(
    client: AsyncQdrantClient,
    urls: list[str],
    chunk_numbers: list[int],
    code_examples: list[str],
    summaries: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    source_ids: list[str] | None = None,
) -> None:
    """Add code examples to Qdrant"""
    if source_ids is None:
        source_ids = [""] * len(urls)

    # Process in batches
    for i in range(0, len(urls), BATCH_SIZE):
        batch_slice = slice(i, min(i + BATCH_SIZE, len(urls)))
        batch_urls = urls[batch_slice]
        batch_chunks = chunk_numbers[batch_slice]
        batch_code_examples = code_examples[batch_slice]
        batch_summaries = summaries[batch_slice]
        batch_metadatas = metadatas[batch_slice]
        batch_embeddings = embeddings[batch_slice]
        batch_source_ids = source_ids[batch_slice]

        # Create points
        points = []
        for (
            url,
            chunk_num,
            code_example,
            summary,
            metadata,
            embedding,
            source_id,
        ) in zip(
            batch_urls,
            batch_chunks,
            batch_code_examples,
            batch_summaries,
            batch_metadatas,
            batch_embeddings,
            batch_source_ids,
            strict=False,
        ):
            # Generate a unique UUID for code examples using a different namespace
            id_string = f"code_{url}_{chunk_num}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, id_string))

            payload = {
                "url": url,
                "chunk_number": chunk_num,
                "code": code_example,
                "summary": summary,
                "metadata": metadata or {},
            }
            if source_id:
                payload["source_id"] = source_id

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        # Upsert to Qdrant
        await client.upsert(
            collection_name=CODE_EXAMPLES,
            points=points,
        )


async def search_code_examples(
    client: AsyncQdrantClient,
    query: str | list[float] | None = None,
    match_count: int = 10,
    filter_metadata: dict[str, Any] | None = None,
    source_filter: str | None = None,
    # Legacy parameter for backward compatibility
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Search for similar code examples"""
    # Handle backward compatibility - prioritize query_embedding if provided
    if query_embedding is not None:
        final_embedding = query_embedding
    elif query is not None:
        # Generate embedding if query is a string
        if isinstance(query, str):
            from src.utils import create_embedding

            final_embedding = create_embedding(query)
        else:
            final_embedding = query
    else:
        msg = "Either 'query' or 'query_embedding' must be provided"
        raise ValueError(msg)

    # Build filter if needed
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
        search_filter = Filter(must=cast("list[Condition]", filter_conditions))

    # Perform search
    results = await client.search(
        collection_name=CODE_EXAMPLES,
        query_vector=final_embedding,
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


async def delete_code_examples_by_url(client: AsyncQdrantClient, urls: list[str]) -> None:
    """Delete all code examples with the given URLs"""
    for url in urls:
        # First, find all points with this URL
        filter_condition = Filter(
            must=cast("list[Condition]", [FieldCondition(key="url", match=MatchValue(value=url))]),
        )

        points, _ = await client.scroll(
            collection_name=CODE_EXAMPLES,
            scroll_filter=filter_condition,
            limit=1000,
        )

        if points:
            # Delete all points for this URL
            point_ids = [point.id for point in points]
            await client.delete(
                collection_name=CODE_EXAMPLES,
                points_selector=PointIdsList(points=point_ids),
            )


async def search_code_examples_by_keyword(
    client: AsyncQdrantClient,
    keyword: str,
    match_count: int = 10,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search code examples by keyword using scroll API"""
    filter_conditions = []

    # Add keyword filter - search in code content
    filter_conditions.append(
        FieldCondition(
            key="code",
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

    search_filter = Filter(must=cast("list[Condition]", filter_conditions))

    # Use scroll to find matching code examples
    points, _ = await client.scroll(
        collection_name=CODE_EXAMPLES,
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


async def get_repository_code_examples(
    client: AsyncQdrantClient,
    repo_name: str,
    code_type: str | None = None,
    match_count: int = 100,
) -> list[dict[str, Any]]:
    """
    Get all code examples for a specific repository.

    Args:
        client: Qdrant client instance
        repo_name: Repository name to filter by
        code_type: Optional code type filter ('class', 'method', 'function')
        match_count: Maximum number of results

    Returns:
        List of code examples from the repository
    """
    filter_conditions = [
        FieldCondition(
            key="metadata.repository_name",
            match=MatchValue(value=repo_name),
        ),
    ]

    if code_type:
        filter_conditions.append(
            FieldCondition(
                key="metadata.code_type",
                match=MatchValue(value=code_type),
            ),
        )

    search_filter = Filter(must=cast("list[Condition]", filter_conditions))

    points, _ = await client.scroll(
        collection_name=CODE_EXAMPLES,
        scroll_filter=search_filter,
        limit=match_count,
    )

    # Format results
    formatted_results = []
    for point in points:
        if point.payload is None:
            continue
        doc = point.payload.copy()
        doc["id"] = point.id
        formatted_results.append(doc)

    return formatted_results


async def delete_repository_code_examples(
    client: AsyncQdrantClient,
    repo_name: str,
) -> None:
    """
    Delete all code examples for a specific repository.

    Args:
        client: Qdrant client instance
        repo_name: Repository name to delete code examples for
    """
    filter_condition = Filter(
        must=cast("list[Condition]", [
            FieldCondition(
                key="metadata.repository_name",
                match=MatchValue(value=repo_name),
            ),
        ]),
    )

    points, _ = await client.scroll(
        collection_name=CODE_EXAMPLES,
        scroll_filter=filter_condition,
        limit=1000,
    )

    if points:
        # Extract point IDs
        point_ids = [point.id for point in points]

        # Delete the points
        await client.delete(
            collection_name=CODE_EXAMPLES,
            points_selector=PointIdsList(points=point_ids),
        )


async def search_code_by_signature(
    client: AsyncQdrantClient,
    method_name: str,
    class_name: str | None = None,
    repo_filter: str | None = None,
    match_count: int = 10,
) -> list[dict[str, Any]]:
    """
    Search for code examples by method/function signature.

    Args:
        client: Qdrant client instance
        method_name: Name of method or function to search for
        class_name: Optional class name to filter by
        repo_filter: Optional repository name to filter by
        match_count: Maximum number of results

    Returns:
        List of matching code examples
    """
    filter_conditions = [
        FieldCondition(
            key="metadata.name",
            match=MatchValue(value=method_name),
        ),
    ]

    if class_name:
        filter_conditions.append(
            FieldCondition(
                key="metadata.class_name",
                match=MatchValue(value=class_name),
            ),
        )

    if repo_filter:
        filter_conditions.append(
            FieldCondition(
                key="metadata.repository_name",
                match=MatchValue(value=repo_filter),
            ),
        )

    search_filter = Filter(must=cast("list[Condition]", filter_conditions))

    points, _ = await client.scroll(
        collection_name=CODE_EXAMPLES,
        scroll_filter=search_filter,
        limit=match_count,
    )

    # Format results
    formatted_results = []
    for point in points:
        if point.payload is None:
            continue
        doc = point.payload.copy()
        doc["id"] = point.id
        formatted_results.append(doc)

    return formatted_results
