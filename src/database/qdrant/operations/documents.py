"""Qdrant document operations.

CRUD operations for documents in Qdrant vector database.
"""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList, PointStruct

from src.core.exceptions import QueryError, VectorStoreError

from .utils import BATCH_SIZE, CODE_EXAMPLES, CRAWLED_PAGES, generate_point_id

logger = logging.getLogger(__name__)


async def add_documents(
    client: AsyncQdrantClient,
    urls: list[str],
    chunk_numbers: list[int],
    contents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    source_ids: list[str] | None = None,
) -> None:
    """Add documents to Qdrant.

    Args:
        client: Qdrant client instance
        urls: List of document URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of metadata dictionaries
        embeddings: List of embedding vectors
        source_ids: Optional list of source identifiers
    """
    if source_ids is None:
        source_ids = [""] * len(urls)

    # First, delete any existing documents with the same URLs
    unique_urls = list(set(urls))
    for url in unique_urls:
        try:
            await delete_documents_by_url(client, [url])
        except (QueryError, VectorStoreError) as e:
            logger.error(f"Failed to delete existing documents for {url}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error deleting documents from Qdrant: {e}")

    # Process documents in batches
    for i in range(0, len(urls), BATCH_SIZE):
        batch_slice = slice(i, min(i + BATCH_SIZE, len(urls)))
        batch_urls = urls[batch_slice]
        batch_chunks = chunk_numbers[batch_slice]
        batch_contents = contents[batch_slice]
        batch_metadatas = metadatas[batch_slice]
        batch_embeddings = embeddings[batch_slice]
        batch_source_ids = source_ids[batch_slice]

        # Create points for Qdrant
        points = []
        for _j, (
            url,
            chunk_num,
            content,
            metadata,
            embedding,
            source_id,
        ) in enumerate(
            zip(
                batch_urls,
                batch_chunks,
                batch_contents,
                batch_metadatas,
                batch_embeddings,
                batch_source_ids,
                strict=False,
            ),
        ):
            point_id = generate_point_id(url, chunk_num)

            # Extract source_id from URL if not provided - same logic as Supabase adapter
            if not source_id:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                source_id = parsed_url.netloc or parsed_url.path
                # Remove 'www.' prefix if present for consistency
                if source_id and source_id.startswith("www."):
                    source_id = source_id[4:]

            # Prepare payload - always include source_id
            payload = {
                "url": url,
                "chunk_number": chunk_num,
                "content": content,
                "metadata": metadata or {},
                "source_id": source_id,  # Always include source_id
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        # Upsert batch to Qdrant
        try:
            await client.upsert(
                collection_name=CRAWLED_PAGES,
                points=points,
            )
        except VectorStoreError as e:
            logger.error(f"Failed to upsert documents to Qdrant: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error upserting documents to Qdrant: {e}")
            raise


async def url_exists(client: AsyncQdrantClient, url: str) -> bool:
    """Check if URL exists in database (efficient existence check).

    Uses count() instead of scroll() for performance.
    Per Qdrant docs: count() only returns number, not point data.

    Args:
        client: Qdrant client instance
        url: URL to check

    Returns:
        True if URL exists, False otherwise
    """
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="url",
                match=MatchValue(value=url),
            ),
        ],
    )

    # Use count for efficient existence check
    count_result = await client.count(
        collection_name=CRAWLED_PAGES,
        count_filter=filter_condition,
        exact=False,  # Approximate count is fine for existence check
    )

    return count_result.count > 0


async def get_documents_by_url(client: AsyncQdrantClient, url: str) -> list[dict[str, Any]]:
    """Get all document chunks for a specific URL.

    Args:
        client: Qdrant client instance
        url: URL to retrieve documents for

    Returns:
        List of document dictionaries sorted by chunk number
    """
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="url",
                match=MatchValue(value=url),
            ),
        ],
    )

    # Use scroll to get all chunks
    points, _ = await client.scroll(
        collection_name=CRAWLED_PAGES,
        scroll_filter=filter_condition,
        limit=1000,  # Large limit to get all chunks
    )

    # Format and sort by chunk number
    results = []
    for point in points:
        if point.payload is None:
            continue
        doc = point.payload.copy()
        doc["id"] = point.id
        results.append(doc)

    # Sort by chunk number
    results.sort(key=lambda x: x.get("chunk_number", 0))

    return results


async def delete_documents_by_url(client: AsyncQdrantClient, urls: list[str]) -> None:
    """Delete all document chunks for the given URLs.

    Args:
        client: Qdrant client instance
        urls: List of URLs to delete documents for
    """
    for url in urls:
        # First, find all points with this URL
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="url",
                    match=MatchValue(value=url),
                ),
            ],
        )

        points, _ = await client.scroll(
            collection_name=CRAWLED_PAGES,
            scroll_filter=filter_condition,
            limit=1000,
        )

        if points:
            # Delete all points for this URL
            point_ids = [point.id for point in points]
            await client.delete(
                collection_name=CRAWLED_PAGES,
                points_selector=PointIdsList(points=point_ids),
            )
