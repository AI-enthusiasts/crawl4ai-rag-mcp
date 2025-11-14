"""
Qdrant operations module.

CRUD operations for documents and sources in Qdrant vector database.
All functions are standalone and accept QdrantClient as first parameter.
"""

import sys
import uuid
from datetime import UTC
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList, PointStruct

# Constants
CRAWLED_PAGES = "crawled_pages"
CODE_EXAMPLES = "code_examples"
SOURCES = "sources"
BATCH_SIZE = 100


def _generate_point_id(url: str, chunk_number: int) -> str:
    """Generate a deterministic UUID for a document point"""
    id_string = f"{url}_{chunk_number}"
    # Use uuid5 to generate a deterministic UUID from the URL and chunk number
    return str(uuid.uuid5(uuid.NAMESPACE_URL, id_string))


async def add_documents(
    client: AsyncQdrantClient,
    urls: list[str],
    chunk_numbers: list[int],
    contents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    source_ids: list[str] | None = None,
) -> None:
    """Add documents to Qdrant"""
    if source_ids is None:
        source_ids = [""] * len(urls)

    # First, delete any existing documents with the same URLs
    unique_urls = list(set(urls))
    for url in unique_urls:
        try:
            await delete_documents_by_url(client, [url])
        except Exception as e:
            print(f"Error deleting documents from Qdrant: {e}")

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
            point_id = _generate_point_id(url, chunk_num)

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
                collection_name=CODE_EXAMPLES,
                points=points,
            )
        except Exception as e:
            print(f"Error upserting code examples to Qdrant: {e}")
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
    """Get all document chunks for a specific URL"""
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
        doc = point.payload.copy()
        doc["id"] = point.id
        results.append(doc)

    # Sort by chunk number
    results.sort(key=lambda x: x.get("chunk_number", 0))

    return results


async def delete_documents_by_url(client: AsyncQdrantClient, urls: list[str]) -> None:
    """Delete all document chunks for the given URLs"""
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


async def add_source(
    client: AsyncQdrantClient,
    source_id: str,
    url: str,
    title: str,
    description: str,
    metadata: dict[str, Any],
    embedding: list[float],
) -> None:
    """Add a source to Qdrant"""
    # Generate a deterministic UUID from source_id
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id))

    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            "source_id": source_id,
            "url": url,
            "title": title,
            "description": description,
            "metadata": metadata or {},
        },
    )

    await client.upsert(
        collection_name=SOURCES,
        points=[point],
    )


async def search_sources(
    client: AsyncQdrantClient,
    query_embedding: list[float],
    match_count: int = 10,
) -> list[dict[str, Any]]:
    """Search for similar sources"""
    results = await client.search(
        collection_name=SOURCES,
        query_vector=query_embedding,
        query_filter=None,
        limit=match_count,
    )

    # Format results
    formatted_results = []
    for result in results:
        doc = result.payload.copy()
        doc["similarity"] = result.score  # Interface expects "similarity"
        doc["id"] = result.id
        formatted_results.append(doc)

    return formatted_results


async def update_source(
    client: AsyncQdrantClient,
    source_id: str,
    updates: dict[str, Any],
) -> None:
    """Update a source's metadata"""
    # Generate a deterministic UUID from source_id
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id))

    # Get existing source
    try:
        existing_points = await client.retrieve(
            collection_name=SOURCES,
            ids=[point_id],
        )

        if not existing_points:
            msg = f"Source {source_id} not found"
            raise ValueError(msg)

        # Update payload
        existing_point = existing_points[0]
        updated_payload = existing_point.payload.copy()
        updated_payload.update(updates)

        # Update the point
        await client.set_payload(
            collection_name=SOURCES,
            payload=updated_payload,
            points=[point_id],
        )
    except Exception as e:
        print(f"Error updating source: {e}", file=sys.stderr)
        raise


async def get_sources(client: AsyncQdrantClient) -> list[dict[str, Any]]:
    """
    Get all available sources.

    Returns:
        List of sources, each containing:
        - source_id: Source identifier
        - summary: Source summary
        - total_word_count: Total word count
        - created_at: Creation timestamp
        - updated_at: Update timestamp
    """
    try:
        # Scroll through all points in the sources collection
        all_sources = []
        offset = None
        limit = 100

        while True:
            # Get a batch of sources
            points, next_offset = await client.scroll(
                collection_name=SOURCES,
                offset=offset,
                limit=limit,
                with_payload=True,
            )

            # Format each source
            for point in points:
                source_data = {
                    "source_id": point.payload.get(
                        "source_id",
                        point.id,
                    ),  # Get from payload, fallback to ID
                    "summary": point.payload.get("summary", ""),
                    "total_word_count": point.payload.get("total_word_count", 0),
                    "created_at": point.payload.get("created_at", ""),
                    "updated_at": point.payload.get("updated_at", ""),
                    "enabled": point.payload.get("enabled", True),
                    "url_count": point.payload.get("url_count", 0),
                }
                all_sources.append(source_data)

            # Check if there are more sources
            if next_offset is None:
                break

            offset = next_offset

        # Sort by source_id for consistency
        all_sources.sort(key=lambda x: x["source_id"])

        return all_sources

    except Exception as e:
        print(f"Error getting sources: {e}", file=sys.stderr)
        return []


async def update_source_info(
    client: AsyncQdrantClient,
    source_id: str,
    summary: str,
    word_count: int,
) -> None:
    """
    Update or insert source information.

    Args:
        client: Qdrant client instance
        source_id: Source identifier
        summary: Source summary
        word_count: Word count for this source
    """
    from datetime import datetime

    timestamp = datetime.now(UTC).isoformat()

    try:
        # Generate a deterministic UUID from source_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id))

        # Try to get existing source
        try:
            existing_points = await client.retrieve(
                collection_name=SOURCES,
                ids=[point_id],
            )

            if existing_points:
                # Update existing source
                existing_point = existing_points[0]
                updated_payload = existing_point.payload.copy()
                updated_payload.update(
                    {
                        "summary": summary,
                        "total_word_count": word_count,
                        "updated_at": timestamp,
                    },
                )

                await client.set_payload(
                    collection_name=SOURCES,
                    payload=updated_payload,
                    points=[point_id],
                )
            else:
                # Create new source
                await _create_new_source(
                    client,
                    source_id,
                    summary,
                    word_count,
                    timestamp,
                    point_id,
                )
        except Exception:
            # Source doesn't exist, create new one
            await _create_new_source(
                client,
                source_id,
                summary,
                word_count,
                timestamp,
                point_id,
            )

    except Exception as e:
        print(f"Error updating source info: {e}", file=sys.stderr)
        raise


async def _create_new_source(
    client: AsyncQdrantClient,
    source_id: str,
    summary: str,
    word_count: int,
    timestamp: str,
    point_id: str,
) -> None:
    """Helper method to create a new source"""
    try:
        # Create new source with a deterministic embedding
        # IMPORTANT: This embedding must be 1536 dimensions to match OpenAI's text-embedding-3-small model
        # Previously this was creating 384-dimensional embeddings which caused vector dimension errors

        # Generate a deterministic embedding from the source_id using SHA256 hash
        import hashlib

        hash_object = hashlib.sha256(source_id.encode())
        hash_bytes = hash_object.digest()  # 32 bytes from SHA256

        # Convert hash bytes to floats between -1 and 1
        # Each byte (0-255) is normalized to the range [-1, 1]
        base_embedding = [(b - 128) / 128.0 for b in hash_bytes]

        # OpenAI embeddings are 1536 dimensions, but SHA256 only gives us 32 values
        # We repeat the pattern to fill all 1536 dimensions deterministically
        embedding: list[float] = []
        while len(embedding) < 1536:
            embedding.extend(base_embedding)

        # Ensure exactly 1536 dimensions (trim any excess from the last repetition)
        embedding = embedding[:1536]

        points = [
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "source_id": source_id,
                    "summary": summary,
                    "total_word_count": word_count,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "enabled": True,
                    "url_count": 1,
                },
            ),
        ]

        await client.upsert(
            collection_name=SOURCES,
            points=points,
        )
    except Exception as e:
        print(f"Error creating new source: {e}", file=sys.stderr)
        raise
