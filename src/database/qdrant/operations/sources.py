"""Qdrant source operations.

Operations for managing sources in Qdrant vector database.
"""

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.core.exceptions import QueryError, VectorStoreError

from .utils import SOURCES

logger = logging.getLogger(__name__)


async def add_source(
    client: AsyncQdrantClient,
    source_id: str,
    url: str,
    title: str,
    description: str,
    metadata: dict[str, Any],
    embedding: list[float],
) -> None:
    """Add a source to Qdrant.

    Args:
        client: Qdrant client instance
        source_id: Source identifier
        url: Source URL
        title: Source title
        description: Source description
        metadata: Additional metadata
        embedding: Embedding vector
    """
    # Generate a deterministic UUID from source_id
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id))

    point = models.PointStruct(
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
    """Search for similar sources.

    Args:
        client: Qdrant client instance
        query_embedding: Query embedding vector
        match_count: Maximum number of results to return

    Returns:
        List of similar sources with similarity scores
    """
    results = await client.search(
        collection_name=SOURCES,
        query_vector=query_embedding,
        query_filter=None,
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


async def update_source(
    client: AsyncQdrantClient,
    source_id: str,
    updates: dict[str, Any],
) -> None:
    """Update a source's metadata.

    Args:
        client: Qdrant client instance
        source_id: Source identifier
        updates: Dictionary of fields to update
    """
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
        if existing_point.payload is None:
            updated_payload: dict[str, Any] = {}
        else:
            updated_payload = existing_point.payload.copy()
        updated_payload.update(updates)

        # Update the point
        await client.set_payload(
            collection_name=SOURCES,
            payload=updated_payload,
            points=[point_id],
        )
    except QueryError as e:
        logger.error("Failed to update source: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error updating source: %s", e)
        raise


async def get_sources(client: AsyncQdrantClient) -> list[dict[str, Any]]:
    """Get all available sources.

    Args:
        client: Qdrant client instance

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
                if point.payload is None:
                    continue
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

    except QueryError as e:
        logger.error("Failed to get sources: %s", e)
        return []
    except Exception as e:
        logger.exception("Unexpected error getting sources: %s", e)
        return []


async def update_source_info(
    client: AsyncQdrantClient,
    source_id: str,
    summary: str,
    word_count: int,
) -> None:
    """Update or insert source information.

    Args:
        client: Qdrant client instance
        source_id: Source identifier
        summary: Source summary
        word_count: Word count for this source
    """
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
                if existing_point.payload is None:
                    updated_payload: dict[str, Any] = {}
                else:
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
        except QueryError:
            # Source doesn't exist, create new one
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

    except QueryError as e:
        logger.error("Failed to update source info: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error updating source info: %s", e)
        raise


async def _create_new_source(
    client: AsyncQdrantClient,
    source_id: str,
    summary: str,
    word_count: int,
    timestamp: str,
    point_id: str,
) -> None:
    """Helper method to create a new source.

    Args:
        client: Qdrant client instance
        source_id: Source identifier
        summary: Source summary
        word_count: Word count
        timestamp: ISO format timestamp
        point_id: Pre-generated point ID
    """
    try:
        # Create new source with a deterministic embedding
        # IMPORTANT: This embedding must be 1536 dimensions to match OpenAI's text-embedding-3-small model
        # Previously this was creating 384-dimensional embeddings which caused vector dimension errors

        # Generate a deterministic embedding from the source_id using SHA256 hash
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
    except VectorStoreError as e:
        logger.error("Failed to create new source: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error creating new source: %s", e)
        raise
