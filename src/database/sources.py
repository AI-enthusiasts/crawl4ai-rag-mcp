"""
Source management functionality for the database.

Handles source tracking, metadata, and statistics.
"""

import logging
from datetime import datetime
from typing import Any

from src.core.exceptions import QueryError

logger = logging.getLogger(__name__)


async def update_source_summary(
    database_client: Any,
    source_id: str,
    total_chunks: int,
    last_crawled: datetime | None = None,
) -> None:
    """
    Update or create a source summary in the database.

    Args:
        database_client: The database client instance
        source_id: The source identifier (usually domain name)
        total_chunks: Total number of chunks for this source
        last_crawled: When the source was last crawled (defaults to now)
    """
    try:
        if last_crawled is None:
            last_crawled = datetime.utcnow()

        await database_client.update_source(
            source_id=source_id,
            total_chunks=total_chunks,
            last_crawled=last_crawled,
        )
        logger.info("Updated source summary for %s", source_id)
    except QueryError as e:
        logger.error("Failed to update source summary for %s: %s", source_id, e)
        raise
    except Exception as e:
        logger.exception("Unexpected error updating source summary for %s: %s", source_id, e)
        raise


async def get_source_statistics(
    database_client: Any,
    source_id: str,
) -> dict[str, Any] | None:
    """
    Get statistics for a specific source.

    Args:
        database_client: The database client instance
        source_id: The source identifier

    Returns:
        Dictionary with source statistics or None if not found
    """
    try:
        sources = await database_client.get_sources()
        for source in sources:
            if source.get("source_id") == source_id:
                return source  # type: ignore[no-any-return]
        return None
    except QueryError as e:
        logger.error("Failed to get source statistics for %s: %s", source_id, e)
        raise
    except Exception as e:
        logger.exception("Unexpected error getting source statistics for %s: %s", source_id, e)
        raise


async def list_all_sources(database_client: Any) -> list[dict[str, Any]]:
    """
    List all sources in the database with their metadata.

    Args:
        database_client: The database client instance

    Returns:
        List of source dictionaries
    """
    try:
        sources = await database_client.get_sources()
        return sources or []
    except QueryError as e:
        logger.error("Failed to list sources: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error listing sources: %s", e)
        raise
