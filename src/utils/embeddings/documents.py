"""Document management and search utilities for embeddings."""

import asyncio
from typing import Any, TypedDict

from src.core.exceptions import EmbeddingError, LLMError
from src.core.logging import logger
from src.database.base import VectorDatabase

from .basic import create_embedding, create_embeddings_batch
from .config import get_contextual_embedding_config
from .contextual import process_chunk_with_context


class SourceData(TypedDict):
    """Structure for source metadata."""

    url: str
    title: str
    description: str
    word_count: int
    metadata: dict[str, Any]


async def add_documents_to_database(
    database: VectorDatabase,
    urls: list[str],
    chunk_numbers: list[int],
    contents: list[str],
    metadatas: list[dict[str, Any]],
    url_to_full_document: dict[str, str] | None = None,
    batch_size: int = 20,
    source_ids: list[str] | None = None,
) -> None:
    """
    Add documents to the database with embeddings.

    This function generates embeddings, stores documents in the vector database,
    and automatically adds source entries for web scraped content.

    Args:
        database: VectorDatabase instance (the database adapter)
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content (optional)
        batch_size: Size of each batch for insertion
        source_ids: Optional list of source IDs
    """
    # Get contextual embedding configuration
    config = get_contextual_embedding_config()
    use_contextual_embeddings = config["enabled"]

    if use_contextual_embeddings and url_to_full_document:
        logger.info("Using contextual embeddings for enhanced retrieval")

        # Process chunks with contextual embeddings using asyncio
        contextual_contents = contents.copy()
        embeddings_dict: dict[int, list[float]] = {}
        successful_contextual_count = 0
        failed_contextual_count = 0
        total_chunks = len(contents)

        try:
            # Process all chunks in parallel with asyncio.gather
            tasks = []
            for i, (url, content) in enumerate(zip(urls, contents, strict=False)):
                full_document = url_to_full_document.get(url, "")
                args = (content, full_document, i, total_chunks)
                tasks.append(process_chunk_with_context(args))

            # Wait for all tasks to complete
            results: list[
                tuple[str, list[float]] | BaseException
            ] = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    logger.warning(
                        f"Error generating contextual embedding for chunk {i}: {result}. Using original content.",
                    )
                    embedding = await create_embedding(contents[i])
                    embeddings_dict[i] = embedding
                    failed_contextual_count += 1
                else:
                    # Result is tuple[str, list[float]] from process_chunk_with_context
                    contextual_text, embedding = result
                    contextual_contents[i] = contextual_text
                    embeddings_dict[i] = embedding
                    successful_contextual_count += 1

            # Update contents to use contextual versions where successful
            contents = contextual_contents

            # Convert dict to list in correct order
            embeddings = [embeddings_dict[i] for i in range(len(contents))]

            # Add contextual embedding flag to metadata for successful ones
            for i, metadata in enumerate(metadatas):
                metadata["contextual_embedding"] = i < successful_contextual_count

            logger.info(
                f"Contextual embedding processing: {successful_contextual_count} successful, {failed_contextual_count} failed",
            )

        except Exception as e:
            logger.error(
                f"Unexpected error during contextual embedding processing: {e}. Falling back to standard embeddings.",
            )
            # Fall back to standard embedding generation for all
            embeddings = []
            for i in range(0, len(contents), batch_size):
                batch_texts = contents[i : i + batch_size]
                batch_embeddings = await create_embeddings_batch(batch_texts)
                embeddings.extend(batch_embeddings)
    else:
        # Generate embeddings for all contents in batches (standard approach)
        embeddings = []
        for i in range(0, len(contents), batch_size):
            batch_texts = contents[i : i + batch_size]
            batch_embeddings = await create_embeddings_batch(batch_texts)
            embeddings.extend(batch_embeddings)

    # Store documents with embeddings using the provided database adapter
    # Use empty list if source_ids not provided
    final_source_ids = source_ids if source_ids is not None else [""] * len(urls)
    await database.add_documents(
        urls=urls,
        chunk_numbers=chunk_numbers,
        contents=contents,
        metadatas=metadatas,
        embeddings=embeddings,
        source_ids=final_source_ids,
    )

    # Add source entries for web scraped content
    if source_ids and url_to_full_document:
        await _add_web_sources_to_database(
            database=database,
            urls=urls,
            source_ids=source_ids,
            url_to_full_document=url_to_full_document,
            contents=contents,
        )


async def search_documents(
    database: VectorDatabase,
    query: str,
    match_count: int = 10,
    filter_metadata: dict[str, Any] | None = None,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search for documents using vector similarity.

    Args:
        database: VectorDatabase instance (the database adapter)
        query: Search query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_filter: Optional source ID filter

    Returns:
        List of documents with similarity scores
    """
    # Generate embedding for the query
    query_embedding = await create_embedding(query)

    # Search using the database adapter
    return await database.search_documents(
        query_embedding=query_embedding,
        match_count=match_count,
        filter_metadata=filter_metadata,
        source_filter=source_filter,
    )


async def _add_web_sources_to_database(
    database: VectorDatabase,
    urls: list[str],
    source_ids: list[str],
    url_to_full_document: dict[str, str],
    contents: list[str],
) -> None:
    """
    Add web sources to the sources table for scraped content.

    Args:
        database: Database adapter
        urls: List of URLs
        source_ids: List of source IDs
        url_to_full_document: Map of URLs to full documents
        contents: List of chunk contents for counting
    """
    try:
        # Group by source_id to create source summaries
        source_data: dict[str, SourceData] = {}

        for _i, (url, source_id) in enumerate(zip(urls, source_ids, strict=False)):
            if source_id and source_id not in source_data:
                # Get full document for this URL
                full_document = url_to_full_document.get(url, "")

                # Count chunks for this source
                chunk_count = sum(1 for sid in source_ids if sid == source_id)

                # Generate a simple summary from first 200 characters
                summary = full_document[:200].strip()
                if len(full_document) > 200:
                    summary += "..."

                # Word count estimation
                word_count = len(full_document.split())

                source_data[source_id] = {
                    "url": url,  # Use the first URL for this source
                    "title": source_id,  # Use source_id as title for web sources
                    "description": summary,
                    "word_count": word_count,
                    "metadata": {
                        "type": "web_scrape",
                        "chunk_count": chunk_count,
                        "total_content_length": len(full_document),
                    },
                }

        # Add each unique source to the database
        for source_id, data in source_data.items():
            try:
                # Add source with vector embedding (all adapters support this now)
                source_embeddings = await create_embeddings_batch([data["description"]])
                source_embedding = source_embeddings[0]
                await database.add_source(
                    source_id=source_id,
                    url=data["url"],
                    title=data["title"],
                    description=data["description"],
                    metadata=data["metadata"],
                    embedding=source_embedding,
                )
                logger.info(f"Added web source: {source_id}")

            except EmbeddingError as e:
                logger.warning(f"Embedding error adding web source {source_id}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error adding web source {source_id}: {e}")

    except EmbeddingError as e:
        logger.error(f"Embedding error adding web sources to database: {e}")
    except Exception as e:
        logger.error(f"Unexpected error adding web sources to database: {e}")
