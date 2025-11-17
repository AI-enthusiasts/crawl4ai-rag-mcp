"""Document management and search utilities for embeddings."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from anyio.to_thread import run_sync as run_in_thread

from src.core.exceptions import EmbeddingError, LLMError
from src.core.logging import logger

from .basic import create_embedding, create_embeddings_batch
from .config import get_contextual_embedding_config
from .contextual import process_chunk_with_context


async def add_documents_to_database(
    database: Any,  # VectorDatabase instance
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

        # Use ThreadPoolExecutor for parallel processing with individual error handling
        with ThreadPoolExecutor(
            max_workers=config["max_workers"],
        ) as executor:
            # Submit tasks individually for better error handling
            future_to_index = {}
            total_chunks = len(contents)

            for i, (url, content) in enumerate(zip(urls, contents, strict=False)):
                full_document = url_to_full_document.get(url, "")
                args = (content, full_document, i, total_chunks)
                future = executor.submit(process_chunk_with_context, args)
                future_to_index[future] = i

            # Process results as they complete, with individual error handling
            contextual_contents = contents.copy()  # Start with original contents
            # Pre-allocate embeddings list with correct size (Python docs: list assignment requires existing index)
            embeddings: list[list[float] | None] = [None] * len(contents)
            successful_contextual_count = 0
            failed_contextual_count = 0

            try:
                for future in as_completed(future_to_index.keys()):
                    index = future_to_index[future]
                    try:
                        contextual_text, embedding = future.result()
                        contextual_contents[index] = contextual_text
                        embeddings[index] = embedding
                        successful_contextual_count += 1
                    except LLMError as e:
                        logger.warning(
                            f"LLM error generating contextual embedding for chunk {index}: {e}. Using original content.",
                        )
                        # Keep original content and generate standard embedding
                        embedding = create_embedding(contents[index])
                        embeddings[index] = embedding
                        failed_contextual_count += 1
                    except EmbeddingError as e:
                        logger.warning(
                            f"Embedding error for chunk {index}: {e}. Using original content.",
                        )
                        # Keep original content and generate standard embedding
                        embedding = create_embedding(contents[index])
                        embeddings[index] = embedding
                        failed_contextual_count += 1
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error generating contextual embedding for chunk {index}: {e}. Using original content.",
                        )
                        # Keep original content and generate standard embedding
                        embedding = create_embedding(contents[index])
                        embeddings[index] = embedding
                        failed_contextual_count += 1

                # Update contents to use contextual versions where successful
                contents = contextual_contents

                # Add contextual embedding flag to metadata for successful ones
                for i, metadata in enumerate(metadatas):
                    metadata["contextual_embedding"] = (
                        embeddings[i] is not None and i < successful_contextual_count
                    )

                logger.info(
                    f"Contextual embedding processing: {successful_contextual_count} successful, {failed_contextual_count} failed",
                )

            except LLMError as e:
                logger.error(
                    f"LLM error during contextual embedding processing: {e}. Falling back to standard embeddings.",
                )
                # Fall back to standard embedding generation for all

                embeddings = []
                for i in range(0, len(contents), batch_size):
                    batch_texts = contents[i : i + batch_size]
                    # Run in thread to avoid blocking event loop
                    batch_embeddings = await run_in_thread(
                        create_embeddings_batch, batch_texts,
                    )
                    embeddings.extend(batch_embeddings)
            except EmbeddingError as e:
                logger.error(
                    f"Embedding error during contextual embedding processing: {e}. Falling back to standard embeddings.",
                )
                # Fall back to standard embedding generation for all

                embeddings = []
                for i in range(0, len(contents), batch_size):
                    batch_texts = contents[i : i + batch_size]
                    # Run in thread to avoid blocking event loop
                    batch_embeddings = await run_in_thread(
                        create_embeddings_batch, batch_texts,
                    )
                    embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(
                    f"Unexpected error during contextual embedding processing: {e}. Falling back to standard embeddings.",
                )
                # Fall back to standard embedding generation for all

                embeddings = []
                for i in range(0, len(contents), batch_size):
                    batch_texts = contents[i : i + batch_size]
                    # Run in thread to avoid blocking event loop
                    batch_embeddings = await run_in_thread(
                        create_embeddings_batch, batch_texts,
                    )
                    embeddings.extend(batch_embeddings)
    else:
        # Generate embeddings for all contents in batches (standard approach)

        embeddings = []
        for i in range(0, len(contents), batch_size):
            batch_texts = contents[i : i + batch_size]
            # Run in thread to avoid blocking event loop
            batch_embeddings = await run_in_thread(create_embeddings_batch, batch_texts)
            embeddings.extend(batch_embeddings)

    # Store documents with embeddings using the provided database adapter
    await database.add_documents(
        urls=urls,
        chunk_numbers=chunk_numbers,
        contents=contents,
        metadatas=metadatas,
        embeddings=embeddings,
        source_ids=source_ids,
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
    database: Any,  # VectorDatabase instance
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
    # Run in thread to avoid blocking event loop
    query_embedding = await run_in_thread(create_embedding, query)

    # Search using the database adapter
    return await database.search_documents(
        query_embedding=query_embedding,
        match_count=match_count,
        filter_metadata=filter_metadata,
        source_filter=source_filter,
    )


async def _add_web_sources_to_database(
    database: Any,
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
        source_data = {}

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
                # Check database adapter type and use appropriate method
                if hasattr(database, "add_source"):
                    # Qdrant adapter - needs embedding
                    # Run in thread to avoid blocking event loop
                    source_embeddings = await run_in_thread(
                        create_embeddings_batch, [data["description"]],
                    )
                    source_embedding = source_embeddings[0]
                    await database.add_source(
                        source_id=source_id,
                        url=data["url"],
                        title=data["title"],
                        description=data["description"],
                        metadata=data["metadata"],
                        embedding=source_embedding,
                    )
                    logger.info(f"Added web source to Qdrant: {source_id}")

                elif hasattr(database, "update_source_info"):
                    # Supabase adapter - simpler interface
                    await database.update_source_info(
                        source_id=source_id,
                        summary=data["description"],
                        word_count=data["word_count"],
                    )
                    logger.info(f"Added web source to Supabase: {source_id}")

                else:
                    logger.warning("Database adapter does not support adding sources")

            except EmbeddingError as e:
                logger.warning(f"Embedding error adding web source {source_id}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error adding web source {source_id}: {e}")

    except EmbeddingError as e:
        logger.error(f"Embedding error adding web sources to database: {e}")
    except Exception as e:
        logger.error(f"Unexpected error adding web sources to database: {e}")
