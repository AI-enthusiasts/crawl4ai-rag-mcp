"""
RAG (Retrieval Augmented Generation) query functionality.

Handles vector search, hybrid search, code example retrieval,
multi-query fusion for improved recall, and recency-based decay.
"""

import asyncio
import json
import logging
import math
import os
import time
from typing import Any

from src.config import get_settings
from src.core.exceptions import QueryError, VectorStoreError

logger = logging.getLogger(__name__)

# Default k parameter for Reciprocal Rank Fusion
# Per RAG best practices: k=60 is standard, balances rank importance
RRF_K_DEFAULT = 60


def calculate_recency_decay(
    crawled_at: float | None,
    half_life_days: float = 14.0,
    min_decay: float = 0.1,
) -> float:
    """Calculate exponential decay factor based on document age.

    Uses exponential decay formula: decay = 0.5 ^ (age / half_life)
    This means after half_life_days, the decay factor is 0.5.

    Args:
        crawled_at: Unix timestamp when document was crawled (None = now)
        half_life_days: Days after which decay factor is 0.5
        min_decay: Minimum decay factor (prevents old docs from being ignored)

    Returns:
        Decay factor between min_decay and 1.0
    """
    if crawled_at is None:
        return 1.0  # No timestamp = assume fresh

    now = time.time()
    age_seconds = now - crawled_at
    age_days = age_seconds / 86400  # Convert to days

    if age_days <= 0:
        return 1.0  # Future or now = no decay

    # Exponential decay: 0.5 ^ (age / half_life)
    decay = math.pow(0.5, age_days / half_life_days)

    # Apply minimum decay floor
    return max(decay, min_decay)


def apply_recency_decay_to_results(
    results: list[dict[str, Any]],
    recency_weight: float = 0.3,
    half_life_days: float = 14.0,
    min_decay: float = 0.1,
) -> list[dict[str, Any]]:
    """Apply recency decay to search results and re-rank.

    Combines semantic similarity with recency using weighted average:
    final_score = (1 - weight) * similarity + weight * recency_factor

    Args:
        results: Search results with 'similarity_score' and 'crawled_at'
        recency_weight: Weight of recency in final score (0-1)
        half_life_days: Half-life for exponential decay
        min_decay: Minimum decay factor

    Returns:
        Results with updated scores, sorted by final score descending
    """
    if not results:
        return results

    for result in results:
        similarity = result.get("similarity_score", 0.0)
        crawled_at = result.get("crawled_at")

        # Calculate recency decay
        recency_factor = calculate_recency_decay(
            crawled_at,
            half_life_days=half_life_days,
            min_decay=min_decay,
        )

        # Weighted combination of similarity and recency
        # Higher recency_weight = more emphasis on fresh content
        final_score = (
            1 - recency_weight
        ) * similarity + recency_weight * recency_factor

        # Store both scores for transparency
        result["recency_factor"] = recency_factor
        result["original_similarity"] = similarity
        result["similarity_score"] = final_score  # Replace with combined score

    # Re-sort by final score
    results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

    return results


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = RRF_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Combine results from multiple queries using Reciprocal Rank Fusion.

    Per RAG best practices: RRF combines results from multiple queries
    by scoring documents based on their ranks across all result lists.

    Formula: RRF_score(d) = sum(1 / (k + rank_i(d))) for all queries i

    Args:
        result_lists: List of result lists from different queries.
                     Each result must have 'url' key for deduplication.
        k: Ranking constant (default 60). Higher k reduces impact of high ranks.

    Returns:
        Combined and re-ranked results with RRF scores.
        Each result includes 'rrf_score' field.
    """
    if not result_lists:
        return []

    # Calculate RRF scores for each document (identified by URL)
    rrf_scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            url = result.get("url", "")
            if not url:
                continue

            # RRF formula: 1 / (k + rank + 1)
            # rank is 0-indexed, so we add 1 to make it 1-indexed
            rrf_scores[url] = rrf_scores.get(url, 0.0) + 1.0 / (k + rank + 1)

            # Keep the result with highest similarity score
            if url not in docs:
                docs[url] = result.copy()
            else:
                current_score = docs[url].get("similarity_score", 0.0)
                new_score = result.get("similarity_score", 0.0)
                if new_score > current_score:
                    docs[url] = result.copy()

    # Sort by RRF score descending
    sorted_urls = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Build final result list with RRF scores
    fused_results = []
    for url in sorted_urls:
        doc = docs[url]
        doc["rrf_score"] = rrf_scores[url]
        fused_results.append(doc)

    logger.info(
        "RRF fusion: %d result lists -> %d unique documents",
        len(result_lists),
        len(fused_results),
    )

    return fused_results


async def perform_multi_query_search(
    database_client: Any,
    queries: list[str],
    source: str | None = None,
    match_count: int = 5,
    use_rrf: bool = True,
    apply_recency: bool = True,
) -> list[dict[str, Any]]:
    """Perform RAG search with multiple queries and combine results.

    Per RAG best practices: Multi-query search improves recall by
    searching with different phrasings and combining results.

    OPTIMIZED: Uses batch embeddings and parallel Qdrant searches
    instead of sequential calls for significant performance improvement.

    Args:
        database_client: The database client instance
        queries: List of query variations to search
        source: Optional source domain to filter results
        match_count: Maximum results per query (default: 5)
        use_rrf: Whether to use RRF for combining (default: True)
        apply_recency: Whether to apply recency decay after fusion (default: True)

    Returns:
        Combined search results (RRF-fused if use_rrf=True, with recency decay)
    """
    if not queries:
        return []

    start_time = time.perf_counter()
    logger.info("Performing multi-query search with %d queries", len(queries))

    # Import here to avoid circular imports
    from src.database.qdrant.search import search_documents
    from src.utils.embeddings.basic import create_embeddings_batch

    # Step 1: Create all embeddings in a single batch API call
    embed_start = time.perf_counter()
    try:
        embeddings = await create_embeddings_batch(queries)
    except Exception as e:
        logger.error("Batch embedding failed: %s", e)
        return []
    embed_time = time.perf_counter() - embed_start
    logger.info(
        "PERF: batch embeddings for %d queries took %.1fs", len(queries), embed_time
    )

    if len(embeddings) != len(queries):
        logger.error(
            "Embedding count mismatch: got %d embeddings for %d queries",
            len(embeddings),
            len(queries),
        )
        return []

    # Step 2: Perform all Qdrant searches in parallel
    search_start = time.perf_counter()

    # Build filter for source if provided
    source_filter = source.strip() if source and source.strip() else None

    async def search_with_embedding(embedding: list[float]) -> list[dict[str, Any]]:
        """Execute single search with pre-computed embedding."""
        try:
            return await search_documents(
                client=database_client.client,
                query_embedding=embedding,
                match_count=match_count,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.warning("Search failed: %s", e)
            return []

    # Execute all searches in parallel
    search_results = await asyncio.gather(
        *[search_with_embedding(emb) for emb in embeddings]
    )
    search_time = time.perf_counter() - search_start
    logger.info("PERF: parallel Qdrant searches took %.1fs", search_time)

    # Collect non-empty results
    all_result_lists: list[list[dict[str, Any]]] = [
        results for results in search_results if results
    ]

    if not all_result_lists:
        logger.warning("No results from any query")
        return []

    # Combine results
    if use_rrf and len(all_result_lists) > 1:
        combined = reciprocal_rank_fusion(all_result_lists)
    elif all_result_lists:
        # Just deduplicate if single query or RRF disabled
        combined = _deduplicate_results(all_result_lists)
    else:
        return []

    # Apply additional recency decay after RRF if enabled
    if apply_recency and combined:
        settings = get_settings()
        if settings.recency_decay_enabled:
            combined = apply_recency_decay_to_results(
                combined,
                recency_weight=settings.recency_decay_weight,
                half_life_days=settings.recency_decay_half_life_days,
                min_decay=settings.recency_decay_min_score,
            )

    total_time = time.perf_counter() - start_time
    logger.info(
        "PERF: multi-query search complete in %.1fs: %d results from %d queries",
        total_time,
        len(combined),
        len(queries),
    )

    return combined


def _deduplicate_results(
    result_lists: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Deduplicate results from multiple queries by URL.

    Args:
        result_lists: List of result lists to deduplicate

    Returns:
        Deduplicated results, keeping highest similarity score per URL
    """
    docs: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for result in results:
            url = result.get("url", "")
            if not url:
                continue

            if url not in docs:
                docs[url] = result
            else:
                current_score = docs[url].get("similarity_score", 0.0)
                new_score = result.get("similarity_score", 0.0)
                if new_score > current_score:
                    docs[url] = result

    # Sort by similarity score descending
    return sorted(
        docs.values(), key=lambda x: x.get("similarity_score", 0.0), reverse=True
    )


async def get_available_sources(database_client: Any) -> str:
    """
    Get all available sources from the sources table.

    This returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics.

    Args:
        database_client: The database client instance

    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Query the sources table
        source_data = await database_client.get_sources()

        # Format the sources with their details
        sources = []
        if source_data:
            for source in source_data:
                sources.append(
                    {
                        "source_id": source.get("source_id"),
                        "summary": source.get("summary"),
                        "total_chunks": source.get("total_chunks"),
                        "first_crawled": source.get("first_crawled"),
                        "last_crawled": source.get("last_crawled"),
                    },
                )

        return json.dumps(
            {
                "success": True,
                "sources": sources,
                "count": len(sources),
                "message": f"Found {len(sources)} unique sources.",
            },
            indent=2,
        )
    except QueryError as e:
        logger.error("Query failed in get_available_sources: %s", e)
        return json.dumps({"success": False, "error": str(e)}, indent=2)
    except Exception as e:
        logger.exception("Unexpected error in get_available_sources: %s", e)
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def perform_rag_query(
    database_client: Any,
    query: str,
    source: str | None = None,
    match_count: int = 5,
) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Args:
        database_client: The database client instance
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"

        # Prepare filter if source is provided and not empty
        # Note: We pass source directly as source_filter, not as filter_metadata
        filter_metadata = None
        source_filter = source.strip() if source and source.strip() else None

        if use_hybrid_search:
            # Use hybrid search (vector + keyword)
            logger.info("Performing hybrid search")
            results = await database_client.hybrid_search(
                query,
                match_count=match_count,
                filter_metadata=filter_metadata,
                source_filter=source_filter,  # Pass source_filter separately
            )
        else:
            # Use standard vector search
            logger.info("Performing standard vector search")
            results = await database_client.search(
                query,
                match_count=match_count,
                filter_metadata=filter_metadata,
                source_filter=source_filter,  # Pass source_filter separately
            )

        # Get recency settings
        settings = get_settings()

        # Group results by URL to avoid duplicates and combine chunks
        url_groups: dict[str, dict[str, Any]] = {}
        for result in results:
            url = result.get("url", "")
            if not url:
                continue

            similarity = result.get("similarity", 0)
            content = result.get("content", "")
            chunk_number = result.get("chunk_number", 0)
            crawled_at = result.get("crawled_at")  # Unix timestamp

            if url not in url_groups:
                url_groups[url] = {
                    "content": content,
                    "source": result.get("source_id"),
                    "url": url,
                    "title": result.get("title"),
                    "chunk_index": chunk_number,
                    "similarity_score": similarity,
                    "crawled_at": crawled_at,  # Preserve for decay calculation
                    "chunks": [(chunk_number, content)],
                }
            else:
                # Add chunk and update score if higher
                url_groups[url]["chunks"].append((chunk_number, content))
                if similarity > url_groups[url]["similarity_score"]:
                    url_groups[url]["similarity_score"] = similarity
                # Keep the most recent crawled_at
                existing_crawled = url_groups[url].get("crawled_at")
                if crawled_at and (
                    not existing_crawled or crawled_at > existing_crawled
                ):
                    url_groups[url]["crawled_at"] = crawled_at

        # Combine chunks for each URL (sorted by chunk number)
        formatted_results = []
        for url_data in url_groups.values():
            chunks = url_data.pop("chunks")
            chunks.sort(key=lambda x: x[0])
            combined_content = "\n\n".join(chunk[1] for chunk in chunks)
            url_data["content"] = combined_content
            formatted_results.append(url_data)

        # Apply recency decay if enabled
        if settings.recency_decay_enabled and formatted_results:
            formatted_results = apply_recency_decay_to_results(
                formatted_results,
                recency_weight=settings.recency_decay_weight,
                half_life_days=settings.recency_decay_half_life_days,
                min_decay=settings.recency_decay_min_score,
            )
            search_type = ("hybrid" if use_hybrid_search else "vector") + "+recency"
        else:
            # Sort by similarity score descending (no decay)
            formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            search_type = "hybrid" if use_hybrid_search else "vector"

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source,
                "match_count": len(formatted_results),
                "results": formatted_results,
                "search_type": search_type,
                "recency_decay_enabled": settings.recency_decay_enabled,
            },
            indent=2,
        )
    except (QueryError, VectorStoreError) as e:
        logger.error("Search failed in perform_rag_query: %s", e)
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)
    except Exception as e:
        logger.exception("Unexpected error in perform_rag_query: %s", e)
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)


async def search_code_examples(
    database_client: Any,
    query: str,
    source_id: str | None = None,
    match_count: int = 5,
) -> str:
    """
    Search for code examples relevant to the query.

    This searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.

    Args:
        database_client: The database client instance
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"

    if not extract_code_examples_enabled:
        return json.dumps(
            {
                "success": False,
                "error": "Code example extraction is disabled. Perform a normal RAG search.",
            },
            indent=2,
        )

    try:
        # Prepare filter if source_id is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source_id": source_id}

        # Search for code examples
        results = await database_client.search_code_examples(
            query,
            match_count=match_count,
            filter_metadata=filter_metadata,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "code": result.get("code"),
                    "summary": result.get("summary"),
                    "source_id": result.get("source_id"),
                    "url": result.get("url"),
                    "programming_language": result.get("programming_language"),
                    "similarity_score": result.get(
                        "similarity", 0
                    ),  # Qdrant returns "similarity"
                },
            )

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source_id,
                "match_count": len(formatted_results),
                "results": formatted_results,
                "search_type": "code_examples",
            },
            indent=2,
        )
    except (QueryError, VectorStoreError) as e:
        logger.error("Search failed in search_code_examples: %s", e)
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)
    except Exception as e:
        logger.exception("Unexpected error in search_code_examples: %s", e)
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)
