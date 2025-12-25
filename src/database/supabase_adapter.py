"""
Supabase adapter implementation for VectorDatabase protocol.

Extracts and refactors existing Supabase functionality.
"""

import logging
import os
import time
from typing import Any, cast
from urllib.parse import urlparse

from supabase import Client, create_client

from src.core.exceptions import QueryError, VectorStoreError

logger = logging.getLogger(__name__)


class SupabaseAdapter:
    """
    Supabase implementation of the VectorDatabase protocol.

    Uses PostgreSQL with pgvector extension for vector similarity search.
    """

    def __init__(self) -> None:
        """Initialize Supabase adapter with environment configuration"""
        self.client: Client | None = None
        self.batch_size = 20  # Default batch size for operations
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial retry delay in seconds

    async def initialize(self) -> None:
        """Initialize Supabase client connection"""
        if self.client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")

            if not url or not key:
                msg = (
                    "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in "
                    "environment variables"
                )
                raise ValueError(msg)

            self.client = create_client(url, key)

    async def add_documents(
        self,
        urls: list[str],
        chunk_numbers: list[int],
        contents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        source_ids: list[str] | None = None,
    ) -> None:
        """Add documents to Supabase with batch processing"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        # Handle None source_ids - create empty strings as placeholders
        if source_ids is None:
            source_ids = [""] * len(urls)

        # Get unique URLs to delete existing records
        unique_urls = list(set(urls))

        # Delete existing records for these URLs
        await self._delete_documents_batch(unique_urls)

        # Process in batches to avoid memory issues
        for i in range(0, len(contents), self.batch_size):
            batch_end = min(i + self.batch_size, len(contents))

            # Prepare batch data
            batch_data = []
            for j in range(i, batch_end):
                # Extract source_id from URL if not provided
                if j < len(source_ids) and source_ids[j]:
                    source_id = source_ids[j]
                else:
                    parsed_url = urlparse(urls[j])
                    source_id = parsed_url.netloc or parsed_url.path

                data = {
                    "url": urls[j],
                    "chunk_number": chunk_numbers[j],
                    "content": contents[j],
                    "metadata": {
                        "chunk_size": len(contents[j]),
                        **(metadatas[j] if j < len(metadatas) else {}),
                    },
                    "source_id": source_id,
                    "embedding": embeddings[j],
                }
                batch_data.append(data)

            # Insert batch with retry logic
            await self._insert_with_retry("crawled_pages", batch_data)

    async def search_documents(
        self,
        query_embedding: list[float],
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents using vector similarity"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            # Build parameters for RPC call
            params = {"query_embedding": query_embedding, "match_count": match_count}

            # Add optional filters
            if filter_metadata:
                params["filter"] = filter_metadata
            if source_filter:
                params["source_filter"] = source_filter

            # Execute search using Supabase RPC function
            result = self.client.rpc("match_crawled_pages", params).execute()
        except (QueryError, VectorStoreError):
            logger.exception("Vector search failed")
            return []
        except Exception:
            logger.exception("Unexpected error searching documents")
            return []
        else:
            return cast(list[dict[str, Any]], result.data) if result.data else []

    async def delete_documents_by_url(self, urls: list[str]) -> None:
        """Delete documents by URL"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        await self._delete_documents_batch(urls)

    async def add_code_examples(
        self,
        urls: list[str],
        chunk_numbers: list[int],
        code_examples: list[str],
        summaries: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        _source_ids: list[str],
    ) -> None:
        """Add code examples to Supabase"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        if not urls:
            return

        # Delete existing records for these URLs
        unique_urls = list(set(urls))
        for url in unique_urls:
            try:
                self.client.table("code_examples").delete().eq("url", url).execute()
            except QueryError:
                logger.exception("Failed to delete existing code examples for %s", url)
            except Exception:
                logger.exception("Unexpected error deleting code examples for %s", url)

        # Process in batches
        for i in range(0, len(urls), self.batch_size):
            batch_end = min(i + self.batch_size, len(urls))

            # Prepare batch data
            batch_data = []
            for j in range(i, batch_end):
                # Extract source_id from URL
                parsed_url = urlparse(urls[j])
                source_id = parsed_url.netloc or parsed_url.path

                batch_data.append(
                    {
                        "url": urls[j],
                        "chunk_number": chunk_numbers[j],
                        "content": code_examples[j],
                        "summary": summaries[j],
                        "metadata": metadatas[j] if j < len(metadatas) else {},
                        "source_id": source_id,
                        "embedding": embeddings[j],
                    },
                )

            # Insert batch with retry logic
            await self._insert_with_retry("code_examples", batch_data)

    async def search_code_examples(
        self,
        query_embedding: list[float],
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search code examples using vector similarity"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            # Build parameters for RPC call
            params = {"query_embedding": query_embedding, "match_count": match_count}

            # Add optional filters
            if filter_metadata:
                params["filter"] = filter_metadata
            if source_filter:
                params["source_filter"] = source_filter

            # Execute search using Supabase RPC function
            result = self.client.rpc("match_code_examples", params).execute()
        except (QueryError, VectorStoreError):
            logger.exception("Code example search failed")
            return []
        except Exception:
            logger.exception("Unexpected error searching code examples")
            return []
        else:
            return cast(list[dict[str, Any]], result.data) if result.data else []

    async def delete_code_examples_by_url(self, urls: list[str]) -> None:
        """Delete code examples by URL"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        for url in urls:
            try:
                self.client.table("code_examples").delete().eq("url", url).execute()
            except QueryError:
                logger.exception("Failed to delete code examples for %s", url)
            except Exception:
                logger.exception("Unexpected error deleting code examples for %s", url)

    async def update_source_info(
        self,
        source_id: str,
        summary: str,
        word_count: int,
    ) -> None:
        """Update or create source information"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            # Try to update existing source
            result = (
                self.client.table("sources")
                .update(
                    {
                        "summary": summary,
                        "total_word_count": word_count,
                        "updated_at": "now()",
                    },
                )
                .eq("source_id", source_id)
                .execute()
            )

            # If no rows were updated, insert new source
            if not result.data:
                self.client.table("sources").insert(
                    {
                        "source_id": source_id,
                        "summary": summary,
                        "total_word_count": word_count,
                    },
                ).execute()
                logger.info("Created new source: %s", source_id)
            else:
                logger.info("Updated source: %s", source_id)

        except QueryError:
            logger.exception("Failed to update source %s", source_id)
        except Exception:
            logger.exception("Unexpected error updating source %s", source_id)

    async def get_documents_by_url(self, url: str) -> list[dict[str, Any]]:
        """Get all document chunks for a specific URL"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            result = (
                self.client.table("crawled_pages").select("*").eq("url", url).execute()
            )
            # Supabase may return None if no results
        except QueryError:
            logger.exception("Query failed for URL %s", url)
            return []
        except Exception:
            logger.exception("Unexpected error getting documents by URL")
            return []
        else:
            return cast(list[dict[str, Any]], result.data) if result.data else []

    async def search_documents_by_keyword(
        self,
        keyword: str,
        match_count: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for documents containing a keyword"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            query = (
                self.client.table("crawled_pages")
                .select("id, url, chunk_number, content, metadata, source_id")
                .ilike("content", f"%{keyword}%")
            )

            if source_filter:
                query = query.eq("source_id", source_filter)

            result = query.limit(match_count).execute()
        except QueryError:
            logger.exception("Keyword search failed for %s", keyword)
            return []
        except Exception:
            logger.exception("Unexpected error searching documents by keyword")
            return []
        else:
            return cast(list[dict[str, Any]], result.data) if result.data else []

    async def search_code_examples_by_keyword(
        self,
        keyword: str,
        match_count: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for code examples containing a keyword"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            query = (
                self.client.table("code_examples")
                .select(
                    "id, url, chunk_number, content, summary, metadata, source_id",
                )
                .or_(f"content.ilike.%{keyword}%,summary.ilike.%{keyword}%")
            )

            if source_filter:
                query = query.eq("source_id", source_filter)

            result = query.limit(match_count).execute()
        except QueryError:
            logger.exception("Keyword search failed for code examples %s", keyword)
            return []
        except Exception:
            logger.exception("Unexpected error searching code examples by keyword")
            return []
        else:
            return cast(list[dict[str, Any]], result.data) if result.data else []

    async def get_sources(self) -> list[dict[str, Any]]:
        """Get all available sources"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            result = (
                self.client.table("sources").select("*").order("source_id").execute()
            )
            # Supabase may return None if no results
        except QueryError:
            logger.exception("Failed to get sources")
            return []
        except Exception:
            logger.exception("Unexpected error getting sources")
            return []
        else:
            return cast(list[dict[str, Any]], result.data) if result.data else []

    # Private helper methods

    async def _delete_documents_batch(self, urls: list[str]) -> None:
        """Delete documents in batch with fallback to individual deletion"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        try:
            if urls:
                # Try batch deletion
                self.client.table("crawled_pages").delete().in_("url", urls).execute()
        except QueryError:
            logger.warning(
                "Batch delete failed. Trying one-by-one deletion as fallback.",
            )
            # Fallback: delete records one by one
            for url in urls:
                try:
                    self.client.table("crawled_pages").delete().eq("url", url).execute()
                except QueryError:
                    logger.exception("Failed to delete record for URL %s", url)
                except Exception:
                    logger.exception("Unexpected error deleting record for URL %s", url)
        except Exception:
            logger.exception("Unexpected error in batch delete")

    async def _insert_with_retry(
        self,
        table_name: str,
        batch_data: list[dict[str, Any]],
    ) -> None:
        """Insert data with retry logic"""
        if not self.client:
            msg = "Database not initialized. Call initialize() first."
            raise RuntimeError(msg)

        retry_delay = self.retry_delay

        for retry in range(self.max_retries):
            try:
                self.client.table(table_name).insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except QueryError:
                if retry < self.max_retries - 1:
                    logger.warning(
                        "Insert failed for %s (attempt %s/%s)",
                        table_name,
                        retry + 1,
                        self.max_retries,
                    )
                    logger.info("Retrying in %s seconds...", retry_delay)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    logger.exception(
                        "Failed to insert batch after %s attempts",
                        self.max_retries,
                    )
                    # Try inserting records one by one as a last resort
                    logger.info("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            self.client.table(table_name).insert(record).execute()
                            successful_inserts += 1
                        except QueryError:
                            logger.exception("Failed to insert individual record")
                        except Exception:
                            logger.exception(
                                "Unexpected error inserting individual record",
                            )

                    if successful_inserts > 0:
                        logger.info(
                            "Successfully inserted %s/%s records individually",
                            successful_inserts,
                            len(batch_data),
                        )
            except Exception:
                logger.exception("Unexpected error inserting into %s", table_name)
                break

    async def url_exists(self, url: str) -> bool:
        """
        Check if URL exists in database.

        Uses maybe_single() for efficient existence check without counting overhead.

        Args:
            url: URL to check

        Returns:
            True if URL exists, False otherwise
        """
        if not self.client:
            msg = "Supabase client not initialized"
            raise VectorStoreError(msg)

        try:
            # Use maybe_single() for minimal overhead (no COUNT(*))
            response = (
                self.client.table("crawled_pages")
                .select("url")
                .eq("url", url)
                .maybe_single()
                .execute()
            )

            return response.data is not None if response else False
        except Exception as e:
            logger.exception("Error checking URL existence: %s", e)
            # Fail open - return False to allow crawling on error
            return False

    async def add_source(
        self,
        source_id: str,
        url: str,
        title: str,
        description: str,
        metadata: dict[str, Any],
        embedding: list[float],
    ) -> None:
        """
        Add a web source with metadata and vector embedding.

        Requires 'sources' table to have 'embedding' column (vector type).

        Args:
            source_id: Unique source identifier
            url: Source URL
            title: Source title
            description: Source description
            metadata: Additional metadata
            embedding: Embedding vector
        """
        if not self.client:
            msg = "Supabase client not initialized"
            raise VectorStoreError(msg)

        try:
            source_data: dict[str, Any] = {
                "source_id": source_id,
                "url": url,
                "title": title,
                "description": description,
                "embedding": embedding,
                "metadata": metadata or {},
                "updated_at": "now()",
            }

            self.client.table("sources").upsert(source_data).execute()
            logger.info(f"Added source to Supabase: {source_id}")

        except Exception as e:
            logger.exception("Error adding source: %s", e)
            raise VectorStoreError(f"Failed to add source {source_id}") from e
