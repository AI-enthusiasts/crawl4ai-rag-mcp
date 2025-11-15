"""
Qdrant adapter implementation for VectorDatabase protocol.

Uses Qdrant vector database for similarity search.
Delegates operations to modular qdrant package functions.
"""

import logging
import os
import sys
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

# Import qdrant package modules
from . import qdrant
from src.core.exceptions import ConnectionError, VectorStoreError

logger = logging.getLogger(__name__)


class QdrantAdapter:
    """
    Qdrant implementation of the VectorDatabase protocol.

    Uses AsyncQdrantClient for native async vector search operations.
    Delegates all operations to modular qdrant package functions.
    """

    def __init__(self, url: str | None = None, api_key: str | None = None):
        """Initialize Qdrant adapter with connection parameters"""
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.client: AsyncQdrantClient = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        self.batch_size = 100  # Qdrant can handle larger batches

        # Collection names
        self.CRAWLED_PAGES = "crawled_pages"
        self.CODE_EXAMPLES = "code_examples"
        self.SOURCES = "sources"

    async def initialize(self) -> None:
        """Initialize Qdrant client and create collections if needed"""
        # Create collections if they don't exist
        await self._ensure_collections()

    async def _ensure_collections(self) -> None:
        """Ensure all required collections exist"""
        collections = [
            (self.CRAWLED_PAGES, 1536),  # OpenAI embedding size
            (self.CODE_EXAMPLES, 1536),
            (self.SOURCES, 1536),  # OpenAI embedding size for consistency
        ]

        for collection_name, vector_size in collections:
            try:
                await self.client.get_collection(collection_name)
            except ConnectionError:
                # Collection doesn't exist, create it
                try:
                    await self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                except VectorStoreError as create_error:
                    logger.warning(
                        f"Could not create collection {collection_name}: {create_error}",
                    )
                except Exception as create_error:
                    logger.exception(
                        f"Unexpected error creating collection {collection_name}: {create_error}",
                    )
            except Exception:
                # Collection doesn't exist or other error, try to create it
                try:
                    await self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                except VectorStoreError as create_error:
                    logger.warning(
                        f"Could not create collection {collection_name}: {create_error}",
                    )
                except Exception as create_error:
                    logger.exception(
                        f"Unexpected error creating collection {collection_name}: {create_error}",
                    )

    # Document operations - delegate to qdrant.operations
    async def add_documents(
        self,
        urls: list[str],
        chunk_numbers: list[int],
        contents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        source_ids: list[str] | None = None,
    ) -> None:
        """Add documents to Qdrant - delegates to qdrant.operations"""
        return await qdrant.operations.add_documents(
            self.client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            embeddings,
            source_ids,
        )

    async def url_exists(self, url: str) -> bool:
        """Check if URL exists - delegates to qdrant.operations"""
        return await qdrant.operations.url_exists(self.client, url)

    async def get_documents_by_url(self, url: str) -> list[dict[str, Any]]:
        """Get all document chunks for a URL - delegates to qdrant.operations"""
        return await qdrant.operations.get_documents_by_url(self.client, url)

    async def delete_documents_by_url(self, urls: list[str]) -> None:
        """Delete document chunks by URL - delegates to qdrant.operations"""
        return await qdrant.operations.delete_documents_by_url(self.client, urls)

    # Search operations - delegate to qdrant.search
    async def search_documents(
        self,
        query_embedding: list[float],
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents - delegates to qdrant.search"""
        return await qdrant.search.search_documents(
            self.client,
            query_embedding,
            match_count,
            filter_metadata,
            source_filter,
        )

    async def search_documents_by_keyword(
        self,
        keyword: str,
        match_count: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents by keyword - delegates to qdrant.search"""
        return await qdrant.search.search_documents_by_keyword(
            self.client,
            keyword,
            match_count,
            source_filter,
        )

    async def search(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generic search with embedding generation - delegates to qdrant.search"""
        return await qdrant.search.search(
            self.client,
            query,
            match_count,
            filter_metadata,
            source_filter,
        )

    async def hybrid_search(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector and keyword - delegates to qdrant.search"""
        return await qdrant.search.hybrid_search(
            self.client,
            query,
            match_count,
            filter_metadata,
            source_filter,
        )

    # Code examples operations - delegate to qdrant.code_examples
    async def add_code_examples(
        self,
        urls: list[str],
        chunk_numbers: list[int],
        code_examples: list[str],
        summaries: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        source_ids: list[str] | None = None,
    ) -> None:
        """Add code examples - delegates to qdrant.code_examples"""
        return await qdrant.code_examples.add_code_examples(
            self.client,
            urls,
            chunk_numbers,
            code_examples,
            summaries,
            metadatas,
            embeddings,
            source_ids,
        )

    async def search_code_examples(
        self,
        query: str | list[float] | None = None,
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Search code examples - delegates to qdrant.code_examples"""
        return await qdrant.code_examples.search_code_examples(
            self.client,
            query,
            match_count,
            filter_metadata,
            source_filter,
            query_embedding,
        )

    async def delete_code_examples_by_url(self, urls: list[str]) -> None:
        """Delete code examples by URL - delegates to qdrant.code_examples"""
        return await qdrant.code_examples.delete_code_examples_by_url(self.client, urls)

    async def search_code_examples_by_keyword(
        self,
        keyword: str,
        match_count: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search code examples by keyword - delegates to qdrant.code_examples"""
        return await qdrant.code_examples.search_code_examples_by_keyword(
            self.client,
            keyword,
            match_count,
            source_filter,
        )

    async def get_repository_code_examples(
        self,
        repo_name: str,
        code_type: str | None = None,
        match_count: int = 100,
    ) -> list[dict[str, Any]]:
        """Get repository code examples - delegates to qdrant.code_examples"""
        return await qdrant.code_examples.get_repository_code_examples(
            self.client,
            repo_name,
            code_type,
            match_count,
        )

    async def delete_repository_code_examples(self, repo_name: str) -> None:
        """Delete repository code examples - delegates to qdrant.code_examples"""
        return await qdrant.code_examples.delete_repository_code_examples(self.client, repo_name)

    async def search_code_by_signature(
        self,
        method_name: str,
        class_name: str | None = None,
        repo_filter: str | None = None,
        match_count: int = 10,
    ) -> list[dict[str, Any]]:
        """Search code by signature - delegates to qdrant.code_examples"""
        return await qdrant.code_examples.search_code_by_signature(
            self.client,
            method_name,
            class_name,
            repo_filter,
            match_count,
        )

    # Source operations - delegate to qdrant.operations
    async def add_source(
        self,
        source_id: str,
        url: str,
        title: str,
        description: str,
        metadata: dict[str, Any],
        embedding: list[float],
    ) -> None:
        """Add a source - delegates to qdrant.operations"""
        return await qdrant.operations.add_source(
            self.client,
            source_id,
            url,
            title,
            description,
            metadata,
            embedding,
        )

    async def search_sources(
        self,
        query_embedding: list[float],
        match_count: int = 10,
    ) -> list[dict[str, Any]]:
        """Search sources - delegates to qdrant.operations"""
        return await qdrant.operations.search_sources(
            self.client,
            query_embedding,
            match_count,
        )

    async def update_source(
        self,
        source_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Update source metadata - delegates to qdrant.operations"""
        return await qdrant.operations.update_source(
            self.client,
            source_id,
            updates,
        )

    async def get_sources(self) -> list[dict[str, Any]]:
        """Get all sources - delegates to qdrant.operations"""
        return await qdrant.operations.get_sources(self.client)

    async def get_all_sources(self) -> list[str]:
        """Get all source IDs - extracts source_id from get_sources()"""
        sources = await self.get_sources()
        return [source.get("source_id", "") for source in sources if source.get("source_id")]

    async def update_source_info(
        self,
        source_id: str,
        summary: str,
        word_count: int,
    ) -> None:
        """Update source information - delegates to qdrant.operations"""
        return await qdrant.operations.update_source_info(
            self.client,
            source_id,
            summary,
            word_count,
        )
