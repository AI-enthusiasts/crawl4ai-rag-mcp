"""
Base protocol/interface for vector database implementations.
All database adapters must implement this protocol.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorDatabase(Protocol):
    """
    Protocol defining the interface for vector database operations.
    Both Supabase and Qdrant adapters must implement this interface.
    """

    async def initialize(self) -> None:
        """
        Initialize the database connection and create necessary collections/tables.
        This should be called once when the application starts.
        """
        ...

    async def add_documents(
        self,
        urls: list[str],
        chunk_numbers: list[int],
        contents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        source_ids: list[str],
    ) -> None:
        """
        Add documents with their embeddings to the database.

        Args:
            urls: List of document URLs
            chunk_numbers: List of chunk numbers for each document
            contents: List of document contents
            metadatas: List of metadata dictionaries for each document
            embeddings: List of embedding vectors (1536 dimensions for OpenAI)
            source_ids: List of source identifiers (usually domain names)
        """
        ...

    async def search_documents(
        self,
        query_embedding: list[float],
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for documents using vector similarity.

        Args:
            query_embedding: Query embedding vector
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_filter: Optional source ID filter

        Returns:
            List of documents with similarity scores, each containing:
            - id: Document ID
            - url: Document URL
            - chunk_number: Chunk number
            - content: Document content
            - metadata: Document metadata
            - source_id: Source identifier
            - similarity: Similarity score (0-1)
        """
        ...

    async def delete_documents_by_url(self, urls: list[str]) -> None:
        """
        Delete all documents with the given URLs.

        Args:
            urls: List of URLs to delete
        """
        ...

    async def add_code_examples(
        self,
        urls: list[str],
        chunk_numbers: list[int],
        code_examples: list[str],
        summaries: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        source_ids: list[str],
    ) -> None:
        """
        Add code examples with their embeddings to the database.

        Args:
            urls: List of source URLs
            chunk_numbers: List of chunk numbers
            code_examples: List of code example contents
            summaries: List of code example summaries
            metadatas: List of metadata dictionaries
            embeddings: List of embedding vectors
            source_ids: List of source identifiers
        """
        ...

    async def search_code_examples(
        self,
        query_embedding: list[float],
        match_count: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for code examples using vector similarity.

        Args:
            query_embedding: Query embedding vector
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_filter: Optional source ID filter

        Returns:
            List of code examples with similarity scores, each containing:
            - id: Example ID
            - url: Source URL
            - chunk_number: Chunk number
            - content: Code content
            - summary: Code summary
            - metadata: Example metadata
            - source_id: Source identifier
            - similarity: Similarity score (0-1)
        """
        ...

    async def delete_code_examples_by_url(self, urls: list[str]) -> None:
        """
        Delete all code examples with the given URLs.

        Args:
            urls: List of URLs to delete
        """
        ...

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

        Args:
            source_id: Unique source identifier
            url: Source URL
            title: Source title
            description: Source description
            metadata: Additional metadata
            embedding: Embedding vector for semantic search
        """
        ...

    async def update_source_info(
        self,
        source_id: str,
        summary: str,
        word_count: int,
    ) -> None:
        """
        Update or create source information (legacy method).

        Deprecated: Use add_source() instead for full functionality.

        Args:
            source_id: Source identifier (usually domain name)
            summary: Summary of the source
            word_count: Total word count for the source
        """
        ...

    async def get_documents_by_url(self, url: str) -> list[dict[str, Any]]:
        """
        Get all document chunks for a specific URL.

        Args:
            url: The URL to retrieve documents for

        Returns:
            List of documents, each containing:
            - id: Document ID
            - url: Document URL
            - chunk_number: Chunk number
            - content: Document content
            - metadata: Document metadata
            - source_id: Source identifier
        """
        ...

    async def url_exists(self, url: str) -> bool:
        """
        Check if URL exists in database.

        This method is used for duplicate detection in crawling operations.
        Both Qdrant and Supabase adapters must implement this.

        Args:
            url: URL to check

        Returns:
            True if URL exists, False otherwise
        """
        ...

    async def search_documents_by_keyword(
        self,
        keyword: str,
        match_count: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for documents containing a keyword.

        Args:
            keyword: Keyword to search for
            match_count: Maximum number of results to return
            source_filter: Optional source ID filter

        Returns:
            List of documents matching the keyword
        """
        ...

    async def search_code_examples_by_keyword(
        self,
        keyword: str,
        match_count: int = 10,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for code examples containing a keyword.

        Args:
            keyword: Keyword to search for
            match_count: Maximum number of results to return
            source_filter: Optional source ID filter

        Returns:
            List of code examples matching the keyword
        """
        ...

    async def get_sources(self) -> list[dict[str, Any]]:
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
        ...

    async def delete_repository_code_examples(self, repository_name: str) -> None:
        """
        Delete all code examples for a specific repository.

        This is used when removing a repository from the knowledge graph
        to clean up associated code examples from the vector database.

        Args:
            repository_name: Name of the repository to delete code examples for
        """
        ...

    async def get_all_sources(self) -> list[str]:
        """
        Get a list of all available source identifiers.

        Returns:
            List of source identifiers (domain names)
        """
        ...


@runtime_checkable
class QdrantDatabase(VectorDatabase, Protocol):
    """
    Extended protocol for Qdrant-specific functionality.

    Qdrant adapter implements additional methods beyond the base VectorDatabase protocol.
    Use isinstance() checks to access these methods in a type-safe way.
    """

    async def store_crawled_page(
        self,
        url: str,
        content: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a crawled page (Qdrant-specific, legacy method).

        Args:
            url: Page URL
            content: Page content
            title: Optional page title
            metadata: Optional metadata

        Returns:
            Document ID
        """
        ...
