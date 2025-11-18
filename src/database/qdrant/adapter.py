"""
Qdrant adapter implementation for VectorDatabase protocol.

Main adapter class that coordinates Qdrant vector database operations.
"""

import logging
import os
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from src.core.exceptions import DatabaseConnectionError, VectorStoreError

logger = logging.getLogger(__name__)


class QdrantAdapter:
    """
    Qdrant implementation of the VectorDatabase protocol.

    Uses AsyncQdrantClient for native async vector search operations.
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
            except Exception:
                # Collection doesn't exist, create it
                try:
                    await self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                except VectorStoreError as create_error:
                    logger.warning(
                        "Could not create collection %s: %s",
                        collection_name,
                        create_error,
                    )
                except Exception as create_error:
                    logger.exception(
                        "Unexpected error creating collection %s: %s",
                        collection_name,
                        create_error,
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
                        "Could not create collection %s: %s",
                        collection_name,
                        create_error,
                    )
                except Exception as create_error:
                    logger.exception(
                        "Unexpected error creating collection %s: %s",
                        collection_name,
                        create_error,
                    )

    def _generate_point_id(self, url: str, chunk_number: int) -> str:
        """Generate a deterministic UUID for a document point"""
        id_string = f"{url}_{chunk_number}"
        # Use uuid5 to generate a deterministic UUID from the URL and chunk number
        return str(uuid.uuid5(uuid.NAMESPACE_URL, id_string))
