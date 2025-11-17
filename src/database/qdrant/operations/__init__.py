"""Qdrant operations module.

CRUD operations for documents and sources in Qdrant vector database.
All functions are standalone and accept QdrantClient as first parameter.
"""

# Document operations
from .documents import (
    add_documents,
    delete_documents_by_url,
    get_documents_by_url,
    url_exists,
)

# Source operations
# Private function exports for backward compatibility
from .sources import (
    _create_new_source,
    add_source,
    get_sources,
    search_sources,
    update_source,
    update_source_info,
)

# Utilities and constants
from .utils import BATCH_SIZE, CODE_EXAMPLES, CRAWLED_PAGES, SOURCES, generate_point_id

# Alias for test compatibility
_generate_point_id = generate_point_id

__all__ = [
    # Document operations
    "add_documents",
    "url_exists",
    "get_documents_by_url",
    "delete_documents_by_url",
    # Source operations
    "add_source",
    "search_sources",
    "update_source",
    "get_sources",
    "update_source_info",
    # Utilities and constants
    "BATCH_SIZE",
    "CODE_EXAMPLES",
    "CRAWLED_PAGES",
    "SOURCES",
    "generate_point_id",
    "_generate_point_id",  # Alias for tests
    # Private (for tests)
    "_create_new_source",
]
