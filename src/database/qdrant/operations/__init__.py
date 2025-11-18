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
    "BATCH_SIZE",
    "CODE_EXAMPLES",
    "CRAWLED_PAGES",
    "SOURCES",
    "_create_new_source",
    "_generate_point_id",
    "add_documents",
    "add_source",
    "delete_documents_by_url",
    "generate_point_id",
    "get_documents_by_url",
    "get_sources",
    "search_sources",
    "update_source",
    "update_source_info",
    "url_exists",
]
