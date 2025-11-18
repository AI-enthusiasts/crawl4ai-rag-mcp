"""
Qdrant database package.

Modular Qdrant adapter implementation with organized operations.
"""

from .adapter import QdrantAdapter
from .code_examples import (
    add_code_examples,
    delete_code_examples_by_url,
    delete_repository_code_examples,
    get_repository_code_examples,
    search_code_by_signature,
    search_code_examples,
    search_code_examples_by_keyword,
)
from .operations import (
    add_documents,
    add_source,
    delete_documents_by_url,
    get_documents_by_url,
    get_sources,
    search_sources,
    update_source,
    update_source_info,
    url_exists,
)
from .search import (
    hybrid_search,
    search,
    search_documents,
    search_documents_by_keyword,
)

__all__ = [
    "QdrantAdapter",
    "add_code_examples",
    "add_documents",
    "add_source",
    "delete_code_examples_by_url",
    "delete_documents_by_url",
    "delete_repository_code_examples",
    "get_documents_by_url",
    "get_repository_code_examples",
    "get_sources",
    "hybrid_search",
    "search",
    "search_code_by_signature",
    "search_code_examples",
    "search_code_examples_by_keyword",
    "search_documents",
    "search_documents_by_keyword",
    "search_sources",
    "update_source",
    "update_source_info",
    "url_exists",
]
