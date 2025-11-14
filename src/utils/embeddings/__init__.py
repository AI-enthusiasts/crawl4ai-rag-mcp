"""Embedding generation and management utilities.

This package provides utilities for creating embeddings using OpenAI,
managing documents and code examples in vector databases, and supporting
contextual embeddings for enhanced retrieval.
"""

from .basic import create_embedding, create_embeddings_batch
from .code_examples import add_code_examples_to_database, search_code_examples
from .config import (
    get_contextual_embedding_config,
    get_embedding_dimensions,
    get_embedding_model,
)
from .contextual import generate_contextual_embedding, process_chunk_with_context
from .documents import (
    _add_web_sources_to_database,
    add_documents_to_database,
    search_documents,
)

# Alias for backward compatibility with tests that expect _get_embedding_dimensions
_get_embedding_dimensions = get_embedding_dimensions

__all__ = [
    # Basic embedding functions
    "create_embedding",
    "create_embeddings_batch",
    # Configuration functions
    "get_embedding_dimensions",
    "get_embedding_model",
    "get_contextual_embedding_config",
    # Contextual embedding functions
    "generate_contextual_embedding",
    "process_chunk_with_context",
    # Document management functions
    "add_documents_to_database",
    "search_documents",
    # Code example functions
    "add_code_examples_to_database",
    "search_code_examples",
    # Private functions (for testing and internal use)
    "_get_embedding_dimensions",
    "_add_web_sources_to_database",
]
