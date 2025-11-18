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
    "_add_web_sources_to_database",
    "_get_embedding_dimensions",
    "add_code_examples_to_database",
    "add_documents_to_database",
    "create_embedding",
    "create_embeddings_batch",
    "generate_contextual_embedding",
    "get_contextual_embedding_config",
    "get_embedding_dimensions",
    "get_embedding_model",
    "process_chunk_with_context",
    "search_code_examples",
    "search_documents",
]
