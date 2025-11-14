"""Embedding configuration and model utilities.

This module provides configuration management for embedding models,
including model dimensions and client initialization.
"""

import os


def get_embedding_dimensions(model: str) -> int:
    """Get the embedding dimensions for a given OpenAI model.

    Args:
        model: The embedding model name

    Returns:
        Number of dimensions for the embedding
    """
    # OpenAI embedding model dimensions
    model_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Default to 1536 for unknown models (most common)
    return model_dimensions.get(model, 1536)


def get_embedding_model() -> str:
    """Get the configured embedding model.

    Returns:
        Embedding model name from environment or default
    """
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_contextual_embedding_config() -> dict[str, any]:
    """Get contextual embedding configuration from environment.

    Returns:
        Dictionary with contextual embedding configuration
    """
    import logging

    logger = logging.getLogger(__name__)

    # Model selection
    model = os.getenv("CONTEXTUAL_EMBEDDING_MODEL", "gpt-4o-mini")

    # Validate and set max_tokens (1-4096 range)
    try:
        max_tokens = int(os.getenv("CONTEXTUAL_EMBEDDING_MAX_TOKENS", "200"))
        if not (1 <= max_tokens <= 4096):
            logger.warning(
                f"CONTEXTUAL_EMBEDDING_MAX_TOKENS ({max_tokens}) out of range 1-4096, using default 200",
            )
            max_tokens = 200
    except ValueError:
        logger.warning(
            "CONTEXTUAL_EMBEDDING_MAX_TOKENS must be an integer, using default 200",
        )
        max_tokens = 200

    # Validate and set temperature (0.0-2.0 range)
    try:
        temperature = float(os.getenv("CONTEXTUAL_EMBEDDING_TEMPERATURE", "0.3"))
        if not (0.0 <= temperature <= 2.0):
            logger.warning(
                f"CONTEXTUAL_EMBEDDING_TEMPERATURE ({temperature}) out of range 0.0-2.0, using default 0.3",
            )
            temperature = 0.3
    except ValueError:
        logger.warning(
            "CONTEXTUAL_EMBEDDING_TEMPERATURE must be a number, using default 0.3",
        )
        temperature = 0.3

    # Validate and set max_doc_chars (positive integer)
    try:
        max_doc_chars = int(os.getenv("CONTEXTUAL_EMBEDDING_MAX_DOC_CHARS", "25000"))
        if max_doc_chars <= 0:
            logger.warning(
                f"CONTEXTUAL_EMBEDDING_MAX_DOC_CHARS ({max_doc_chars}) must be positive, using default 25000",
            )
            max_doc_chars = 25000
    except ValueError:
        logger.warning(
            "CONTEXTUAL_EMBEDDING_MAX_DOC_CHARS must be a positive integer, using default 25000",
        )
        max_doc_chars = 25000

    # Max workers for parallel processing
    try:
        max_workers = int(os.getenv("CONTEXTUAL_EMBEDDING_MAX_WORKERS", "10"))
        if max_workers <= 0:
            logger.warning(
                f"CONTEXTUAL_EMBEDDING_MAX_WORKERS ({max_workers}) must be positive, using default 10",
            )
            max_workers = 10
    except ValueError:
        logger.warning(
            "CONTEXTUAL_EMBEDDING_MAX_WORKERS must be a positive integer, using default 10",
        )
        max_workers = 10

    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_doc_chars": max_doc_chars,
        "max_workers": max_workers,
        "enabled": os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false").lower() == "true",
    }
