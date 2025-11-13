"""Type guards for common validation patterns.

Type guards help mypy understand type narrowing after validation checks,
reducing the need for type: ignore comments and improving type safety.
"""

from typing import Any, TypeGuard


def is_valid_url(url: str | None) -> TypeGuard[str]:
    """Type guard to check if URL is valid and not None.

    Args:
        url: URL string to validate

    Returns:
        True if URL is a non-empty string starting with http:// or https://

    Example:
        >>> url = settings.searxng_url
        >>> if not is_valid_url(url):
        ...     return []
        >>> # mypy now knows url is str, not str | None
        >>> process_url(url)
    """
    return (
        url is not None
        and isinstance(url, str)
        and url.strip() != ""
        and url.startswith(("http://", "https://"))
    )


def is_non_empty_string(value: str | None) -> TypeGuard[str]:
    """Type guard for non-empty string validation.

    Args:
        value: String to check

    Returns:
        True if value is a non-empty string

    Example:
        >>> api_key = os.getenv("API_KEY")
        >>> if not is_non_empty_string(api_key):
        ...     raise ValueError("API key required")
        >>> # mypy now knows api_key is str
        >>> client = Client(api_key)
    """
    return value is not None and isinstance(value, str) and value.strip() != ""


def is_valid_embedding(embedding: list[float] | None) -> TypeGuard[list[float]]:
    """Type guard for valid embedding vector.

    Args:
        embedding: Embedding vector to validate

    Returns:
        True if embedding is a non-empty list of floats

    Example:
        >>> embedding = generate_embedding(text)
        >>> if not is_valid_embedding(embedding):
        ...     return None
        >>> # mypy knows embedding is list[float]
        >>> store_embedding(embedding)
    """
    return (
        embedding is not None
        and isinstance(embedding, list)
        and len(embedding) > 0
        and all(isinstance(x, (int, float)) for x in embedding)
    )


def is_search_result(data: Any) -> TypeGuard[dict[str, Any]]:
    """Type guard for valid search result structure.

    Args:
        data: Data to validate as search result

    Returns:
        True if data has required search result fields

    Example:
        >>> result = await search(query)
        >>> if is_search_result(result):
        ...     content = result["content"]  # Safe access
    """
    return (
        isinstance(data, dict)
        and "content" in data
        and "score" in data
        and isinstance(data.get("score"), (int, float))
    )


def is_document_chunk(data: Any) -> TypeGuard[dict[str, Any]]:
    """Type guard for valid document chunk structure.

    Args:
        data: Data to validate as document chunk

    Returns:
        True if data has required document chunk fields

    Example:
        >>> chunk = process_document(doc)
        >>> if is_document_chunk(chunk):
        ...     text = chunk["content"]
        ...     metadata = chunk["metadata"]
    """
    return (
        isinstance(data, dict)
        and "content" in data
        and "metadata" in data
        and isinstance(data.get("content"), str)
        and isinstance(data.get("metadata"), dict)
    )


def has_neo4j_config(
    uri: str | None, username: str | None, password: str | None,
) -> TypeGuard[tuple[str, str, str]]:
    """Type guard for complete Neo4j configuration.

    Args:
        uri: Neo4j URI
        username: Neo4j username
        password: Neo4j password

    Returns:
        True if all configuration values are non-empty strings

    Example:
        >>> if has_neo4j_config(uri, user, pwd):
        ...     # mypy knows all three are str
        ...     driver = GraphDatabase.driver(uri, auth=(user, pwd))
    """
    return (
        is_non_empty_string(uri)
        and is_non_empty_string(username)
        and is_non_empty_string(password)
    )


def is_point_id_list(data: Any) -> TypeGuard[list[str | int]]:
    """Type guard for list of valid point IDs.

    Args:
        data: Data to validate as point ID list

    Returns:
        True if data is a list of strings or integers

    Example:
        >>> ids = extract_ids(results)
        >>> if is_point_id_list(ids):
        ...     await db.delete(ids)
    """
    return (
        isinstance(data, list)
        and len(data) > 0
        and all(isinstance(x, (str, int)) for x in data)
    )


def is_embedding_dimension_valid(embedding: list[float], expected_dim: int) -> bool:
    """Check if embedding has expected dimensionality.

    Args:
        embedding: Embedding vector
        expected_dim: Expected dimension (1536, 3072, etc.)

    Returns:
        True if embedding dimension matches expected

    Note:
        This is not a TypeGuard as it doesn't narrow the type,
        just validates the dimension.

    Example:
        >>> if is_embedding_dimension_valid(emb, 1536):
        ...     store_in_database(emb)
    """
    return len(embedding) == expected_dim


def is_confidence_score(value: Any) -> TypeGuard[float]:
    """Type guard for valid confidence score (0.0 to 1.0).

    Args:
        value: Value to check

    Returns:
        True if value is a float between 0.0 and 1.0

    Example:
        >>> score = calculate_confidence(result)
        >>> if is_confidence_score(score):
        ...     if score > 0.8:
        ...         high_confidence_action()
    """
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
