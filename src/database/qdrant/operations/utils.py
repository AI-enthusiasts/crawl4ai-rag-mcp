"""Qdrant operations utilities.

Provides constants and helper functions for Qdrant operations.
"""

import uuid

# Collection names
CRAWLED_PAGES = "crawled_pages"
CODE_EXAMPLES = "code_examples"
SOURCES = "sources"

# Batch processing
BATCH_SIZE = 100


def generate_point_id(url: str, chunk_number: int) -> str:
    """Generate a deterministic UUID for a document point.

    Args:
        url: Document URL
        chunk_number: Chunk number

    Returns:
        Deterministic UUID string
    """
    id_string = f"{url}_{chunk_number}"
    # Use uuid5 to generate a deterministic UUID from the URL and chunk number
    return str(uuid.uuid5(uuid.NAMESPACE_URL, id_string))
