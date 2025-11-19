"""Basic embedding creation utilities using OpenAI."""

import os

from openai import AsyncOpenAI

from src.core.exceptions import EmbeddingError
from src.core.logging import logger

from .config import get_embedding_dimensions, get_embedding_model

# Global async client instance (reused across calls)
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Get or create AsyncOpenAI client instance."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


async def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []

    max_retries = 3
    model = get_embedding_model()
    client = _get_client()

    for retry in range(max_retries):
        try:
            response = await client.embeddings.create(
                model=model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except EmbeddingError as e:
            if retry < max_retries - 1:
                logger.error(
                    "Embedding error (attempt %d/%d): %s",
                    retry + 1,
                    max_retries,
                    str(e),
                )
                # OpenAI SDK handles retries automatically, just continue
                continue
            else:
                logger.error(
                    "Failed to create batch embeddings after %d attempts: %s",
                    max_retries,
                    str(e),
                )
                # Try creating embeddings one by one as fallback
                logger.info("Attempting to create embeddings individually...")
                embeddings: list[list[float]] = []
                successful_count = 0

                for i, text in enumerate(texts):
                    try:
                        individual_response = await client.embeddings.create(
                            model=model,
                            input=[text],
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except EmbeddingError as individual_error:
                        logger.error(
                            "Embedding error for text %d: %s",
                            i,
                            str(individual_error),
                        )
                        # Add zero embedding as fallback
                        dimensions = get_embedding_dimensions(model)
                        embeddings.append([0.0] * dimensions)
                    except Exception as individual_error:
                        logger.error(
                            "Unexpected error creating embedding for text %d: %s",
                            i,
                            str(individual_error),
                        )
                        # Add zero embedding as fallback
                        dimensions = get_embedding_dimensions(model)
                        embeddings.append([0.0] * dimensions)

                logger.info(
                    "Successfully created %d/%d embeddings individually",
                    successful_count,
                    len(texts),
                )
                return embeddings
        except Exception as e:
            if retry < max_retries - 1:
                logger.error(
                    "Error creating batch embeddings (attempt %d/%d): %s",
                    retry + 1,
                    max_retries,
                    str(e),
                )
                continue
            else:
                logger.error(
                    "Failed to create batch embeddings after %d attempts: %s",
                    max_retries,
                    str(e),
                )

    # Fallback: return zero embeddings
    dimensions = get_embedding_dimensions(model)
    return [[0.0] * dimensions for _ in texts]


async def create_embedding(text: str) -> list[float]:
    """
    Create an embedding for a single text using OpenAI's API.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = await create_embeddings_batch([text])
        if embeddings:
            return embeddings[0]
        model = get_embedding_model()
        dimensions = get_embedding_dimensions(model)
        return [0.0] * dimensions
    except EmbeddingError as e:
        logger.error("Embedding error creating single embedding: %s", str(e))
        model = get_embedding_model()
        dimensions = get_embedding_dimensions(model)
        return [0.0] * dimensions
    except Exception as e:
        logger.error("Unexpected error creating embedding: %s", str(e))
        model = get_embedding_model()
        dimensions = get_embedding_dimensions(model)
        return [0.0] * dimensions
