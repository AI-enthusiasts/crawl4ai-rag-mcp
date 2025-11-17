"""Basic embedding creation utilities using OpenAI."""

import os
import time

import openai

from src.core.exceptions import EmbeddingError
from src.core.logging import logger

from .config import get_embedding_dimensions, get_embedding_model


def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
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
    retry_delay = 1.0  # Start with 1 second delay

    # Use the embedding model from environment or default
    model = get_embedding_model()

    # Create OpenAI client instance once
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for retry in range(max_retries):
        try:
            response = client.embeddings.create(
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
                logger.info("Retrying in %s seconds...", retry_delay)
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
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
                        individual_response = client.embeddings.create(
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
                logger.info("Retrying in %s seconds...", retry_delay)
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
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
                    # This block is now handled above in the EmbeddingError catch
                    pass

                logger.info(
                    "Successfully created %d/%d embeddings individually",
                    successful_count,
                    len(texts),
                )
                return embeddings

    # This should never be reached, but added for type safety
    return []


def create_embedding(text: str) -> list[float]:
    """
    Create an embedding for a single text using OpenAI's API.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        if embeddings:
            return embeddings[0]
        # Fallback with dynamic dimensions
        model = get_embedding_model()
        dimensions = get_embedding_dimensions(model)
        return [0.0] * dimensions
    except EmbeddingError as e:
        logger.error("Embedding error creating single embedding: %s", str(e))
        # Return empty embedding with dynamic dimensions
        model = get_embedding_model()
        dimensions = get_embedding_dimensions(model)
        return [0.0] * dimensions
    except Exception as e:
        logger.error("Unexpected error creating embedding: %s", str(e))
        # Return empty embedding with dynamic dimensions
        model = get_embedding_model()
        dimensions = get_embedding_dimensions(model)
        return [0.0] * dimensions
