"""Contextual embedding generation utilities."""

import os

import openai

from src.core.exceptions import LLMError
from src.core.logging import logger

from .basic import create_embedding
from .config import get_contextual_embedding_config


def generate_contextual_embedding(
    chunk: str,
    full_document: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> str:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Args:
        chunk: The specific chunk of text to generate context for
        full_document: The complete document text
        chunk_index: Index of the current chunk (optional)
        total_chunks: Total number of chunks (optional)

    Returns:
        The contextual text that situates the chunk within the document
    """
    # Get configuration for contextual embeddings
    config = get_contextual_embedding_config()
    model_choice = config["model"]
    max_tokens = config["max_tokens"]
    temperature = config["temperature"]
    max_doc_chars = config["max_doc_chars"]

    try:
        # Truncate full document if it's too long
        truncated_document = full_document[:max_doc_chars]

        # Create position info if available
        position_info = ""
        if chunk_index >= 0 and total_chunks > 1:
            position_info = f" (chunk {chunk_index + 1} of {total_chunks})"

        # Create the prompt for generating contextual information
        prompt = f"""<document>
{truncated_document}
</document>

Here is the chunk{position_info} we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Create OpenAI client instance
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Call the OpenAI API to generate contextual information
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise contextual information.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract the generated context
        content = response.choices[0].message.content
        context = content.strip() if content else ""

        # Combine the context with the original chunk
        return f"{context}\n---\n{chunk}"

    except LLMError as e:
        logger.error(
            "LLM error generating contextual embedding: %s. Using original chunk instead.",
            str(e),
        )
        return chunk
    except Exception as e:
        logger.error(
            "Unexpected error generating contextual embedding: %s. Using original chunk instead.",
            str(e),
        )
        return chunk


def process_chunk_with_context(args) -> tuple[str, list[float]]:
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (chunk, full_document, chunk_index, total_chunks)

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - The embedding for the contextual text
    """
    chunk, full_document, chunk_index, total_chunks = args
    contextual_text = generate_contextual_embedding(
        chunk, full_document, chunk_index, total_chunks,
    )
    embedding = create_embedding(contextual_text)
    return contextual_text, embedding
