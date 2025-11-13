"""Type stubs for OpenAI library to reduce type: ignore usage."""

from typing import Any

class Completion:
    """OpenAI completion response."""

    choices: list[CompletionChoice]

class CompletionChoice:
    """Single completion choice."""

    message: CompletionMessage

class CompletionMessage:
    """Completion message with content."""

    content: str
    role: str

class ChatCompletion:
    """OpenAI chat completion response."""

    choices: list[CompletionChoice]
    id: str
    model: str

class Embedding:
    """OpenAI embedding response."""

    data: list[EmbeddingData]

class EmbeddingData:
    """Single embedding data."""

    embedding: list[float]
    index: int

class Completions:
    """OpenAI completions endpoint."""

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        **kwargs: Any,
    ) -> ChatCompletion: ...

class Chat:
    """OpenAI chat endpoint."""

    completions: Completions

class Embeddings:
    """OpenAI embeddings endpoint."""

    def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> Embedding: ...

class OpenAI:
    """OpenAI client."""

    chat: Chat
    embeddings: Embeddings

    def __init__(self, api_key: str | None = None) -> None: ...
