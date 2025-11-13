"""Pydantic models for agentic search pipeline.

This module contains all type-safe data structures used throughout
the agentic search workflow with strict validation.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchStatus(str, Enum):
    """Status of agentic search execution."""

    COMPLETE = "complete"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    ERROR = "error"


class ActionType(str, Enum):
    """Type of action performed in search iteration."""

    LOCAL_CHECK = "local_check"
    WEB_SEARCH = "web_search"
    CRAWL = "crawl"
    QUERY_REFINEMENT = "query_refinement"


class CompletenessEvaluation(BaseModel):
    """LLM evaluation of answer completeness.

    This model represents the LLM's assessment of whether the current
    knowledge is sufficient to answer the user's query.
    """

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Completeness score from 0.0 (incomplete) to 1.0 (complete)",
    )
    reasoning: str = Field(
        min_length=1,
        description="LLM's explanation of the score",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="List of missing information or knowledge gaps",
    )

    @field_validator("gaps")
    @classmethod
    def validate_gaps(cls, v: list[str]) -> list[str]:
        """Ensure gaps are non-empty strings."""
        return [gap.strip() for gap in v if gap and gap.strip()]


class URLRanking(BaseModel):
    """LLM ranking of a single URL's relevance.

    Represents how promising a URL is for filling knowledge gaps.
    """

    url: str = Field(
        min_length=1,
        description="The URL being ranked",
    )
    title: str = Field(
        default="",
        description="Page title from search results",
    )
    snippet: str = Field(
        default="",
        description="Content snippet from search results",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant)",
    )
    reasoning: str = Field(
        min_length=1,
        description="LLM's explanation of the relevance score",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is non-empty and stripped."""
        url = v.strip()
        if not url:
            raise ValueError("URL cannot be empty")
        return url


class SearchIteration(BaseModel):
    """Record of a single search iteration.

    Tracks all actions and results from one cycle of the agentic search.
    """

    iteration: int = Field(
        ge=1,
        description="Iteration number (1-indexed)",
    )
    query: str = Field(
        min_length=1,
        description="Query used in this iteration",
    )
    action: ActionType = Field(
        description="Type of action performed",
    )
    completeness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Completeness score if local_check was performed",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Knowledge gaps identified",
    )
    urls_found: int = Field(
        default=0,
        ge=0,
        description="Number of URLs found in web search",
    )
    urls_ranked: int = Field(
        default=0,
        ge=0,
        description="Number of URLs that were ranked",
    )
    promising_urls: int = Field(
        default=0,
        ge=0,
        description="Number of URLs selected for crawling",
    )
    urls: list[str] = Field(
        default_factory=list,
        description="List of URLs crawled in this iteration",
    )
    urls_stored: int = Field(
        default=0,
        ge=0,
        description="Number of URLs successfully stored",
    )
    chunks_stored: int = Field(
        default=0,
        ge=0,
        description="Total chunks stored in vector database",
    )

    @field_validator("gaps")
    @classmethod
    def validate_gaps(cls, v: list[str]) -> list[str]:
        """Ensure gaps are non-empty strings."""
        return [gap.strip() for gap in v if gap and gap.strip()]

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, v: list[str]) -> list[str]:
        """Ensure URLs are non-empty strings."""
        return [url.strip() for url in v if url and url.strip()]


class RAGResult(BaseModel):
    """Single result from RAG query.

    Represents a chunk of content retrieved from the vector database.
    """

    content: str = Field(
        min_length=1,
        description="Content of the retrieved chunk",
    )
    url: str = Field(
        default="",
        description="Source URL of the content",
    )
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score from vector search",
    )
    chunk_index: int = Field(
        ge=0,
        description="Index of the chunk within its source document",
    )


class AgenticSearchResult(BaseModel):
    """Complete result from agentic search.

    This is the top-level response containing all information about
    the search execution and its results.
    """

    success: bool = Field(
        description="Whether the search completed successfully",
    )
    query: str = Field(
        min_length=1,
        description="Original user query",
    )
    iterations: int = Field(
        ge=0,
        description="Number of search-crawl cycles performed",
    )
    completeness: float = Field(
        ge=0.0,
        le=1.0,
        description="Final completeness score",
    )
    results: list[RAGResult] = Field(
        default_factory=list,
        description="RAG results from vector database",
    )
    search_history: list[SearchIteration] = Field(
        default_factory=list,
        description="Detailed log of all actions taken",
    )
    status: SearchStatus = Field(
        description="Final status of the search",
    )
    error: str | None = Field(
        default=None,
        description="Error message if search failed",
    )

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override to provide consistent JSON serialization."""
        return super().model_dump_json(indent=2, **kwargs)


class QueryRefinement(BaseModel):
    """LLM-generated query refinements.

    Contains alternative or refined queries to try when current results
    are incomplete.
    """

    original_query: str = Field(
        min_length=1,
        description="Original user query",
    )
    current_query: str = Field(
        min_length=1,
        description="Current query being executed",
    )
    refined_queries: list[str] = Field(
        min_length=1,
        description="List of refined queries to try",
    )
    reasoning: str = Field(
        min_length=1,
        description="LLM's explanation of the refinements",
    )

    @field_validator("refined_queries")
    @classmethod
    def validate_refined_queries(cls, v: list[str]) -> list[str]:
        """Ensure refined queries are non-empty and unique."""
        queries = [q.strip() for q in v if q and q.strip()]
        if not queries:
            raise ValueError("At least one refined query is required")
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        return unique_queries


class SearchMetadata(BaseModel):
    """Metadata extracted from crawled content.

    Optional structured information that can be used to generate
    smarter Qdrant queries.
    """

    title: str = Field(
        default="",
        description="Page title",
    )
    description: str = Field(
        default="",
        description="Page description or summary",
    )
    main_topics: list[str] = Field(
        default_factory=list,
        description="Main topics identified in content",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="Page headers/sections",
    )
    code_blocks_count: int = Field(
        default=0,
        ge=0,
        description="Number of code blocks found",
    )
    code_languages: list[str] = Field(
        default_factory=list,
        description="Programming languages detected",
    )

    @field_validator("main_topics", "headers", "code_languages")
    @classmethod
    def validate_string_lists(cls, v: list[str]) -> list[str]:
        """Ensure list items are non-empty strings."""
        return [item.strip() for item in v if item and item.strip()]


class SearchHints(BaseModel):
    """LLM-generated search hints for Qdrant queries.

    Based on crawled content metadata, these are optimized queries
    for retrieving relevant information from the vector database.
    """

    original_query: str = Field(
        min_length=1,
        description="Original user query",
    )
    hints: list[str] = Field(
        min_length=1,
        description="Generated search hints/queries",
    )
    reasoning: str = Field(
        min_length=1,
        description="LLM's explanation of the hints",
    )

    @field_validator("hints")
    @classmethod
    def validate_hints(cls, v: list[str]) -> list[str]:
        """Ensure hints are non-empty and unique."""
        hints = [h.strip() for h in v if h and h.strip()]
        if not hints:
            raise ValueError("At least one search hint is required")
        # Remove duplicates while preserving order
        seen = set()
        unique_hints = []
        for h in hints:
            if h.lower() not in seen:
                seen.add(h.lower())
                unique_hints.append(h)
        return unique_hints
