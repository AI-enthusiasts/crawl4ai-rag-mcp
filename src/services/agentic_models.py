"""Pydantic models for agentic search pipeline.

This module contains all type-safe data structures used throughout
the agentic search workflow with strict validation.
"""

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is finite (not NaN or infinity).

        Per Pydantic best practices: Field constraints (ge/le) don't catch NaN/infinity.
        Must explicitly validate with math.isnan() and math.isinf().
        """
        if math.isnan(v):
            msg = "Score cannot be NaN (Not a Number)"
            raise ValueError(msg)
        if math.isinf(v):
            msg = "Score cannot be infinity"
            raise ValueError(msg)
        return v

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

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is finite (not NaN or infinity).

        Per Pydantic best practices: Field constraints (ge/le) don't catch NaN/infinity.
        Must explicitly validate with math.isnan() and math.isinf().
        """
        if math.isnan(v):
            msg = "Score cannot be NaN (Not a Number)"
            raise ValueError(msg)
        if math.isinf(v):
            msg = "Score cannot be infinity"
            raise ValueError(msg)
        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is non-empty and stripped."""
        url = v.strip()
        if not url:
            msg = "URL cannot be empty"
            raise ValueError(msg)
        return url


class URLRankingList(BaseModel):
    """List of URL rankings from LLM structured output.

    This is used as the response format for OpenAI's structured outputs API.
    """

    rankings: list[URLRanking] = Field(
        default_factory=list,
        description="List of ranked URLs with scores and reasoning",
    )


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

    @field_validator("completeness")
    @classmethod
    def validate_completeness(cls, v: float | None) -> float | None:
        """Validate completeness is finite (not NaN or infinity)."""
        if v is None:
            return v
        if math.isnan(v):
            msg = "Completeness cannot be NaN (Not a Number)"
            raise ValueError(msg)
        if math.isinf(v):
            msg = "Completeness cannot be infinity"
            raise ValueError(msg)
        return v

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

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        """Validate similarity score is finite (not NaN or infinity)."""
        if math.isnan(v):
            msg = "Similarity score cannot be NaN (Not a Number)"
            raise ValueError(msg)
        if math.isinf(v):
            msg = "Similarity score cannot be infinity"
            raise ValueError(msg)
        return v


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

    @field_validator("completeness")
    @classmethod
    def validate_completeness(cls, v: float) -> float:
        """Validate completeness is finite (not NaN or infinity)."""
        if math.isnan(v):
            msg = "Completeness cannot be NaN (Not a Number)"
            raise ValueError(msg)
        if math.isinf(v):
            msg = "Completeness cannot be infinity"
            raise ValueError(msg)
        return v

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
            msg = "At least one refined query is required"
            raise ValueError(msg)
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
            msg = "At least one search hint is required"
            raise ValueError(msg)
        # Remove duplicates while preserving order
        seen = set()
        unique_hints = []
        for h in hints:
            if h.lower() not in seen:
                seen.add(h.lower())
                unique_hints.append(h)
        return unique_hints


class QueryComplexity(str, Enum):
    """Complexity level of a query for effort scaling."""

    SIMPLE = "simple"  # 1 agent, 3-10 tool calls
    MODERATE = "moderate"  # 2-4 subagents, 10-15 calls each
    COMPLEX = "complex"  # 10+ subagents, comprehensive research


class TopicDecomposition(BaseModel):
    """Decomposition of query into required topics.

    Per Anthropic research: "Search strategy should mirror expert human research:
    explore the landscape before drilling into specifics."

    This model breaks down a query into essential topics that MUST be covered
    for a complete answer, enabling topic-based completeness evaluation.
    """

    original_query: str = Field(
        min_length=1,
        description="Original user query",
    )
    topics: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Essential topics that must be covered (3-6 typically)",
    )
    topic_queries: dict[str, str] = Field(
        default_factory=dict,
        description="Specific search query for each topic",
    )
    complexity: QueryComplexity = Field(
        default=QueryComplexity.MODERATE,
        description="Query complexity for effort scaling",
    )

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        """Ensure topics are non-empty and unique."""
        topics = [t.strip() for t in v if t and t.strip()]
        if not topics:
            msg = "At least one topic is required"
            raise ValueError(msg)
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for t in topics:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_topics.append(t)
        return unique_topics

    @model_validator(mode="after")
    def ensure_topic_queries_populated(self) -> "TopicDecomposition":
        """Ensure topic_queries has an entry for each topic.

        If LLM doesn't provide topic_queries, generate default queries
        based on the original query and topic names.
        """
        if not self.topic_queries:
            # Generate default queries for each topic
            self.topic_queries = {
                topic: f"{self.original_query} {topic}" for topic in self.topics
            }
        else:
            # Ensure all topics have queries
            for topic in self.topics:
                if topic not in self.topic_queries:
                    self.topic_queries[topic] = f"{self.original_query} {topic}"
        return self


class MultiQueryExpansion(BaseModel):
    """Multiple query variations for better recall.

    Per RAG best practices: Generate 3-4 query variations BEFORE search
    to improve recall through different phrasings and perspectives.
    """

    original_query: str = Field(
        min_length=1,
        description="Original query to expand",
    )
    variations: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Query variations (rephrased, synonyms)",
    )
    broad_query: str = Field(
        min_length=1,
        description="More general version (step-back query)",
    )
    specific_query: str = Field(
        min_length=1,
        description="More specific version (with context)",
    )

    @field_validator("variations")
    @classmethod
    def validate_variations(cls, v: list[str]) -> list[str]:
        """Ensure variations are non-empty and unique."""
        variations = [q.strip() for q in v if q and q.strip()]
        if not variations:
            msg = "At least one variation is required"
            raise ValueError(msg)
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for q in variations:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_variations.append(q)
        return unique_variations


class TopicCompleteness(BaseModel):
    """Completeness evaluation for a single topic.

    Per Perplexity research: Evaluate coverage per topic, not single score.
    This enables gap-driven iteration - search only for uncovered topics.
    """

    topic: str = Field(
        min_length=1,
        description="Topic being evaluated",
    )
    covered: bool = Field(
        description="Whether topic is adequately covered",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Coverage score from 0.0 (not covered) to 1.0 (fully covered)",
    )
    evidence: str = Field(
        default="",
        description="Quote from results proving coverage (empty if not covered)",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="What's missing for this topic",
    )

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is finite (not NaN or infinity)."""
        if math.isnan(v):
            msg = "Score cannot be NaN (Not a Number)"
            raise ValueError(msg)
        if math.isinf(v):
            msg = "Score cannot be infinity"
            raise ValueError(msg)
        return v

    @field_validator("gaps")
    @classmethod
    def validate_gaps(cls, v: list[str]) -> list[str]:
        """Ensure gaps are non-empty strings."""
        return [gap.strip() for gap in v if gap and gap.strip()]


class TopicCompletenessEvaluation(BaseModel):
    """Complete topic-based evaluation result.

    Aggregates individual topic evaluations into overall assessment
    with gap-driven iteration support.
    """

    topics: list[TopicCompleteness] = Field(
        default_factory=list,
        description="Evaluation for each required topic",
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted average of topic scores",
    )
    reasoning: str = Field(
        min_length=1,
        description="Overall assessment reasoning",
    )
    uncovered_topics: list[str] = Field(
        default_factory=list,
        description="Topics that need more information",
    )
    gap_queries: list[str] = Field(
        default_factory=list,
        description="Specific queries to fill gaps",
    )

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float) -> float:
        """Validate overall score is finite (not NaN or infinity)."""
        if math.isnan(v):
            msg = "Overall score cannot be NaN (Not a Number)"
            raise ValueError(msg)
        if math.isinf(v):
            msg = "Overall score cannot be infinity"
            raise ValueError(msg)
        return v

    @field_validator("uncovered_topics", "gap_queries")
    @classmethod
    def validate_string_lists(cls, v: list[str]) -> list[str]:
        """Ensure list items are non-empty strings."""
        return [item.strip() for item in v if item and item.strip()]
