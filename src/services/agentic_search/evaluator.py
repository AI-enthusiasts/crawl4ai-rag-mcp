"""Local knowledge evaluation for agentic search (Stage 1).

This module handles:
- Querying Qdrant vector database for existing knowledge
- LLM-based completeness evaluation (simple and topic-based)
- Gap identification for guiding web search
- Topic-based completeness for gap-driven iteration
"""

import json
import logging
from typing import Any

from fastmcp import Context
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.core import MCPToolError
from src.core.constants import MAX_RETRIES_DEFAULT
from src.core.context import get_app_context
from src.core.exceptions import LLMError
from src.database import perform_rag_query, perform_multi_query_search
from src.services.agentic_models import (
    ActionType,
    CompletenessEvaluation,
    RAGResult,
    SearchIteration,
    TopicCompleteness,
    TopicCompletenessEvaluation,
    TopicDecomposition,
)

from .config import AgenticSearchConfig

logger = logging.getLogger(__name__)


class LocalKnowledgeEvaluator:
    """Evaluates existing local knowledge using Qdrant + LLM.

    Supports both simple completeness evaluation and topic-based evaluation
    for gap-driven iteration.
    """

    def __init__(self, config: AgenticSearchConfig) -> None:
        """Initialize evaluator with shared configuration.

        Args:
            config: Shared agentic search configuration with agents
        """
        self.config = config

        # Create topic completeness agent for topic-based evaluation
        self.topic_completeness_agent = Agent(
            model=config.openai_model,
            output_type=TopicCompletenessEvaluation,
            output_retries=MAX_RETRIES_DEFAULT,
            model_settings=config.base_model_settings,
        )

    async def evaluate_local_knowledge(
        self,
        ctx: Context,
        query: str,
        iteration: int,
        search_history: list[SearchIteration],
        is_recheck: bool = False,
    ) -> tuple[CompletenessEvaluation, list[RAGResult]]:
        """STAGE 1: Check local knowledge and evaluate completeness.

        Args:
            ctx: FastMCP context
            query: Search query
            iteration: Current iteration number
            search_history: History to append to
            is_recheck: Whether this is a re-check after crawling

        Returns:
            Tuple of (evaluation, RAG results)
        """
        logger.info(
            "STAGE 1: Local knowledge check %s", "(recheck)" if is_recheck else ""
        )

        # Get app context for database client
        app_ctx = get_app_context()
        if not app_ctx or not app_ctx.database_client:
            msg = "Database client not available"
            raise MCPToolError(msg)

        # Query Qdrant with all RAG enhancements
        rag_response = await perform_rag_query(
            app_ctx.database_client,
            query=query,
            source=None,
            match_count=self.config.max_qdrant_results,
        )

        # Parse RAG results
        rag_data = json.loads(rag_response)
        rag_results = []

        if rag_data.get("success") and rag_data.get("results"):
            for result in rag_data["results"]:
                # Skip invalid results (empty content or missing chunk_index)
                # Prevents Pydantic errors when Qdrant returns malformed data
                content = result.get("content", "")
                chunk_index = result.get("chunk_index")
                url = result.get("url", "unknown")

                if not content or content.strip() == "":
                    logger.warning(
                        "Skipping RAG result with empty content from %s",
                        url,
                    )
                    continue

                if chunk_index is None:
                    logger.warning(
                        "Skipping RAG result with missing chunk_index from %s",
                        url,
                    )
                    continue

                try:
                    rag_results.append(
                        RAGResult(
                            content=content,
                            url=result.get("url", ""),
                            similarity_score=result.get("similarity_score", 0.0),
                            chunk_index=chunk_index,
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to create RAGResult from %s: %s",
                        url,
                        e,
                    )
                    continue

        # LLM evaluation of completeness
        evaluation = await self._evaluate_completeness(query, rag_results)

        # Record iteration
        if not is_recheck:
            search_history.append(
                SearchIteration(
                    iteration=iteration,
                    query=query,
                    action=ActionType.LOCAL_CHECK,
                    completeness=evaluation.score,
                    gaps=evaluation.gaps,
                ),
            )

        return evaluation, rag_results

    async def _evaluate_completeness(
        self,
        query: str,
        results: list[RAGResult],
    ) -> CompletenessEvaluation:
        """Evaluate answer completeness using LLM with Pydantic structured output.

        Args:
            query: User's query
            results: RAG results from Qdrant

        Returns:
            Completeness evaluation from LLM
        """
        # Format results for LLM
        results_text = "\n\n".join(
            [
                f"[Result {i + 1}]\n{r.content[:500]}..."
                for i, r in enumerate(results[:5])
            ],
        )

        prompt = f"""You are evaluating whether the provided information is sufficient to answer a user's query.

User Query: {query}

Available Information:
{results_text if results_text else "[No information available]"}

Evaluate the completeness of the available information on a scale of 0.0 to 1.0:
- 0.0: No relevant information
- 0.5: Partial information, significant gaps
- 0.8: Most information present, minor gaps
- 1.0: Complete and comprehensive answer possible

Be strict in your evaluation. Score should be 0.95 or higher only if the information is truly comprehensive.

Provide:
- score: float between 0.0 and 1.0
- reasoning: Brief explanation of the score
- gaps: List of missing information or knowledge gaps (empty list if complete)"""

        try:
            # Per Pydantic AI docs: agent.run() returns RunResult with typed output
            # Retries and validation handled automatically
            result = await self.config.completeness_agent.run(prompt)
            return result.output

        except UnexpectedModelBehavior as e:
            # Per Pydantic AI docs: Raised when retries exhausted
            logger.error("Completeness evaluation failed after retries: %s", e)
            raise LLMError("LLM completeness evaluation failed after retries") from e

        except Exception as e:
            # Handle specific error types with more helpful messages
            error_type = type(e).__name__
            error_msg = str(e)

            logger.exception(
                "Unexpected error in completeness evaluation: %s: %s",
                error_type,
                error_msg,
            )

            # Provide more specific error messages based on error type
            if "APIConnectionError" in error_type or "ConnectError" in error_type:
                raise LLMError(
                    f"Failed to connect to OpenAI API. Check network connectivity and API access. "
                    f"Error: {error_msg}",
                ) from e
            if "AuthenticationError" in error_type or "Invalid API" in error_msg:
                raise LLMError(
                    "OpenAI API authentication failed. Check OPENAI_API_KEY environment variable.",
                ) from e
            if "RateLimitError" in error_type:
                raise LLMError(
                    "OpenAI API rate limit exceeded. Please try again later."
                ) from e
            raise LLMError(
                f"Completeness evaluation failed: {error_type}: {error_msg}"
            ) from e

    async def evaluate_with_topics(
        self,
        ctx: Context,
        query: str,
        decomposition: TopicDecomposition,
        queries: list[str],
        iteration: int,
        search_history: list[SearchIteration],
        is_recheck: bool = False,
    ) -> tuple[TopicCompletenessEvaluation, list[RAGResult]]:
        """Evaluate local knowledge with topic-based completeness.

        Per Perplexity research: Evaluate coverage per topic, not single score.
        This enables gap-driven iteration - search only for uncovered topics.

        Args:
            ctx: FastMCP context
            query: Original user query
            decomposition: Topic decomposition from query enhancer
            queries: All query variations to search
            iteration: Current iteration number
            search_history: History to append to
            is_recheck: Whether this is a re-check after crawling

        Returns:
            Tuple of (topic-based evaluation, RAG results)
        """
        logger.info(
            "STAGE 1: Topic-based knowledge check %s (%d topics, %d queries)",
            "(recheck)" if is_recheck else "",
            len(decomposition.topics),
            len(queries),
        )

        # Get app context for database client
        app_ctx = get_app_context()
        if not app_ctx or not app_ctx.database_client:
            msg = "Database client not available"
            raise MCPToolError(msg)

        # Perform multi-query search with RRF fusion
        results = await perform_multi_query_search(
            app_ctx.database_client,
            queries=queries,
            source=None,
            match_count=self.config.max_qdrant_results,
            use_rrf=True,
        )

        # Convert to RAGResult objects
        rag_results = self._convert_to_rag_results(results)

        # LLM evaluation of topic completeness
        evaluation = await self._evaluate_topic_completeness(
            query,
            decomposition.topics,
            rag_results,
        )

        # Record iteration
        if not is_recheck:
            search_history.append(
                SearchIteration(
                    iteration=iteration,
                    query=query,
                    action=ActionType.LOCAL_CHECK,
                    completeness=evaluation.overall_score,
                    gaps=evaluation.uncovered_topics,
                ),
            )

        return evaluation, rag_results

    def _convert_to_rag_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[RAGResult]:
        """Convert raw search results to RAGResult objects.

        Args:
            results: Raw results from multi-query search

        Returns:
            List of validated RAGResult objects
        """
        rag_results = []

        for result in results:
            content = result.get("content", "")
            chunk_index = result.get("chunk_index", 0)
            url = result.get("url", "unknown")

            if not content or content.strip() == "":
                logger.warning(
                    "Skipping RAG result with empty content from %s",
                    url,
                )
                continue

            try:
                rag_results.append(
                    RAGResult(
                        content=content,
                        url=result.get("url", ""),
                        similarity_score=result.get("similarity_score", 0.0),
                        chunk_index=chunk_index,
                    ),
                )
            except Exception as e:
                logger.warning(
                    "Failed to create RAGResult from %s: %s",
                    url,
                    e,
                )
                continue

        return rag_results

    async def _evaluate_topic_completeness(
        self,
        query: str,
        topics: list[str],
        results: list[RAGResult],
    ) -> TopicCompletenessEvaluation:
        """Evaluate completeness for each topic using LLM.

        Per Perplexity research: Evaluate coverage per topic, not single score.

        Args:
            query: User's original query
            topics: Required topics from decomposition
            results: RAG results from multi-query search

        Returns:
            Topic-based completeness evaluation
        """
        # If no results, return empty evaluation without LLM call
        if not results:
            logger.info("No results to evaluate, returning empty topic evaluation")
            return TopicCompletenessEvaluation(
                topics=[
                    TopicCompleteness(
                        topic=topic,
                        covered=False,
                        score=0.0,
                        evidence="",
                        gaps=[f"No information found for {topic}"],
                    )
                    for topic in topics
                ],
                overall_score=0.0,
                reasoning="No information available in knowledge base",
                uncovered_topics=topics,
                gap_queries=[f"{query} {topic}" for topic in topics[:3]],
            )

        # Format results for LLM (more content for topic evaluation)
        results_text = "\n\n".join(
            [
                f"[Source {i + 1}: {r.url}]\n{r.content[:1000]}..."
                for i, r in enumerate(results[:10])
            ],
        )

        topics_text = "\n".join([f"- {topic}" for topic in topics])

        prompt = f"""You are evaluating whether retrieved information covers all required topics for a user's query.

User Query: "{query}"

Required Topics:
{topics_text}

Retrieved Information:
{results_text if results_text else "[No information available]"}

For EACH topic, evaluate:
1. covered: Is the topic adequately addressed? (true/false)
2. score: Coverage quality from 0.0 (not covered) to 1.0 (fully covered)
3. evidence: If covered, quote the relevant text (max 100 chars). Empty if not covered.
4. gaps: What specific information is missing for this topic?

Then provide:
- overall_score: Weighted average of topic scores (0.0-1.0)
- reasoning: Brief overall assessment
- uncovered_topics: List of topic names that need more information
- gap_queries: 2-3 specific search queries to fill the gaps

Be strict: A topic is only "covered" if there's substantial, relevant information.
Score 0.8+ only if the topic is well-explained with examples or details."""

        try:
            result = await self.topic_completeness_agent.run(prompt)
            evaluation = result.output

            logger.info(
                "Topic evaluation: overall=%.2f, covered=%d/%d, uncovered=%s",
                evaluation.overall_score,
                sum(1 for t in evaluation.topics if t.covered),
                len(evaluation.topics),
                evaluation.uncovered_topics,
            )

            return evaluation

        except UnexpectedModelBehavior as e:
            logger.error("Topic completeness evaluation failed after retries: %s", e)
            raise LLMError(
                "LLM topic completeness evaluation failed after retries"
            ) from e

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.exception(
                "Unexpected error in topic completeness evaluation: %s: %s",
                error_type,
                error_msg,
            )
            raise LLMError(
                f"Topic completeness evaluation failed: {error_type}: {error_msg}"
            ) from e
