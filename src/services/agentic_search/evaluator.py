"""Local knowledge evaluation for agentic search (Stage 1).

This module handles:
- Querying Qdrant vector database for existing knowledge
- LLM-based completeness evaluation
- Gap identification for guiding web search
"""

import json
import logging

from fastmcp import Context
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.core import MCPToolError
from src.core.context import get_app_context
from src.core.exceptions import LLMError
from src.database import perform_rag_query
from src.services.agentic_models import (
    ActionType,
    CompletenessEvaluation,
    RAGResult,
    SearchIteration,
)

from .config import AgenticSearchConfig

logger = logging.getLogger(__name__)


class LocalKnowledgeEvaluator:
    """Evaluates existing local knowledge using Qdrant + LLM."""

    def __init__(self, config: AgenticSearchConfig) -> None:
        """Initialize evaluator with shared configuration.

        Args:
            config: Shared agentic search configuration with agents
        """
        self.config = config

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
        logger.info(f"STAGE 1: Local knowledge check {'(recheck)' if is_recheck else ''}")

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
        self, query: str, results: list[RAGResult],
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
                f"[Result {i+1}]\n{r.content[:500]}..."
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
            logger.error(f"Completeness evaluation failed after retries: {e}")
            raise LLMError("LLM completeness evaluation failed after retries") from e

        except Exception as e:
            # Handle specific error types with more helpful messages
            error_type = type(e).__name__
            error_msg = str(e)

            logger.exception(f"Unexpected error in completeness evaluation: {error_type}: {error_msg}")

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
                raise LLMError("OpenAI API rate limit exceeded. Please try again later.") from e
            raise LLMError(f"Completeness evaluation failed: {error_type}: {error_msg}") from e
