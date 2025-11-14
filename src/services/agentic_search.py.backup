"""Agentic search service implementation.

This module implements the complete agentic search pipeline:
1. Local Knowledge Check (Qdrant + LLM completeness evaluation)
2. Web Search (SearXNG + LLM URL ranking)
3. Selective Crawling (Crawl4AI + Qdrant indexing)
4. Query Refinement (LLM-based iteration)

The implementation is production-ready with:
- Full type safety using Pydantic AI agents with structured outputs
- Automatic retry handling with validation
- Comprehensive error handling
- Detailed logging
- Configurable thresholds and limits
"""

import json
import logging
from typing import Any

from fastmcp import Context
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from src.config import get_settings
from src.core import MCPToolError
from src.core.exceptions import DatabaseError, LLMError
from src.core.constants import (
    LLM_API_TIMEOUT_DEFAULT,
    MAX_RETRIES_DEFAULT,
    SCORE_IMPROVEMENT_THRESHOLD,
)
from src.core.context import get_app_context
from src.database import perform_rag_query
from src.services.crawling import crawl_urls_for_agentic_search
from src.services.search import _search_searxng

from .agentic_models import (
    ActionType,
    AgenticSearchResult,
    CompletenessEvaluation,
    QueryRefinement,
    RAGResult,
    SearchIteration,
    SearchStatus,
    URLRanking,
    URLRankingList,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class AgenticSearchService:
    """Service for executing agentic search with iterative refinement.

    This service coordinates all stages of the agentic search pipeline,
    using LLM-driven decisions to minimize costs while maximizing answer quality.
    """

    def __init__(self) -> None:
        """Initialize the agentic search service with Pydantic AI agents."""
        # Per Pydantic AI docs: Use ModelSettings for timeout and temperature
        # Create OpenAI model instance with API key
        # Per Pydantic AI docs: OpenAIModel wraps the OpenAI client
        model = OpenAIModel(
            model_name=settings.model_choice,
            api_key=settings.openai_api_key,
        )

        # Shared model settings for all agents
        # Per Pydantic AI docs: timeout, temperature configured via ModelSettings
        self.base_model_settings = ModelSettings(
            temperature=settings.agentic_search_llm_temperature,
            timeout=LLM_API_TIMEOUT_DEFAULT,  # 60s timeout
        )

        # Create specialized agents for each LLM task
        # Per Pydantic AI docs: Agent with output_type for structured outputs
        self.completeness_agent = Agent(
            model=model,
            output_type=CompletenessEvaluation,
            output_retries=MAX_RETRIES_DEFAULT,  # Retry 3 times for validation errors
            model_settings=self.base_model_settings,
        )

        self.ranking_agent = Agent(
            model=model,
            output_type=URLRankingList,
            output_retries=MAX_RETRIES_DEFAULT,
            model_settings=self.base_model_settings,
        )

        # Query refinement agent with custom temperature for creativity
        # Per Pydantic AI docs: Use ModelSettings to override temperature per agent
        # Note: output_type defined inline in _stage4_query_refinement (dynamic model)
        refinement_settings = ModelSettings(
            temperature=0.5,  # More creative for query generation
            timeout=LLM_API_TIMEOUT_DEFAULT,
        )
        self.refinement_model_settings = refinement_settings
        self.openai_model = model  # Store for dynamic agent creation

        # Configuration parameters
        self.model_name = settings.model_choice
        self.temperature = settings.agentic_search_llm_temperature
        self.completeness_threshold = settings.agentic_search_completeness_threshold
        self.max_iterations = settings.agentic_search_max_iterations
        self.max_urls_per_iteration = settings.agentic_search_max_urls_per_iteration
        self.max_pages_per_iteration = settings.agentic_search_max_pages_per_iteration
        self.url_score_threshold = settings.agentic_search_url_score_threshold
        self.use_search_hints = settings.agentic_search_use_search_hints
        self.enable_url_filtering = settings.agentic_search_enable_url_filtering
        self.max_urls_to_rank = settings.agentic_search_max_urls_to_rank
        self.max_qdrant_results = settings.agentic_search_max_qdrant_results

    async def execute_search(
        self,
        ctx: Context,
        query: str,
        completeness_threshold: float | None = None,
        max_iterations: int | None = None,
        max_urls_per_iteration: int | None = None,
        url_score_threshold: float | None = None,
        use_search_hints: bool | None = None,
    ) -> AgenticSearchResult:
        """Execute agentic search with automatic refinement.

        Args:
            ctx: FastMCP context
            query: User's search query
            completeness_threshold: Override default completeness threshold
            max_iterations: Override default max iterations
            max_urls_per_iteration: Override default max URLs per iteration
            url_score_threshold: Override default URL score threshold
            use_search_hints: Override default search hints setting

        Returns:
            Complete search result with all iterations tracked

        Raises:
            MCPToolError: If search fails critically
        """
        # Use overrides or defaults
        threshold = completeness_threshold or self.completeness_threshold
        max_iter = max_iterations or self.max_iterations
        max_urls = max_urls_per_iteration or self.max_urls_per_iteration
        url_threshold = url_score_threshold or self.url_score_threshold
        use_hints = use_search_hints if use_search_hints is not None else self.use_search_hints

        search_history: list[SearchIteration] = []
        current_query = query
        iteration = 0
        final_completeness = 0.0
        final_results: list[RAGResult] = []

        try:
            logger.info(f"Starting agentic search for query: {query}")
            logger.info(
                f"Parameters: threshold={threshold}, max_iter={max_iter}, "
                f"max_urls={max_urls}, url_threshold={url_threshold}",
            )

            while iteration < max_iter:
                iteration += 1
                logger.info(f"Iteration {iteration}/{max_iter}: Query='{current_query}'")

                # STAGE 1: Local Knowledge Check
                evaluation, rag_results = await self._stage1_local_check(
                    ctx, current_query, iteration, search_history,
                )

                final_completeness = evaluation.score
                final_results = rag_results

                # Check if we have sufficient answer
                if evaluation.score >= threshold:
                    logger.info(
                        f"Completeness threshold met: {evaluation.score:.2f} >= {threshold:.2f}",
                    )
                    return AgenticSearchResult(
                        success=True,
                        query=query,
                        iterations=iteration,
                        completeness=evaluation.score,
                        results=rag_results,
                        search_history=search_history,
                        status=SearchStatus.COMPLETE,
                    )

                logger.info(
                    f"Completeness insufficient: {evaluation.score:.2f} < {threshold:.2f}",
                )
                logger.info(f"Knowledge gaps: {evaluation.gaps}")

                # STAGE 2: Web Search
                promising_urls = await self._stage2_web_search(
                    current_query,
                    evaluation.gaps,
                    max_urls,
                    url_threshold,
                    iteration,
                    search_history,
                )

                if not promising_urls:
                    logger.warning("No promising URLs found, attempting query refinement")
                    # STAGE 4: Query Refinement (when no good URLs)
                    if iteration < max_iter:
                        refined = await self._stage4_query_refinement(
                            query, current_query, evaluation.gaps,
                        )
                        current_query = refined.refined_queries[0]
                        logger.info(f"Refined query: {current_query}")
                        continue
                    logger.info("Max iterations reached with no promising URLs")
                    break

                # STAGE 3: Selective Crawling & Indexing
                previous_score = final_completeness  # Save score before crawling
                urls_stored = await self._stage3_selective_crawl(
                    ctx,
                    promising_urls,
                    current_query,
                    use_hints,
                    iteration,
                    search_history,
                )

                # OPTIMIZATION 1: Skip re-check if no content was stored
                # Saves 1 LLM call per failed crawl
                if urls_stored > 0:
                    logger.info(f"Re-checking completeness after storing {urls_stored} URLs")
                    evaluation, rag_results = await self._stage1_local_check(
                        ctx, current_query, iteration, search_history, is_recheck=True,
                    )

                    final_completeness = evaluation.score
                    final_results = rag_results

                    if evaluation.score >= threshold:
                        logger.info(
                            f"Completeness threshold met after crawling: {evaluation.score:.2f}",
                        )
                        return AgenticSearchResult(
                            success=True,
                            query=query,
                            iterations=iteration,
                            completeness=evaluation.score,
                            results=rag_results,
                            search_history=search_history,
                            status=SearchStatus.COMPLETE,
                        )
                else:
                    logger.info(
                        "No content stored during crawl, skipping re-check (optimization)",
                    )

                # OPTIMIZATION 2: Skip refinement if score improved significantly
                # Saves 1 LLM call when making good progress
                score_improvement = final_completeness - previous_score
                if iteration < max_iter:
                    if score_improvement >= SCORE_IMPROVEMENT_THRESHOLD:
                        logger.info(
                            f"Score improved significantly ({previous_score:.2f} → "
                            f"{final_completeness:.2f}, +{score_improvement:.2f}), "
                            f"skipping refinement (optimization)",
                        )
                    else:
                        refined = await self._stage4_query_refinement(
                            query, current_query, evaluation.gaps,
                        )
                        current_query = refined.refined_queries[0]

            # Max iterations reached
            logger.info(f"Max iterations reached: {iteration}/{max_iter}")
            return AgenticSearchResult(
                success=True,
                query=query,
                iterations=iteration,
                completeness=final_completeness,
                results=final_results,
                search_history=search_history,
                status=SearchStatus.MAX_ITERATIONS_REACHED,
            )

        except Exception as e:
            logger.exception(f"Agentic search failed: {e}")
            return AgenticSearchResult(
                success=False,
                query=query,
                iterations=iteration,
                completeness=final_completeness,
                results=final_results,
                search_history=search_history,
                status=SearchStatus.ERROR,
                error=str(e),
            )

    async def _stage1_local_check(
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
            match_count=self.max_qdrant_results,
        )

        # Parse RAG results
        rag_data = json.loads(rag_response)
        rag_results = []

        if rag_data.get("success") and rag_data.get("results"):
            for result in rag_data["results"]:
                rag_results.append(
                    RAGResult(
                        content=result.get("chunk", ""),
                        url=result.get("url", ""),
                        similarity_score=result.get("similarity_score", 0.0),
                        chunk_index=result.get("chunk_index", 0),
                    ),
                )

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
            result = await self.completeness_agent.run(prompt)
            return result.output

        except UnexpectedModelBehavior as e:
            # Per Pydantic AI docs: Raised when retries exhausted
            logger.error(f"Completeness evaluation failed after retries: {e}")
            raise LLMError("LLM completeness evaluation failed after retries") from e

        except Exception as e:
            logger.exception(f"Unexpected error in completeness evaluation: {e}")
            raise LLMError("Completeness evaluation failed") from e

    async def _stage2_web_search(
        self,
        query: str,
        gaps: list[str],
        max_urls: int,
        url_threshold: float,
        iteration: int,
        search_history: list[SearchIteration],
    ) -> list[str]:
        """STAGE 2: Perform web search and rank URLs.

        Args:
            query: Search query
            gaps: Knowledge gaps to fill
            max_urls: Maximum URLs to select
            url_threshold: Minimum relevance score
            iteration: Current iteration number
            search_history: History to append to

        Returns:
            List of promising URLs to crawl
        """
        logger.info("STAGE 2: Web search and URL ranking")

        # Search SearXNG
        search_results = await _search_searxng(query, num_results=20)

        if not search_results:
            logger.warning("No search results found")
            search_history.append(
                SearchIteration(
                    iteration=iteration,
                    query=query,
                    action=ActionType.WEB_SEARCH,
                    urls_found=0,
                    urls_ranked=0,
                    promising_urls=0,
                ),
            )
            return []

        logger.info(f"Found {len(search_results)} search results")

        # OPTIMIZATION 3: Limit URLs to rank (configurable)
        # Reduces LLM tokens and speeds up ranking
        if len(search_results) > self.max_urls_to_rank:
            logger.info(
                f"Limiting ranking to top {self.max_urls_to_rank} of {len(search_results)} results",
            )
            search_results = search_results[: self.max_urls_to_rank]

        # LLM ranking of URLs
        rankings = await self._rank_urls(query, gaps, search_results)

        # Filter promising URLs
        promising = [r for r in rankings if r.score >= url_threshold]
        promising = promising[:max_urls]  # Limit to max URLs

        logger.info(
            f"Ranked {len(rankings)} URLs, {len(promising)} above threshold {url_threshold}",
        )

        search_history.append(
            SearchIteration(
                iteration=iteration,
                query=query,
                action=ActionType.WEB_SEARCH,
                urls_found=len(search_results),
                urls_ranked=len(rankings),
                promising_urls=len(promising),
            ),
        )

        return [r.url for r in promising]

    async def _rank_urls(
        self, query: str, gaps: list[str], search_results: list[dict[str, Any]],
    ) -> list[URLRanking]:
        """Rank URLs by relevance using LLM with Pydantic structured output.

        Args:
            query: User's query
            gaps: Knowledge gaps to fill
            search_results: Search results from SearXNG

        Returns:
            List of ranked URLs sorted by score descending
        """
        # Format search results for LLM
        results_text = "\n".join(
            [
                f"{i+1}. {r['title']}\n   URL: {r['url']}\n   Snippet: {r.get('snippet', '')[:200]}"
                for i, r in enumerate(search_results)
            ],
        )

        gaps_text = "\n".join([f"- {gap}" for gap in gaps])

        prompt = f"""You are evaluating which URLs are most likely to contain information that fills specific knowledge gaps.

User Query: {query}

Knowledge Gaps to Fill:
{gaps_text if gaps_text else "[General information needed]"}

Search Results:
{results_text}

For each URL, provide a relevance score (0.0-1.0) indicating how likely it is to contain valuable information:
- 0.0-0.3: Unlikely to be relevant
- 0.4-0.6: Possibly relevant
- 0.7-0.8: Likely relevant
- 0.9-1.0: Highly relevant

Be selective - most URLs should score below 0.7.

Return a list of rankings with url, title, snippet, score, and reasoning for each."""

        try:
            # Per Pydantic AI docs: agent.run() returns RunResult with typed output
            result = await self.ranking_agent.run(prompt)
            rankings = result.output.rankings

            # Sort by score descending
            rankings.sort(key=lambda r: r.score, reverse=True)
            return rankings

        except UnexpectedModelBehavior as e:
            # Per Pydantic AI docs: Raised when retries exhausted
            logger.error(f"URL ranking failed after retries: {e}")
            raise LLMError("LLM URL ranking failed after retries") from e

        except Exception as e:
            logger.exception(f"Unexpected error in URL ranking: {e}")
            raise LLMError("URL ranking failed") from e

    async def _stage3_selective_crawl(
        self,
        ctx: Context,
        urls: list[str],
        query: str,
        use_hints: bool,
        iteration: int,
        search_history: list[SearchIteration],
    ) -> int:
        """STAGE 3: Crawl promising URLs recursively with smart filtering and limits.

        Args:
            ctx: FastMCP context
            urls: URLs to crawl (starting points)
            query: Original query
            use_hints: Whether to use search hints
            iteration: Current iteration number
            search_history: History to append to

        Returns:
            Number of URLs successfully stored in Qdrant
        """
        logger.info(f"STAGE 3: Recursively crawling {len(urls)} promising URLs")

        # HIGH PRIORITY FIX #10: Duplicate detection - filter out already crawled URLs
        # Uses Qdrant count() for efficient existence check (per Qdrant docs)
        app_ctx = get_app_context(ctx)
        database_client = app_ctx.database_client
        urls_to_crawl = []
        urls_skipped = 0

        for url in urls:
            # Check if URL already exists in Qdrant (efficient count-based check)
            try:
                exists = await database_client.url_exists(url)
                if exists:
                    logger.info(f"Skipping duplicate URL (already in database): {url}")
                    urls_skipped += 1
                else:
                    urls_to_crawl.append(url)
            except DatabaseError as e:
                # On database error, include URL (fail open)
                logger.warning(f"Database error checking duplicate for {url}: {e}")
                urls_to_crawl.append(url)
            except Exception as e:
                # On unexpected error, include URL (fail open)
                logger.warning(f"Unexpected error checking duplicate for {url}: {e}")
                urls_to_crawl.append(url)

        if urls_skipped > 0:
            logger.info(
                f"Filtered {urls_skipped}/{len(urls)} duplicate URLs, "
                f"crawling {len(urls_to_crawl)} new URLs",
            )

        if not urls_to_crawl:
            logger.info("All URLs already in database, skipping crawl")
            return 0

        # Crawl recursively with smart limits and filtering
        crawl_result = await crawl_urls_for_agentic_search(
            ctx=ctx,
            urls=urls_to_crawl,  # Use filtered URLs (duplicates removed)
            max_pages=self.max_pages_per_iteration,
            enable_url_filtering=self.enable_url_filtering,
        )

        # Extract results
        urls_crawled = crawl_result.get("urls_crawled", 0)
        urls_stored = crawl_result.get("urls_stored", 0)
        chunks_stored = crawl_result.get("chunks_stored", 0)
        urls_filtered = crawl_result.get("urls_filtered", 0)

        logger.info(
            f"Crawled {urls_crawled} pages, stored {urls_stored} URLs, "
            f"{chunks_stored} chunks, filtered {urls_filtered} URLs",
        )

        search_history.append(
            SearchIteration(
                iteration=iteration,
                query=query,
                action=ActionType.CRAWL,
                urls=urls,
                urls_stored=urls_stored,
                chunks_stored=chunks_stored,
            ),
        )

        # Note: Search hints feature requires Crawl4AI metadata capabilities
        # Currently not implemented - would generate optimized Qdrant queries from metadata
        if use_hints:
            logger.info("Search hints requested but not yet implemented")

        return urls_stored

    async def _stage4_query_refinement(
        self, original_query: str, current_query: str, gaps: list[str],
    ) -> QueryRefinement:
        """STAGE 4: Generate refined queries using LLM with Pydantic structured output.

        Args:
            original_query: Original user query
            current_query: Current query being used
            gaps: Knowledge gaps identified

        Returns:
            Query refinement with alternative queries
        """
        logger.info("STAGE 4: Query refinement")

        gaps_text = "\n".join([f"- {gap}" for gap in gaps])

        prompt = f"""You are helping refine a search query to fill knowledge gaps.

Original Query: {original_query}
Current Query: {current_query}

Knowledge Gaps:
{gaps_text}

Generate 2-3 refined search queries that are more likely to find information filling these gaps.
Make queries more specific, use different terminology, or approach from different angles.

Provide:
- refined_queries: list of 2-3 alternative search queries
- reasoning: Brief explanation of the refinement strategy"""

        try:
            # Create a temporary model for this response
            # Per Pydantic AI docs: Can create Agent with any Pydantic model as output_type
            class QueryRefinementResponse(BaseModel):
                """Response model for query refinement."""

                refined_queries: list[str] = Field(
                    min_length=1,
                    max_length=3,
                    description="List of refined search queries",
                )
                reasoning: str = Field(
                    min_length=1,
                    description="Explanation of refinement strategy",
                )

            # Create temporary agent with inline response model
            # Per Pydantic AI docs: Create Agent instance with specific output_type
            refinement_agent = Agent(
                model=self.openai_model,
                output_type=QueryRefinementResponse,
                output_retries=MAX_RETRIES_DEFAULT,
                model_settings=self.refinement_model_settings,
            )

            # Per Pydantic AI docs: agent.run() returns RunResult with typed output
            result = await refinement_agent.run(prompt)
            parsed = result.output

            return QueryRefinement(
                original_query=original_query,
                current_query=current_query,
                refined_queries=parsed.refined_queries,
                reasoning=parsed.reasoning,
            )

        except UnexpectedModelBehavior as e:
            # Per Pydantic AI docs: Raised when retries exhausted
            logger.error(f"Query refinement failed after retries: {e}")
            raise LLMError("LLM query refinement failed after retries") from e

        except Exception as e:
            logger.exception(f"Unexpected error in query refinement: {e}")
            raise LLMError("Query refinement failed") from e


# Singleton instance for connection pooling (per Pydantic AI best practices)
_agentic_search_service: AgenticSearchService | None = None


def get_agentic_search_service() -> AgenticSearchService:
    """Get singleton instance of AgenticSearchService.

    Per Pydantic AI docs: Reuse Agent instances to benefit from connection pooling.
    Creating new agents for every request wastes resources and degrades performance.

    Returns:
        Singleton AgenticSearchService instance with cached Pydantic AI agents
    """
    global _agentic_search_service
    if _agentic_search_service is None:
        _agentic_search_service = AgenticSearchService()
    return _agentic_search_service


async def agentic_search_impl(
    ctx: Context,
    query: str,
    completeness_threshold: float | None = None,
    max_iterations: int | None = None,
    max_urls_per_iteration: int | None = None,
    url_score_threshold: float | None = None,
    use_search_hints: bool | None = None,
) -> str:
    """Execute agentic search and return JSON result.

    This is the main entry point called by the MCP tool.

    Args:
        ctx: FastMCP context
        query: User's search query
        completeness_threshold: Override default completeness threshold (0-1)
        max_iterations: Override default max iterations (1-10)
        max_urls_per_iteration: Override default max URLs per iteration (1-20)
        url_score_threshold: Override default URL score threshold (0-1)
        use_search_hints: Override default search hints setting

    Returns:
        JSON string with complete search results

    Raises:
        MCPToolError: If search fails
    """
    if not settings.agentic_search_enabled:
        msg = "Agentic search is not enabled. Set AGENTIC_SEARCH_ENABLED=true in your environment."
        raise MCPToolError(
            msg,
        )

    try:
        # Get singleton service instance (connection pooling optimization)
        service = get_agentic_search_service()
        result = await service.execute_search(
            ctx=ctx,
            query=query,
            completeness_threshold=completeness_threshold,
            max_iterations=max_iterations,
            max_urls_per_iteration=max_urls_per_iteration,
            url_score_threshold=url_score_threshold,
            use_search_hints=use_search_hints,
        )
        return result.model_dump_json()

    except Exception as e:
        logger.exception("Agentic search implementation failed")
        msg = f"Agentic search failed: {e!s}"
        raise MCPToolError(msg) from e
