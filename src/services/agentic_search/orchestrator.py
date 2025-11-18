"""Agentic search orchestrator - main pipeline orchestration.

This module handles the core orchestration of the agentic search pipeline,
coordinating the stages of search:

**Standard Flow (execute_search):**
1. Local Knowledge Check (evaluates existing knowledge completeness)
2. Web Search (searches the web and ranks promising URLs)
3. Selective Crawling (crawls and indexes selected URLs)
4. Query Refinement (refines search queries for subsequent iterations)

**Deep Research Flow (execute_deep_search):**
1. Topic Decomposition (breaks query into essential topics)
2. Multi-Query Generation (creates query variations per topic)
3. Topic-Based Evaluation (evaluates coverage per topic)
4. Gap-Driven Iteration (searches only for uncovered topics)

Per Anthropic research: "Search strategy should mirror expert human research:
explore the landscape before drilling into specifics."
"""

import logging
from typing import Any

from fastmcp import Context
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from src.config import get_settings
from src.core.constants import (
    LLM_API_TIMEOUT_DEFAULT,
    MAX_RETRIES_DEFAULT,
    SCORE_IMPROVEMENT_THRESHOLD,
)
from src.core.exceptions import LLMError
from src.services.agentic_models import (
    AgenticSearchResult,
    QueryRefinement,
    RAGResult,
    SearchIteration,
    SearchStatus,
    TopicDecomposition,
)

from .query_enhancer import QueryEnhancer

logger = logging.getLogger(__name__)
settings = get_settings()


class AgenticSearchService:
    """Service for executing agentic search with iterative refinement.

    This service coordinates all stages of the agentic search pipeline,
    using LLM-driven decisions to minimize costs while maximizing answer quality.
    It uses modular components for each stage and manages the iteration loop.
    """

    def __init__(
        self,
        evaluator: Any,
        ranker: Any,
        crawler: Any,
        config: Any | None = None,
    ) -> None:
        """Initialize the agentic search service with modular components.

        Args:
            evaluator: LocalKnowledgeEvaluator instance for completeness evaluation
            ranker: URLRanker instance for URL ranking
            crawler: SelectiveCrawler instance for URL crawling
            config: Optional AgenticSearchConfig instance
        """
        # Store modular components
        self.evaluator = evaluator
        self.ranker = ranker
        self.crawler = crawler

        # Per Pydantic AI docs: Use ModelSettings for timeout and temperature
        # Create OpenAI model instance
        # API key is automatically read from OPENAI_API_KEY environment variable
        model = OpenAIModel(
            model_name=settings.model_choice,
        )

        # Shared model settings for all agents
        # Per Pydantic AI docs: timeout, temperature configured via ModelSettings
        self.base_model_settings = ModelSettings(
            temperature=settings.agentic_search_llm_temperature,
            timeout=LLM_API_TIMEOUT_DEFAULT,  # 60s timeout
        )

        # Query refinement agent with custom temperature for creativity
        # Per Pydantic AI docs: Use ModelSettings to override temperature per agent
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
        use_hints = (
            use_search_hints if use_search_hints is not None else self.use_search_hints
        )

        search_history: list[SearchIteration] = []
        current_query = query
        iteration = 0
        final_completeness = 0.0
        final_results: list[RAGResult] = []

        try:
            logger.info("Starting agentic search for query: %s", query)
            logger.info(
                "Parameters: threshold=%s, max_iter=%s, max_urls=%s, url_threshold=%s",
                threshold,
                max_iter,
                max_urls,
                url_threshold,
            )

            while iteration < max_iter:
                iteration += 1
                logger.info(
                    "Iteration %d/%d: Query='%s'", iteration, max_iter, current_query
                )

                # STAGE 1: Local Knowledge Check
                evaluation, rag_results = await self.evaluator.evaluate_local_knowledge(
                    ctx,
                    current_query,
                    iteration,
                    search_history,
                )

                final_completeness = evaluation.score
                final_results = rag_results

                # Check if we have sufficient answer
                if evaluation.score >= threshold:
                    logger.info(
                        "Completeness threshold met: %.2f >= %.2f",
                        evaluation.score,
                        threshold,
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
                    "Completeness insufficient: %.2f < %.2f",
                    evaluation.score,
                    threshold,
                )
                logger.info("Knowledge gaps: %s", evaluation.gaps)

                # STAGE 2: Web Search
                promising_urls = await self.ranker.search_and_rank_urls(
                    current_query,
                    evaluation.gaps,
                    max_urls,
                    url_threshold,
                    iteration,
                    search_history,
                )

                if not promising_urls:
                    logger.warning(
                        "No promising URLs found, attempting query refinement"
                    )
                    # STAGE 4: Query Refinement (when no good URLs)
                    if iteration < max_iter:
                        refined = await self._stage4_query_refinement(
                            query,
                            current_query,
                            evaluation.gaps,
                        )
                        current_query = refined.refined_queries[0]
                        logger.info("Refined query: %s", current_query)
                        continue
                    logger.info("Max iterations reached with no promising URLs")
                    break

                # STAGE 3: Selective Crawling & Indexing
                previous_score = final_completeness  # Save score before crawling
                urls_stored = await self.crawler.crawl_and_store(
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
                    logger.info(
                        "Re-checking completeness after storing %d URLs", urls_stored
                    )
                    (
                        evaluation,
                        rag_results,
                    ) = await self.evaluator.evaluate_local_knowledge(
                        ctx,
                        current_query,
                        iteration,
                        search_history,
                        is_recheck=True,
                    )

                    final_completeness = evaluation.score
                    final_results = rag_results

                    if evaluation.score >= threshold:
                        logger.info(
                            "Completeness threshold met after crawling: %.2f",
                            evaluation.score,
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
                            "Score improved significantly (%.2f → %.2f, +%.2f), skipping refinement (optimization)",
                            previous_score,
                            final_completeness,
                            score_improvement,
                        )
                    else:
                        refined = await self._stage4_query_refinement(
                            query,
                            current_query,
                            evaluation.gaps,
                        )
                        current_query = refined.refined_queries[0]

            # Max iterations reached
            logger.info("Max iterations reached: %d/%d", iteration, max_iter)
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
            logger.exception("Agentic search failed: %s", e)
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

    async def _stage4_query_refinement(
        self,
        original_query: str,
        current_query: str,
        gaps: list[str],
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
            # Per Pydantic AI docs: Can create Agent with any Pydantic model as result_type
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
            logger.error("Query refinement failed after retries: %s", e)
            raise LLMError("LLM query refinement failed after retries") from e

        except Exception as e:
            logger.exception("Unexpected error in query refinement: %s", e)
            raise LLMError("Query refinement failed") from e

    async def execute_deep_search(
        self,
        ctx: Context,
        query: str,
        completeness_threshold: float | None = None,
        max_iterations: int | None = None,
        max_urls_per_iteration: int | None = None,
        url_score_threshold: float | None = None,
        use_search_hints: bool | None = None,
    ) -> AgenticSearchResult:
        """Execute deep research with topic decomposition and gap-driven iteration.

        Per Anthropic research: "Search strategy should mirror expert human research:
        explore the landscape before drilling into specifics."

        This method implements the enhanced flow:
        1. Topic Decomposition - break query into essential topics
        2. Multi-Query Generation - create query variations per topic
        3. Topic-Based Evaluation - evaluate coverage per topic
        4. Gap-Driven Iteration - search only for uncovered topics

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
        """
        # Use overrides or defaults
        threshold = completeness_threshold or self.completeness_threshold
        max_iter = max_iterations or self.max_iterations
        max_urls = max_urls_per_iteration or self.max_urls_per_iteration
        url_threshold = url_score_threshold or self.url_score_threshold
        use_hints = (
            use_search_hints if use_search_hints is not None else self.use_search_hints
        )

        search_history: list[SearchIteration] = []
        iteration = 0
        final_completeness = 0.0
        final_results: list[RAGResult] = []

        try:
            logger.info("Starting DEEP RESEARCH for query: %s", query)
            logger.info(
                "Parameters: threshold=%s, max_iter=%s, max_urls=%s",
                threshold,
                max_iter,
                max_urls,
            )

            # Create query enhancer for this search
            # Import config here to avoid circular import
            from .config import AgenticSearchConfig

            config = AgenticSearchConfig()
            query_enhancer = QueryEnhancer(config)

            # PHASE 1: Topic Decomposition + Multi-Query Generation
            logger.info("PHASE 1: Topic decomposition and multi-query generation")
            decomposition, all_queries = await query_enhancer.enhance_query(
                query,
                decompose=True,
                expand=True,
            )

            if not decomposition:
                logger.warning(
                    "Topic decomposition failed, falling back to standard search"
                )
                return await self.execute_search(
                    ctx,
                    query,
                    completeness_threshold,
                    max_iterations,
                    max_urls_per_iteration,
                    url_score_threshold,
                    use_search_hints,
                )

            logger.info(
                "Decomposed into %d topics, generated %d queries",
                len(decomposition.topics),
                len(all_queries),
            )

            # PHASE 2: Initial Topic-Based Evaluation
            logger.info("PHASE 2: Initial topic-based evaluation")
            iteration = 1

            evaluation, rag_results = await self.evaluator.evaluate_with_topics(
                ctx,
                query,
                decomposition,
                all_queries,
                iteration,
                search_history,
            )

            final_completeness = evaluation.overall_score
            final_results = rag_results

            # Check if we have sufficient answer
            if evaluation.overall_score >= threshold:
                logger.info(
                    "Completeness threshold met on first check: %.2f >= %.2f",
                    evaluation.overall_score,
                    threshold,
                )
                return AgenticSearchResult(
                    success=True,
                    query=query,
                    iterations=iteration,
                    completeness=evaluation.overall_score,
                    results=rag_results,
                    search_history=search_history,
                    status=SearchStatus.COMPLETE,
                )

            logger.info(
                "Initial completeness: %.2f < %.2f, uncovered topics: %s",
                evaluation.overall_score,
                threshold,
                evaluation.uncovered_topics,
            )

            # PHASE 3: Gap-Driven Iteration
            while iteration < max_iter:
                iteration += 1
                logger.info(
                    "PHASE 3: Gap-driven iteration %d/%d",
                    iteration,
                    max_iter,
                )

                # Generate queries for uncovered topics
                if evaluation.uncovered_topics:
                    gap_queries = await query_enhancer.generate_gap_queries(
                        evaluation.uncovered_topics,
                        query,
                    )
                    # Also use LLM-suggested gap queries
                    gap_queries.extend(evaluation.gap_queries)
                else:
                    # Use LLM-suggested gap queries if no uncovered topics
                    gap_queries = evaluation.gap_queries

                if not gap_queries:
                    logger.warning("No gap queries generated, using original gaps")
                    gap_queries = [
                        f"{query} {gap}" for gap in evaluation.uncovered_topics[:3]
                    ]

                logger.info("Gap queries: %s", gap_queries[:5])

                # Web search for gaps
                promising_urls = await self.ranker.search_and_rank_urls(
                    gap_queries[0] if gap_queries else query,
                    evaluation.uncovered_topics,
                    max_urls,
                    url_threshold,
                    iteration,
                    search_history,
                )

                if not promising_urls:
                    logger.warning("No promising URLs found for gaps")
                    # Try with different gap query
                    if len(gap_queries) > 1:
                        promising_urls = await self.ranker.search_and_rank_urls(
                            gap_queries[1],
                            evaluation.uncovered_topics,
                            max_urls,
                            url_threshold,
                            iteration,
                            search_history,
                        )

                if promising_urls:
                    # Crawl and store
                    urls_stored = await self.crawler.crawl_and_store(
                        ctx,
                        promising_urls,
                        query,
                        use_hints,
                        iteration,
                        search_history,
                    )

                    if urls_stored > 0:
                        logger.info("Stored %d URLs, re-evaluating topics", urls_stored)

                        # Re-evaluate with all queries (including gap queries)
                        combined_queries = list(set(all_queries + gap_queries))
                        (
                            evaluation,
                            rag_results,
                        ) = await self.evaluator.evaluate_with_topics(
                            ctx,
                            query,
                            decomposition,
                            combined_queries,
                            iteration,
                            search_history,
                            is_recheck=True,
                        )

                        final_completeness = evaluation.overall_score
                        final_results = rag_results

                        if evaluation.overall_score >= threshold:
                            logger.info(
                                "Completeness threshold met: %.2f >= %.2f",
                                evaluation.overall_score,
                                threshold,
                            )
                            return AgenticSearchResult(
                                success=True,
                                query=query,
                                iterations=iteration,
                                completeness=evaluation.overall_score,
                                results=rag_results,
                                search_history=search_history,
                                status=SearchStatus.COMPLETE,
                            )

                        logger.info(
                            "After iteration %d: completeness=%.2f, uncovered=%s",
                            iteration,
                            evaluation.overall_score,
                            evaluation.uncovered_topics,
                        )
                else:
                    logger.warning("No URLs to crawl in iteration %d", iteration)

            # Max iterations reached
            logger.info(
                "Max iterations reached: %d/%d, final completeness: %.2f",
                iteration,
                max_iter,
                final_completeness,
            )
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
            logger.exception("Deep research failed: %s", e)
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
