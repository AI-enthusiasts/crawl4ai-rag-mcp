"""Agentic search service implementation.

This module implements the complete agentic search pipeline:
1. Local Knowledge Check (Qdrant + LLM completeness evaluation)
2. Web Search (SearXNG + LLM URL ranking)
3. Selective Crawling (Crawl4AI + Qdrant indexing)
4. Query Refinement (LLM-based iteration)

The implementation is production-ready with:
- Full type safety using Pydantic models
- Comprehensive error handling
- Detailed logging
- Configurable thresholds and limits
"""

import json
import logging
from typing import Any

import openai
from fastmcp import Context
from openai import AsyncOpenAI

from config import get_settings
from core import MCPToolError
from core.context import get_app_context
from database import perform_rag_query
from services.crawling import process_urls_for_mcp
from services.search import _search_searxng

from .agentic_models import (
    ActionType,
    AgenticSearchResult,
    CompletenessEvaluation,
    QueryRefinement,
    RAGResult,
    SearchHints,
    SearchIteration,
    SearchMetadata,
    SearchStatus,
    URLRanking,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class AgenticSearchService:
    """Service for executing agentic search with iterative refinement.

    This service coordinates all stages of the agentic search pipeline,
    using LLM-driven decisions to minimize costs while maximizing answer quality.
    """

    def __init__(self) -> None:
        """Initialize the agentic search service."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.model_choice
        self.temperature = settings.agentic_search_llm_temperature
        self.completeness_threshold = settings.agentic_search_completeness_threshold
        self.max_iterations = settings.agentic_search_max_iterations
        self.max_urls_per_iteration = settings.agentic_search_max_urls_per_iteration
        self.url_score_threshold = settings.agentic_search_url_score_threshold
        self.use_search_hints = settings.agentic_search_use_search_hints
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
                f"max_urls={max_urls}, url_threshold={url_threshold}"
            )

            while iteration < max_iter:
                iteration += 1
                logger.info(f"Iteration {iteration}/{max_iter}: Query='{current_query}'")

                # STAGE 1: Local Knowledge Check
                evaluation, rag_results = await self._stage1_local_check(
                    ctx, current_query, iteration, search_history
                )

                final_completeness = evaluation.score
                final_results = rag_results

                # Check if we have sufficient answer
                if evaluation.score >= threshold:
                    logger.info(
                        f"Completeness threshold met: {evaluation.score:.2f} >= {threshold:.2f}"
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
                    f"Completeness insufficient: {evaluation.score:.2f} < {threshold:.2f}"
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
                            query, current_query, evaluation.gaps
                        )
                        current_query = refined.refined_queries[0]
                        logger.info(f"Refined query: {current_query}")
                        continue
                    else:
                        logger.info("Max iterations reached with no promising URLs")
                        break

                # STAGE 3: Selective Crawling & Indexing
                await self._stage3_selective_crawl(
                    ctx,
                    promising_urls,
                    current_query,
                    use_hints,
                    iteration,
                    search_history,
                )

                # Re-evaluate after crawling
                evaluation, rag_results = await self._stage1_local_check(
                    ctx, current_query, iteration, search_history, is_recheck=True
                )

                final_completeness = evaluation.score
                final_results = rag_results

                if evaluation.score >= threshold:
                    logger.info(
                        f"Completeness threshold met after crawling: {evaluation.score:.2f}"
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

                # Still incomplete, try refining query for next iteration
                if iteration < max_iter:
                    refined = await self._stage4_query_refinement(
                        query, current_query, evaluation.gaps
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
            raise MCPToolError("Database client not available")

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
                    )
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
                )
            )

        return evaluation, rag_results

    async def _evaluate_completeness(
        self, query: str, results: list[RAGResult]
    ) -> CompletenessEvaluation:
        """Evaluate answer completeness using LLM.

        Args:
            query: User's query
            results: RAG results from Qdrant

        Returns:
            Completeness evaluation from LLM
        """
        # Format results for LLM
        results_text = "\n\n".join(
            [f"[Result {i+1}]\n{r.content[:500]}..." for i, r in enumerate(results[:5])]
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

Respond with a JSON object:
{{
  "score": 0.0-1.0,
  "reasoning": "Brief explanation of the score",
  "gaps": ["missing topic 1", "unclear aspect 2", ...]
}}

Be strict in your evaluation. Score should be 0.95 or higher only if the information is truly comprehensive."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            data = json.loads(content)
            return CompletenessEvaluation(**data)

        except (openai.OpenAIError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Completeness evaluation failed: {e}")
            # Return safe default
            return CompletenessEvaluation(
                score=0.0,
                reasoning=f"Evaluation failed: {e}",
                gaps=["Unable to evaluate completeness"],
            )

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
                )
            )
            return []

        logger.info(f"Found {len(search_results)} search results")

        # LLM ranking of URLs
        rankings = await self._rank_urls(query, gaps, search_results)

        # Filter promising URLs
        promising = [r for r in rankings if r.score >= url_threshold]
        promising = promising[:max_urls]  # Limit to max URLs

        logger.info(
            f"Ranked {len(rankings)} URLs, {len(promising)} above threshold {url_threshold}"
        )

        search_history.append(
            SearchIteration(
                iteration=iteration,
                query=query,
                action=ActionType.WEB_SEARCH,
                urls_found=len(search_results),
                urls_ranked=len(rankings),
                promising_urls=len(promising),
            )
        )

        return [r.url for r in promising]

    async def _rank_urls(
        self, query: str, gaps: list[str], search_results: list[dict[str, Any]]
    ) -> list[URLRanking]:
        """Rank URLs by relevance using LLM.

        Args:
            query: User's query
            gaps: Knowledge gaps to fill
            search_results: Search results from SearXNG

        Returns:
            List of ranked URLs
        """
        # Format search results for LLM
        results_text = "\n".join(
            [
                f"{i+1}. {r['title']}\n   URL: {r['url']}\n   Snippet: {r.get('snippet', '')[:200]}"
                for i, r in enumerate(search_results)
            ]
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

Respond with a JSON array:
[
  {{
    "url": "...",
    "title": "...",
    "snippet": "...",
    "score": 0.0-1.0,
    "reasoning": "Brief explanation"
  }},
  ...
]

Be selective - most URLs should score below 0.7."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            # Handle both array and object with "rankings" key
            data = json.loads(content)
            rankings_data = data if isinstance(data, list) else data.get("rankings", [])

            rankings = []
            for item in rankings_data:
                try:
                    rankings.append(URLRanking(**item))
                except Exception as e:
                    logger.warning(f"Failed to parse ranking: {e}")
                    continue

            # Sort by score descending
            rankings.sort(key=lambda r: r.score, reverse=True)
            return rankings

        except (openai.OpenAIError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"URL ranking failed: {e}")
            # Return all URLs with neutral score
            return [
                URLRanking(
                    url=r["url"],
                    title=r["title"],
                    snippet=r.get("snippet", ""),
                    score=0.5,
                    reasoning="Ranking failed, using neutral score",
                )
                for r in search_results
            ]

    async def _stage3_selective_crawl(
        self,
        ctx: Context,
        urls: list[str],
        query: str,
        use_hints: bool,
        iteration: int,
        search_history: list[SearchIteration],
    ) -> None:
        """STAGE 3: Crawl promising URLs and index in Qdrant.

        Args:
            ctx: FastMCP context
            urls: URLs to crawl
            query: Original query
            use_hints: Whether to use search hints
            iteration: Current iteration number
            search_history: History to append to
        """
        logger.info(f"STAGE 3: Crawling {len(urls)} promising URLs")

        # Crawl and index URLs (process_urls_for_mcp handles full indexing)
        crawl_result = await process_urls_for_mcp(
            ctx=ctx,
            urls=urls,
            batch_size=20,
            return_raw_markdown=False,  # Store in database
        )

        # Parse crawl results
        crawl_data = json.loads(crawl_result)
        urls_stored = sum(1 for r in crawl_data.get("results", []) if r.get("success"))
        chunks_stored = sum(r.get("chunks_stored", 0) for r in crawl_data.get("results", []))

        logger.info(f"Stored {urls_stored}/{len(urls)} URLs, {chunks_stored} chunks total")

        search_history.append(
            SearchIteration(
                iteration=iteration,
                query=query,
                action=ActionType.CRAWL,
                urls=urls,
                urls_stored=urls_stored,
                chunks_stored=chunks_stored,
            )
        )

        # TODO: Implement search hints generation if use_hints=True
        # This requires investigating Crawl4AI's metadata capabilities
        if use_hints:
            logger.info("Search hints requested but not yet implemented")

    async def _stage4_query_refinement(
        self, original_query: str, current_query: str, gaps: list[str]
    ) -> QueryRefinement:
        """STAGE 4: Generate refined queries for next iteration.

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

Respond with JSON:
{{
  "refined_queries": ["query 1", "query 2", "query 3"],
  "reasoning": "Brief explanation of the refinement strategy"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,  # Slightly more creative for query generation
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            data = json.loads(content)
            return QueryRefinement(
                original_query=original_query,
                current_query=current_query,
                refined_queries=data.get("refined_queries", [current_query]),
                reasoning=data.get("reasoning", ""),
            )

        except (openai.OpenAIError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Query refinement failed: {e}")
            # Return current query as fallback
            return QueryRefinement(
                original_query=original_query,
                current_query=current_query,
                refined_queries=[current_query],
                reasoning=f"Refinement failed: {e}",
            )


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
        raise MCPToolError(
            "Agentic search is not enabled. Set AGENTIC_SEARCH_ENABLED=true in your environment."
        )

    try:
        service = AgenticSearchService()
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
        logger.exception(f"Agentic search implementation failed: {e}")
        msg = f"Agentic search failed: {e!s}"
        raise MCPToolError(msg)
