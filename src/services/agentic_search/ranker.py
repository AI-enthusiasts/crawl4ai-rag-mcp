"""Web search and URL ranking for agentic search (Stage 2).

This module handles:
- SearXNG web search execution
- LLM-based URL relevance ranking
- Filtering promising URLs above threshold
"""

import logging
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.core.exceptions import LLMError
from src.services.agentic_models import ActionType, SearchIteration, URLRanking
from src.services.search import _search_searxng

from .config import AgenticSearchConfig

logger = logging.getLogger(__name__)


class URLRanker:
    """Performs web search and ranks URLs by relevance using LLM."""

    def __init__(self, config: AgenticSearchConfig) -> None:
        """Initialize ranker with shared configuration.

        Args:
            config: Shared agentic search configuration with agents
        """
        self.config = config

    async def search_and_rank_urls(
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
        if len(search_results) > self.config.max_urls_to_rank:
            logger.info(
                f"Limiting ranking to top {self.config.max_urls_to_rank} of {len(search_results)} results",
            )
            search_results = search_results[: self.config.max_urls_to_rank]

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
            result = await self.config.ranking_agent.run(prompt)
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
