"""Agentic search service - modular implementation.

This package implements the complete agentic search pipeline with modular design:
- config: Agent initialization and configuration
- evaluator: Local knowledge evaluation (Stage 1)
- ranker: Web search and URL ranking (Stage 2)
- crawler: Selective crawling (Stage 3)
- orchestrator: Main pipeline orchestration and query refinement (Stage 4)
"""

from .config import AgenticSearchConfig
from .crawler import SelectiveCrawler
from .evaluator import LocalKnowledgeEvaluator
from .orchestrator import AgenticSearchService
from .ranker import URLRanker

# Re-export main service and singleton for backward compatibility
__all__ = [
    "AgenticSearchConfig",
    "AgenticSearchService",
    "get_agentic_search_service",
    "agentic_search_impl",
]

# Singleton instance
_service_instance: AgenticSearchService | None = None


def get_agentic_search_service() -> AgenticSearchService:
    """Get singleton instance of AgenticSearchService.

    Returns:
        Singleton AgenticSearchService instance with cached agents.
    """
    global _service_instance
    if _service_instance is None:
        # Create modular components
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)
        ranker = URLRanker(config)
        crawler = SelectiveCrawler(config)

        # Initialize service with components
        _service_instance = AgenticSearchService(
            evaluator=evaluator,
            ranker=ranker,
            crawler=crawler,
            config=config,
        )
    return _service_instance


# Import MCP wrapper after service definition to avoid circular imports
from .mcp_wrapper import agentic_search_impl

__all__.append("agentic_search_impl")
