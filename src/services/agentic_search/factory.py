"""Factory for agentic search service singleton.

This module provides the factory function for creating the AgenticSearchService singleton,
separated to avoid circular imports with mcp_wrapper.
"""

from .config import AgenticSearchConfig
from .crawler import SelectiveCrawler
from .evaluator import LocalKnowledgeEvaluator
from .orchestrator import AgenticSearchService
from .ranker import URLRanker

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
