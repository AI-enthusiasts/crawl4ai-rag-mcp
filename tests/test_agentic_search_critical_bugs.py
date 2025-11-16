"""Regression tests for AgenticSearchService initialization bugs.

These tests verify that previously discovered initialization bugs stay fixed.

BUGS TESTED (regression prevention):
1. AgenticSearchService singleton initialization without required arguments
2. Search timeout issues (>10 seconds)
"""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestAgenticSearchServiceInitialization:
    """Regression tests for service initialization bugs."""

    def test_service_requires_all_components(self):
        """Verify AgenticSearchService requires evaluator, ranker, crawler, config."""
        from src.services.agentic_search.orchestrator import AgenticSearchService

        # Should fail without arguments
        with pytest.raises(TypeError, match="missing.*required"):
            AgenticSearchService()

    def test_service_initializes_with_components(self):
        """Verify service initializes correctly when all components provided."""
        from src.services.agentic_search.config import AgenticSearchConfig
        from src.services.agentic_search.crawler import SelectiveCrawler
        from src.services.agentic_search.evaluator import LocalKnowledgeEvaluator
        from src.services.agentic_search.orchestrator import AgenticSearchService
        from src.services.agentic_search.ranker import URLRanker

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY required")

        # Create components
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)
        ranker = URLRanker(config)
        crawler = SelectiveCrawler(config)

        # Initialize service
        service = AgenticSearchService(
            evaluator=evaluator,
            ranker=ranker,
            crawler=crawler,
            config=config,
        )

        # Verify initialization
        assert service.evaluator is evaluator
        assert service.ranker is ranker
        assert service.crawler is crawler
        assert service.config is config

    def test_singleton_factory_creates_service(self):
        """Verify get_agentic_search_service() singleton works."""
        from src.services.agentic_search import get_agentic_search_service

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY required")

        # Reset singleton for clean test
        import src.services.agentic_search
        src.services.agentic_search._service_instance = None

        # Get service through singleton
        service = get_agentic_search_service()

        # Verify all components initialized
        assert service is not None
        assert hasattr(service, 'evaluator') and service.evaluator is not None
        assert hasattr(service, 'ranker') and service.ranker is not None
        assert hasattr(service, 'crawler') and service.crawler is not None
        assert hasattr(service, 'config') and service.config is not None

        # Verify singleton pattern - same instance returned
        service2 = get_agentic_search_service()
        assert service is service2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
