"""Critical bug verification tests for agentic search.

This test suite verifies the critical bugs found in production WITHOUT using mocks.
These tests MUST fail to verify the bugs exist, then pass after fixes are applied.

CRITICAL BUGS BEING TESTED:
1. AgenticSearchService.__init__() missing 3 required positional arguments:
   'evaluator', 'ranker', and 'crawler'
   - Symptom: get_agentic_search_service() fails with TypeError
   - Root cause: singleton calls AgenticSearchService() without required arguments

2. Regular Search - TIMEOUT ISSUES
   - Symptom: Search tool takes >12.5 seconds and gets canceled
   - Root cause: TBD (needs investigation)

Prerequisites:
    - OPENAI_API_KEY must be set in environment
    - NO MOCKS - these are real integration tests
    - Tests should FAIL initially, verifying bugs exist

Run with: pytest tests/test_agentic_search_critical_bugs.py -v -s
"""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestCriticalBug1_ServiceInitialization:
    """Test Critical Bug #1: AgenticSearchService initialization failure.

    BUG: get_agentic_search_service() fails with:
    TypeError: AgenticSearchService.__init__() missing 3 required positional
    arguments: 'evaluator', 'ranker', and 'crawler'

    Expected: This test should FAIL, proving the bug exists.
    """

    def test_singleton_service_initialization_without_mocks(self):
        """Test that singleton can be created without errors.

        This test uses NO MOCKS and should FAIL with TypeError,
        proving that the singleton is broken.
        """
        # Import here to avoid module-level import errors
        from src.services.agentic_search import get_agentic_search_service

        # Reset singleton to force fresh initialization
        import src.services.agentic_search
        src.services.agentic_search._service_instance = None

        # This should FAIL with TypeError about missing arguments
        # Proves bug: AgenticSearchService() called without evaluator, ranker, crawler
        try:
            service = get_agentic_search_service()

            # If we get here without error, check that service is properly initialized
            assert service is not None, "Service should not be None"
            assert hasattr(service, 'evaluator'), "Service should have evaluator"
            assert hasattr(service, 'ranker'), "Service should have ranker"
            assert hasattr(service, 'crawler'), "Service should have crawler"
            assert service.evaluator is not None, "Evaluator should be initialized"
            assert service.ranker is not None, "Ranker should be initialized"
            assert service.crawler is not None, "Crawler should be initialized"

        except TypeError as e:
            # Expected failure - bug is verified
            error_msg = str(e)
            assert "missing" in error_msg and "required positional argument" in error_msg, \
                f"Expected TypeError about missing arguments, got: {error_msg}"
            pytest.fail(
                f"BUG VERIFIED: Singleton initialization fails with TypeError: {e}\n"
                f"Root cause: get_agentic_search_service() calls AgenticSearchService() "
                f"without required arguments (evaluator, ranker, crawler)"
            )

    def test_direct_service_initialization_without_mocks(self):
        """Test that AgenticSearchService requires evaluator, ranker, crawler.

        This test verifies that the service __init__ signature requires 3 arguments.
        """
        from src.services.agentic_search import AgenticSearchService

        # Attempt to create service without arguments - should fail
        try:
            service = AgenticSearchService()
            pytest.fail(
                "AgenticSearchService() should fail without arguments, but it didn't. "
                "This means __init__ signature was changed."
            )
        except TypeError as e:
            # Expected - verifies that evaluator, ranker, crawler are required
            error_msg = str(e)
            assert "evaluator" in error_msg or "missing" in error_msg, \
                f"Expected TypeError mentioning evaluator, got: {error_msg}"

    def test_service_initialization_with_proper_components(self):
        """Test that service CAN be initialized with proper components.

        This test creates the components manually and verifies service works.
        This test should PASS even before bug fix (proves components work).
        """
        from src.services.agentic_search.config import AgenticSearchConfig
        from src.services.agentic_search.crawler import SelectiveCrawler
        from src.services.agentic_search.evaluator import LocalKnowledgeEvaluator
        from src.services.agentic_search.orchestrator import AgenticSearchService
        from src.services.agentic_search.ranker import URLRanker

        # Verify OPENAI_API_KEY is set (required for AgenticSearchConfig)
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - cannot test real initialization")

        # Create components manually (this is what singleton SHOULD do)
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)
        ranker = URLRanker(config)
        crawler = SelectiveCrawler(config)

        # Now create service with components
        service = AgenticSearchService(
            evaluator=evaluator,
            ranker=ranker,
            crawler=crawler,
            config=config,
        )

        # Verify service is properly initialized
        assert service is not None
        assert service.evaluator is evaluator
        assert service.ranker is ranker
        assert service.crawler is crawler


class TestCriticalBug2_SearchTimeout:
    """Test Critical Bug #2: Regular search timeout issues.

    BUG: Search tool takes >12.5 seconds and gets canceled.

    Expected: This test should FAIL with timeout, proving the bug exists.
    """

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.getenv("RUN_EXPENSIVE_TESTS"),
        reason="Expensive test - set RUN_EXPENSIVE_TESTS=true to run",
    )
    def test_search_performance_without_mocks(self):
        """Test that search completes within reasonable time.

        This test uses real search (no mocks) and measures performance.
        Should FAIL with timeout if bug exists.
        """
        import asyncio
        import time
        from unittest.mock import MagicMock

        from src.tools.search import search

        # Create mock context
        ctx = MagicMock()

        # Measure search time
        start_time = time.time()

        async def run_search():
            try:
                # Use simple query to minimize variability
                result = await search(
                    ctx=ctx,
                    query="python programming",
                    return_raw_markdown=False,
                    num_results=3,  # Small number for speed
                    batch_size=10,
                )
                return result
            except Exception as e:
                return f"Error: {e}"

        # Run search with timeout
        try:
            result = asyncio.run(
                asyncio.wait_for(run_search(), timeout=10.0)
            )
            elapsed = time.time() - start_time

            # Search should complete in <10 seconds
            if elapsed > 10.0:
                pytest.fail(
                    f"BUG VERIFIED: Search took {elapsed:.2f}s (>10s threshold)\n"
                    f"Expected: <10 seconds\n"
                    f"This confirms the timeout issue."
                )

            # If it completes quickly, bug might be fixed
            assert elapsed < 10.0, f"Search took too long: {elapsed:.2f}s"

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            pytest.fail(
                f"BUG VERIFIED: Search timed out after {elapsed:.2f}s\n"
                f"This confirms the timeout issue in regular search."
            )


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_agentic_search_critical_bugs.py -v -s
    pytest.main([__file__, "-v", "-s"])
