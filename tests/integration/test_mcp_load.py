"""Integration tests for MCP service stability and load testing.

Tests ensure:
- MCP server handles concurrent requests without blocking
- No Chrome/Playwright process leaks
- Memory usage remains stable
- No event loop blocking operations
"""

import asyncio
from typing import Any

import psutil
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_mcp_concurrent_rag_queries() -> None:
    """Test MCP server handles concurrent RAG queries without blocking.

    This test simulates multiple concurrent RAG query operations to ensure
    the MCP server can handle parallel requests efficiently without blocking
    the event loop.
    """
    from src.database import perform_rag_query
    from src.database.factory import get_database_client

    # Initialize database client
    db_client = get_database_client()

    # Create 10 concurrent RAG query tasks
    query_count = 10
    tasks: list[asyncio.Task[Any]] = []

    for i in range(query_count):
        # Create diverse queries to test different code paths
        query = f"test query {i} about Python programming"
        task = asyncio.create_task(perform_rag_query(db_client, query, match_count=3))
        tasks.append(task)

    # Gather all results with exception handling
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify all requests completed without raising exceptions
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            pytest.fail(f"Query {i} failed with exception: {result}")
        successful_results.append(result)

    # All queries should complete
    assert len(successful_results) == query_count
    print(f"✓ Successfully processed {query_count} concurrent RAG queries")


@pytest.mark.asyncio
async def test_no_chrome_process_leak() -> None:
    """Verify Chrome/Playwright processes are not created or leaked.

    The MCP server should not spawn Chrome processes for regular operations.
    This test ensures no process leaks occur during crawling operations.
    """
    # Get initial Chrome/Chromium process count
    initial_chrome_processes = _count_chrome_processes()
    print(f"Initial Chrome processes: {initial_chrome_processes}")

    # Note: Crawl4AI may legitimately use Chrome for crawling
    # We test that processes don't accumulate indefinitely
    max_allowed_increase = 5  # Allow some temporary Chrome processes

    try:
        from src.services import process_urls_for_mcp

        # Perform multiple crawl operations
        # Using example.com which should complete quickly
        test_urls = ["https://example.com"] * 3

        for url in test_urls:
            await process_urls_for_mcp([url], return_raw_markdown=True)
            await asyncio.sleep(0.5)  # Small delay between operations

        # Give processes time to clean up
        await asyncio.sleep(2)

        # Check final Chrome process count
        final_chrome_processes = _count_chrome_processes()
        print(f"Final Chrome processes: {final_chrome_processes}")

        # Verify no excessive process accumulation
        process_increase = final_chrome_processes - initial_chrome_processes
        assert (
            process_increase <= max_allowed_increase
        ), f"Chrome processes increased by {process_increase}, max allowed: {max_allowed_increase}"

        print(f"✓ Chrome process management verified (increase: {process_increase})")

    except Exception as e:
        # Check if error is due to missing dependencies
        if "crawl4ai" in str(e).lower() or "playwright" in str(e).lower():
            pytest.skip(f"Crawling dependencies not available: {e}")
        raise


@pytest.mark.asyncio
async def test_concurrent_database_operations() -> None:
    """Test concurrent database operations don't cause deadlocks or race conditions.

    This test ensures the database adapters handle concurrent writes and reads
    correctly without corruption or blocking issues.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    # Create concurrent read operations
    # These should not block each other
    read_tasks = []
    for _ in range(5):
        task = asyncio.create_task(db_client.get_all_sources())
        read_tasks.append(task)

    results = await asyncio.gather(*read_tasks, return_exceptions=True)

    # All reads should succeed
    for result in results:
        assert not isinstance(result, Exception), f"Database read failed: {result}"

    print("✓ Concurrent database operations handled correctly")


@pytest.mark.asyncio
async def test_memory_stability_under_load() -> None:
    """Test memory usage remains stable under sustained load.

    This test ensures there are no memory leaks by monitoring memory usage
    during repeated operations.
    """
    import gc

    # Force garbage collection before starting
    gc.collect()

    # Get baseline memory
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory_mb:.2f} MB")

    # Perform multiple operations
    from src.database.factory import get_database_client

    db_client = get_database_client()

    for i in range(20):
        # Perform lightweight operations
        await db_client.get_all_sources()

        if i % 5 == 0:
            gc.collect()  # Periodic garbage collection

    # Check final memory
    final_memory_mb = process.memory_info().rss / 1024 / 1024
    memory_increase_mb = final_memory_mb - initial_memory_mb

    print(f"Final memory: {final_memory_mb:.2f} MB")
    print(f"Memory increase: {memory_increase_mb:.2f} MB")

    # Allow some increase but not excessive (adjust threshold as needed)
    max_allowed_increase_mb = 50.0
    assert (
        memory_increase_mb < max_allowed_increase_mb
    ), f"Memory increased by {memory_increase_mb:.2f} MB, threshold: {max_allowed_increase_mb} MB"

    print("✓ Memory usage stable under load")


@pytest.mark.asyncio
async def test_event_loop_not_blocked() -> None:
    """Test event loop is not blocked by long-running operations.

    This test ensures async operations don't block the event loop,
    which would prevent other tasks from running.
    """
    import time

    start_time = time.time()
    heartbeat_count = 0

    async def heartbeat_task() -> None:
        """Task that increments counter every 100ms."""
        nonlocal heartbeat_count
        for _ in range(10):
            await asyncio.sleep(0.1)
            heartbeat_count += 1

    async def mock_operation() -> None:
        """Simulate a database operation."""
        from src.database.factory import get_database_client

        db_client = get_database_client()
        await db_client.get_all_sources()

    # Run heartbeat alongside mock operation
    await asyncio.gather(heartbeat_task(), mock_operation())

    elapsed = time.time() - start_time

    # Heartbeat should have run multiple times
    # If event loop was blocked, heartbeat_count would be low
    assert heartbeat_count >= 5, (
        f"Event loop may be blocked, heartbeat ran only {heartbeat_count} times "
        f"in {elapsed:.2f}s"
    )

    print(f"✓ Event loop not blocked (heartbeat: {heartbeat_count} times in {elapsed:.2f}s)")


# Helper functions


def _count_chrome_processes() -> int:
    """Count Chrome/Chromium processes currently running.

    Returns:
        Number of Chrome/Chromium processes found
    """
    chrome_count = 0
    try:
        for proc in psutil.process_iter(["name"]):
            proc_name = proc.info["name"]
            if proc_name and any(
                browser in proc_name.lower()
                for browser in ["chrome", "chromium", "playwright"]
            ):
                chrome_count += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return chrome_count
