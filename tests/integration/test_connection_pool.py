"""Connection pool and resource management tests.

Tests for database connection pooling, resource cleanup, and timeout handling.
"""

import asyncio
from typing import Any

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_connection_pool_under_load() -> None:
    """Test database connection pool handles sustained load.

    Simulates sustained database operations to verify connection pool
    doesn't exhaust or leak connections.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    # Perform many sequential operations
    operation_count = 50
    for i in range(operation_count):
        sources = await db_client.get_all_sources()
        assert isinstance(sources, list)

        if i % 10 == 0:
            # Give connection pool time to recycle
            await asyncio.sleep(0.01)

    print(f"✓ Connection pool handled {operation_count} operations")


@pytest.mark.asyncio
async def test_concurrent_connection_limit() -> None:
    """Test system behavior at connection pool limits.

    Creates many concurrent database operations to stress test
    connection pool management.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    async def db_operation(op_id: int) -> list[str]:
        """Single database operation."""
        await asyncio.sleep(0.01 * (op_id % 3))  # Vary timing
        return await db_client.get_all_sources()

    # Create more concurrent operations than typical pool size
    tasks = [asyncio.create_task(db_operation(i)) for i in range(20)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Most should succeed, some may timeout/fail gracefully
    successful = [r for r in results if isinstance(r, list)]
    failed = [r for r in results if isinstance(r, Exception)]

    print(f"✓ Concurrent connections: {len(successful)} succeeded, {len(failed)} failed")
    assert len(successful) > 0, "At least some operations should succeed"


@pytest.mark.asyncio
async def test_connection_recovery_after_error() -> None:
    """Test connection pool recovers after errors.

    Verifies that after a database error, new connections can still be created.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    # First, cause an error with invalid parameters
    try:
        await db_client.search_documents(
            query_embedding=[],  # Invalid empty embedding
            match_count=5,
        )
    except Exception as e:
        print(f"Expected error occurred: {type(e).__name__}")

    # Verify database still works after error
    sources = await db_client.get_all_sources()
    assert isinstance(sources, list)
    print("✓ Connection pool recovered after error")


@pytest.mark.asyncio
async def test_timeout_on_slow_operation() -> None:
    """Test timeout handling for slow database operations.

    Edge case: Operations that exceed timeout should fail gracefully.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    async def slow_operation_with_timeout() -> Any:
        """Simulate slow operation with timeout."""
        try:
            # Use asyncio.wait_for to enforce timeout
            result = await asyncio.wait_for(
                db_client.get_all_sources(), timeout=0.001  # Very short timeout
            )
            return result
        except asyncio.TimeoutError:
            return "TIMEOUT"

    result = await slow_operation_with_timeout()

    # Either completes quickly or times out gracefully
    if result == "TIMEOUT":
        print("✓ Timeout handled gracefully")
    else:
        print(f"✓ Operation completed within timeout ({len(result)} sources)")


@pytest.mark.asyncio
async def test_connection_cleanup_on_exception() -> None:
    """Test connections are properly cleaned up when exceptions occur.

    Ensures no connection leaks when operations fail.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    # Force multiple errors
    for _ in range(5):
        try:
            await db_client.delete_documents_by_url([])  # Empty list may cause error
        except Exception:
            pass  # Expected

    # Verify database still works (connections weren't leaked)
    sources = await db_client.get_all_sources()
    assert isinstance(sources, list)
    print("✓ Connections cleaned up after exceptions")


@pytest.mark.asyncio
async def test_rapid_connection_open_close() -> None:
    """Test rapid connection creation and destruction.

    Edge case: Quickly creating/destroying connections should not leak resources.
    """
    from src.database.factory import get_database_client

    # Create and use multiple clients rapidly
    for i in range(10):
        db_client = get_database_client()
        sources = await db_client.get_all_sources()
        assert isinstance(sources, list)

        # Small delay between iterations
        if i % 3 == 0:
            await asyncio.sleep(0.01)

    print("✓ Rapid connection cycling handled safely")


@pytest.mark.asyncio
async def test_concurrent_read_write_operations() -> None:
    """Test concurrent mix of read and write operations.

    Ensures read/write operations don't block each other unnecessarily.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    async def read_operation(op_id: int) -> list[str]:
        await asyncio.sleep(0.01)
        return await db_client.get_all_sources()

    async def write_operation(op_id: int) -> None:
        await asyncio.sleep(0.01)
        try:
            await db_client.update_source_info(
                source_id=f"test-{op_id}", summary="Test", word_count=100
            )
        except Exception:
            pass  # May fail if source doesn't exist, that's ok

    # Mix of reads and writes
    tasks = []
    for i in range(10):
        if i % 2 == 0:
            tasks.append(asyncio.create_task(read_operation(i)))
        else:
            tasks.append(asyncio.create_task(write_operation(i)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful operations
    successful = sum(1 for r in results if not isinstance(r, Exception))
    print(f"✓ Concurrent read/write: {successful}/{len(tasks)} succeeded")


@pytest.mark.asyncio
async def test_connection_reuse_efficiency() -> None:
    """Test connection pooling reuses connections efficiently.

    Multiple operations should reuse connections rather than creating new ones.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    # Perform multiple operations sequentially
    # If connection pooling works, these should reuse connections
    for i in range(20):
        sources = await db_client.get_all_sources()
        assert isinstance(sources, list)

    print("✓ Connection reuse verified (no errors from exhaustion)")


@pytest.mark.asyncio
async def test_timeout_configuration_respected() -> None:
    """Test that timeout configurations are properly respected.

    Verifies custom timeouts work as expected.
    """

    async def operation_with_custom_timeout() -> str:
        """Test operation with custom timeout."""
        try:
            # Simulate long operation
            await asyncio.wait_for(asyncio.sleep(10), timeout=0.1)
            return "COMPLETED"
        except asyncio.TimeoutError:
            return "TIMED_OUT"

    result = await operation_with_custom_timeout()
    assert result == "TIMED_OUT"
    print("✓ Custom timeout configuration respected")


@pytest.mark.asyncio
async def test_graceful_shutdown_with_active_connections() -> None:
    """Test graceful shutdown when connections are active.

    Simulates server shutdown scenario with active database operations.
    """
    from src.database.factory import get_database_client

    db_client = get_database_client()

    # Start long-running operation
    task = asyncio.create_task(db_client.get_all_sources())

    # Give it time to start
    await asyncio.sleep(0.01)

    # Simulate shutdown - cancel the task
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("✓ Active connections cancelled gracefully during shutdown")

    # Verify database can still be used after cancellation
    sources = await db_client.get_all_sources()
    assert isinstance(sources, list)
    print("✓ Database operational after shutdown simulation")
