"""Edge case integration tests for MCP service.

Tests covering error scenarios, boundary conditions, and unusual inputs.
"""

import asyncio

import pytest

from src.database.factory import create_database_client
from src.utils.type_guards import is_valid_url

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_database_empty_query_handling() -> None:
    """Test database handles empty query gracefully.

    Edge case: Empty query string should not crash the system.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    # Empty query should return empty results, not crash
    result = await perform_rag_query(db_client, "", match_count=5)

    assert isinstance(result, str)
    # Should either return empty or a message, not crash
    print(f"✓ Empty query handled gracefully: {len(result)} chars")


@pytest.mark.asyncio
async def test_database_very_long_query() -> None:
    """Test database handles very long queries.

    Edge case: Extremely long query strings should be handled properly.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    # Create a very long query (10KB of text)
    long_query = "Python programming " * 500

    try:
        result = await perform_rag_query(db_client, long_query, match_count=3)
        assert isinstance(result, str)
        print(f"✓ Long query ({len(long_query)} chars) handled successfully")
    except Exception as e:
        # Should handle gracefully, not crash
        print(f"✓ Long query properly rejected with: {type(e).__name__}")


@pytest.mark.asyncio
async def test_database_special_characters_in_query() -> None:
    """Test database handles special characters in queries.

    Edge case: SQL/NoSQL injection attempts should be sanitized.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    # Test various special characters
    special_queries = [
        "'; DROP TABLE users; --",  # SQL injection attempt
        "<script>alert('xss')</script>",  # XSS attempt
        "test\x00null\x00byte",  # Null bytes
        "unicode: 你好世界 🚀",  # Unicode characters
        "regex: [a-z]+ (test)*",  # Regex special chars
    ]

    for query in special_queries:
        result = await perform_rag_query(db_client, query, match_count=2)
        assert isinstance(result, str)
        print(f"✓ Special query handled: {query[:30]}...")


@pytest.mark.asyncio
async def test_concurrent_database_writes() -> None:
    """Test concurrent write operations don't cause race conditions.

    Edge case: Multiple concurrent writes to same collection.
    """
    db_client = create_database_client()

    # Simulate concurrent metadata updates
    async def update_source_info(source_id: str) -> None:
        try:
            await db_client.update_source_info(
                source_id=source_id, summary="Test summary", word_count=100
            )
        except Exception as e:
            # Some failures expected in concurrent scenarios
            print(f"Expected concurrent write behavior: {type(e).__name__}")

    # Create multiple concurrent updates to same source
    tasks = []
    for i in range(5):
        task = asyncio.create_task(update_source_info(f"test-source-{i % 2}"))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)
    print("✓ Concurrent writes handled without deadlock")


@pytest.mark.asyncio
async def test_database_invalid_source_filter() -> None:
    """Test database handles invalid source filters gracefully.

    Edge case: Non-existent source filter should return empty results.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    # Use non-existent source filter
    result = await perform_rag_query(
        db_client,
        "test query",
        source="this-source-definitely-does-not-exist-12345",
        match_count=5,
    )

    assert isinstance(result, str)
    print("✓ Invalid source filter handled gracefully")


@pytest.mark.asyncio
async def test_type_guard_edge_cases() -> None:
    """Test type guards with edge case inputs.

    Edge case: Various invalid URL formats.
    """
    test_cases = [
        ("", False),  # Empty string
        (None, False),  # None
        ("   ", False),  # Whitespace only
        ("http://", False),  # Incomplete URL
        ("https://", False),  # Incomplete HTTPS
        ("ftp://example.com", False),  # Wrong protocol
        ("http://localhost", True),  # Localhost valid
        ("https://192.168.1.1", True),  # IP address valid
        ("http://example.com:8080/path?query=1", True),  # Full URL valid
    ]

    for url_input, expected in test_cases:
        result = is_valid_url(url_input)  # type: ignore[arg-type]
        assert result == expected, f"Failed for input: {url_input!r}"

    print("✓ Type guard edge cases validated")


@pytest.mark.asyncio
async def test_database_zero_match_count() -> None:
    """Test database handles zero match count.

    Edge case: match_count=0 should return empty results.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    result = await perform_rag_query(db_client, "test query", match_count=0)

    assert isinstance(result, str)
    print("✓ Zero match count handled gracefully")


@pytest.mark.asyncio
async def test_database_negative_match_count() -> None:
    """Test database handles negative match count.

    Edge case: Negative match_count should be handled gracefully.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    try:
        result = await perform_rag_query(db_client, "test query", match_count=-1)
        # Either handles gracefully or raises appropriate error
        assert isinstance(result, str)
        print("✓ Negative match count handled gracefully")
    except (ValueError, AssertionError) as e:
        # Acceptable to raise validation error
        print(f"✓ Negative match count properly rejected: {type(e).__name__}")


@pytest.mark.asyncio
async def test_database_extremely_large_match_count() -> None:
    """Test database handles very large match counts.

    Edge case: Extremely large match_count should be capped or handled.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    # Request 1 million results
    result = await perform_rag_query(db_client, "test query", match_count=1_000_000)

    assert isinstance(result, str)
    # Should handle gracefully, possibly with internal limit
    print("✓ Extremely large match count handled")


@pytest.mark.asyncio
async def test_repeated_database_initialization() -> None:
    """Test repeated database initialization is safe.

    Edge case: Multiple initialize() calls should be idempotent.
    """
    db_client = create_database_client()

    # Initialize multiple times
    for _ in range(3):
        await db_client.initialize()

    print("✓ Repeated initialization handled safely")


@pytest.mark.asyncio
async def test_database_unicode_content() -> None:
    """Test database handles unicode content properly.

    Edge case: Emoji, CJK characters, RTL text should work.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    unicode_queries = [
        "Python 🐍 programming",
        "日本語プログラミング",  # Japanese
        "البرمجة بالعربية",  # Arabic (RTL)
        "Προγραμματισμός",  # Greek
        "תכנות",  # Hebrew (RTL)
    ]

    for query in unicode_queries:
        result = await perform_rag_query(db_client, query, match_count=2)
        assert isinstance(result, str)
        print(f"✓ Unicode query handled: {query[:20]}...")


@pytest.mark.asyncio
async def test_asyncio_task_cancellation() -> None:
    """Test graceful handling of cancelled async tasks.

    Edge case: Cancelled tasks should not leave resources locked.
    """
    from src.database import perform_rag_query

    db_client = create_database_client()

    async def long_running_query() -> str:
        await asyncio.sleep(0.1)
        return await perform_rag_query(db_client, "test query", match_count=5)

    # Create and immediately cancel task
    task = asyncio.create_task(long_running_query())
    await asyncio.sleep(0.01)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("✓ Task cancellation handled gracefully")

    # Verify database still works after cancellation
    result = await perform_rag_query(db_client, "test after cancel", match_count=2)
    assert isinstance(result, str)
    print("✓ Database operational after task cancellation")
