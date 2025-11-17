"""
Shared pytest fixtures and configuration for all tests.

Environment Variables for Test Control:
- ALLOW_OPENAI_TESTS=true : Enable tests that make real OpenAI API calls (embeddings)
- ALLOW_EXPENSIVE_TESTS=true : Enable tests with expensive LLM inference (GPT-4, etc)
- By default, all OpenAI calls are mocked to prevent accidental costs
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Create module alias for backward compatibility with tests
# Tests import from 'crawl4ai_mcp', but the actual module is 'src'
import src as crawl4ai_mcp_module

sys.modules["crawl4ai_mcp"] = crawl4ai_mcp_module

# Set test environment variables BEFORE any imports
# DO NOT override OPENAI_API_KEY - use real key from environment
os.environ["VECTOR_DATABASE"] = "qdrant"
os.environ["QDRANT_URL"] = "http://localhost:6333"

# Load test environment

# Import Neo4j fixtures to make them available to all tests
from .fixtures.neo4j_fixtures import *

# Register the performance monitoring plugin
pytest_plugins = ["tests.performance_plugin"]


@pytest.fixture(scope="session")
async def get_adapter():
    """Factory fixture to get database adapters by name - session scoped for performance"""
    adapters = {}

    async def _get_adapter(adapter_name: str):
        if adapter_name in adapters:
            return adapters[adapter_name]

        if adapter_name == "supabase":
            # Import will be available after we create the adapter
            from src.database.supabase_adapter import SupabaseAdapter

            adapter = SupabaseAdapter()
            # Mock the Supabase client for testing
            adapter.client = MagicMock()

            # Track deleted URLs for Supabase too
            deleted_urls_supabase = set()

            # Mock table operations
            table_mock = MagicMock()
            delete_mock = MagicMock()

            def mock_eq(field, value):
                # Track deletion
                if field == "url" and value == "https://delete-test.com/page":
                    deleted_urls_supabase.add(value)
                return MagicMock(execute=MagicMock())

            delete_mock.eq = mock_eq
            table_mock.delete = MagicMock(return_value=delete_mock)

            # Mock insert to check embedding sizes
            def mock_insert(data):
                # Create a mock that has an execute method that raises on invalid sizes
                mock_obj = MagicMock()

                def mock_execute():
                    # Check embedding sizes
                    if isinstance(data, list):
                        for item in data:
                            if "embedding" in item and len(item["embedding"]) != 1536:
                                raise ValueError(
                                    f"Invalid embedding size: expected 1536, got {len(item['embedding'])}",
                                )
                    elif "embedding" in data and len(data["embedding"]) != 1536:
                        raise ValueError(
                            f"Invalid embedding size: expected 1536, got {len(data['embedding'])}",
                        )
                    return MagicMock()

                mock_obj.execute = mock_execute
                return mock_obj

            table_mock.insert = mock_insert

            # Override _insert_with_retry to directly call insert and propagate exceptions
            async def direct_insert(table_name, batch_data):
                # This will directly raise the exception without retries
                adapter.client.table(table_name).insert(batch_data).execute()

            adapter._insert_with_retry = direct_insert
            adapter.client.table = MagicMock(return_value=table_mock)

            # Override search_documents to raise exceptions for invalid embeddings
            original_search_supabase = adapter.search_documents

            async def search_with_validation_supabase(
                query_embedding,
                match_count=10,
                filter_metadata=None,
                source_filter=None,
            ):
                # Check embedding size and raise
                if query_embedding and len(query_embedding) != 1536:
                    raise ValueError(
                        f"Invalid embedding size: expected 1536, got {len(query_embedding)}",
                    )
                # Call original method
                return await original_search_supabase(
                    query_embedding,
                    match_count,
                    filter_metadata,
                    source_filter,
                )

            adapter.search_documents = search_with_validation_supabase

            # Mock RPC operations for search
            def mock_rpc(function_name, params=None):
                mock_result = MagicMock()

                # Check embedding size for search operations
                if params and "query_embedding" in params:
                    if len(params["query_embedding"]) != 1536:
                        raise ValueError(
                            f"Invalid embedding size: expected 1536, got {len(params['query_embedding'])}",
                        )

                if function_name == "match_crawled_pages":
                    # Dynamic results based on query embedding value
                    if params and params.get("query_embedding"):
                        qe = params["query_embedding"]
                        if qe[0] == 0.45:  # Test with metadata filter
                            results = [
                                {
                                    "id": "test-python-1",
                                    "url": "https://test.com/1",
                                    "chunk_number": 1,
                                    "content": "Python content",
                                    "metadata": {"language": "python"},
                                    "source_id": "test.com",
                                    "similarity": 0.95,
                                },
                                {
                                    "id": "test-js-1",
                                    "url": "https://test.com/2",
                                    "chunk_number": 1,
                                    "content": "JavaScript content",
                                    "metadata": {"language": "javascript"},
                                    "source_id": "test.com",
                                    "similarity": 0.85,
                                },
                            ]
                            # Apply metadata filter if provided
                            if params.get("filter"):
                                results = [
                                    r
                                    for r in results
                                    if all(
                                        r["metadata"].get(k) == v
                                        for k, v in params["filter"].items()
                                    )
                                ]
                        elif qe[0] == 0.65:  # Test with source filter
                            results = [
                                {
                                    "id": "test-source1-1",
                                    "url": "https://source1.com/page",
                                    "chunk_number": 1,
                                    "content": "Content from source 1",
                                    "metadata": {},
                                    "source_id": "source1.com",
                                    "similarity": 0.95,
                                },
                                {
                                    "id": "test-source2-1",
                                    "url": "https://source2.com/page",
                                    "chunk_number": 1,
                                    "content": "Content from source 2",
                                    "metadata": {},
                                    "source_id": "source2.com",
                                    "similarity": 0.85,
                                },
                            ]
                            # Apply source filter if provided
                            if params.get("source_filter"):
                                results = [
                                    r
                                    for r in results
                                    if r["source_id"] == params["source_filter"]
                                ]
                        elif qe[0] == 0.8:  # Test for delete
                            if (
                                "https://delete-test.com/page"
                                not in deleted_urls_supabase
                            ):
                                results = [
                                    {
                                        "id": "test-delete-1",
                                        "url": "https://delete-test.com/page",
                                        "chunk_number": 1,
                                        "content": "Content to be deleted",
                                        "metadata": {},
                                        "source_id": "delete-test.com",
                                        "similarity": 0.95,
                                    },
                                ]
                            else:
                                results = []
                        elif qe[0] == 0.25:  # Test for batch operations
                            results = []
                            for i in range(15):
                                results.append(
                                    {
                                        "id": f"test-batch-{i}",
                                        "url": f"https://batch-test.com/page{i}",
                                        "chunk_number": i,
                                        "content": f"Content {i}",
                                        "metadata": {"index": i},
                                        "source_id": "batch-test.com",
                                        "similarity": 0.9 - (i * 0.01),
                                    },
                                )
                        else:  # Default case
                            results = [
                                {
                                    "id": "test-id-1",
                                    "url": "https://example.com/page1",
                                    "chunk_number": 1,
                                    "content": "This is the first document",
                                    "metadata": {"title": "Page 1"},
                                    "source_id": "example.com",
                                    "similarity": 0.95,
                                },
                            ]
                        mock_result.execute.return_value.data = results
                    else:
                        mock_result.execute.return_value.data = []
                elif function_name == "match_code_examples":
                    # Return sample code example results
                    mock_result.execute.return_value.data = [
                        {
                            "content": "def hello():\n    return 'world'",
                            "summary": "A simple hello world function",
                            "metadata": {"language": "python"},
                            "similarity": 0.92,
                        },
                    ]
                elif function_name == "update_source_summary":
                    mock_result.execute.return_value = MagicMock()
                    # Track that source was updated
                    if not hasattr(adapter, "_test_sources"):
                        adapter._test_sources = {}
                    if params:
                        adapter._test_sources[
                            params.get("p_source_id", "test-source.com")
                        ] = {
                            "source_id": params.get("p_source_id", "test-source.com"),
                            "summary": params.get(
                                "p_summary",
                                "A test source for unit tests",
                            ),
                            "total_word_count": params.get("p_word_count", 1000),
                        }
                elif function_name == "get_source_summaries":
                    # Return tracked sources or default
                    if hasattr(adapter, "_test_sources"):
                        mock_result.execute.return_value.data = list(
                            adapter._test_sources.values(),
                        )
                    else:
                        mock_result.execute.return_value.data = [
                            {
                                "source_id": "test-source.com",
                                "summary": "A test source for unit tests",
                                "total_word_count": 1000,
                            },
                        ]
                else:
                    mock_result.execute.return_value.data = []
                return mock_result

            adapter.client.rpc = MagicMock(side_effect=mock_rpc)

        elif adapter_name == "qdrant":
            # Import will be available after we create the adapter
            from src.database.qdrant_adapter import QdrantAdapter

            adapter = QdrantAdapter(url="http://localhost:6333")
            # Mock the Qdrant client for testing - Qdrant client is SYNCHRONOUS
            adapter.client = MagicMock()

            # Simple state tracking for deleted URLs
            deleted_urls = set()

            # Mock collection operations
            adapter.client.get_collection = MagicMock(return_value=MagicMock())
            adapter.client.create_collection = MagicMock()

            # Mock delete to track deleted URLs
            def mock_delete(collection_name, points_selector):
                # In the delete test, this gets called with point IDs
                # We'll just track that deletion happened
                if collection_name == "crawled_pages" and isinstance(
                    points_selector,
                    list,
                ):
                    # Mark URL as deleted based on the test scenario
                    if "test-delete-1" in points_selector:
                        deleted_urls.add("https://delete-test.com/page")

            adapter.client.delete = mock_delete

            # Add delete tracking to delete_documents_by_url
            original_delete_by_url = adapter.delete_documents_by_url

            async def tracked_delete_by_url(urls):
                # Add URLs to deleted set
                for url in urls:
                    deleted_urls.add(url)
                # Call original method
                await original_delete_by_url(urls)

            adapter.delete_documents_by_url = tracked_delete_by_url

            # Mock upsert to check embedding sizes - Qdrant client is SYNCHRONOUS
            def mock_upsert(collection_name, points):
                # Check embedding sizes in points
                for point in points:
                    if hasattr(point, "vector") and len(point.vector) != 1536:
                        raise ValueError(
                            f"Invalid embedding size: expected 1536, got {len(point.vector)}",
                        )

            adapter.client.upsert = mock_upsert

            # Override search_documents to raise exceptions for invalid embeddings
            original_search_documents = adapter.search_documents

            async def search_with_validation(
                query_embedding,
                match_count=10,
                filter_metadata=None,
                source_filter=None,
            ):
                # Check embedding size and raise
                if query_embedding and len(query_embedding) != 1536:
                    raise ValueError(
                        f"Invalid embedding size: expected 1536, got {len(query_embedding)}",
                    )
                # Call original method
                return await original_search_documents(
                    query_embedding,
                    match_count,
                    filter_metadata,
                    source_filter,
                )

            adapter.search_documents = search_with_validation

            # Mock search operations - Qdrant client is SYNCHRONOUS
            def mock_search(
                collection_name,
                query_vector,
                limit=10,
                with_payload=True,
                query_filter=None,
            ):
                # Check embedding size (but don't raise here, let search_documents handle it)
                if query_vector and len(query_vector) != 1536:
                    # Return empty results to trigger search_documents to handle it
                    return []

                # Special handling for delete operations (when searching with dummy vector [0.0] * 1536)
                if (
                    collection_name == "crawled_pages"
                    and query_vector
                    and all(v == 0.0 for v in query_vector)
                    and query_filter
                ):
                    # This is a search for deletion - check the URL filter
                    if hasattr(query_filter, "must") and query_filter.must:
                        for condition in query_filter.must:
                            if (
                                hasattr(condition, "key")
                                and condition.key == "url"
                                and hasattr(condition, "match")
                            ):
                                url = condition.match.value
                                if url == "https://delete-test.com/page":
                                    return [MagicMock(id="test-delete-1")]
                    return []

                # Return sample search results based on collection
                if collection_name == "crawled_pages":
                    # Create dynamic results based on query vector to simulate different test scenarios
                    if query_vector[0] == 0.45:  # Test with metadata filter
                        results = [
                            MagicMock(
                                id="test-python-1",
                                score=0.95,
                                payload={
                                    "url": "https://test.com/1",
                                    "chunk_number": 1,
                                    "content": "Python content",
                                    "metadata": {"language": "python"},
                                    "source_id": "test.com",
                                },
                            ),
                            MagicMock(
                                id="test-js-1",
                                score=0.85,
                                payload={
                                    "url": "https://test.com/2",
                                    "chunk_number": 1,
                                    "content": "JavaScript content",
                                    "metadata": {"language": "javascript"},
                                    "source_id": "test.com",
                                },
                            ),
                        ]
                    elif query_vector[0] == 0.65:  # Test with source filter
                        results = [
                            MagicMock(
                                id="test-source1-1",
                                score=0.95,
                                payload={
                                    "url": "https://source1.com/page",
                                    "chunk_number": 1,
                                    "content": "Content from source 1",
                                    "metadata": {},
                                    "source_id": "source1.com",
                                },
                            ),
                            MagicMock(
                                id="test-source2-1",
                                score=0.85,
                                payload={
                                    "url": "https://source2.com/page",
                                    "chunk_number": 1,
                                    "content": "Content from source 2",
                                    "metadata": {},
                                    "source_id": "source2.com",
                                },
                            ),
                        ]
                    elif query_vector[0] == 0.8:  # Test for delete
                        # Check if URL has been deleted
                        if "https://delete-test.com/page" not in deleted_urls:
                            results = [
                                MagicMock(
                                    id="test-delete-1",
                                    score=0.95,
                                    payload={
                                        "url": "https://delete-test.com/page",
                                        "chunk_number": 1,
                                        "content": "Content to be deleted",
                                        "metadata": {},
                                        "source_id": "delete-test.com",
                                    },
                                ),
                            ]
                        else:
                            results = []  # URL has been deleted
                    elif query_vector[0] == 0.25:  # Test for batch operations
                        # Return multiple results for batch test
                        batch_results = []
                        for i in range(15):  # Return 15 results
                            batch_results.append(
                                MagicMock(
                                    id=f"test-batch-{i}",
                                    score=0.9 - (i * 0.01),
                                    payload={
                                        "url": f"https://batch-test.com/page{i}",
                                        "chunk_number": i,
                                        "content": f"Content {i}",
                                        "metadata": {"index": i},
                                        "source_id": "batch-test.com",
                                    },
                                ),
                            )
                        results = batch_results
                    else:  # Default case
                        results = [
                            MagicMock(
                                id="test-id-1",
                                score=0.95,
                                payload={
                                    "url": "https://example.com/page1",
                                    "chunk_number": 1,
                                    "content": "This is the first document",
                                    "metadata": {"title": "Page 1"},
                                    "source_id": "example.com",
                                },
                            ),
                        ]
                elif collection_name == "code_examples":
                    results = [
                        MagicMock(
                            score=0.92,
                            payload={
                                "content": "def hello():\n    return 'world'",
                                "summary": "A simple hello world function",
                                "metadata": {"language": "python"},
                            },
                        ),
                    ]
                else:
                    results = []

                # Apply filters if provided
                if (
                    query_filter
                    and hasattr(query_filter, "must")
                    and query_filter.must
                    and results
                ):
                    # Simple filter simulation
                    filtered_results = []
                    for result in results:
                        include = True
                        # Check each condition in the filter
                        for condition in query_filter.must:
                            if hasattr(condition, "key") and hasattr(
                                condition,
                                "match",
                            ):
                                key = condition.key
                                expected_value = condition.match.value

                                # Handle metadata.* keys
                                if key.startswith("metadata."):
                                    metadata_key = key.replace("metadata.", "")
                                    if (
                                        result.payload.get("metadata", {}).get(
                                            metadata_key,
                                        )
                                        != expected_value
                                    ):
                                        include = False
                                        break
                                # Handle direct keys like source_id
                                elif (
                                    key == "source_id"
                                    and result.payload.get("source_id")
                                    != expected_value
                                ) or (
                                    key == "url"
                                    and result.payload.get("url") != expected_value
                                ):
                                    include = False
                                    break

                        if include:
                            filtered_results.append(result)
                    results = filtered_results

                return results

            adapter.client.search = mock_search

            # Mock other Qdrant client operations - all are SYNCHRONOUS
            adapter.client.retrieve = MagicMock(return_value=[])
            adapter.client.set_payload = MagicMock()

            # Track sources for Qdrant
            adapter._test_sources_qdrant = {}

            # Mock update_source_info
            original_update_source = adapter.update_source_info

            async def tracked_update_source(source_id, summary, word_count):
                adapter._test_sources_qdrant[source_id] = {
                    "source_id": source_id,
                    "summary": summary,
                    "total_word_count": word_count,
                }
                # Don't call original as it will fail with mocked client

            adapter.update_source_info = tracked_update_source

            # Mock get_sources to return tracked sources
            async def mock_get_sources():
                if adapter._test_sources_qdrant:
                    return list(adapter._test_sources_qdrant.values())
                return [
                    {
                        "source_id": "test-source.com",
                        "summary": "A test source for unit tests",
                        "total_word_count": 1000,
                    },
                ]

            adapter.get_sources = mock_get_sources

            # Mock scroll operations for get_all_payloads - Qdrant client is SYNCHRONOUS
            def mock_scroll(
                collection_name,
                scroll_filter=None,
                offset=None,
                limit=100,
                with_payload=True,
            ):
                # Default empty response for most cases
                return ([], None)

            adapter.client.scroll = MagicMock(side_effect=mock_scroll)

        else:
            raise ValueError(f"Unknown adapter: {adapter_name}")

        adapters[adapter_name] = adapter
        return adapter

    yield _get_adapter

    # Cleanup
    adapters.clear()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Cache for expensive mock objects
_MOCK_CACHE = {}


@pytest.fixture
def mock_openai_embeddings():
    """
    Mock OpenAI embeddings for testing - with caching.

    This fixture is applied by default UNLESS ALLOW_OPENAI_TESTS=true is set.
    Tests using this fixture will use mocked embeddings (all zeros).

    To use real OpenAI API: set ALLOW_OPENAI_TESTS=true environment variable.
    """
    # Check if real OpenAI is allowed
    allow_openai = os.getenv("ALLOW_OPENAI_TESTS", "false").lower() == "true"

    if allow_openai:
        # Don't mock - let real OpenAI calls through
        # But ensure API key is set
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith(
            "test-"
        ):
            pytest.fail(
                "ALLOW_OPENAI_TESTS=true but no valid OPENAI_API_KEY found. "
                "Set a real API key in .env or environment.",
            )
        yield None  # No mock, real API will be used
        return

    # Mock mode (default)
    # DO NOT override OPENAI_API_KEY globally - only mock for specific tests
    from unittest.mock import MagicMock, patch

    # Use cached mock if available
    if "openai_embeddings" not in _MOCK_CACHE:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        _MOCK_CACHE["openai_embeddings"] = mock_response

    # Mock both OpenAI client instance and module-level access
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _MOCK_CACHE["openai_embeddings"]
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="test context"))],
    )

    with patch("openai.OpenAI", return_value=mock_client) as mock_openai_class:
        yield mock_openai_class


@pytest.fixture
async def clean_test_data():
    """Cleanup test data before and after tests"""
    # This will be implemented when we have real adapters
    return
    # Cleanup after test


# ============================================================================
# Docker Logs Collection for Load Tests
# ============================================================================

import subprocess
from datetime import datetime
from pathlib import Path


@pytest.fixture(scope="function", autouse=False)
def docker_logs_collector(request):
    """
    Collect Docker logs for MCP container during test execution.

    Usage:
        @pytest.mark.usefixtures("docker_logs_collector")
        async def test_something():
            ...
    """
    # Get container name
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mcp-crawl4ai", "--format", "{{.Names}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        container_name = result.stdout.strip()
    except Exception:
        container_name = None

    if not container_name:
        print("⚠️  MCP container not found, skipping log collection")
        yield
        return

    # Record start time
    start_time = datetime.now()
    test_name = request.node.name

    print(f"\n📋 Collecting logs for: {test_name}")
    print(f"🐳 Container: {container_name}")
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run test
    yield

    # Collect logs after test
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"⏱️  Test duration: {duration:.2f}s")
    print("📝 Collecting Docker logs...")

    try:
        # Get logs since start time
        since_param = start_time.strftime("%Y-%m-%dT%H:%M:%S")

        result = subprocess.run(
            ["docker", "logs", "--since", since_param, "--timestamps", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        logs = result.stdout + result.stderr

        # Save logs to file
        logs_dir = Path("tests/results/docker_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{test_name}_{timestamp}.log"

        with log_file.open("w") as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Container: {container_name}\n")
            f.write(f"Start: {start_time}\n")
            f.write(f"End: {end_time}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write("=" * 80 + "\n\n")
            f.write(logs)

        print(f"✅ Logs saved to: {log_file}")

        # Analyze logs for errors
        error_count = logs.count("ERROR")
        warning_count = logs.count("WARNING")
        timeout_count = logs.count("timeout") + logs.count("Timeout")

        if error_count > 0 or warning_count > 0 or timeout_count > 0:
            print("⚠️  Log analysis:")
            if error_count > 0:
                print(f"   - Errors: {error_count}")
            if warning_count > 0:
                print(f"   - Warnings: {warning_count}")
            if timeout_count > 0:
                print(f"   - Timeouts: {timeout_count}")
        else:
            print("✅ No errors found in logs")

    except subprocess.TimeoutExpired:
        print("❌ Timeout while collecting logs")
    except Exception as e:
        print(f"❌ Error collecting logs: {e}")


@pytest.fixture(scope="session")
def docker_container_name():
    """Get MCP Docker container name."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mcp-crawl4ai", "--format", "{{.Names}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        names = result.stdout.strip().split("\n")
        return names[0] if names and names[0] else None
    except Exception as e:
        print(f"⚠️  Could not get container name: {e}")
        return None


# ========================================
# TEST CONTROL: Cost Protection
# ========================================


def pytest_configure(config):
    """Configure custom markers for cost control."""
    config.addinivalue_line(
        "markers",
        "requires_openai: Test requires real OpenAI API calls (embeddings, ~$0.0001)",
    )
    config.addinivalue_line(
        "markers",
        "expensive: Test uses expensive LLM inference (GPT-4, etc, $$$)",
    )
    config.addinivalue_line(
        "markers",
        "cheap: Test uses only cheap operations (embeddings, $0.0001)",
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip expensive tests unless explicitly enabled.

    Protection against accidental costs:
    - By default: all OpenAI calls are mocked
    - ALLOW_OPENAI_TESTS=true: enable cheap tests (embeddings)
    - ALLOW_EXPENSIVE_TESTS=true: enable expensive tests (LLM inference)
    """
    allow_openai = os.getenv("ALLOW_OPENAI_TESTS", "false").lower() == "true"
    allow_expensive = os.getenv("ALLOW_EXPENSIVE_TESTS", "false").lower() == "true"

    skip_openai = pytest.mark.skip(
        reason="OpenAI tests disabled. Set ALLOW_OPENAI_TESTS=true to enable (cheap embeddings, ~$0.0001 per test)",
    )
    skip_expensive = pytest.mark.skip(
        reason="Expensive tests disabled. Set ALLOW_EXPENSIVE_TESTS=true to enable ($$$ LLM inference)",
    )

    for item in items:
        # Skip expensive LLM tests unless explicitly allowed
        if "expensive" in item.keywords and not allow_expensive:
            item.add_marker(skip_expensive)

        # Skip all OpenAI tests (including cheap ones) unless explicitly allowed
        elif "requires_openai" in item.keywords and not allow_openai:
            item.add_marker(skip_openai)
