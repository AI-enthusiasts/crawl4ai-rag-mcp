"""
Load testing for Crawl4AI MCP Server using FastMCP Client.

This module provides comprehensive load testing for the MCP server using the
official FastMCP Client to invoke MCP tools under various load conditions.

Test Categories:
- Throughput tests: Measure requests/second under sustained load
- Latency tests: Measure response times under different load levels
- Concurrency tests: Test behavior with multiple concurrent users
- Stress tests: Push system to limits to find breaking points
- Endurance tests: Sustained load over extended periods
"""

import asyncio
import gc
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil
import pytest
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

# Load environment variables
load_dotenv()


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_seconds: float = 0.0
    response_times_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_percent: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def throughput_rps(self) -> float:
        """Calculate requests per second."""
        if self.total_duration_seconds == 0:
            return 0.0
        return self.total_requests / self.total_duration_seconds

    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time."""
        if not self.response_times_ms:
            return 0.0
        return statistics.mean(self.response_times_ms)

    @property
    def p50_response_time_ms(self) -> float:
        """Calculate 50th percentile response time."""
        if not self.response_times_ms:
            return 0.0
        return statistics.median(self.response_times_ms)

    @property
    def p95_response_time_ms(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times_ms:
            return 0.0
        sorted_times = sorted(self.response_times_ms)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]

    @property
    def p99_response_time_ms(self) -> float:
        """Calculate 99th percentile response time."""
        if not self.response_times_ms:
            return 0.0
        sorted_times = sorted(self.response_times_ms)
        index = int(len(sorted_times) * 0.99)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]

    @property
    def max_response_time_ms(self) -> float:
        """Get maximum response time."""
        if not self.response_times_ms:
            return 0.0
        return max(self.response_times_ms)

    @property
    def min_response_time_ms(self) -> float:
        """Get minimum response time."""
        if not self.response_times_ms:
            return 0.0
        return min(self.response_times_ms)

    @property
    def avg_memory_mb(self) -> float:
        """Calculate average memory usage."""
        if not self.memory_usage_mb:
            return 0.0
        return statistics.mean(self.memory_usage_mb)

    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage."""
        if not self.memory_usage_mb:
            return 0.0
        return max(self.memory_usage_mb)

    @property
    def avg_cpu_percent(self) -> float:
        """Calculate average CPU usage."""
        if not self.cpu_percent:
            return 0.0
        return statistics.mean(self.cpu_percent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": round(self.success_rate, 2),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "throughput_rps": round(self.throughput_rps, 2),
            "response_times": {
                "avg_ms": round(self.avg_response_time_ms, 2),
                "min_ms": round(self.min_response_time_ms, 2),
                "max_ms": round(self.max_response_time_ms, 2),
                "p50_ms": round(self.p50_response_time_ms, 2),
                "p95_ms": round(self.p95_response_time_ms, 2),
                "p99_ms": round(self.p99_response_time_ms, 2),
            },
            "resource_usage": {
                "avg_memory_mb": round(self.avg_memory_mb, 2),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            },
            "errors": self.errors[:10],  # First 10 errors
            "error_count": len(self.errors),
        }


class MCPLoadTester:
    """Load tester for FastMCP HTTP server using FastMCP Client."""

    def __init__(self, init_timeout: float = 5.0, timeout: float = 30.0, tool_timeout: float = 60.0):
        """Initialize load tester.
        
        Args:
            init_timeout: Connection initialization timeout in seconds (default: 5s)
            timeout: General operation timeout in seconds (default: 30s)
            tool_timeout: Tool invocation timeout in seconds (default: 60s)
        """
        self.process = psutil.Process(os.getpid())
        
        # Get MCP server configuration from environment
        self.mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8051")
        self.mcp_api_key = os.getenv("MCP_API_KEY", "")
        
        # Timeouts
        self.init_timeout = init_timeout
        self.timeout = timeout
        self.tool_timeout = tool_timeout
        
        # FastMCP Client instance
        self.client: Optional[Client] = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Create FastMCP Client with HTTP transport, authentication, and timeouts
        # Client automatically infers HTTP transport from URL
        self.client = Client(
            self.mcp_url,
            auth=BearerAuth(self.mcp_api_key) if self.mcp_api_key else None,
            timeout=self.timeout,
            init_timeout=self.init_timeout  # 5s for connection
        )
        
        # Enter client context
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def invoke_mcp_tool(
        self, tool_name: str, args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Invoke an MCP tool via FastMCP Client.

        Args:
            tool_name: Name of the MCP tool to invoke
            args: Arguments to pass to the tool

        Returns:
            Tool invocation result
        """
        try:
            if not self.client:
                raise RuntimeError("Client not initialized. Use async context manager.")

            # Call tool using FastMCP Client with timeout
            # Disable automatic error raising to handle errors manually
            result = await self.client.call_tool(
                tool_name, 
                args or {}, 
                raise_on_error=False,
                timeout=self.tool_timeout
            )
            
            # FastMCP Client returns a CallToolResult object
            # Check if the call was successful
            if result.is_error:
                # Extract error message from content
                error_msg = "Unknown error"
                if result.content:
                    content_item = result.content[0]
                    # Safely get text attribute if it exists
                    error_msg = getattr(content_item, 'text', str(content_item))
                return {
                    "success": False,
                    "error": error_msg,
                    "tool": tool_name,
                    "args": args,
                }
            else:
                return {
                    "success": True,
                    "result": result.data,  # Use .data for structured output
                    "tool": tool_name,
                    "args": args or {},
                }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout",
                "tool": tool_name,
                "args": args,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "args": args,
            }

    async def run_load_test(
        self,
        tool_name: str,
        args_generator,
        num_requests: int,
        concurrency: int = 1,
        duration_seconds: Optional[float] = None,
        ramp_up_seconds: float = 0,
    ) -> LoadTestMetrics:
        """Run a load test against an MCP tool.

        Args:
            tool_name: Name of the MCP tool to test
            args_generator: Function that generates arguments for each request
            num_requests: Total number of requests to make
            concurrency: Number of concurrent requests
            duration_seconds: Optional duration limit (overrides num_requests)
            ramp_up_seconds: Time to gradually increase load

        Returns:
            LoadTestMetrics with test results
        """
        metrics = LoadTestMetrics()
        start_time = time.time()
        request_count = 0
        active_tasks = set()

        # Calculate ramp-up delay between starting new workers
        ramp_up_delay = ramp_up_seconds / concurrency if concurrency > 0 else 0

        async def worker(worker_id: int, start_delay: float = 0):
            """Worker coroutine that makes requests."""
            nonlocal request_count

            # Ramp-up delay
            if start_delay > 0:
                await asyncio.sleep(start_delay)

            while True:
                # Check if we should stop
                if duration_seconds:
                    if time.time() - start_time >= duration_seconds:
                        break
                else:
                    if request_count >= num_requests:
                        break

                request_count += 1
                current_request = request_count

                # Generate arguments for this request
                args = args_generator(current_request, worker_id)

                # Make request and measure time
                request_start = time.time()
                try:
                    result = await self.invoke_mcp_tool(tool_name, args)
                    request_duration = (time.time() - request_start) * 1000

                    metrics.response_times_ms.append(request_duration)
                    metrics.total_requests += 1

                    if result.get("success", False):
                        metrics.successful_requests += 1
                    else:
                        metrics.failed_requests += 1
                        error_msg = result.get("error", "Unknown error")
                        metrics.errors.append(
                            f"Request {current_request}: {error_msg}"
                        )

                except Exception as e:
                    request_duration = (time.time() - request_start) * 1000
                    metrics.response_times_ms.append(request_duration)
                    metrics.total_requests += 1
                    metrics.failed_requests += 1
                    metrics.errors.append(f"Request {current_request}: {str(e)}")

                # Collect resource metrics periodically
                if current_request % 10 == 0:
                    memory_info = self.process.memory_info()
                    metrics.memory_usage_mb.append(memory_info.rss / 1024 / 1024)
                    metrics.cpu_percent.append(self.process.cpu_percent())

        # Start workers with ramp-up
        for i in range(concurrency):
            delay = i * ramp_up_delay
            task = asyncio.create_task(worker(i, delay))
            active_tasks.add(task)

        # Wait for all workers to complete
        await asyncio.gather(*active_tasks, return_exceptions=True)

        # Calculate total duration
        metrics.total_duration_seconds = time.time() - start_time

        return metrics


@pytest.fixture
async def load_tester():
    """Provide load tester instance with reasonable timeouts."""
    async with MCPLoadTester(init_timeout=5.0, timeout=30.0, tool_timeout=60.0) as tester:
        yield tester


@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for validation."""
    return {
        "min_success_rate_percent": 95.0,
        "max_avg_response_time_ms": 2000,
        "max_p95_response_time_ms": 5000,
        "max_p99_response_time_ms": 10000,
        "min_throughput_rps": 1.0,
        "max_memory_growth_mb": 500,
        "max_cpu_percent": 80,
    }


@pytest.mark.integration
@pytest.mark.load
@pytest.mark.usefixtures("docker_logs_collector")
class TestMCPThroughput:
    """Throughput tests for MCP tools."""

    @pytest.mark.asyncio
    async def test_search_tool_throughput(self, load_tester, performance_thresholds):
        """Test throughput of the search tool under sustained load."""

        def args_generator(request_num, worker_id):
            """Generate search queries."""
            queries = [
                "Python async programming",
                "FastAPI performance",
                "Vector database optimization",
                "RAG best practices",
                "Web scraping techniques",
            ]
            return {"query": queries[request_num % len(queries)], "num_results": 5}

        # Run load test
        metrics = await load_tester.run_load_test(
            tool_name="search",
            args_generator=args_generator,
            num_requests=100,
            concurrency=5,
            ramp_up_seconds=2,
        )

        # Print results
        print("\n" + "=" * 60)
        print("SEARCH TOOL THROUGHPUT TEST")
        print("=" * 60)
        print(json.dumps(metrics.to_dict(), indent=2))

        # Validate against thresholds
        assert (
            metrics.success_rate >= performance_thresholds["min_success_rate_percent"]
        ), f"Success rate {metrics.success_rate}% below threshold"

        assert (
            metrics.throughput_rps >= performance_thresholds["min_throughput_rps"]
        ), f"Throughput {metrics.throughput_rps} RPS below threshold"

        assert (
            metrics.avg_response_time_ms
            <= performance_thresholds["max_avg_response_time_ms"]
        ), f"Average response time {metrics.avg_response_time_ms}ms above threshold"

    @pytest.mark.asyncio
    async def test_scrape_urls_throughput(self, load_tester, performance_thresholds):
        """Test throughput of the scrape_urls tool."""

        def args_generator(request_num, worker_id):
            """Generate URLs to scrape."""
            return {
                "url": f"https://example.com/page{request_num}",
                "return_raw_markdown": True,
            }

        metrics = await load_tester.run_load_test(
            tool_name="scrape_urls",
            args_generator=args_generator,
            num_requests=50,
            concurrency=3,
            ramp_up_seconds=1,
        )

        print("\n" + "=" * 60)
        print("SCRAPE_URLS THROUGHPUT TEST")
        print("=" * 60)
        print(json.dumps(metrics.to_dict(), indent=2))

        assert metrics.success_rate >= performance_thresholds["min_success_rate_percent"]
        assert metrics.throughput_rps >= 0.5  # Lower threshold for scraping

    @pytest.mark.asyncio
    async def test_perform_rag_query_throughput(
        self, load_tester, performance_thresholds
    ):
        """Test throughput of RAG query operations."""

        def args_generator(request_num, worker_id):
            """Generate RAG queries."""
            queries = [
                "How to implement async functions?",
                "Best practices for error handling",
                "Database connection pooling",
                "API rate limiting strategies",
            ]
            return {"query": queries[request_num % len(queries)], "match_count": 5}

        metrics = await load_tester.run_load_test(
            tool_name="perform_rag_query",
            args_generator=args_generator,
            num_requests=100,
            concurrency=10,
            ramp_up_seconds=2,
        )

        print("\n" + "=" * 60)
        print("RAG QUERY THROUGHPUT TEST")
        print("=" * 60)
        print(json.dumps(metrics.to_dict(), indent=2))

        assert metrics.success_rate >= performance_thresholds["min_success_rate_percent"]
        assert metrics.throughput_rps >= performance_thresholds["min_throughput_rps"]


@pytest.mark.integration
@pytest.mark.load
@pytest.mark.usefixtures("docker_logs_collector")
class TestMCPLatency:
    """Latency tests for MCP tools under different load conditions."""

    @pytest.mark.asyncio
    async def test_search_latency_single_user(
        self, load_tester, performance_thresholds
    ):
        """Test search latency with single user."""

        def args_generator(request_num, worker_id):
            return {"query": "test query", "num_results": 5}

        metrics = await load_tester.run_load_test(
            tool_name="search",
            args_generator=args_generator,
            num_requests=20,
            concurrency=1,
        )

        print("\n" + "=" * 60)
        print("SEARCH LATENCY - SINGLE USER")
        print("=" * 60)
        print(json.dumps(metrics.to_dict(), indent=2))

        # Single user should have low latency
        assert metrics.p95_response_time_ms <= 1000, "P95 latency too high for single user"
        assert metrics.p99_response_time_ms <= 2000, "P99 latency too high for single user"

    @pytest.mark.asyncio
    async def test_search_latency_concurrent_users(
        self, load_tester, performance_thresholds
    ):
        """Test search latency with multiple concurrent users."""

        def args_generator(request_num, worker_id):
            return {"query": f"query from user {worker_id}", "num_results": 5}

        # Test with increasing concurrency
        concurrency_levels = [1, 5, 10, 20]
        results = {}

        for concurrency in concurrency_levels:
            metrics = await load_tester.run_load_test(
                tool_name="search",
                args_generator=args_generator,
                num_requests=50,
                concurrency=concurrency,
                ramp_up_seconds=1,
            )

            results[concurrency] = {
                "p50_ms": metrics.p50_response_time_ms,
                "p95_ms": metrics.p95_response_time_ms,
                "p99_ms": metrics.p99_response_time_ms,
                "throughput_rps": metrics.throughput_rps,
            }

            print(f"\nConcurrency {concurrency}: {json.dumps(results[concurrency], indent=2)}")

        # Validate latency doesn't degrade too much with concurrency
        baseline_p95 = results[1]["p95_ms"]
        max_p95 = results[20]["p95_ms"]

        # P95 shouldn't increase more than 5x with 20x concurrency
        assert max_p95 <= baseline_p95 * 5, "Latency degradation too severe under load"

    @pytest.mark.asyncio
    async def test_rag_query_latency_distribution(
        self, load_tester, performance_thresholds
    ):
        """Test RAG query latency distribution."""

        def args_generator(request_num, worker_id):
            return {"query": "test query for latency", "match_count": 10}

        metrics = await load_tester.run_load_test(
            tool_name="perform_rag_query",
            args_generator=args_generator,
            num_requests=100,
            concurrency=5,
        )

        print("\n" + "=" * 60)
        print("RAG QUERY LATENCY DISTRIBUTION")
        print("=" * 60)
        print(json.dumps(metrics.to_dict(), indent=2))

        # Validate latency percentiles
        assert (
            metrics.p95_response_time_ms
            <= performance_thresholds["max_p95_response_time_ms"]
        )
        assert (
            metrics.p99_response_time_ms
            <= performance_thresholds["max_p99_response_time_ms"]
        )

        # Check latency consistency (p99 shouldn't be too far from p95)
        latency_spread = metrics.p99_response_time_ms - metrics.p95_response_time_ms
        assert latency_spread <= metrics.p95_response_time_ms, "Latency too inconsistent"


@pytest.mark.integration
@pytest.mark.load
@pytest.mark.usefixtures("docker_logs_collector")
class TestMCPConcurrency:
    """Concurrency tests for MCP tools."""

    @pytest.mark.asyncio
    async def test_mixed_workload_concurrency(self, load_tester):
        """Test mixed workload with multiple tools concurrently."""

        async def run_tool_workload(tool_name, num_requests, args_gen):
            """Run workload for a specific tool."""
            return await load_tester.run_load_test(
                tool_name=tool_name,
                args_generator=args_gen,
                num_requests=num_requests,
                concurrency=3,
            )

        # Define workloads for different tools
        workloads = [
            (
                "search",
                30,
                lambda req, wid: {"query": f"search query {req}", "num_results": 5},
            ),
            (
                "perform_rag_query",
                30,
                lambda req, wid: {"query": f"rag query {req}", "match_count": 5},
            ),
            (
                "scrape_urls",
                20,
                lambda req, wid: {
                    "url": f"https://example.com/page{req}",
                    "return_raw_markdown": True,
                },
            ),
        ]

        # Run all workloads concurrently
        start_time = time.time()
        results = await asyncio.gather(
            *[run_tool_workload(tool, num, args_gen) for tool, num, args_gen in workloads]
        )
        total_duration = time.time() - start_time

        # Analyze results
        print("\n" + "=" * 60)
        print("MIXED WORKLOAD CONCURRENCY TEST")
        print("=" * 60)
        print(f"Total duration: {total_duration:.2f}s")

        for (tool_name, _, _), metrics in zip(workloads, results):
            print(f"\n{tool_name}:")
            print(f"  Success rate: {metrics.success_rate:.2f}%")
            print(f"  Throughput: {metrics.throughput_rps:.2f} RPS")
            print(f"  Avg latency: {metrics.avg_response_time_ms:.2f}ms")
            print(f"  P95 latency: {metrics.p95_response_time_ms:.2f}ms")

            # All tools should maintain good success rate
            assert metrics.success_rate >= 90, f"{tool_name} success rate too low"

    @pytest.mark.asyncio
    async def test_concurrent_users_simulation(self, load_tester):
        """Simulate realistic concurrent user behavior."""

        async def simulate_user(user_id: int, duration_seconds: float):
            """Simulate a single user's behavior."""
            user_metrics = LoadTestMetrics()
            start_time = time.time()
            request_count = 0

            while time.time() - start_time < duration_seconds:
                request_count += 1

                # User performs different actions
                if request_count % 3 == 0:
                    # Search action
                    tool = "search"
                    args = {"query": f"user {user_id} search {request_count}", "num_results": 5}
                elif request_count % 3 == 1:
                    # RAG query action
                    tool = "perform_rag_query"
                    args = {"query": f"user {user_id} question {request_count}", "match_count": 5}
                else:
                    # Get sources action
                    tool = "get_available_sources"
                    args = {}

                # Make request
                req_start = time.time()
                try:
                    result = await load_tester.invoke_mcp_tool(tool, args)
                    req_duration = (time.time() - req_start) * 1000

                    user_metrics.response_times_ms.append(req_duration)
                    user_metrics.total_requests += 1

                    if result.get("success", False):
                        user_metrics.successful_requests += 1
                    else:
                        user_metrics.failed_requests += 1

                except Exception as e:
                    user_metrics.total_requests += 1
                    user_metrics.failed_requests += 1
                    user_metrics.errors.append(str(e))

                # Think time between requests
                await asyncio.sleep(0.5)

            user_metrics.total_duration_seconds = time.time() - start_time
            return user_metrics

        # Simulate 10 concurrent users for 10 seconds
        num_users = 10
        duration = 10

        print("\n" + "=" * 60)
        print(f"SIMULATING {num_users} CONCURRENT USERS FOR {duration}s")
        print("=" * 60)

        user_tasks = [simulate_user(i, duration) for i in range(num_users)]
        user_results = await asyncio.gather(*user_tasks)

        # Aggregate results
        total_requests = sum(m.total_requests for m in user_results)
        total_successful = sum(m.successful_requests for m in user_results)
        all_response_times = []
        for m in user_results:
            all_response_times.extend(m.response_times_ms)

        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0

        print(f"\nOverall Results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Success rate: {overall_success_rate:.2f}%")
        print(f"  Avg response time: {avg_response_time:.2f}ms")
        print(f"  Throughput: {total_requests / duration:.2f} RPS")

        # Validate
        assert overall_success_rate >= 90, "Overall success rate too low"
        assert total_requests >= num_users * 10, "Not enough requests generated"


@pytest.mark.integration
@pytest.mark.load
@pytest.mark.slow
@pytest.mark.load
@pytest.mark.usefixtures("docker_logs_collector")
class TestMCPEndurance:
    """Endurance tests for sustained load over time."""

    @pytest.mark.asyncio
    async def test_sustained_load_endurance(self, load_tester):
        """Test system behavior under sustained load for extended period."""

        def args_generator(request_num, worker_id):
            queries = [
                "endurance test query 1",
                "endurance test query 2",
                "endurance test query 3",
            ]
            return {"query": queries[request_num % len(queries)], "num_results": 5}

        # Run for 60 seconds with moderate load
        duration_seconds = 60
        concurrency = 5

        print("\n" + "=" * 60)
        print(f"ENDURANCE TEST - {duration_seconds}s with {concurrency} concurrent users")
        print("=" * 60)

        # Collect metrics at intervals
        interval_metrics = []
        interval_duration = 10  # Collect metrics every 10 seconds

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            interval_start = time.time()

            # Run load for this interval
            metrics = await load_tester.run_load_test(
                tool_name="search",
                args_generator=args_generator,
                duration_seconds=interval_duration,
                concurrency=concurrency,
            )

            interval_elapsed = time.time() - start_time
            interval_metrics.append((interval_elapsed, metrics))

            print(f"\nInterval {len(interval_metrics)} ({interval_elapsed:.0f}s):")
            print(f"  Requests: {metrics.total_requests}")
            print(f"  Success rate: {metrics.success_rate:.2f}%")
            print(f"  Throughput: {metrics.throughput_rps:.2f} RPS")
            print(f"  Avg latency: {metrics.avg_response_time_ms:.2f}ms")
            print(f"  Memory: {metrics.avg_memory_mb:.2f}MB")

        # Analyze stability over time
        success_rates = [m.success_rate for _, m in interval_metrics]
        throughputs = [m.throughput_rps for _, m in interval_metrics]
        latencies = [m.avg_response_time_ms for _, m in interval_metrics]

        print("\n" + "=" * 60)
        print("ENDURANCE TEST SUMMARY")
        print("=" * 60)
        print(f"Success rate - Min: {min(success_rates):.2f}%, Max: {max(success_rates):.2f}%")
        print(f"Throughput - Min: {min(throughputs):.2f}, Max: {max(throughputs):.2f} RPS")
        print(f"Latency - Min: {min(latencies):.2f}ms, Max: {max(latencies):.2f}ms")

        # Validate stability
        assert min(success_rates) >= 90, "Success rate dropped below 90% during endurance test"

        # Throughput should remain relatively stable (within 50% variance)
        throughput_variance = (max(throughputs) - min(throughputs)) / statistics.mean(throughputs)
        assert throughput_variance < 0.5, f"Throughput too unstable: {throughput_variance:.2%} variance"

        # Latency shouldn't increase significantly over time (no degradation)
        first_half_latency = statistics.mean(latencies[: len(latencies) // 2])
        second_half_latency = statistics.mean(latencies[len(latencies) // 2 :])
        latency_increase = (second_half_latency - first_half_latency) / first_half_latency

        assert latency_increase < 0.3, f"Latency degraded {latency_increase:.2%} over time"

        print(f"\n✅ System stable over {duration_seconds}s endurance test")
