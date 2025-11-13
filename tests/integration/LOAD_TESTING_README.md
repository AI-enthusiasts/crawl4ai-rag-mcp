# MCP Load Testing Guide

## Overview

This directory contains comprehensive load testing suite for the Crawl4AI MCP Server. The tests are designed to measure performance, identify bottlenecks, and validate system behavior under various load conditions.

## Test Categories

### 1. Throughput Tests (`TestMCPThroughput`)

Measure the system's ability to handle sustained request load.

**Tests:**
- `test_search_tool_throughput` - Search tool performance under load
- `test_scrape_urls_throughput` - URL scraping throughput
- `test_perform_rag_query_throughput` - RAG query performance

**Key Metrics:**
- Requests per second (RPS)
- Success rate percentage
- Average response time
- P95/P99 latency

**Thresholds:**
- Minimum success rate: 95%
- Minimum throughput: 1.0 RPS
- Maximum average response time: 2000ms

### 2. Latency Tests (`TestMCPLatency`)

Measure response times under different load conditions.

**Tests:**
- `test_search_latency_single_user` - Baseline latency with single user
- `test_search_latency_concurrent_users` - Latency degradation with concurrency
- `test_rag_query_latency_distribution` - Latency percentile distribution

**Key Metrics:**
- P50 (median) latency
- P95 latency
- P99 latency
- Maximum latency

**Thresholds:**
- P95 latency: < 5000ms
- P99 latency: < 10000ms
- Latency degradation: < 5x with 20x concurrency

### 3. Concurrency Tests (`TestMCPConcurrency`)

Test system behavior with multiple concurrent users and mixed workloads.

**Tests:**
- `test_mixed_workload_concurrency` - Multiple tools running concurrently
- `test_concurrent_users_simulation` - Realistic user behavior simulation

**Key Metrics:**
- Concurrent request handling
- Resource contention
- Success rate under concurrency
- Throughput with mixed workloads

**Validation:**
- All tools maintain >90% success rate
- No deadlocks or race conditions
- Graceful degradation under load

### 4. Stress Tests (`TestMCPStress`)

Push the system to its limits to find breaking points.

**Tests:**
- `test_search_stress_increasing_load` - Incrementally increase load until failure
- `test_memory_stress` - Memory usage under sustained load

**Key Metrics:**
- Maximum sustainable concurrency
- Memory growth patterns
- System degradation points
- Resource exhaustion thresholds

**Validation:**
- Memory growth < 500MB
- No memory leaks (final memory < 1.5x baseline)
- Identify optimal concurrency level

### 5. Endurance Tests (`TestMCPEndurance`)

Validate system stability over extended periods.

**Tests:**
- `test_sustained_load_endurance` - 60-second sustained load test

**Key Metrics:**
- Performance stability over time
- Throughput variance
- Latency degradation
- Resource leak detection

**Validation:**
- Success rate stays >90% throughout
- Throughput variance < 50%
- Latency increase < 30% over time

## Running Load Tests

### Prerequisites

```bash
# Ensure MCP server is running
docker-compose up -d

# Install test dependencies
uv sync
```

### Run All Load Tests

```bash
# Run all load tests
uv run pytest tests/integration/test_mcp_load_testing.py -v -m load

# Run specific test category
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPThroughput -v

# Run with detailed output
uv run pytest tests/integration/test_mcp_load_testing.py -v -s
```

### Run Individual Tests

```bash
# Throughput test
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPThroughput::test_search_tool_throughput -v

# Latency test
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPLatency::test_search_latency_concurrent_users -v

# Stress test
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPStress::test_search_stress_increasing_load -v
```

### Skip Slow Tests

```bash
# Skip endurance and stress tests
uv run pytest tests/integration/test_mcp_load_testing.py -v -m "load and not slow"
```

## Test Configuration

### Performance Thresholds

Thresholds are defined in the `performance_thresholds` fixture:

```python
{
    "min_success_rate_percent": 95.0,
    "max_avg_response_time_ms": 2000,
    "max_p95_response_time_ms": 5000,
    "max_p99_response_time_ms": 10000,
    "min_throughput_rps": 1.0,
    "max_memory_growth_mb": 500,
    "max_cpu_percent": 80,
}
```

### Load Test Parameters

Customize load test parameters:

```python
metrics = await load_tester.run_load_test(
    tool_name="search",
    args_generator=args_generator,
    num_requests=100,        # Total requests
    concurrency=5,           # Concurrent workers
    ramp_up_seconds=2,       # Gradual load increase
    duration_seconds=None,   # Optional time limit
)
```

## Metrics Collected

### Response Time Metrics
- Average response time
- Minimum/Maximum response time
- P50 (median)
- P95 (95th percentile)
- P99 (99th percentile)

### Throughput Metrics
- Total requests
- Successful requests
- Failed requests
- Success rate percentage
- Requests per second (RPS)

### Resource Metrics
- Memory usage (average and peak)
- CPU usage percentage
- Memory growth over time

### Error Metrics
- Error count
- Error messages
- Failure patterns

## Interpreting Results

### Good Performance Indicators

✅ **Success Rate**: >95%
✅ **Throughput**: Scales with concurrency
✅ **Latency**: P95 < 5s, P99 < 10s
✅ **Memory**: Stable, no leaks
✅ **CPU**: < 80% average

### Warning Signs

⚠️ **Success Rate**: 90-95%
⚠️ **Latency**: P95 > 5s
⚠️ **Memory**: Growing over time
⚠️ **Throughput**: Decreasing with concurrency

### Critical Issues

🔴 **Success Rate**: < 90%
🔴 **Latency**: P99 > 10s
🔴 **Memory**: Continuous growth (leak)
🔴 **Errors**: Increasing error rate

## Example Output

```
==============================================================
SEARCH TOOL THROUGHPUT TEST
==============================================================
{
  "total_requests": 100,
  "successful_requests": 98,
  "failed_requests": 2,
  "success_rate_percent": 98.0,
  "total_duration_seconds": 12.45,
  "throughput_rps": 8.03,
  "response_times": {
    "avg_ms": 623.45,
    "min_ms": 102.34,
    "max_ms": 1234.56,
    "p50_ms": 567.89,
    "p95_ms": 987.65,
    "p99_ms": 1123.45
  },
  "resource_usage": {
    "avg_memory_mb": 245.67,
    "peak_memory_mb": 289.34,
    "avg_cpu_percent": 45.23
  },
  "errors": [],
  "error_count": 2
}
```

## Integration with FastMCP Client

### Current Implementation

The load tester uses the official **FastMCP Client** to connect to the MCP server via HTTP transport:

```python
from fastmcp import Client

# Client automatically infers HTTP transport from URL
client = Client("http://localhost:8051")

async with client:
    # Call tools using the FastMCP Client API
    result = await client.call_tool("search", {"query": "test", "num_results": 5})
    
    # Access structured data
    print(result.data)  # Fully hydrated Python objects
    
    # Check for errors
    if result.is_error:
        print(f"Error: {result.content}")
```

### Configuration

Set environment variables to configure the MCP server connection:

```bash
# MCP server URL (HTTP transport)
export MCP_SERVER_URL="http://localhost:8051"

# Optional: API key for authentication
export MCP_API_KEY="your-api-key"
```

### FastMCP Client Features

The load tests leverage FastMCP Client's features:

- **Automatic transport detection** - HTTP transport inferred from URL
- **Structured data access** - `.data` property provides fully hydrated Python objects
- **Error handling** - `.is_error` property for checking failures
- **Async context manager** - Proper connection lifecycle management
- **Type safety** - Full type hints and IDE support

## Continuous Integration

### GitHub Actions Integration

Add to `.github/workflows/load-tests.yml`:

```yaml
name: Load Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Run load tests
        run: |
          uv run pytest tests/integration/test_mcp_load_testing.py \
            -v -m "load and not slow" \
            --json-report --json-report-file=load-test-results.json
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: load-test-results.json
```

## Performance Optimization Tips

### 1. Increase Concurrency

If tests show good performance with low concurrency, try increasing:

```python
concurrency=20  # Instead of 5
```

### 2. Adjust Timeouts

For slower operations, increase thresholds:

```python
"max_avg_response_time_ms": 5000,  # Instead of 2000
```

### 3. Optimize Resource Usage

Monitor memory and CPU usage to identify bottlenecks:

```python
# Check resource metrics in test output
print(f"Peak memory: {metrics.peak_memory_mb}MB")
print(f"Avg CPU: {metrics.avg_cpu_percent}%")
```

### 4. Database Optimization

- Enable connection pooling
- Optimize vector search parameters
- Use batch operations where possible

### 5. Caching

- Enable response caching for frequently accessed data
- Use Redis for distributed caching

## Troubleshooting

### High Failure Rate

**Symptoms:** Success rate < 95%

**Solutions:**
- Check server logs for errors
- Verify database connectivity
- Increase timeout values
- Reduce concurrency level

### High Latency

**Symptoms:** P95 > 5s

**Solutions:**
- Optimize database queries
- Enable caching
- Increase server resources
- Use connection pooling

### Memory Leaks

**Symptoms:** Continuous memory growth

**Solutions:**
- Check for unclosed connections
- Review async task cleanup
- Use memory profiler
- Implement proper resource cleanup

### Throughput Degradation

**Symptoms:** RPS decreases with concurrency

**Solutions:**
- Check for lock contention
- Optimize async operations
- Increase worker pool size
- Review database connection limits

## Best Practices

1. **Run load tests regularly** - Catch performance regressions early
2. **Test realistic scenarios** - Use production-like workloads
3. **Monitor resources** - Track memory, CPU, and network usage
4. **Set appropriate thresholds** - Based on SLA requirements
5. **Document baselines** - Track performance over time
6. **Test edge cases** - High concurrency, large payloads, etc.
7. **Isolate tests** - Avoid interference between tests
8. **Clean up resources** - Prevent resource exhaustion

## Contributing

When adding new load tests:

1. Follow existing test structure
2. Use descriptive test names
3. Document expected behavior
4. Set appropriate thresholds
5. Add to relevant test class
6. Update this README

## References

- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [psutil documentation](https://psutil.readthedocs.io/)
- [Load Testing Best Practices](https://www.nginx.com/blog/load-testing-best-practices/)
- [Python Async Performance](https://docs.python.org/3/library/asyncio-dev.html)
