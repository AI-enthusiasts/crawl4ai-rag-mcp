# Load Testing Guide

## Quick Start (Recommended)

**Using Makefile** (easiest way):
```bash
# 1. Ensure .env file has credentials (auto-loaded by Makefile)
cat .env | grep MCP_

# 2. Run fast tests (~4 min)
make load-test
```

**Using pytest directly**:
```bash
# 1. Configure server URL
export MCP_SERVER_URL="https://rag.melo.eu.org/mcp"
export MCP_API_KEY="your-api-key"

# 2. Run tests
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPThroughput -v
```

## Running Tests

### Using Makefile (Recommended)

Environment variables are loaded automatically from `.env` file.

```bash
# Fast tests (Throughput + Latency, ~4 min) - RECOMMENDED
make load-test              # Alias for load-test-fast
make load-test-fast         # Same as above

# Individual test suites
make load-test-throughput   # ~2 min - Measures requests/second
make load-test-latency      # ~2 min - Response time distribution
make load-test-concurrency  # ~5 min - Parallel request handling
make load-test-endurance    # ~15 min - 60s sustained load (SLOW)

# All tests including endurance (~20 min)
make load-test-all
```

### Using pytest directly

```bash
# All tests (9 tests)
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py -v

# Exclude slow tests (endurance)
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py -v -m "not slow"

# Specific category
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPThroughput -v
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPLatency -v
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPConcurrency -v
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPEndurance -v

# Specific test
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPThroughput::test_search_tool_throughput -v
```

## Test Categories

### 1. Throughput (3 tests, ~2 min)
Measures requests per second under sustained load.

```bash
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPThroughput -v
```

**Tests:**
- `test_search_tool_throughput` - Search performance
- `test_scrape_urls_throughput` - URL scraping
- `test_perform_rag_query_throughput` - RAG queries

### 2. Latency (3 tests, ~2 min)
Measures response times and percentiles.

```bash
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPLatency -v
```

**Tests:**
- `test_search_latency_single_user` - Baseline latency
- `test_search_latency_concurrent_users` - Latency under load
- `test_rag_query_latency_distribution` - P50/P95/P99

### 3. Concurrency (2 tests, ~5 min)
Tests multi-user scenarios.

```bash
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPConcurrency -v
```

**Tests:**
- `test_mixed_workload_concurrency` - Multiple tools
- `test_concurrent_users_simulation` - User simulation

### 4. Endurance (1 test, ~15 min, slow)
Long-term stability testing.

```bash
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPEndurance -v
```

**Tests:**
- `test_sustained_load_endurance` - 60s sustained load

## Understanding Results

### Good Performance ✅
- Success rate: >95%
- Throughput: Scales with concurrency
- P95 latency: <5s
- Memory: Stable, no leaks

### Warning Signs ⚠️
- Success rate: 90-95%
- P95 latency: >5s
- Memory: Growing over time

### Critical Issues 🔴
- Success rate: <90%
- P99 latency: >10s
- Memory: Continuous growth

### Sample Output

```json
{
  "total_requests": 100,
  "successful_requests": 98,
  "success_rate_percent": 98.0,
  "throughput_rps": 8.03,
  "response_times": {
    "avg_ms": 623.45,
    "p95_ms": 987.65,
    "p99_ms": 1123.45
  },
  "resource_usage": {
    "peak_memory_mb": 289.34,
    "avg_cpu_percent": 45.23
  }
}
```

## Configuration

### Environment Variables

**Using Makefile** (auto-loaded from `.env`):
```bash
# Edit .env file (Makefile loads it automatically)
cat >> .env << 'EOF'
MCP_SERVER_URL=https://rag.melo.eu.org/mcp
MCP_API_KEY=your-api-key
EOF

# Run tests (no export needed)
make load-test
```

**Using pytest directly** (manual export):
```bash
# Required: MCP server URL (with /mcp endpoint)
export MCP_SERVER_URL="https://rag.melo.eu.org/mcp"

# Required: API key for authentication
export MCP_API_KEY="your-api-key"

# Run tests
~/.local/bin/uv run pytest tests/integration/test_mcp_load_testing.py -v
```

### Performance Thresholds

Default thresholds (can be adjusted in `performance_thresholds` fixture):

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

## Troubleshooting

### Tests fail to connect

```bash
# Verify environment variables
echo $MCP_SERVER_URL
echo $MCP_API_KEY

# Check Docker logs
docker logs --tail 100 $(docker ps --filter "name=mcp-crawl4ai" --format "{{.Names}}")

# Test logs are automatically saved to:
# tests/results/docker_logs/
```

### High failure rate

- Check Docker logs (automatically collected in `tests/results/docker_logs/`)
- Reduce concurrency in tests
- Increase timeout thresholds (see `tests/LOAD_TESTING_TIMEOUTS.md`)
- Check server health

### Slow tests

```bash
# Skip slow tests (endurance)
uv run pytest tests/integration/test_mcp_load_testing.py -v -m "not slow"

# Run only fast tests (Throughput + Latency)
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPThroughput -v
uv run pytest tests/integration/test_mcp_load_testing.py::TestMCPLatency -v
```

## Files

- **Tests:** `tests/integration/test_mcp_load_testing.py` (9 tests, 4 classes)
- **Docker Logs:** `tests/results/docker_logs/` (auto-collected)
- **Timeouts Guide:** `tests/LOAD_TESTING_TIMEOUTS.md`
- **Full Guide:** `tests/integration/LOAD_TESTING_README.md`

## Technical Details

**Uses FastMCP Client** for MCP server communication:
- Automatic HTTP transport detection from URL
- Bearer token authentication
- Structured data via `.data` property
- Built-in error handling with `.is_error`
- Configurable timeouts: init=5s, general=30s, tool=60s

**Automatic Docker Logs Collection:**
- Logs collected for each test via `docker_logs_collector` fixture
- Saved to `tests/results/docker_logs/`
- Automatic analysis for errors, warnings, timeouts

For detailed documentation, see:
- `tests/integration/LOAD_TESTING_README.md` - Full guide
- `tests/LOAD_TESTING_TIMEOUTS.md` - Timeout configuration
