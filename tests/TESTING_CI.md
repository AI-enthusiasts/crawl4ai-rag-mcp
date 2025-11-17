# CI/CD Testing Infrastructure

This document describes the CI/CD testing setup for integration tests.

## Overview

The project uses GitHub Actions CI with Docker Compose to run integration tests against real services (NO MOCKS policy).

## Test Environments

### Local Testing

Run tests locally with real services:

```bash
# Start services
docker-compose -f docker-compose.test.yml up -d --wait

# Run integration tests
uv run pytest tests/ -m integration -v

# Stop services
docker-compose -f docker-compose.test.yml down -v
```

### CI Testing (GitHub Actions)

CI automatically runs:
1. **Unit tests** (always) - Tests without external service dependencies
2. **Integration tests** (always) - Tests with real Qdrant, SearXNG, MCP server

## Services in Test Environment

### docker-compose.test.yml

Lightweight configuration for CI/CD:

- **Qdrant** (localhost:6333) - Vector database
  - No password for simplicity
  - In-memory storage (fast startup)
  - Health check enabled

- **SearXNG** (localhost:8080) - Search engine
  - No authentication
  - Minimal configuration
  - Limiter disabled for faster tests

- **Valkey/Redis** (localhost:6379) - Cache
  - No password
  - LRU eviction
  - Persistence disabled (fast)

- **MCP Server** (localhost:8051) - Main application
  - Test configuration
  - Connects to all services
  - OPENAI_API_KEY from CI secrets

## CI Workflow (.github/workflows/ci.yml)

### Steps

1. **Lint & Format** - Ruff checks
2. **Unit Tests** - No external services, fast
3. **Integration Tests**:
   - Start all services via docker-compose
   - Wait for health checks (120s timeout)
   - Verify connectivity
   - Run pytest with `integration` marker
   - Show logs on failure
   - Clean up containers

### Environment Variables

Required for integration tests:

```yaml
QDRANT_URL: http://localhost:6333
SEARXNG_URL: http://localhost:8080
VALKEY_URL: redis://localhost:6379
OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # Set in GitHub repo settings
```

## Test Markers

Tests use pytest markers to categorize:

```python
@pytest.mark.integration  # Requires external services
@pytest.mark.slow         # Takes >5 seconds
```

Run specific test categories:

```bash
# Only integration tests
pytest -m integration

# Only non-integration tests (unit)
pytest -m "not integration"

# Slow tests
pytest -m slow
```

## Skip Policy (NO MOCKS)

Per project policy, integration tests:
- Use **real services** only
- Skip via `pytest.skip()` if service unavailable
- **NEVER mock** external services

Example:

```python
@pytest.fixture
async def qdrant_client():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/collections")
            if response.status_code != 200:
                pytest.skip("Qdrant not available")
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")

    return QdrantAdapter(url="http://localhost:6333")
```

## Troubleshooting

### Services not starting in CI

Check logs:
```bash
docker-compose -f docker-compose.test.yml logs
```

Common issues:
- Port already in use (kill existing services)
- Health check timeout (increase wait time)
- Missing environment variables

### Tests failing locally but passing in CI

Likely issues:
- Different service versions
- Different environment variables
- Data persistence (clean volumes: `docker-compose down -v`)

### Integration tests skipped in CI

- Services failed to start (check docker logs)
- Network connectivity issues
- Health checks timing out

## Performance

Current CI test times:
- Unit tests: ~30-60s
- Integration tests: ~2-5 min (including service startup)
- Total CI time: ~5-10 min

Optimization tips:
- Use `--maxfail=10` to stop early on failures
- Parallel test execution with `pytest-xdist` (future)
- Cache Docker images in CI

## Adding New Services

To add a new service for tests:

1. Add to `docker-compose.test.yml`:
   ```yaml
   new-service:
     image: service/image:tag
     ports:
       - "PORT:PORT"
     healthcheck:
       test: ["CMD", "curl", "-f", "http://localhost:PORT/health"]
   ```

2. Update CI environment variables in `.github/workflows/ci.yml`

3. Add fixture in `tests/integration/conftest.py`:
   ```python
   @pytest.fixture
   async def new_service():
       # Check availability, skip if not ready
       ...
   ```

4. Document in this file

## References

- [Project Testing Policy](../AGENTS.md#testing)
- [Integration Tests README](integration/README.md)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [pytest Markers](https://docs.pytest.org/en/stable/example/markers.html)
