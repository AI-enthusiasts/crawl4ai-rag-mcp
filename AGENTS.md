# Crawl4AI RAG MCP

**IMPORTANT**: This project uses **uv** for ALL Python operations (not pip/poetry/conda)

## Repository Structure

```
crawl4ai-rag-mcp/
├── src/
│   ├── services/     # Crawling, search logic
│   ├── database/     # Vector DB adapters
│   ├── utils/        # Helper functions
│   └── tools.py      # MCP tool definitions
├── docs/             # Documentation
├── pyproject.toml    # Project configuration
└── uv.lock          # uv lockfile
```

## Deployment Workflow

**IMPORTANT**: This project does NOT auto-deploy from git push. Manual deployment required.

```bash
# 1. Make changes
vim openmemory/api/app/routers/memories.py

# 2. Commit changes
git add .
git commit -m "fix: your message"
git push origin mcp-service

# 3. Build Docker images locally
cd ~/mem0
make docker-build          # Builds all 3 images (API, MCP, MCP Bearer)

# 4. Test images
make docker-test           # Verify imports work

# 5. Push to GHCR
make docker-push           # Push all images to GitHub Container Registry

# 6. Deploy via Coolify dashboard
# Go to Coolify → Find mem0 services → Click "Redeploy" on each:
#   - openmemory-api
#   - openmemory-mcp
#   - openmemory-mcp-bearer (if changed)
```

**Why manual deployment?**
- Uses path dependencies (mem0ai fork)
- Custom Docker builds with uv
- Coolify can't build from git (needs local context)

## Testing

**Unit & Integration Tests**:
```bash
# Run unit tests
make test-unit

# Run integration tests
make test-integration

# Run all tests
make test-all

# Run tests with coverage
make test-coverage
```

**Health Checks**:
```bash
# Check MCP server health (requires auth, returns error is OK)
curl -X POST https://rag.melo.eu.org/mcp

# Check Docker logs (suffix changes on each deploy)
docker ps | grep crawl4ai
docker logs -f mcp-crawl4ai-<suffix>

# Check container creation time to verify new deploy
docker ps --format "{{.CreatedAt}}\t{{.Names}}" | grep crawl4ai
```

## Load Testing

**Prerequisites**: 
- Environment variables are loaded automatically from `.env` file
- Ensure `MCP_SERVER_URL` and `MCP_API_KEY` are set in `.env`:
  ```bash
  MCP_SERVER_URL=https://rag.melo.eu.org/mcp
  MCP_API_KEY=your-api-key
  ```

**Quick Start**:
```bash
cd ~/crawl4ai-rag-mcp
make load-test              # Fast tests (~4 min)
```

**Available Commands**:
```bash
# Fast tests (Throughput + Latency, ~4 min) - RECOMMENDED
make load-test              # Alias for load-test-fast
make load-test-fast         # Same as above

# Individual test suites
make load-test-throughput   # ~2 min - Measures requests/second
make load-test-latency      # ~2 min - Response time distribution (p50/p95/p99)
make load-test-concurrency  # ~5 min - Parallel request handling
make load-test-endurance    # ~15 min - 60s sustained load (SLOW)

# All tests including endurance (~20 min)
make load-test-all          # Runs everything
```

**Test Categories**:
- **Throughput**: Measures requests/second capacity under load
  - `test_search_tool_throughput` - Search performance
  - `test_scrape_urls_throughput` - URL scraping
  - `test_perform_rag_query_throughput` - RAG queries
  
- **Latency**: Measures response time distribution
  - `test_search_latency_single_user` - Baseline latency
  - `test_search_latency_concurrent_users` - Latency under load
  - `test_rag_query_latency_distribution` - P50/P95/P99 percentiles
  
- **Concurrency**: Tests multi-user scenarios
  - `test_mixed_workload_concurrency` - Multiple tools simultaneously
  - `test_concurrent_users_simulation` - User simulation
  
- **Endurance**: Long-term stability (marked as `slow`)
  - `test_sustained_load_endurance` - 60s sustained load

**Understanding Results**:
- ✅ **Good**: Success rate >95%, P95 latency <5s, stable memory
- ⚠️ **Warning**: Success rate 90-95%, P95 latency >5s, growing memory
- 🔴 **Critical**: Success rate <90%, P99 latency >10s, continuous memory growth

**Troubleshooting**:
```bash
# Check if server is running
docker ps --filter "name=crawl4ai"

# View server logs
docker logs -f $(docker ps --filter "name=crawl4ai" --format "{{.Names}}")

# Verify environment variables
cd ~/crawl4ai-rag-mcp && cat .env | grep MCP_
```

**Documentation**:
- Full guide: `LOAD_TESTING.md`
- Test file: `tests/integration/test_mcp_load_testing.py`
- Results: `tests/results/load_tests/`

## Health Checks

- **MCP endpoint**: `/mcp` - requires auth, any response = server alive
- **Docker healthcheck**: TCP connection to port (not HTTP)

## File Management Rules

**IMPORTANT**:

- Only create files inside the repository (`~/crawl4ai-rag-mcp/`)
- Follow project structure:
  - Code: `src/`
  - Tests: `tests/`
  - Docs: `docs/`
  - Scripts: `scripts/`
- Never create random files in `~` (home directory)
- This AGENTS.md is the ONLY exception (instructions file)

**Deployment Warning**:

- Every `git push` to `feat/deployment-improvements` triggers Coolify redeploy
- Commit often locally: `git commit -m "message"`
- Push only when ready to deploy: `git push origin feat/deployment-improvements`

## Notes

- Deploy time: ~2-5 minutes
- Monitor: Coolify dashboard
- Server: https://rag.melo.eu.org
- Docker container name has dynamic suffix (changes on redeploy)
- Healthcheck: TCP connection to port, not HTTP endpoint
