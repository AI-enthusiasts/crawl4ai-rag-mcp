# Crawl4AI RAG MCP - Developer Guide

**Stack**: Docker + uv + Pydantic AI + FastMCP + Qdrant + Neo4j

## Core Rules

### Package Manager: uv ONLY
**CRITICAL**: ALL Python operations MUST use `uv`. Never use `python`, `pip`, `poetry`, or `conda` directly.

```bash
# Dependencies
uv sync                          # Install/sync all dependencies
uv add package-name              # Add production dependency
uv add --dev package-name        # Add dev dependency
uv lock --upgrade                # Update lockfile

# Execution
uv run pytest tests/             # Run tests
uv run python src/main.py        # Run scripts
uv run ruff check src/           # Lint code
uv run mypy src/                 # Type check
```

### Testing Workflow
```bash
# Pre-commit hook (auto-runs on git commit)
scripts/install-hooks.sh         # Install once

# Manual testing
uv run pytest                    # All tests
uv run pytest tests/unit/ -v    # Unit tests only
uv run pytest tests/test_imports.py  # Import verification
uv run pytest --cov=src --cov-report=html  # With coverage
```

### Docker Workflow
```bash
# Development (local, no Docker needed for most work)
uv sync && uv run pytest

# Docker build/test (only when testing Docker-specific issues)
docker-compose -f docker-compose.dev.yml up
docker-compose logs -f mcp
docker-compose build --no-cache mcp
```

## Architecture

### Technology Stack
- **Package Manager**: uv
- **MCP Framework**: FastMCP
- **LLM**: Pydantic AI (NOT OpenAI SDK)
- **Vector DB**: Qdrant (primary), Supabase (legacy)
- **Knowledge Graph**: Neo4j (optional)
- **Crawler**: Crawl4AI
- **Search**: SearXNG

### Repository Structure
```
src/
├── services/          # Business logic (agentic_search, crawling, search)
├── database/          # Vector DB adapters (qdrant, supabase)
├── knowledge_graph/   # Neo4j code analysis
├── config/            # Settings (settings.py)
├── core/              # Core utilities (context, exceptions, logging)
├── utils/             # Helpers (embeddings, validation, text_processing)
├── tools.py           # MCP tool registration
└── main.py            # FastMCP server entry

tests/                 # Test suite
├── unit/              # Unit tests
├── integration/       # Integration tests
├── fixtures/          # Test fixtures
└── test_imports.py    # Import verification (pre-commit)
```

## Code Patterns

### Pydantic AI Agents
**Always use Pydantic AI for LLM calls**:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

# Create agent with typed output
agent = Agent(
    model=OpenAIModel(model_name=settings.model_choice, api_key=settings.openai_api_key),
    output_type=YourPydanticModel,
    output_retries=3,
    model_settings=ModelSettings(temperature=0.7, timeout=60),
)

# Use agent
result = await agent.run(prompt)
return result.output  # Fully typed
```

### Singleton for Agents
```python
_instance: Service | None = None

def get_service() -> Service:
    global _instance
    if _instance is None:
        _instance = Service()
    return _instance
```

### FastMCP Lifespan
```python
@asynccontextmanager
async def lifespan(app: FastMCP):
    context = await initialize_resources()
    yield context
    await cleanup_resources(context)

mcp = FastMCP("Server Name", lifespan=lifespan)
```

## Configuration

### Environment Variables (`.env`)
```bash
# LLM
OPENAI_API_KEY=sk-...
MODEL_CHOICE=gpt-4o-mini

# Agentic Search
AGENTIC_SEARCH_ENABLED=true
AGENTIC_SEARCH_COMPLETENESS_THRESHOLD=0.85
AGENTIC_SEARCH_MAX_ITERATIONS=5

# Vector DB
USE_QDRANT=true
QDRANT_URL=http://qdrant:6333

# Services
SEARXNG_URL=http://searxng:8080
CRAWL4AI_URL=http://crawl4ai:8000

# Optional
USE_KNOWLEDGE_GRAPH=true
USE_OAUTH2=false
```

## Git Workflow

### Branches
- `main` - production
- `feat/*` - features
- `fix/*` - bug fixes
- `refactor/*` - refactoring

### Commits
**Conventional commits, ~500 lines max per commit**:

```bash
git commit -m "feat: add feature description"
git commit -m "fix: resolve bug description"
git commit -m "refactor: improve code structure"
git commit -m "test: add test coverage"
```

**Pre-commit hook automatically runs**:
1. Import verification (91 tests)
2. Ruff linting

### Squash WIP Commits
```bash
git reset --soft HEAD~5
git add <files>
git commit -m "feat: complete feature description"
```

## Common Tasks

### Add MCP Tool
1. Define model in `src/services/agentic_models.py`
2. Implement in service file
3. Register in `src/tools.py`:

```python
@mcp.tool()
async def your_tool(ctx: Context, param: str) -> str:
    """Tool description."""
    service = get_service()
    return await service.method(ctx, param)
```

### Add Pydantic AI Agent
```python
# 1. Define output model
class Output(BaseModel):
    field: str = Field(description="...")

# 2. Create agent in service __init__
self.agent = Agent(model=self.model, output_type=Output)

# 3. Use agent
result = await self.agent.run(prompt)
return result.output
```

## Debugging

### Logs
```bash
# MCP server
docker-compose logs -f mcp

# Claude Desktop (macOS)
tail -f ~/Library/Logs/Claude/mcp*.log

# Claude Desktop (Linux)
tail -f ~/.config/Claude/logs/mcp*.log
```

### Common Issues

**Module not found**:
```bash
uv sync --reinstall
```

**Docker build fails**:
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

**Import errors**:
```bash
uv run pytest tests/test_imports.py -v
```

## Deployment

**Manual deployment only** (no auto-deploy):

1. Test locally: `uv run pytest`
2. Commit and push
3. On server:
   ```bash
   git pull origin main
   docker-compose build --no-cache mcp
   docker-compose up -d
   docker-compose logs -f mcp | grep ERROR
   ```

## Resources

- [Project Roadmap](docs/PROJECT_ROADMAP.md)
- [Agentic Search Architecture](docs/AGENTIC_SEARCH_ARCHITECTURE.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [FastMCP Docs](https://github.com/jlowin/fastmcp)
- [uv Docs](https://github.com/astral-sh/uv)
