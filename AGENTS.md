# Crawl4AI RAG MCP - Developer Guide

**Stack**: Docker + uv + Pydantic AI + FastMCP + Qdrant + Neo4j

---

## ⚠️ **КРИТИЧЕСКИ ВАЖНО: ЗАПРЕТ АВТОМАТИЧЕСКИХ ПРАВОК** ⚠️

**КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** использовать любые виды автоматических замен/правок:

### ЗАПРЕЩЕНО:
- ❌ `sed` команды
- ❌ `awk` команды
- ❌ `perl -pi -e` или любые in-place редакторы
- ❌ Regex замены через bash
- ❌ Python скрипты для массовых замен
- ❌ Task/Agent tool с инструкциями делать автозамены
- ❌ `ruff --fix` для **unsafe fixes** (только [*] marked safe fixes разрешены)
- ❌ Любые другие способы автоматической модификации кода

### РАЗРЕШЕНО ТОЛЬКО:
- ✅ **Edit tool** - ручное редактирование каждого файла
- ✅ **Write tool** - для создания новых файлов
- ✅ **ruff --fix** - ТОЛЬКО для safe fixes помеченных [*] (не --unsafe-fixes)

### ПОЧЕМУ ЭТО КРИТИЧЕСКИ ВАЖНО:

1. **СОЗДАНИЕ СКРЫТЫХ ОШИБОК**: Автозамены создают миллион скрытых багов из-за:
   - Неправильного понимания контекста
   - Неучета edge cases (форматирование, многострочность, escape sequences)
   - Порчи синтаксиса (неправильные кавычки, скобки, отступы)

2. **ПОТЕРЯ ПРОГРЕССА**: При обнаружении ошибок приходится делать `git checkout`, что:
   - Откатывает ВСЕ ручные правки
   - Уничтожает часы работы
   - Требует переделывать все заново

3. **ТРАТА ДЕНЕГ И ТОКЕНОВ**: Каждый откат означает:
   - Повторное чтение тысяч строк кода
   - Повторные правки
   - Множественные git операции
   - Потраченные API токены OpenAI/Claude

4. **НЕВОЗМОЖНОСТЬ ВЕРИФИКАЦИИ**: После массовой автозамены:
   - Невозможно проверить все изменения вручную
   - Тесты могут не покрывать все случаи
   - Баги проявляются в production

### ПРАВИЛЬНЫЙ ПОДХОД:

```python
# ❌ НЕПРАВИЛЬНО - автозамена через sed/awk/regex
subprocess.run(["sed", "-i", "s/pattern/replacement/g", file])

# ✅ ПРАВИЛЬНО - Edit tool для каждого файла
Edit(
    file_path="/path/to/file.py",
    old_string="logger.info(f'Processing {count} items')",
    new_string='logger.info("Processing %s items", count)',
)
```

**ЕСЛИ НУЖНО ИСПРАВИТЬ 1000 ОШИБОК - ИСПОЛЬЗУЙ Edit tool 1000 РАЗ, ПО ОДНОЙ ЗА РАЗ.**

---

## ⚠️ **КРИТИЧЕСКИ ВАЖНО: ЗАПРЕТ КОСТЫЛЕЙ И WORKAROUNDS** ⚠️

**КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** использовать временные решения/костыли/workarounds:

### ЗАПРЕЩЕНО:
- ❌ **`# noqa` комментарии** - подавляют линтер вместо исправления проблемы
- ❌ **`# type: ignore`** - скрывают проблемы типизации
- ❌ **Импорты в конце файла** - нарушают PEP 8, ломаются при ruff auto-fix
- ❌ **"Временные" решения** - становятся постоянными и ломают проект
- ❌ **Комментарии "TODO: fix later"** - никто не фиксит, баги накапливаются
- ❌ **Любые workarounds** которые можно правильно исправить

### РАЗРЕШЕНО ТОЛЬКО:
- ✅ **Правильная архитектура** - рефакторинг, новые модули, паттерны
- ✅ **Исправление root cause** - устранение причины, не симптома
- ✅ **Clean code** - следование PEP 8, best practices

### ПОЧЕМУ ЭТО КРИТИЧЕСКИ ВАЖНО:

1. **КОСТЫЛИ ЛОМАЮТ ПРОЕКТ**: Временные решения создают критические баги:
   - Линтер/formatter ломает "хитрые" workarounds при auto-fix
   - Импорты переставляются, circular imports возвращаются
   - `# noqa` скрывает реальные проблемы, которые проявятся в production

2. **ПОТЕРЯ ДЕНЕГ**: Каждый костыль означает:
   - Повторная оплата за переделывание
   - Debugging сломанного кода
   - Откаты и потеря прогресса
   - Трата токенов на исправление исправлений

3. **ТЕХНИЧЕСКИЙ ДОЛГ**: Workarounds накапливаются:
   - Код становится unmaintainable
   - Новые разработчики не понимают логику
   - Рефакторинг становится невозможным

4. **НЕСТАБИЛЬНОСТЬ**: Проект даже не стартует:
   - Circular imports из-за неправильного порядка импортов
   - Runtime errors из-за подавленных warnings
   - Непредсказуемое поведение

### ПРАВИЛЬНЫЙ ПОДХОД:

```python
# ❌ НЕПРАВИЛЬНО - костыль с noqa
# Import at end to avoid circular import
from .mcp_wrapper import agentic_search_impl  # noqa: E402

# ✅ ПРАВИЛЬНО - архитектурное решение
# Create factory.py to break circular dependency
# src/services/agentic_search/factory.py
def get_agentic_search_service() -> AgenticSearchService:
    # Singleton factory
    ...

# src/services/agentic_search/__init__.py
from .factory import get_agentic_search_service  # Clean import
from .mcp_wrapper import agentic_search_impl      # No circular dependency
```

**ЕСЛИ ЛИНТЕР РУГАЕТСЯ - ИСПРАВЬ ПРОБЛЕМУ, НЕ ПОДАВЛЯЙ WARNING.**

**ЕСЛИ ЕСТЬ CIRCULAR IMPORT - РЕФАКТОРИ АРХИТЕКТУРУ, НЕ ДВИГАЙ ИМПОРТЫ В КОНЕЦ.**

**ЕСЛИ ЕСТЬ TYPE ERROR - ИСПРАВЬ ТИПЫ, НЕ ДОБАВЛЯЙ `# type: ignore`.**

---

## Technology Stack

- **Package Manager**: uv (NOT pip/poetry/conda)
- **MCP Framework**: FastMCP
- **LLM Framework**: Pydantic AI (NOT OpenAI SDK directly)
- **Vector DB**: Qdrant (primary), Supabase (legacy)
- **Knowledge Graph**: Neo4j (optional)
- **Web Crawler**: Crawl4AI
- **Search Engine**: SearXNG (self-hosted)

## Repository Structure

```
src/
├── services/          # Business logic (agentic_search, crawling, search)
├── database/          # Vector DB adapters (qdrant, supabase)
├── knowledge_graph/   # Neo4j code analysis
├── config/            # Settings management
├── core/              # Core utilities (context, exceptions, logging)
├── utils/             # Helpers (embeddings, validation, text_processing)
├── tools.py           # MCP tool registration
└── main.py            # FastMCP server entry

tests/                 # Test suite
├── unit/              # Unit tests
├── integration/       # Integration tests
├── fixtures/          # Test fixtures
└── test_imports.py    # Import verification (pre-commit)

docs/                  # Documentation
├── AGENTIC_SEARCH_ARCHITECTURE.md
├── PROJECT_ROADMAP.md
└── CONFIGURATION.md
```

## Development Workflow

### Package Management with uv

**CRITICAL**: ALL Python operations MUST use `uv`. Never use `python`, `pip`, `poetry`, or `conda` directly.

```bash
# Dependencies
uv sync                          # Install/sync
uv add package-name              # Add production dependency
uv add --dev package-name        # Add dev dependency
uv lock --upgrade                # Update lockfile

# Execution
uv run pytest tests/             # Run tests
uv run python src/main.py        # Run scripts
uv run ruff check src/           # Lint
uv run mypy src/                 # Type check
```

### Docker Development

```bash
# Development (hot reload, debug logs)
make dev
docker-compose -f docker-compose.dev.yml up

# Production
make prod
docker-compose -f docker-compose.prod.yml up -d

# Logs
docker-compose logs -f mcp
docker-compose logs -f crawl4ai

# Rebuild after dependency changes
docker-compose build --no-cache mcp
```

### Testing

**CRITICAL: NO MOCKS POLICY**

All tests MUST use real services. Mocking is FORBIDDEN.

**Test Execution Model**:
- Tests run LOCALLY via `uv run pytest`
- Services must be accessible on localhost (Qdrant on :6333, SearXNG on :8080)

**Requirements**:
- Real Qdrant (localhost:6333)
- Real SearXNG (localhost:8080 or skip test)
- Real OpenAI API (or skip test if no key)
- E2E tests MUST pre-populate Qdrant with test data
- Use `pytest.skip()` if service unavailable - NEVER mock

**Setup Local Services**:
```bash
# Qdrant (binary, :6333)
cd ~ && curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
nohup ~/qdrant > ~/qdrant.log 2>&1 &

# SearXNG - Docker (runs on :8080)
docker run -d -p 8080:8080 --name searxng searxng/searxng

# SearXNG - Local install (runs on :8888)
cd ~ && git clone https://github.com/searxng/searxng.git
cd ~/searxng && ./manage pyenv.install
nohup ./manage webapp.run > ~/searxng.log 2>&1 &
```

**Running Tests**:
```bash
# All tests (requires services)
uv run pytest

# Integration tests with real services
uv run pytest tests/integration/ -v

# E2E with real Qdrant data
uv run pytest tests/integration/test_agentic_search_e2e.py -v

# With coverage
uv run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Linting
uv run ruff check src/
uv run ruff check src/ --fix

# Type checking
uv run mypy src/

# Formatting
uv run ruff format src/
```

### Type Stubs Generation

For libraries without built-in types (e.g. `crawl4ai`), stubs are needed in `stubs/`.

```bash
# Generate (saves to stubs/ via pyproject.toml [tool.pyright] stubPath)
uv run pyright --createstub crawl4ai

# Check for errors in generated stubs
uv run mypy src/ --no-error-summary 2>&1 | grep "stubs/"

# Validate stubs match runtime API
uv run python -m mypy.stubtest crawl4ai
```

**Generated stubs require manual fixes:**

```python
# Add return types
def foo(self) -> ReturnType: ...

# Type **kwargs
def foo(self, **kwargs: Any) -> None: ...

# Add class attributes (generator only creates __init__ params)
class Config:
    verbose: bool
    def __init__(self, verbose: bool = ...) -> None: ...
```

### Pre-commit Hooks

Install once to auto-run tests before commits:

```bash
scripts/install-hooks.sh
```

Hook runs:
1. Import verification (91 module tests, ~0.3s)
2. Ruff linting (warnings only)

## Architecture Patterns

### Pydantic AI Integration

**Use Pydantic AI, NOT OpenAI SDK directly**:

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

# Create model wrapper
model = OpenAIModel(
    model_name=settings.model_choice,
    api_key=settings.openai_api_key,
)

# Configure model settings
model_settings = ModelSettings(
    temperature=0.7,
    timeout=60,
)

# Create agent with structured output
agent = Agent(
    model=model,
    output_type=YourPydanticModel,
    output_retries=3,
    model_settings=model_settings,
)

# Run agent
try:
    result = await agent.run(prompt)
    return result.output  # Fully typed
except UnexpectedModelBehavior:
    logger.exception("Agent failed after retries")
    raise
```

### Singleton Pattern for Agents

**Reuse agent instances for connection pooling**:

```python
_service_instance: AgenticSearchService | None = None

def get_service() -> AgenticSearchService:
    """Get singleton instance with cached Pydantic AI agents."""
    global _service_instance
    if _service_instance is None:
        _service_instance = AgenticSearchService()
    return _service_instance
```

### Async Context Management

**FastMCP handles lifespan automatically**:

```python
from fastmcp import FastMCP

@asynccontextmanager
async def lifespan(app: FastMCP):
    """Lifespan manager for resource initialization/cleanup."""
    # Startup
    context = await initialize_resources()
    yield context
    # Cleanup
    await cleanup_resources(context)

# FastMCP handles lifespan internally
mcp = FastMCP("Server Name", lifespan=lifespan)
```

## Configuration

### Environment Variables

Core settings in `src/config/settings.py`:

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
MODEL_CHOICE=gpt-4o-mini

# Agentic Search
AGENTIC_SEARCH_ENABLED=true
AGENTIC_SEARCH_COMPLETENESS_THRESHOLD=0.85
AGENTIC_SEARCH_MAX_ITERATIONS=5
AGENTIC_SEARCH_LLM_TEMPERATURE=0.3

# Vector Database
USE_QDRANT=true
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=crawled_pages

# Services
SEARXNG_URL=http://searxng:8080
CRAWL4AI_URL=http://crawl4ai:8000

# Optional Features
USE_KNOWLEDGE_GRAPH=true      # Neo4j code analysis
USE_AGENTIC_RAG=true          # Advanced RAG features
USE_OAUTH2=false              # OAuth provider for Claude Web
```

## Common Tasks

### Adding a New MCP Tool

1. Define Pydantic model in `src/services/agentic_models.py`
2. Implement logic in relevant service file
3. Register tool in `src/tools.py`:

```python
@mcp.tool()
async def your_tool(
    ctx: Context,
    param: str,
) -> str:
    """Tool description for AI agents."""
    service = get_service()
    result = await service.your_method(ctx, param)
    return result.model_dump_json()
```

### Adding a New Pydantic AI Agent

1. Define output model:

```python
class YourOutput(BaseModel):
    field: str = Field(description="Field description")
```

2. Create agent in service `__init__`:

```python
self.your_agent = Agent(
    model=self.openai_model,
    output_type=YourOutput,
    output_retries=MAX_RETRIES_DEFAULT,
    model_settings=self.base_model_settings,
)
```

3. Use agent:

```python
async def your_method(self, prompt: str) -> YourOutput:
    try:
        result = await self.your_agent.run(prompt)
        return result.output
    except UnexpectedModelBehavior:
        logger.exception("Agent failed")
        raise
```

### Debugging

**MCP Server Logs**:
```bash
docker-compose logs -f mcp

# Or in container
docker exec -it crawl4aimcp-mcp-1 bash
tail -f /app/logs/mcp.log
```

**Claude Desktop Logs**:
```bash
# macOS
tail -f ~/Library/Logs/Claude/mcp*.log

# Linux
tail -f ~/.config/Claude/logs/mcp*.log
```

**Test MCP Server Directly**:
```bash
docker exec -it crawl4aimcp-mcp-1 uv run python src/main.py
```

## Git Workflow

### Branch Strategy

- `main` - production-ready code
- `feat/*` - new features
- `fix/*` - bug fixes
- `refactor/*` - code improvements

### Commit Messages

Follow conventional commits:

```bash
git commit -m "feat: add pydantic-ai migration"
git commit -m "fix: resolve memory leak in crawler"
git commit -m "refactor: improve type safety"
git commit -m "docs: update architecture guide"
git commit -m "test: add integration tests"
```

**Commit Size Rules**:

- **One commit = one logical change** (feature, fix, refactor)
- **Maximum ~500 lines changed** per commit (guideline, not strict)
- **Group related changes together** (don't split one feature into 10 tiny commits)
- **Squash work-in-progress commits** before pushing

**Examples of GOOD commits**:
- `feat: agentic search with recursive crawling` - 500 lines, complete feature
- `fix: security improvements` - 250 lines, related security fixes
- `refactor: migrate to pydantic-ai` - 300 lines, complete migration

**Examples of BAD commits**:
- `fix: add import` - 1 line (too small, should be squashed)
- `refactor: massive cleanup` - 5000 lines (too big, split by feature)
- `wip: testing stuff` - don't push WIP commits

**How to squash commits**:

```bash
# Squash last 5 commits into logical groups
git reset --soft HEAD~5
git add <files for feature 1>
git commit -m "feat: feature description"
git add <files for feature 2>
git commit -m "fix: bug description"
# ... repeat for other changes
```

## Troubleshooting

### Module not found Errors

```bash
# Rebuild uv lockfile
uv lock --upgrade

# Reinstall all dependencies
uv sync --reinstall
```

### Docker Build Issues

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Pydantic AI Issues

```bash
# Check version
uv run python -c "import pydantic_ai; print(pydantic_ai.__version__)"

# Should be >= 0.0.14
uv add "pydantic-ai>=0.0.14"
```

### Service Connection Issues

```bash
# Check service health
docker-compose ps

# Test internal networking
docker exec crawl4aimcp-mcp-1 curl http://searxng:8080
docker exec crawl4aimcp-mcp-1 curl http://qdrant:6333/collections
```

### Port Conflicts with docker-compose.test.yml

`docker-compose.test.yml` uses standard ports that may conflict with other services:
- **6379** (Valkey/Redis) - often used by `joern-redis` or other Redis instances
- **6333** (Qdrant) - may conflict with standalone Qdrant
- **8080** (SearXNG) - common port for web services

**Solution**: Stop conflicting services or use alternative ports:

```bash
# Check what's using the port
sudo lsof -i :6379
docker ps --format "{{.Names}}\t{{.Ports}}" | grep 6379

# Option 1: Stop conflicting container
docker stop joern-redis

# Option 2: Start only needed services (skip valkey if Redis already running)
docker compose -f docker-compose.test.yml up -d qdrant-test searxng-test

# Option 3: Use existing Redis (tests don't require Valkey specifically)
# Just ensure Qdrant and SearXNG are running
```

**Note**: Use `docker compose` (v2), not `docker-compose` (v1) - v1 may not be installed.

## File Management Rules

**IMPORTANT**:

- Only create files inside the repository
- Follow project structure:
  - Code: `src/`
  - Tests: `tests/`
  - Docs: `docs/`
  - Scripts: `scripts/`
- Never create random files in `~` (home directory)
- This AGENTS.md is the ONLY exception (instructions file)

## Deployment

**This project does NOT auto-deploy**. Manual deployment required:

1. Make and test changes locally
2. Commit and push to repository
3. Pull changes on server: `git pull origin main`
4. Rebuild: `docker-compose build --no-cache mcp`
5. Restart: `docker-compose up -d`
6. Verify: `docker-compose logs -f mcp`

**Production checklist**:
- [ ] Environment variables configured
- [ ] Services healthy: `docker-compose ps`
- [ ] Logs clean: `docker-compose logs mcp | grep ERROR`
- [ ] MCP tools responding: Test via Claude Desktop
- [ ] Qdrant accessible: Check dashboard at port 6333
- [ ] SearXNG responding: Check at port 8080 (internal)

## Resources

- [Project Roadmap](docs/PROJECT_ROADMAP.md)
- [Agentic Search Architecture](docs/AGENTIC_SEARCH_ARCHITECTURE.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Main README](README.md)
- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [FastMCP Docs](https://github.com/jlowin/fastmcp)
- [uv Docs](https://github.com/astral-sh/uv)
