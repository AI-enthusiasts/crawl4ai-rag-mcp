# Quick Reference Guide

**For**: Developers working on crawl4ai-rag-mcp  
**Updated**: 2025-10-23

---

## Project Status

### ✅ Completed
- Phases 1-7 refactoring (modular structure)
- Browser leak fixed (main.py: 419→113 lines)
- OAuth2 & middleware extracted
- Core infrastructure established

### ❌ Critical Issues
- **5 files >1000 lines** (target: <400)
- **Test coverage 20%** (target: 80%)
- **172 broad exceptions** (target: <20)
- **18 skipped tests** (target: 0)

---

## File Size Violations

| File | Lines | Action Required |
|------|-------|-----------------|
| `knowledge_graph/parse_repo_into_neo4j.py` | 2050 | Split into 7 modules |
| `tools.py` | 1689 | Split into 7 tool groups |
| `knowledge_graph/knowledge_graph_validator.py` | 1256 | Split into 4 validators |
| `database/qdrant_adapter.py` | 1168 | Split into 4 operation modules |
| `knowledge_graph/enhanced_validation.py` | 1020 | Split into 3 validators |

**See**: `docs/PROJECT_CLEANUP_PLAN.md` Phase 1

---

## Test Coverage by Module

```
database/          ████████████░░░░░░░░ 60%  ⚠️
config/            ██████░░░░░░░░░░░░░░ 30%  ⚠️
utils/             ████░░░░░░░░░░░░░░░░ 20%  ❌
core/              ███░░░░░░░░░░░░░░░░░ 15%  ❌
tools.py           ██░░░░░░░░░░░░░░░░░░ 10%  🔥
services/          █░░░░░░░░░░░░░░░░░░░  5%  🔥
knowledge_graph/   █░░░░░░░░░░░░░░░░░░░  5%  🔥
```

**See**: `docs/PROJECT_CLEANUP_PLAN.md` Phase 2

---

## Development Workflow

### Before Starting Work

```bash
# 1. Check current state
pytest --cov=src --cov-report=term-missing

# 2. Find large files
find src -name "*.py" -exec wc -l {} + | awk '$1 > 400 {print}'

# 3. Check for issues
grep -r "except Exception" src/ --include="*.py" | wc -l
```

### Writing Code

**Rules**:
1. **File size**: Keep <400 lines
2. **Tests first**: Write tests before implementation
3. **No mocking**: Use real services (Neo4j, Qdrant)
4. **Coverage**: Achieve 80%+ per module
5. **Exceptions**: Use specific exceptions, not `except Exception`
6. **Logging**: Include context in all logs
7. **Commits**: Frequent, atomic commits

**Example**:
```python
# ❌ BAD
try:
    result = await operation()
except Exception as e:
    logger.error(f"Error: {e}")

# ✅ GOOD
try:
    result = await operation()
except (ValueError, KeyError) as e:
    logger.error(
        "Validation failed",
        extra={"operation": "crawl", "url": url},
        exc_info=True
    )
    raise ValidationError(f"Invalid input: {e}") from e
```

### Testing

```bash
# Run tests with coverage
pytest tests/services/test_crawling.py --cov=src/services/crawling --cov-report=term-missing

# Run specific test
pytest tests/services/test_crawling.py::TestCrawlMarkdownFile::test_successful_crawl -v

# Run with real services
docker-compose up -d neo4j qdrant
pytest tests/integration/ -v
```

### Committing

```bash
# 1. Run tests
pytest --cov=src --cov-fail-under=80

# 2. Check file sizes
python scripts/check_file_size.py --max-lines 400

# 3. Stage changes
git add <files>

# 4. Commit with conventional format
git commit -m "refactor(kg): split parse_repo_into_neo4j into modules

- Extract Neo4jCodeAnalyzer to analyzers/base.py
- Move Python analysis to analyzers/python.py
- Extract Neo4j operations to neo4j/writer.py
- Add tests with 85% coverage

Closes #123"

# 5. Push
git push origin <branch>
```

---

## Common Tasks

### Split Large File

```bash
# 1. Create target structure
mkdir -p src/knowledge_graph/analyzers
touch src/knowledge_graph/analyzers/{__init__,base,python,javascript,go}.py

# 2. Extract classes/functions
# Move code to new files, update imports

# 3. Write tests for each new module
pytest tests/knowledge_graph/analyzers/ --cov=src/knowledge_graph/analyzers --cov-report=term-missing

# 4. Verify no regressions
pytest tests/ -v

# 5. Commit
git commit -m "refactor(kg): split parse_repo_into_neo4j into analyzers"
```

### Add Test Coverage

```bash
# 1. Identify untested code
pytest --cov=src/services --cov-report=term-missing

# 2. Write tests
# tests/services/test_crawling.py

# 3. Run tests
pytest tests/services/test_crawling.py --cov=src/services/crawling --cov-report=term-missing

# 4. Verify 80%+ coverage
# Coverage should show >80%

# 5. Commit
git commit -m "test(services): add crawling service tests

- Test markdown file crawling
- Test sitemap crawling
- Test recursive crawling
- Achieve 85% coverage"
```

### Fix Broad Exception

```bash
# 1. Find broad exceptions
grep -n "except Exception" src/services/crawling.py

# 2. Replace with specific exceptions
# Change except Exception to except (ValueError, KeyError)

# 3. Add custom exceptions if needed
# core/exceptions.py

# 4. Update tests
pytest tests/services/test_crawling.py -v

# 5. Commit
git commit -m "refactor(services): use specific exceptions in crawling

- Replace broad Exception with ValueError, KeyError
- Add CrawlError custom exception
- Update error logging with context"
```

---

## Quality Gates

### Pre-commit Checks

```bash
# File size
find src -name "*.py" -exec wc -l {} + | awk '$1 > 400 {print; exit 1}'

# Test coverage
pytest --cov=src --cov-fail-under=80

# Linting
pylint src/ --fail-under=8.0

# Type checking
mypy src/ --strict
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
- name: Test Coverage
  run: pytest --cov=src --cov-fail-under=80
  
- name: File Size Check
  run: python scripts/check_file_size.py --max-lines 400
  
- name: Lint
  run: pylint src/ --fail-under=8.0
```

---

## Documentation

### Key Documents

| Document | Purpose |
|----------|---------|
| `PROJECT_CLEANUP_PLAN.md` | Detailed improvement plan (4 weeks) |
| `REFACTORING_GUIDE.md` | Completed refactoring history |
| `ARCHITECTURE.md` | System architecture & guidelines |
| `QA/UNIT_TESTING_PLAN.md` | Testing strategy & patterns |

### When to Update Docs

- **After refactoring**: Update `REFACTORING_GUIDE.md`
- **New architecture**: Update `ARCHITECTURE.md`
- **New patterns**: Update `UNIT_TESTING_PLAN.md`
- **Completed phase**: Update `PROJECT_CLEANUP_PLAN.md`

---

## Getting Help

### Check Logs

```bash
# Application logs
docker logs crawl4ai-mcp -f

# Test logs
pytest tests/ -v --log-cli-level=DEBUG

# Coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Debug Issues

```bash
# 1. Verify services running
docker-compose ps

# 2. Check database connections
curl http://localhost:6333/dashboard  # Qdrant
curl http://localhost:7474            # Neo4j

# 3. Run specific test with debug
pytest tests/services/test_crawling.py::test_name -vv --log-cli-level=DEBUG

# 4. Check for import errors
python -c "from src.services.crawling import crawl_markdown_file"
```

---

## Metrics Tracking

### Weekly Review

```bash
# Test coverage trend
pytest --cov=src --cov-report=term | grep TOTAL

# File size violations
find src -name "*.py" -exec wc -l {} + | awk '$1 > 400' | wc -l

# Broad exceptions
grep -r "except Exception" src/ --include="*.py" | wc -l

# Skipped tests
pytest --collect-only -m skip | grep "test_" | wc -l
```

### Progress Dashboard

```bash
# Generate metrics
python scripts/generate_metrics.py

# View dashboard
open metrics/dashboard.html
```

---

## Quick Links

- **Cleanup Plan**: `docs/PROJECT_CLEANUP_PLAN.md`
- **Architecture**: `ARCHITECTURE.md`
- **Testing Guide**: `docs/QA/UNIT_TESTING_PLAN.md`
- **Refactoring History**: `docs/REFACTORING_GUIDE.md`
- **CI/CD**: `.github/workflows/`
