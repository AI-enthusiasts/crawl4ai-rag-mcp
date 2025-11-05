# 🗺️ Project Roadmap

**Status**: 🎯 Active - Primary Development Plan  
**Last Updated**: 2025-11-05  
**Purpose**: Single source of truth for project development priorities

> **This is the ONLY official roadmap for the project.**  
> All development work should follow the priorities outlined here.

---

## 🎯 Development Priorities

### Priority 1: 🔥 Agentic Search (HIGHEST PRIORITY)

**Status**: Not Started  
**Timeline**: 3 weeks  
**Effort**: High  
**Impact**: 🚀 Revolutionary

**Why This is #1:**
- **Unique Value Proposition**: No other MCP server offers intelligent, iterative search
- **Cost Efficiency**: 50-70% reduction in crawling costs through selective crawling
- **Quality**: LLM-driven decisions ensure high-quality, complete answers
- **Market Position**: Establishes project as the most advanced RAG-MCP solution
- **User Experience**: Comprehensive answers without manual iteration

**What It Does:**
Intelligent search system that:
1. Checks local knowledge (Qdrant) first
2. Searches web (SearXNG) only if needed
3. Uses LLM to rank URLs by relevance
4. Crawls only promising URLs (selective crawling)
5. Iteratively refines queries to fill knowledge gaps
6. Returns comprehensive, high-quality answers

**Documentation**: [AGENTIC_SEARCH_ARCHITECTURE.md](AGENTIC_SEARCH_ARCHITECTURE.md)

**Implementation Plan**:
- **Week 1**: Core pipeline (4 stages: local check, web search, selective crawl, refinement)
- **Week 2**: Metadata enhancement (research Crawl4AI capabilities for smart queries)
- **Week 3**: Optimization, monitoring, and production readiness

**Success Metrics**:
- ✅ 80%+ answer completeness in 90% of queries
- ✅ <30% of search results crawled (vs 100% baseline)
- ✅ 50-70% cost reduction vs exhaustive crawling
- ✅ <60 seconds per iteration (average)

**Dependencies**: None - uses existing infrastructure

**Blockers**: None

---

### Priority 2: Code Quality & Refactoring

**Status**: Planned (after Agentic Search)  
**Timeline**: 4-6 weeks  
**Effort**: High  
**Impact**: Maintainability, testability

**Note**: This work is postponed until Agentic Search is complete.

---

## 📊 Current State Assessment

### Code Quality Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Test Coverage | 20% | 80% | ❌ Critical |
| Largest File | 2050 lines | 400 lines | ❌ Critical |
| Files >1000 lines | 5 files | 0 files | ❌ Critical |
| Skipped Tests | 18 tests | 0 tests | ⚠️ Medium |
| Broad Exceptions | 172 cases | <20 cases | ❌ High |
| Root Test Files | 5 files | 0 files | ⚠️ Medium |

### Module Status

| Module | Lines | Files | Coverage | Priority |
|--------|-------|-------|----------|----------|
| `knowledge_graph/parse_repo_into_neo4j.py` | 2050 | 1 | 5% | 🔥 Critical |
| `tools.py` | 1689 | 1 | 10% | 🔥 Critical |
| `knowledge_graph/knowledge_graph_validator.py` | 1256 | 1 | 5% | 🔥 Critical |
| `database/qdrant_adapter.py` | 1168 | 1 | 60% | ⚠️ High |
| `knowledge_graph/enhanced_validation.py` | 1020 | 1 | 5% | 🔥 Critical |
| `services/*` | 524+ | 4 | 5% | 🔥 Critical |

---

## Phase 1: File Size Refactoring (Week 1)

**Goal**: Break down monolithic files into manageable modules (<400 lines each)

### 1.1 Split `parse_repo_into_neo4j.py` (2050 → 300 lines)

**Priority**: 🔥 Critical  
**Effort**: 8 hours  
**Impact**: High maintainability, testability

**Target Structure**:
```
knowledge_graph/
├── parse_repo_into_neo4j.py (300 lines) - Main orchestrator
├── analyzers/
│   ├── base.py (200 lines) - Base analyzer interface
│   ├── python.py (400 lines) - Python AST analysis
│   ├── javascript.py (400 lines) - JS/TS analysis
│   └── go.py (400 lines) - Go analysis
├── neo4j/
│   ├── writer.py (300 lines) - Neo4j write operations
│   └── queries.py (200 lines) - Cypher query builders
└── git/
    └── operations.py (300 lines) - Git operations
```

**Implementation Steps**:
1. Extract `Neo4jCodeAnalyzer` → `analyzers/base.py`
2. Move language-specific logic → `analyzers/{python,javascript,go}.py`
3. Extract Neo4j operations → `neo4j/writer.py`
4. Extract Git operations → `git/operations.py`
5. Update imports in dependent files
6. Write unit tests for each new module (80% coverage)
7. Commit: `refactor(kg): split parse_repo_into_neo4j into modular structure`

### 1.2 Split `tools.py` (1689 → 200 lines)

**Priority**: 🔥 Critical  
**Effort**: 6 hours  
**Impact**: Better organization, easier testing

**Target Structure**:
```
tools/
├── __init__.py (50 lines) - Tool registration
├── search.py (200 lines) - search, scrape_urls
├── crawl.py (200 lines) - smart_crawl_url
├── rag.py (200 lines) - perform_rag_query, get_available_sources, search_code_examples
├── knowledge_graph.py (300 lines) - query_knowledge_graph, parse_github_repository
├── validation.py (300 lines) - check_ai_script_hallucinations, smart_code_search
└── repository.py (200 lines) - parse_local_repository, extract_and_index_repository_code
```

**Implementation Steps**:
1. Create `tools/` directory structure
2. Move tool groups to respective files
3. Update `register_tools()` to import from submodules
4. Write integration tests for tool registration
5. Commit: `refactor(tools): split monolithic tools.py into domain modules`

### 1.3 Split `knowledge_graph_validator.py` (1256 → 400 lines)

**Priority**: 🔥 Critical  
**Effort**: 4 hours

**Target Structure**:
```
knowledge_graph/
├── validation/
│   ├── __init__.py
│   ├── import_validator.py (300 lines)
│   ├── method_validator.py (300 lines)
│   ├── class_validator.py (300 lines)
│   └── function_validator.py (300 lines)
```

**Commit**: `refactor(kg): split validator into specialized validators`

### 1.4 Split `qdrant_adapter.py` (1168 → 400 lines)

**Priority**: ⚠️ High  
**Effort**: 4 hours

**Target Structure**:
```
database/
├── qdrant/
│   ├── adapter.py (300 lines) - Main adapter
│   ├── operations.py (300 lines) - CRUD operations
│   ├── search.py (300 lines) - Search operations
│   └── code_examples.py (200 lines) - Code example operations
```

**Commit**: `refactor(db): split qdrant_adapter into operation modules`

### 1.5 Split `enhanced_validation.py` (1020 → 400 lines)

**Priority**: 🔥 Critical  
**Effort**: 3 hours

**Target Structure**:
```
knowledge_graph/
├── validation/
│   ├── enhanced.py (300 lines) - Main orchestrator
│   ├── semantic_validator.py (300 lines) - Semantic validation
│   └── structural_validator.py (300 lines) - Structural validation
```

**Commit**: `refactor(kg): split enhanced_validation into focused modules`

---

## Phase 2: Test Coverage Improvement (Week 2)

**Goal**: Achieve 80%+ test coverage across all modules

### 2.1 Services Module Testing (5% → 80%)

**Priority**: 🔥 Critical  
**Effort**: 12 hours

**Files to Test**:
- `services/crawling.py` (524 lines)
- `services/search.py`
- `services/scraping.py`
- `services/smart_crawl.py`
- `services/validated_search.py` (752 lines)

**Test Strategy**:
```python
# tests/services/test_crawling.py
class TestCrawlMarkdownFile:
    """Test markdown file crawling."""
    
    @pytest.mark.asyncio
    async def test_successful_crawl(self, mock_crawler):
        """Test successful markdown file crawl."""
        # Real test with actual AsyncWebCrawler mock
        result = await crawl_markdown_file(mock_crawler, "https://example.com/file.txt")
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/file.txt"
        assert "markdown" in result[0]
    
    @pytest.mark.asyncio
    async def test_failed_crawl(self, mock_crawler):
        """Test failed crawl handling."""
        mock_crawler.arun.return_value.success = False
        result = await crawl_markdown_file(mock_crawler, "https://example.com/fail.txt")
        assert result == []
```

**Coverage Target**: 80% per file  
**Commits**: One per service file tested

### 2.2 Knowledge Graph Testing (5% → 80%)

**Priority**: 🔥 Critical  
**Effort**: 16 hours

**Files to Test**:
- `knowledge_graph/parse_repo_into_neo4j.py` (after split)
- `knowledge_graph/knowledge_graph_validator.py` (after split)
- `knowledge_graph/enhanced_validation.py` (after split)
- `knowledge_graph/code_extractor.py` (961 lines)
- `knowledge_graph/git_manager.py` (714 lines)

**Test Strategy**:
- Use real Neo4j test container (not mocked)
- Test with actual Git repositories
- Validate AST parsing with real Python files
- Test cross-language analysis

**Commits**: One per major component

### 2.3 Core & Utils Testing (15-20% → 80%)

**Priority**: ⚠️ High  
**Effort**: 8 hours

**Files to Test**:
- `core/decorators.py`
- `core/stdout_utils.py`
- `core/context.py`
- `utils/embeddings.py` (617 lines)
- `utils/integration_helpers.py` (538 lines)
- `utils/text_processing.py`
- `utils/url_helpers.py`

**Test Strategy**:
- Unit tests for pure functions
- Integration tests for context management
- Performance tests for embeddings

**Commits**: One per module

### 2.4 Tools Integration Testing (10% → 60%)

**Priority**: 🔥 Critical  
**Effort**: 10 hours

**Approach**:
- Test each tool with real MCP protocol
- Use actual database connections (not mocked)
- Validate tool responses match schema
- Test error handling and edge cases

**Test Structure**:
```python
# tests/tools/test_search_tool.py
class TestSearchTool:
    """Integration tests for search tool."""
    
    @pytest.mark.asyncio
    async def test_search_with_real_searxng(self, mcp_context, searxng_container):
        """Test search with actual SearXNG instance."""
        result = await search(
            ctx=mcp_context,
            query="Python testing",
            num_results=3
        )
        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) <= 3
```

**Commits**: One per tool group

---

## Phase 3: Code Quality Improvements (Week 3)

### 3.1 Exception Handling Refinement

**Priority**: ⚠️ High  
**Effort**: 6 hours

**Current**: 172 broad `except Exception` handlers  
**Target**: <20 broad handlers, rest use specific exceptions

**Strategy**:
```python
# Bad (current)
try:
    result = await operation()
except Exception as e:
    logger.error(f"Error: {e}")

# Good (target)
try:
    result = await operation()
except (ValueError, KeyError) as e:
    logger.error(f"Validation error: {e}", exc_info=True)
    raise ValidationError(f"Invalid input: {e}") from e
except ConnectionError as e:
    logger.error(f"Database connection failed: {e}", exc_info=True)
    raise DatabaseError(f"Cannot connect: {e}") from e
```

**Implementation**:
1. Define custom exception hierarchy in `core/exceptions.py`
2. Replace broad exceptions module by module
3. Add proper error context and logging
4. Update tests to verify exception types

**Commits**: One per module refactored

### 3.2 Logging Standardization

**Priority**: ⚠️ Medium  
**Effort**: 4 hours

**Current Issues**:
- Inconsistent log levels
- Missing context in error logs
- No structured logging

**Target**:
```python
# Structured logging with context
logger.info(
    "Crawling URL",
    extra={
        "url": url,
        "max_depth": max_depth,
        "operation": "smart_crawl"
    }
)

logger.error(
    "Crawl failed",
    extra={
        "url": url,
        "error_type": type(e).__name__,
        "operation": "smart_crawl"
    },
    exc_info=True
)
```

**Commits**: `refactor(logging): standardize logging across modules`

### 3.3 Move Root Test Files

**Priority**: ⚠️ Medium  
**Effort**: 30 minutes

**Files to Move**:
```bash
mv test_batch_simple.py tests/integration/
mv test_hallucination_script.py tests/knowledge_graph/
mv test_mixed_hallucinations.py tests/knowledge_graph/
mv test_neo4j_batching.py tests/knowledge_graph/
mv verify_neo4j_fix.py tests/knowledge_graph/
```

**Commit**: `chore(tests): move root test files to proper directories`

### 3.4 Fix Skipped Tests

**Priority**: ⚠️ Medium  
**Effort**: 4 hours

**Current**: 18 skipped tests  
**Target**: 0 skipped tests

**Strategy**:
1. Review each `@pytest.mark.skip` reason
2. Fix underlying issues or remove obsolete tests
3. Re-enable tests one by one
4. Verify all pass in CI

**Commit**: `test: fix and re-enable skipped tests`

---

## Phase 4: Documentation Cleanup (Week 4)

### 4.1 Remove Obsolete Documentation

**Priority**: ⚠️ Medium  
**Effort**: 2 hours

**Files to Remove** (completed plans):
- `docs/MAIN_PY_REFACTORING_PLAN.md` (Phase 7 completed)
- `docs/PHASE_6_SUMMARY.md` (Phase 6 completed)
- `docs/REFACTORING_GUIDE.md` (merge into ARCHITECTURE.md)

**Files to Archive**:
- `docs/MIGRATION.md` → `docs/archive/`
- `docs/MIGRATION_GUIDE.md` → `docs/archive/`

**Commit**: `docs: remove obsolete refactoring plans`

### 4.2 Consolidate Documentation

**Priority**: ⚠️ Medium  
**Effort**: 3 hours

**Merge Similar Docs**:
- `NEO4J_QDRANT_BRIDGE.md` + `NEO4J_QDRANT_INTEGRATION.md` → `NEO4J_INTEGRATION.md`
- Update `ARCHITECTURE.md` with current state
- Update `README.md` with simplified setup

**Commit**: `docs: consolidate and update documentation`

### 4.3 Create Missing Documentation

**Priority**: ⚠️ Low  
**Effort**: 4 hours

**New Docs Needed**:
- `docs/TESTING_GUIDE.md` - How to write tests
- `docs/CONTRIBUTING.md` - Contribution guidelines
- `docs/API_REFERENCE.md` - Tool API documentation

**Commit**: `docs: add testing guide and contribution guidelines`

---

## Phase 5: CI/CD & Quality Gates (Week 4)

### 5.1 Enforce Quality Gates

**Priority**: ⚠️ High  
**Effort**: 3 hours

**Pre-commit Hooks**:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-coverage
        name: pytest with coverage
        entry: pytest --cov=src --cov-fail-under=80
        language: system
        pass_filenames: false
        always_run: true
      
      - id: file-size-check
        name: check file size
        entry: python scripts/check_file_size.py --max-lines 400
        language: system
        pass_filenames: false
```

**Commit**: `ci: add quality gates for coverage and file size`

### 5.2 Automated Testing

**Priority**: ⚠️ High  
**Effort**: 2 hours

**GitHub Actions**:
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:latest
      qdrant:
        image: qdrant/qdrant:latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**Commit**: `ci: add automated testing workflow`

---

## Implementation Timeline

### Week 1: File Size Refactoring
- **Mon-Tue**: Split `parse_repo_into_neo4j.py` (8h)
- **Wed-Thu**: Split `tools.py` (6h)
- **Fri**: Split `knowledge_graph_validator.py` (4h)

### Week 2: Test Coverage
- **Mon-Tue**: Services testing (12h)
- **Wed-Thu**: Knowledge graph testing (16h)
- **Fri**: Core & utils testing (8h)

### Week 3: Code Quality
- **Mon-Tue**: Exception handling (6h)
- **Wed**: Logging standardization (4h)
- **Thu**: Move test files + fix skipped tests (4.5h)
- **Fri**: Tools integration testing (10h)

### Week 4: Documentation & CI
- **Mon**: Documentation cleanup (5h)
- **Tue**: Create missing docs (4h)
- **Wed**: Quality gates (3h)
- **Thu**: Automated testing (2h)
- **Fri**: Final review and deployment

---

## Success Criteria

### Code Quality
- ✅ All files <400 lines
- ✅ Test coverage >80%
- ✅ 0 skipped tests
- ✅ <20 broad exception handlers
- ✅ 0 test files in project root

### Testing
- ✅ All services have unit tests
- ✅ All tools have integration tests
- ✅ Knowledge graph has real Neo4j tests
- ✅ CI passes on all commits

### Documentation
- ✅ No obsolete documentation
- ✅ Clear testing guide
- ✅ Updated architecture docs
- ✅ API reference complete

### CI/CD
- ✅ Pre-commit hooks enforced
- ✅ Automated testing in CI
- ✅ Coverage reports generated
- ✅ Quality gates prevent regressions

---

## Rollback Strategy

Each phase is independent and can be rolled back:

```bash
# Rollback specific phase
git revert <phase-commit-range>

# Rollback to before cleanup
git reset --hard <pre-cleanup-commit>
```

**Mitigation**:
- Small, atomic commits
- Each commit passes tests
- Feature flags for major changes
- Comprehensive test coverage before refactoring

---

## Monitoring & Metrics

Track progress with:

```bash
# Test coverage
pytest --cov=src --cov-report=term-missing

# File size violations
find src -name "*.py" -exec wc -l {} + | awk '$1 > 400 {print}'

# Broad exceptions
grep -r "except Exception" src/ --include="*.py" | wc -l

# Skipped tests
pytest --collect-only -m skip | grep "test_" | wc -l
```

**Weekly Review**:
- Coverage trend
- File size compliance
- Test pass rate
- Code review feedback

---

## Notes

- **No Mocking**: Use real services (Neo4j, Qdrant) in tests
- **Frequent Commits**: Commit after each logical change
- **Test First**: Write tests before refactoring
- **Document Changes**: Update docs with each phase
- **Review Code**: Self-review before committing
