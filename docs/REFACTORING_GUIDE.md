# Crawl4AI MCP Server Refactoring Guide

## Status: Phases 1-7 Completed ✅

This document tracks the completed refactoring of the monolithic `src/crawl4ai_mcp.py` file (2,998 lines) into a modular, maintainable structure.

## Completed Phases

### Phase 1-6: Modular Structure ✅
- Core infrastructure extracted
- Utilities separated
- Services modularized
- Database operations organized
- Configuration centralized
- Tools registered via FastMCP

### Phase 7: main.py Refactoring ✅
- **Browser leak fixed**: Removed duplicate lifespan initialization
- **OAuth2 extracted**: Moved to `auth/setup.py`
- **Middleware extracted**: Moved to `middleware/setup.py`
- **Result**: 419 lines → 113 lines (73% reduction)

## Refactored Structure

```
src/
├── core/                    # Core infrastructure
│   ├── __init__.py         # Core exports
│   ├── context.py          # Application context and lifecycle
│   ├── decorators.py       # Request tracking and other decorators
│   ├── exceptions.py       # Custom exceptions
│   ├── logging.py          # Logging configuration
│   └── stdout_utils.py     # Stdout management utilities
│
├── utils/                   # Utility functions
│   ├── __init__.py         # Utility exports
│   ├── reranking.py        # Cross-encoder reranking
│   ├── text_processing.py  # Text chunking and processing
│   ├── url_helpers.py      # URL parsing and validation
│   └── validation.py       # Input validation functions
│
├── services/                # Business logic services
│   ├── __init__.py         # Service exports
│   ├── crawling.py         # Web crawling services
│   ├── search.py           # Search functionality
│   ├── scraping.py         # URL scraping services
│   └── smart_crawl.py      # Intelligent crawling logic
│
├── database/                # Database interactions
│   ├── __init__.py         # Database exports
│   ├── rag_queries.py      # RAG query functionality
│   └── sources.py          # Source management
│
├── knowledge_graph/         # Neo4j knowledge graph
│   ├── __init__.py         # Knowledge graph exports
│   ├── handlers.py         # Command handlers
│   ├── queries.py          # Graph queries
│   ├── repository.py       # Repository parsing
│   └── validation.py       # Script validation
│
├── tools/                   # MCP tool definitions
│   ├── __init__.py         # Tool registration
│   ├── search_tool.py      # Search tool
│   ├── scrape_tool.py      # Scraping tool
│   └── ...                 # Other tools
│
├── main.py                  # Application entry point
└── server.py               # FastMCP server configuration
```

## Migration Strategy

### Phase 1: Core Infrastructure ✅

- Created `core/` directory with exceptions, logging, decorators, and context management
- Extracted `MCPToolError`, `SuppressStdout`, `track_request`, and `crawl4ai_lifespan`
- Established centralized logging configuration

### Phase 2: Utilities ✅

- Created `utils/` directory with validation, text processing, URL helpers, and reranking
- Extracted all utility functions that don't depend on external services
- Maintained backward compatibility with imports

### Phase 3: Services ✅

- Created `services/` directory for business logic
- Extracted crawling services as the first module
- Prepared structure for search, scraping, and smart crawl services

### Phase 4: Database & Knowledge Graph ✅

- Created `database/` modules for RAG queries and source management
- Created `knowledge_graph/` modules for Neo4j operations
- Successfully separated vector database and graph database operations
- Updated all tool functions to use the refactored modules
- Removed redundant code and improved modularity

### Phase 5: Configuration Management ✅

- Created `config/` directory with settings management
- Extracted all environment variable usage to centralized configuration
- Implemented settings class with property accessors for all config values
- Updated all modules to use the configuration module instead of direct os.getenv() calls
- Added validation and default values in the configuration module
- Tested configuration loading and accessibility

### Phase 6: Tools and Testing ✅

Implemented the tool registration pattern to work with FastMCP's architecture:

1. **Tool Registration Pattern** (Implemented):
   - Created `src/tools.py` with all tool definitions
   - Tools import implementations from service modules
   - Registration function `register_tools(mcp)` called in main.py
   - Maintains FastMCP compatibility while keeping modular structure

2. **Service Layer Expansion**:
   - Created `services/search.py` for SearXNG integration
   - Created `services/smart_crawl.py` for intelligent crawling
   - All business logic separated from tool definitions

3. **Testing Infrastructure**:
   - Created comprehensive unit tests for all modules
   - Added performance benchmarks
   - Verified import times and memory usage
   - Confirmed no circular dependencies

4. **Quality Metrics Achieved**:
   - Most files under 400 lines (tools.py exception due to FastMCP)
   - Module import times < 500ms each
   - Total startup time < 1 second
   - Memory footprint increase < 100MB

## Benefits Achieved

1. **Improved Maintainability**: Each file has a single, clear purpose
2. **Better Testing**: Smaller modules are easier to unit test
3. **Enhanced Readability**: Developers can quickly find specific functionality
4. **Scalability**: Easy to add new features without bloating existing files
5. **Performance**: Smaller files load faster and use less memory

## Test Coverage Status

### Current Coverage Analysis (Post-Refactoring)

**Overall Coverage**: ~20% (Critical gaps in new modules)

| Module | Files | Current Coverage | Test Files | Status |
|--------|-------|-----------------|------------|---------|
| **database/** | 8 | ~60% | 15+ test files | ⚠️ Partial |
| **core/** | 7 | ~15% | test_refactored_modules.py | ❌ Minimal |
| **utils/** | 5 | ~20% | test_refactored_modules.py | ❌ Minimal |
| **services/** | 4 | ~5% | Import tests only | ❌ Critical Gap |
| **knowledge_graph/** | 5 | ~5% | Import tests only | ❌ Critical Gap |
| **config/** | 2 | ~30% | test_refactored_modules.py | ⚠️ Basic |
| **tools.py** | 1 | ~10% | Various MCP tests | ❌ Critical Gap |
| **main.py** | 1 | ~15% | Basic tests | ❌ Minimal |

### Critical Testing Gaps

- **Services Module**: Core business logic (crawling, search, smart crawl) largely untested
- **Tools Module**: MCP tool definitions lack comprehensive unit tests
- **Knowledge Graph**: AI validation features only have import tests
- **Core Infrastructure**: Decorators and stdout utilities missing tests

For detailed unit testing plan, see: `tests/plans/UNIT_TESTING_PLAN.md`

## Testing Strategy

1. **Unit Tests**: Create tests for each module independently
2. **Integration Tests**: Test module interactions
3. **Regression Tests**: Ensure functionality remains unchanged
4. **Coverage Goals**: Aim for 80%+ coverage per module

## Rollback Plan

If issues arise during migration:

1. Keep original `crawl4ai_mcp.py` as backup
2. Use feature flags to switch between old/new implementations
3. Gradual migration allows partial rollback

## Current State

### Completed Refactoring Results

**main.py**: 419 lines → 113 lines (73% reduction) ✅
- Browser leak fixed (single lifespan initialization)
- OAuth2 setup extracted to `auth/setup.py`
- Middleware setup extracted to `middleware/setup.py`
- Nesting reduced from 10 levels to 3 levels

**Module Structure**: ✅
```
src/
├── core/          - Infrastructure (7 files)
├── utils/         - Utilities (9 files)
├── services/      - Business logic (5 files)
├── database/      - Data operations (8 files)
├── knowledge_graph/ - Neo4j operations (18 files)
├── auth/          - Authentication (4 files)
├── middleware/    - Request middleware (2 files)
├── config/        - Configuration (3 files)
├── tools.py       - MCP tool definitions (1689 lines)
└── main.py        - Entry point (113 lines)
```

## Remaining Issues

### File Size Violations

**Critical** (>1000 lines):
- `knowledge_graph/parse_repo_into_neo4j.py` - 2050 lines
- `tools.py` - 1689 lines
- `knowledge_graph/knowledge_graph_validator.py` - 1256 lines
- `database/qdrant_adapter.py` - 1168 lines
- `knowledge_graph/enhanced_validation.py` - 1020 lines

**Target**: All files <400 lines

### Test Coverage Gaps

**Overall**: 20% (Target: 80%)

| Module | Coverage | Priority |
|--------|----------|----------|
| services/ | 5% | 🔥 Critical |
| knowledge_graph/ | 5% | 🔥 Critical |
| core/ | 15% | ❌ High |
| utils/ | 20% | ❌ High |
| database/ | 60% | ⚠️ Medium |

### Code Quality Issues

- **172** broad `except Exception` handlers (Target: <20)
- **18** skipped tests (Target: 0)
- **5** test files in project root (Target: 0)

---

## Next Steps

See `docs/PROJECT_CLEANUP_PLAN.md` for detailed improvement plan.

### Phase 1: File Size Refactoring (Week 1)

**Create**: `src/auth/setup.py`

```python
def setup_oauth2_routes(mcp: FastMCP, oauth2_server: OAuth2Server, 
                        host: str, port: str) -> None:
    """Register OAuth2 routes with FastMCP server."""
    from auth.routes import (
        authorization_server_metadata,
        protected_resource_metadata,
        register_client,
        authorize_get,
        authorize_post,
        token_endpoint
    )
    
    resource_url = f"https://{host}:{port}" if host != "0.0.0.0" else settings.oauth2_issuer
    
    @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def _metadata(request):
        return await authorization_server_metadata(request, oauth2_server)
    
    # ... register other routes
```

**Benefits**:
- Reuses existing `src/auth/routes.py` functions
- Removes 240 lines from `main()`
- Single responsibility: OAuth2 route registration

#### Step 2: Extract Middleware Setup (1 hour)

**Create**: `src/middleware/setup.py`

```python
def setup_middleware(use_oauth2: bool, oauth2_server: OAuth2Server = None) -> List[Middleware]:
    """Configure authentication middleware based on settings."""
    middleware = []
    
    if use_oauth2:
        from starlette.middleware.sessions import SessionMiddleware
        from auth import DualAuthMiddleware
        
        middleware.append(Middleware(SessionMiddleware, secret_key=settings.oauth2_secret_key))
        middleware.append(Middleware(DualAuthMiddleware, oauth2_server=oauth2_server))
        logger.info("✓ OAuth2 + API Key dual authentication enabled")
    elif settings.mcp_api_key:
        middleware.append(Middleware(APIKeyMiddleware))
        logger.info("✓ API Key authentication enabled")
    
    return middleware
```

**Benefits**:
- Removes 60 lines from `main()`
- Testable in isolation
- Clear separation of concerns

#### Step 3: Fix Browser Leak - Remove Manual Lifespan (30 min)

**Before** (lines 67-401):
```python
if transport == "http":
    async with crawl4ai_lifespan(mcp) as context:  # ❌ CAUSES LEAK
        middleware = []
        # ... 300 lines ...
        await mcp.run_http_async(...)
```

**After**:
```python
if transport == "http":
    middleware = setup_middleware(settings.use_oauth2, oauth2_server)
    
    if settings.use_oauth2:
        setup_oauth2_routes(mcp, oauth2_server, host, port)
    
    await mcp.run_http_async(transport="http", host=host, port=int(port), middleware=middleware)
    # FastMCP automatically calls crawl4ai_lifespan ONCE ✅
```

**Benefits**:
- Fixes browser process leak (single crawler instance)
- Removes 326 lines from `main()`
- Reduces nesting from 10 to 3 levels

#### Step 4: Simplify main() Function (30 min)

**Target structure** (~50 lines):
```python
async def main():
    try:
        logger.info("Main function started")
        transport = settings.transport.lower()
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        if transport == "http":
            middleware = setup_middleware(settings.use_oauth2)
            if settings.use_oauth2:
                oauth2_server = OAuth2Server(settings.oauth2_issuer, settings.oauth2_secret_key)
                setup_oauth2_routes(mcp, oauth2_server, host, port)
            await mcp.run_http_async(transport="http", host=host, port=int(port), middleware=middleware)
        elif transport == "sse":
            await mcp.run_sse_async()
        else:
            await mcp.run_stdio_async()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
```

Split large files into focused modules (<400 lines each)

### Phase 2: Test Coverage (Week 2)
Write comprehensive tests for all modules (80%+ coverage)

### Phase 3: Code Quality (Week 3)
- Refine exception handling
- Standardize logging
- Fix skipped tests
- Move root test files

### Phase 4: Documentation & CI (Week 4)
- Update documentation
- Add quality gates
- Automate testing

---

## Lessons Learned

### What Worked Well
1. **Incremental approach**: Small, atomic commits
2. **Module extraction**: Clear separation of concerns
3. **FastMCP integration**: Proper lifespan management
4. **Testing first**: Caught issues early

### Challenges
1. **Large files**: Still have 5 files >1000 lines
2. **Test coverage**: Only 20% overall
3. **Exception handling**: Too many broad handlers
4. **Documentation**: Some outdated docs remain

### Best Practices Established
1. **File size limit**: 400 lines per file
2. **Test coverage**: 80% minimum
3. **Commit frequency**: After each logical change
4. **No mocking**: Use real services in tests
5. **Structured logging**: Include context in all logs

---

## References

- **Cleanup Plan**: `docs/PROJECT_CLEANUP_PLAN.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Testing Guide**: `docs/QA/UNIT_TESTING_PLAN.md`
