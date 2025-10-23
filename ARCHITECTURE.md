# ARCHITECTURE.md

## MCP Server Architecture

The main entry point is `src/main.py` which orchestrates the MCP server using FastMCP. Key components:

1. **Modular Structure**: Clean separation of concerns across multiple modules
2. **Tool Registration**: All MCP tools defined in `src/tools.py` and registered via `register_tools()`
3. **Async Design**: All crawling operations are async for performance
4. **RAG Pipeline**:
   - Web crawling → Content chunking → Embedding generation → Vector storage
   - Retrieval uses semantic search with optional reranking

## Service Architecture

```
┌─────────────┐                          ┌─────────┐
│ MCP Clients │─────────────────────────▶│   MCP   │
└─────────────┘                          │ Server  │
                                         └────┬────┘
                                              │
                    ┌──────────────┐     ┌────▼────┐
                    │   SearXNG    │◀────│ Crawl4AI│
                    │(Search Engine)│     │         │
                    └──────┬───────┘     └────┬────┘
                           │                  │
                    ┌──────▼───────┐     ┌────▼────┐     ┌─────────┐
                    │    Valkey    │     │ Qdrant  │     │ Neo4j   │
                    │   (Cache)    │     │(Vectors)│     │ (Graph) │
                    └──────────────┘     └─────────┘     └─────────┘
```

## Key Design Patterns

1. **Environment-Based Configuration**: All settings via `.env` file
2. **Containerized Services**: Each component will run in an isolated Docker container, the MCP server will be connected via HTTP protocol in production. For Development we run the MCP server manually and use the stdio protocol.
3. **Vector Storage**: Qdrant as default (Supabase with pgvector as alternative)
4. **Caching Layer**: Valkey (Redis fork) for performance
5. **Knowledge Graph**: Neo4j integration for AI hallucination detection

## Module Responsibilities

- **core/**: Infrastructure components including lifecycle management, logging, decorators, and exception handling
- **utils/**: Pure utility functions for text processing, URL handling, validation, and reranking
- **services/**: Business logic for crawling, searching, and intelligent crawl operations
- **database/**: Database adapters and operations for both vector databases and traditional storage
- **knowledge_graph/**: Neo4j operations for AI hallucination detection and repository analysis
- **config/**: Centralized configuration management loading from environment variables
- **tools.py**: All MCP tool definitions using FastMCP decorators
- **main.py**: Application entry point that initializes the server and registers tools

## Code Structure Guidelines

### Adding New MCP Tools

Tools are defined in `src/tools.py` and registered with the FastMCP server. To add a new tool:

1. Add the tool function to `src/tools.py`:

```python
@mcp.tool()
async def tool_name(param: str) -> str:
    """Tool description for MCP clients"""
    # Import implementation from appropriate service module
    from services.your_service import your_function
    return await your_function(param)
```

2. The tool will be automatically registered when `register_tools(mcp)` is called in `main.py`

3. Keep business logic in the appropriate service module (services/, database/, knowledge_graph/)

### Database Operations

Database operations are now organized in the `database/` module:

- **Factory Pattern**: `database/factory.py` creates appropriate database adapter (Qdrant or Supabase)
- **RAG Queries**: `database/rag_queries.py` handles RAG-specific operations
- **Source Management**: `database/sources.py` manages crawled sources
- **Adapters**: `database/qdrant_adapter.py` and `database/supabase_adapter.py` implement the database interface

Key functions:

- `store_crawled_page()`: Store crawled content with embeddings
- `search_crawled_pages()`: Semantic search with reranking
- `store_code_example()`: Store extracted code snippets

### RAG Strategy Configuration

RAG strategies are configured via environment variables:

- `ENHANCED_CONTEXT`: Enable contextual embeddings
- `USE_RERANKING`: Enable cross-encoder reranking
- `ENABLE_AGENTIC_RAG`: Enable code extraction
- `ENABLE_HYBRID_SEARCH`: Combine vector + keyword search

## Testing Approach

### Current Status

**Overall Coverage**: 20% (Target: 80%)

| Module | Coverage | Test Files | Priority |
|--------|----------|------------|----------|
| database/ | 60% | 15+ | ⚠️ Medium |
| config/ | 30% | 3 | ⚠️ Medium |
| core/ | 15% | 2 | ❌ High |
| utils/ | 20% | 2 | ❌ High |
| services/ | 5% | 1 | 🔥 Critical |
| knowledge_graph/ | 5% | 1 | 🔥 Critical |
| tools.py | 10% | 5 | 🔥 Critical |

### Testing Guidelines

**Required for All New Code**:
1. Write tests BEFORE implementation
2. Use real services (Neo4j, Qdrant) - NO mocking
3. Achieve 80%+ coverage per module
4. Test in Docker environment
5. Verify with actual MCP clients

**Test Structure**:
```python
# tests/services/test_crawling.py
class TestCrawlMarkdownFile:
    """Test markdown file crawling with real AsyncWebCrawler."""
    
    @pytest.mark.asyncio
    async def test_successful_crawl(self, real_crawler):
        """Test with actual crawler instance."""
        result = await crawl_markdown_file(real_crawler, "https://example.com/file.txt")
        assert len(result) == 1
        assert "markdown" in result[0]
```

**What NOT to Do**:
- ❌ Mock database connections
- ❌ Mock external services
- ❌ Skip tests with `@pytest.mark.skip`
- ❌ Use broad `except Exception` in tests
- ❌ Write tests without assertions

**Test Organization**:
```
tests/
├── services/        - Service layer tests
├── database/        - Database operation tests
├── knowledge_graph/ - Neo4j integration tests
├── tools/           - MCP tool tests
├── integration/     - End-to-end tests
└── fixtures/        - Shared test fixtures
```

See `docs/PROJECT_CLEANUP_PLAN.md` for detailed testing roadmap.

## Common Development Tasks

### Modifying RAG Behavior

1. Edit strategy flags in `.env`
2. Restart service: `make dev-restart`
3. Test with sample queries
4. RAG logic is in `database/rag_queries.py` for modifications

### Adding New Search Capabilities

1. Update `searxng/settings.yml` for new search engines
2. Modify search logic in `services/search.py`
3. Update the corresponding tool in `src/tools.py` if needed
4. Rebuild: `make dev-rebuild`

### Adding New Services

1. Create a new service module in `services/` directory
2. Implement business logic following existing patterns
3. Import and use in `src/tools.py` for MCP exposure
4. Add tests in the corresponding test file

### Modifying Database Operations

1. For adapter changes: Edit `database/qdrant_adapter.py` or `database/supabase_adapter.py`
2. For new database operations: Add to appropriate module in `database/`
3. Ensure factory pattern is maintained in `database/factory.py`

### Using Hallucination Detection Tools

The hallucination detection tools analyze Python scripts for AI-generated code issues. Scripts must be accessible to the Docker container through volume mounts:

**Script Directories:**

- `./analysis_scripts/user_scripts/` - Your Python scripts for analysis
- `./analysis_scripts/test_scripts/` - Test scripts
- `./analysis_scripts/validation_results/` - Results storage
- `/tmp/` - Temporary scripts (mapped to container's `/app/tmp_scripts/`)

**Path Translation:**
The tools automatically translate host paths to container paths:

- `analysis_scripts/user_scripts/script.py` → `/app/analysis_scripts/user_scripts/script.py`
- `/tmp/test.py` → `/app/tmp_scripts/test.py`
- `script.py` → `/app/analysis_scripts/user_scripts/script.py` (default)

**Example Usage:**

```python
# Place script at: ./analysis_scripts/user_scripts/ai_generated.py
result = check_ai_script_hallucinations(
    script_path="analysis_scripts/user_scripts/ai_generated.py"
)
```

### Debugging Issues

1. Check container logs: `make logs` (choose environment when prompted)
2. Verify environment variables are loaded
3. Test individual components in isolation
4. Check Qdrant connection (<http://localhost:6333/dashboard>)
5. Check Neo4j browser (<http://localhost:7474>)
6. For hallucination detection issues: Use `get_script_analysis_info()` tool to verify setup

## Performance Considerations

1. **Embedding Generation**: Batch operations when possible
2. **Vector Search**: Use appropriate `match_count` limits
3. **Caching**: Valkey caches search results automatically
4. **Chunking**: Balance chunk size vs context (default: 2000 chars)

## Security Notes

1. Never commit `.env` files with real credentials
2. Configure Qdrant API keys for production
3. Set Neo4j authentication properly in production
4. SearXNG provides rate limiting and bot protection
