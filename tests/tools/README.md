# MCP Tools Unit Tests

Comprehensive unit tests for all MCP tools in the `src/tools/` directory.

## Test Files Created

1. **`test_search_tools.py`** - Tests for search-related MCP tools
   - `search`: Basic web search with SearXNG integration
   - `agentic_search`: Advanced autonomous search
   - `analyze_code_cross_language`: Cross-language code analysis

2. **`test_crawl_tools.py`** - Tests for crawling MCP tools
   - `scrape_urls`: Scrape one or more URLs
   - `smart_crawl_url`: Intelligent URL crawling with type detection

3. **`test_rag_tools.py`** - Tests for RAG-related MCP tools
   - `get_available_sources`: List all indexed sources
   - `perform_rag_query`: Semantic search over indexed content
   - `search_code_examples`: Search for code examples

4. **`test_kg_tools.py`** - Tests for knowledge graph MCP tools
   - `query_knowledge_graph`: Query and explore Neo4j knowledge graph
   - `parse_github_repository`: Parse GitHub repos into Neo4j
   - `parse_repository_branch`: Parse specific branches
   - `get_repository_info`: Get repository metadata
   - `update_parsed_repository`: Update already parsed repos
   - `parse_local_repository`: Parse local Git repositories

5. **`test_validation_tools.py`** - Tests for validation MCP tools
   - `extract_and_index_repository_code`: Index code from Neo4j to Qdrant
   - `smart_code_search`: Validated semantic code search
   - `check_ai_script_hallucinations_enhanced`: Enhanced hallucination detection
   - `get_script_analysis_info`: Helper for script analysis setup

## Test Statistics

- **Total Tests**: 67
- **Currently Passing**: 38 (56.7%)
- **Currently Failing**: 29 (43.3%)

## Test Coverage

All MCP tool functions are tested with:
- ✅ Success path tests
- ✅ Error handling tests (MCPToolError wrapping)
- ✅ Context handling tests
- ✅ Parameter validation tests
- ✅ JSON response formatting tests

## Running Tests

```bash
# Run all tool tests
uv run pytest tests/tools/ -v

# Run specific test file
uv run pytest tests/tools/test_search_tools.py -v

# Run specific test class
uv run pytest tests/tools/test_search_tools.py::TestSearchTool -v

# Run with coverage
uv run pytest tests/tools/ --cov=src/tools --cov-report=html

# Run only passing tests
uv run pytest tests/tools/ -v -k "not test_scrape_urls_single"
```

## Test Patterns Used

### 1. Mock FastMCP Context
```python
@pytest.fixture
def mock_context():
    """Create a mock FastMCP Context."""
    ctx = MagicMock(spec=Context)
    return ctx
```

### 2. Mock Tool Registration
```python
mcp_instance = MagicMock()
registered_funcs = {}

def mock_tool_decorator():
    def decorator(func):
        registered_funcs[func.__name__] = func
        return func
    return decorator

mcp_instance.tool = mock_tool_decorator
```

### 3. Mock Service Dependencies
```python
with patch("src.tools.search.search_and_process") as mock_search:
    mock_search.return_value = json.dumps({"success": True})
    # Test code here
```

### 4. Error Handling Tests
```python
with pytest.raises(MCPToolError) as exc_info:
    await tool_func(ctx=mock_context, invalid_param="value")

assert "Expected error message" in str(exc_info.value)
```

## Known Issues

Some tests require fixing import paths for mocking. The main issues are:

1. **Import Path Mismatches**: Some mocks patch modules at the definition location instead of the usage location
2. **Nested Imports**: Functions imported within other functions need special handling
3. **Module Aliases**: Some imports use aliases that need to be tracked

### Example Fix Pattern

If you see errors like:
```
AttributeError: <module 'src.tools.crawl'> does not have the attribute 'clean_url'
```

The fix is to patch where the function is **used**, not where it's **defined**:

```python
# ❌ Wrong - patches at import location
with patch("src.tools.crawl.clean_url") as mock_clean:
    pass

# ✅ Correct - patches at definition location
with patch("src.utils.url_helpers.clean_url") as mock_clean:
    pass
```

## Test Configuration

Tests use the following fixtures from `tests/conftest.py`:

- `mock_openai_embeddings`: Mocks OpenAI API calls (default, unless `ALLOW_OPENAI_TESTS=true`)
- `get_adapter`: Factory for database adapters (Qdrant/Supabase)
- `event_loop`: Asyncio event loop for async tests

## Cost Protection

All tests are designed to run without real OpenAI API calls by default:
- OpenAI embeddings are mocked
- LLM inference calls are mocked
- No real API costs incurred during testing

To enable real API calls (for integration testing):
```bash
export ALLOW_OPENAI_TESTS=true  # Enable cheap tests (~$0.0001/test)
export ALLOW_EXPENSIVE_TESTS=true  # Enable expensive tests ($$$)
```

## Contributing

When adding new MCP tools, please:

1. Create corresponding unit tests in `tests/tools/`
2. Follow the established test patterns (see examples above)
3. Mock all external dependencies
4. Test both success and error paths
5. Aim for >60% coverage (MCP tools are thin wrappers)
6. Use descriptive test names that explain what is being tested

## Additional Resources

- Main test documentation: `/tests/README.md`
- Integration tests: `/tests/integration/`
- Project testing guide: `/docs/TESTING.md`
- MCP tool implementations: `/src/tools/`
