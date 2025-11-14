# Agentic Search Refactoring Analysis

**File**: `/home/user/crawl4ai-rag-mcp/src/services/agentic_search.py`
**Current Size**: 807 LOC
**Target**: 5 focused modules, each <200 LOC
**Analysis Date**: 2025-11-14

---

## Executive Summary

The `agentic_search.py` file implements a complete LLM-driven search pipeline with 4 stages and extensive configuration. It can be cleanly decomposed into 5 focused modules:

1. **config.py** (110 LOC) - Agent initialization & configuration
2. **evaluator.py** (130 LOC) - Local knowledge evaluation
3. **ranker.py** (150 LOC) - Web search & URL ranking
4. **crawler.py** (110 LOC) - Selective URL crawling
5. **orchestrator.py** (200 LOC) - Main execution pipeline

This refactoring improves **testability**, **maintainability**, and **reusability** while maintaining **100% backward compatibility**.

---

## Module Breakdown

### 1. `config.py` - Agent Initialization & Configuration

**Purpose**: Centralize Pydantic AI agent creation and configuration management.

**Size**: ~110 LOC
**Responsibility**: Initialize OpenAI model, create specialized agents, store configuration

**Contains**:
- `AgenticSearchConfig.__init__()` (lines 64-118, 55 LOC)

**Creates**:
```python
- self.completeness_agent: Agent[CompletenessEvaluation]
- self.ranking_agent: Agent[URLRankingList]
- self.openai_model: OpenAIModel
- self.base_model_settings: ModelSettings
- self.refinement_model_settings: ModelSettings
- 9 configuration parameters (model_name, temperature, thresholds, limits)
```

**Dependencies**:
- `pydantic_ai` (Agent, ModelSettings, OpenAIModel)
- `src.config` (get_settings)
- `src.core.constants` (LLM_API_TIMEOUT_DEFAULT, MAX_RETRIES_DEFAULT)
- `agentic_models` (CompletenessEvaluation, URLRankingList)

**Key Design Decisions**:
- Pydantic AI agents use `output_retries=3` for validation robustness
- Shared `base_model_settings` for most agents
- Custom temperature (0.5) for query refinement agent
- All configuration parameters stored as instance attributes for dependency injection

---

### 2. `evaluator.py` - Local Knowledge Evaluation

**Purpose**: Evaluate completeness of local knowledge using Qdrant RAG queries and LLM analysis.

**Size**: ~130 LOC
**Responsibility**: Query Qdrant, parse results, evaluate completeness

**Contains**:
- `CompletionEvaluator._stage1_local_check()` (lines 303-369, 67 LOC)
- `CompletionEvaluator._evaluate_completeness()` (lines 371-424, 54 LOC)

**Flow**:
```
1. Get database client from app context
2. Query Qdrant with RAG enhancements (max_qdrant_results configurable)
3. Parse JSON response → Create RAGResult objects
4. Call LLM to evaluate completeness (score 0.0-1.0)
5. Record iteration in search_history (unless is_recheck=True)
```

**Completeness Scoring**:
- 0.0: No relevant information
- 0.5: Partial information, significant gaps
- 0.8: Most information present, minor gaps
- 1.0: Complete and comprehensive

**Dependencies**:
- `src.core.context` (get_app_context)
- `src.database` (perform_rag_query)
- `src.core.exceptions` (LLMError)
- `agentic_models` (RAGResult, CompletenessEvaluation, ActionType)

**Error Handling**:
- `UnexpectedModelBehavior` → `LLMError` (LLM failed after retries)
- Database errors handled gracefully

---

### 3. `ranker.py` - Web Search & URL Ranking

**Purpose**: Execute web search and rank URLs by relevance using LLM.

**Size**: ~150 LOC
**Responsibility**: Search SearXNG, filter/rank URLs, return promising results

**Contains**:
- `URLRanker._stage2_web_search()` (lines 426-499, 74 LOC)
- `URLRanker._rank_urls()` (lines 501-560, 60 LOC)

**Flow**:
```
1. Search SearXNG with query (20 results)
2. If no results: return [] and record iteration
3. OPTIMIZATION 3: Limit to max_urls_to_rank (reduce LLM tokens)
4. Call LLM to rank URLs by relevance (0.0-1.0 score)
5. Filter: score >= url_threshold
6. Limit to max_urls
7. Record iteration in search_history
```

**URL Scoring**:
- 0.0-0.3: Unlikely to be relevant
- 0.4-0.6: Possibly relevant
- 0.7-0.8: Likely relevant
- 0.9-1.0: Highly relevant (selective scoring)

**Optimizations**:
- Configurable limit before ranking (reduces LLM calls)
- Sorting by score for faster filtering
- Returns sorted results (descending by score)

**Dependencies**:
- `src.services.search` (_search_searxng)
- `src.core.exceptions` (LLMError)
- `agentic_models` (URLRanking, URLRankingList, ActionType)

---

### 4. `crawler.py` - Selective URL Crawling

**Purpose**: Crawl URLs with deduplication and store results in Qdrant.

**Size**: ~110 LOC
**Responsibility**: Detect duplicates, crawl URLs, record metrics

**Contains**:
- `SelectiveCrawler._stage3_selective_crawl()` (lines 562-656, 95 LOC)

**Flow**:
```
1. Get database client from app context
2. For each URL:
   - Check if already in Qdrant (url_exists)
   - Skip if duplicate (with logging)
   - Fail-open: Include URL if database check fails
3. Call crawl_urls_for_agentic_search with filtered URLs
4. Extract metrics:
   - urls_crawled, urls_stored, chunks_stored, urls_filtered
5. Record iteration in search_history
```

**Deduplication Strategy**:
- Uses `database_client.url_exists(url)` for efficient checking
- Fail-open error handling (include URL if check fails)
- Logging for duplicates and errors

**Configuration Parameters**:
- `max_pages_per_iteration` (limit crawl depth)
- `enable_url_filtering` (domain filtering)

**Note**: Search hints feature planned but not yet implemented.

**Dependencies**:
- `src.core.context` (get_app_context)
- `src.core.exceptions` (DatabaseError)
- `src.services.crawling` (crawl_urls_for_agentic_search)
- `agentic_models` (ActionType, SearchIteration)

---

### 5. `orchestrator.py` - Main Execution Pipeline

**Purpose**: Orchestrate entire agentic search with iterative refinement.

**Size**: ~200 LOC
**Responsibility**: Coordinate all 4 stages, apply optimizations, handle failures

**Contains**:
- `SearchOrchestrator.execute_search()` (lines 120-301, 182 LOC)
- `SearchOrchestrator._stage4_query_refinement()` (lines 658-733, 76 LOC)

**Execution Flow**:
```
WHILE iteration < max_iterations:
  1. LOCAL CHECK: Evaluate local knowledge completeness
     - IF score >= threshold: RETURN SUCCESS

  2. WEB SEARCH: Find promising URLs from web search
     - IF no URLs: Try query refinement, continue

  3. CRAWLING: Crawl URLs and store in Qdrant
     - OPTIMIZATION 1: Skip re-check if urls_stored == 0

  4. QUERY REFINEMENT: Generate refined queries
     - OPTIMIZATION 2: Skip if score improvement >= threshold
```

**Return Cases**:
- `COMPLETE` (score >= threshold)
- `MAX_ITERATIONS_REACHED` (max iterations hit)
- `ERROR` (uncaught exception)

**Query Refinement**:
- Creates temporary Agent with inline `QueryRefinementResponse` model
- Temperature 0.5 for creativity (different from evaluation)
- Generates 2-3 alternative queries based on gaps

**Optimizations**:
1. **Skip re-check after crawling** if no new content stored (saves 1 LLM call)
2. **Skip refinement** if score improvement >= threshold (saves 1 LLM call)
3. **Limit URLs before ranking** to configurable max_urls_to_rank

**Dependencies**:
- `pydantic_ai` (Agent for query refinement)
- `src.core` (MCPToolError)
- `src.core.constants` (SCORE_IMPROVEMENT_THRESHOLD)
- `src.core.exceptions` (LLMError)
- Composed dependencies: CompletionEvaluator, URLRanker, SelectiveCrawler

---

## Module Dependencies

```
config.py
├── (No internal dependencies)
└── External: pydantic_ai, src.config, src.core.constants

evaluator.py
├── Depends on: config (receives config)
└── External: src.core, src.database, pydantic_ai.exceptions

ranker.py
├── Depends on: config (receives config)
└── External: src.services.search, src.core.exceptions, pydantic_ai.exceptions

crawler.py
├── Depends on: config (receives config)
└── External: src.core.context, src.core.exceptions, src.services.crawling

orchestrator.py
├── Depends on: config, evaluator, ranker, crawler
└── External: pydantic_ai, src.core, src.core.constants, src.core.exceptions
```

---

## Class Composition

### New `AgenticSearchService` Structure

```python
class AgenticSearchService:
    def __init__(self):
        self.config = AgenticSearchConfig()
        self.evaluator = CompletionEvaluator(self.config)
        self.ranker = URLRanker(self.config)
        self.crawler = SelectiveCrawler(self.config)
        self.orchestrator = SearchOrchestrator(
            self.config,
            self.evaluator,
            self.ranker,
            self.crawler
        )

    async def execute_search(self, ctx, query, **kwargs):
        """Delegates to orchestrator"""
        return await self.orchestrator.execute_search(ctx, query, **kwargs)
```

### Module-Level Functions

**Preserved for backward compatibility**:

1. **`get_agentic_search_service()`** (lines 740-752)
   - Returns singleton `AgenticSearchService` instance
   - Location: agentic_search.py (module level)
   - Purpose: Connection pooling optimization

2. **`agentic_search_impl()`** (lines 755-806)
   - MCP tool entry point
   - Checks `agentic_search_enabled` setting
   - Calls `get_agentic_search_service().execute_search()`
   - Location: agentic_search.py or tools.py

---

## Migration Strategy

### Phase 1: Setup
1. Create `src/services/agentic_search/` directory
2. Create `__init__.py`
3. Create `config.py` with `AgenticSearchConfig` class

### Phase 2: Implement Modules
1. Create `evaluator.py` with `CompletionEvaluator` class
2. Create `ranker.py` with `URLRanker` class
3. Create `crawler.py` with `SelectiveCrawler` class
4. Create `orchestrator.py` with `SearchOrchestrator` class

### Phase 3: Composition
1. Create `agentic_search.py` that composes all modules into `AgenticSearchService`
2. Update `get_agentic_search_service()` and `agentic_search_impl()`
3. Ensure 100% backward compatibility

### Phase 4: Cleanup
1. Move `agentic_models.py` to same directory (if not already there)
2. Update imports in `tools.py` and `main.py`
3. Run full test suite
4. Update documentation
5. Delete old monolithic `agentic_search.py`

---

## Testing Strategy

### Unit Tests
- `tests/unit/services/agentic_search/test_config.py` - Agent initialization
- `tests/unit/services/agentic_search/test_evaluator.py` - Completeness evaluation
- `tests/unit/services/agentic_search/test_ranker.py` - URL ranking
- `tests/unit/services/agentic_search/test_crawler.py` - Crawling logic
- `tests/unit/services/agentic_search/test_orchestrator.py` - Main pipeline

### Integration Tests
- `tests/integration/services/test_agentic_search_integration.py` - Full pipeline
- Test all optimization paths
- Test error handling and recovery

### Backward Compatibility
- Ensure `AgenticSearchService` API unchanged
- Ensure `get_agentic_search_service()` works identically
- Ensure `agentic_search_impl()` produces same results

---

## Benefits

### Code Organization
- Each module has clear single responsibility
- Related code grouped together logically
- Easier to navigate and understand

### Testability
- Each component independently testable
- Mock/stub components easily
- Faster unit test execution

### Maintainability
- Changes localized to relevant module
- Reduced cognitive load per file
- Easier code review

### Reusability
- Components can be used separately
- `URLRanker` can rank any URLs
- `CompletionEvaluator` can evaluate any RAG results
- `SelectiveCrawler` can crawl any URL set

### Extensibility
- Easy to add new evaluation strategies
- Easy to add new ranking algorithms
- Easy to add new crawling behavior

---

## Estimated Metrics

### Current File
- Total Lines: 807
- Class Lines: 750
- Function Lines: 57

### Refactored Modules
- config.py: ~110 LOC
- evaluator.py: ~130 LOC
- ranker.py: ~150 LOC
- crawler.py: ~110 LOC
- orchestrator.py: ~200 LOC
- agentic_search.py (composition): ~80 LOC
- **Total**: ~780 LOC (similar total, much better organized)

### Improvement
- **Max file size**: 807 LOC → max 200 LOC per module
- **Max LOC per class**: 750 → 182 (largest class)
- **Cohesion**: Improved from mixed concerns to single responsibility
- **Cyclomatic complexity**: Distributed across modules

---

## Key Design Decisions

### 1. Composition Over Inheritance
- `AgenticSearchService` composes all modules
- Modules don't inherit from common base
- Allows independent evolution

### 2. Configuration Injection
- All modules receive `AgenticSearchConfig`
- Configuration centralized in one place
- Easy to override per instance

### 3. Dependency Injection
- Modules depend on configuration, not globals
- Easy to test with mock configurations
- Supports dependency inversion

### 4. Backward Compatibility
- `AgenticSearchService` API unchanged
- Singleton factory function preserved
- MCP entry point works identically

### 5. Error Handling
- Specific exception types (LLMError, DatabaseError)
- Fail-open deduplication strategy
- Comprehensive logging at each stage

---

## Next Steps

1. **Create JSON schema** (✓ Done - `refactoring_analysis_agentic_search.json`)
2. **Create markdown analysis** (✓ Done - this file)
3. **Implement refactoring** following Phase 1-4 strategy
4. **Update tests** for each module
5. **Update documentation** (AGENTIC_SEARCH_ARCHITECTURE.md)
6. **Code review** before merging
7. **Performance testing** to verify no regression

---

## File Structure

```
src/services/agentic_search/
├── __init__.py                      # Package exports
├── config.py                        # AgenticSearchConfig
├── evaluator.py                     # CompletionEvaluator
├── ranker.py                        # URLRanker
├── crawler.py                       # SelectiveCrawler
├── orchestrator.py                  # SearchOrchestrator
├── agentic_search.py                # AgenticSearchService composition
└── agentic_models.py                # Pydantic models (moved here)

tests/unit/services/agentic_search/
├── test_config.py
├── test_evaluator.py
├── test_ranker.py
├── test_crawler.py
└── test_orchestrator.py

tests/integration/services/
└── test_agentic_search_integration.py
```

---

## Conclusion

This refactoring decomposes 807 lines of complex orchestration logic into 5 focused, independently testable modules. Each module has clear responsibilities, manageable size, and minimal dependencies. The refactoring maintains 100% backward compatibility while significantly improving code quality, testability, and maintainability.

The detailed JSON analysis (`refactoring_analysis_agentic_search.json`) provides line-by-line mappings, dependency graphs, and implementation guidance for each module.
