# File Size Refactoring Plan

**Goal**: Reduce 13 files >400 LOC to manageable sizes

**Target**: Each file <400 LOC (ideally <300 LOC)

---

## Priority List (Sorted by LOC)

### 🔴 Priority 1: Critical (>700 LOC)

#### 1. `src/services/agentic_search.py` - 806 LOC
**Current**: Pydantic AI agents for agentic search
**Refactor Plan**:
- Extract agent configuration → `agentic_search/config.py`
- Extract search orchestration → `agentic_search/orchestrator.py`
- Extract iteration logic → `agentic_search/iteration.py`
- Extract URL ranking → `agentic_search/ranking.py`
- Keep main service in `agentic_search/service.py`

**Target**: 5 files @ ~160 LOC each

#### 2. `src/services/crawling.py` - 803 LOC
**Current**: Crawl4AI integration with batch processing
**Refactor Plan**:
- Extract batch processing → `crawling/batch.py`
- Extract recursive crawling → `crawling/recursive.py`
- Extract markdown processing → `crawling/markdown.py`
- Extract URL filtering → `crawling/filters.py`
- Keep main service in `crawling/service.py`

**Target**: 5 files @ ~160 LOC each

#### 3. `src/services/validated_search.py` - 798 LOC
**Current**: Validated search with LLM verification
**Refactor Plan**:
- Extract validation logic → `validated_search/validator.py`
- Extract search execution → `validated_search/executor.py`
- Extract result processing → `validated_search/processor.py`
- Keep main service in `validated_search/service.py`

**Target**: 4 files @ ~200 LOC each

#### 4. `src/utils/embeddings.py` - 714 LOC
**Current**: OpenAI embeddings + batch processing
**Refactor Plan**:
- Extract OpenAI client → `embeddings/openai_client.py`
- Extract batch processing → `embeddings/batch.py`
- Extract caching logic → `embeddings/cache.py`
- Extract retry logic → `embeddings/retry.py`
- Keep main API in `embeddings/api.py`

**Target**: 5 files @ ~140 LOC each

---

### 🟡 Priority 2: Medium (500-700 LOC)

#### 5. `src/utils/integration_helpers.py` - 558 LOC
**Current**: Integration test helpers
**Refactor Plan**:
- Extract mock factories → `integration_helpers/mocks.py`
- Extract fixture helpers → `integration_helpers/fixtures.py`
- Extract assertion helpers → `integration_helpers/assertions.py`

**Target**: 3 files @ ~185 LOC each

#### 6. `src/database/qdrant/operations.py` - 532 LOC
**Current**: Qdrant operations (CRUD)
**Refactor Plan**:
- Extract collection management → `qdrant/collections.py`
- Extract document operations → `qdrant/documents.py`
- Extract batch operations → `qdrant/batch.py`

**Target**: 3 files @ ~175 LOC each

#### 7. `src/tools/validation.py` - 527 LOC
**Current**: Validation MCP tools
**Refactor Plan**:
- Already modular (FastMCP registration)
- Could extract validators → `validation/validators.py`
- Keep tool registration in `validation.py`

**Target**: 2 files @ ~260 LOC each (borderline)

---

### 🟢 Priority 3: Low (400-500 LOC)

#### 8. `src/services/smart_crawl.py` - 495 LOC
**Status**: Borderline - may refactor later

#### 9. `src/database/supabase_adapter.py` - 461 LOC
**Status**: Legacy support - low priority

#### 10. `src/services/agentic_models.py` - 436 LOC
**Status**: Pydantic models - already clean

#### 11. `src/utils/validation.py` - 431 LOC
**Status**: Security validators - keep together

#### 12. `src/config/settings.py` - 419 LOC
**Status**: Configuration - acceptable size

---

## Execution Plan

### Phase 1: Critical Files (Weeks 7-8)
- [x] Week 7: Refactor `agentic_search.py` (806 → ~160 LOC each)
- [x] Week 7: Refactor `crawling.py` (803 → ~160 LOC each)
- [ ] Week 8: Refactor `validated_search.py` (798 → ~200 LOC each)
- [ ] Week 8: Refactor `embeddings.py` (714 → ~140 LOC each)

### Phase 2: Medium Files (Week 9)
- [ ] Refactor `integration_helpers.py` (558 → ~185 LOC each)
- [ ] Refactor `qdrant/operations.py` (532 → ~175 LOC each)
- [ ] Review `tools/validation.py` (527 LOC - decide if refactor needed)

### Phase 3: Review & Optimize (Week 10)
- [ ] Review all Priority 3 files (decide if refactoring needed)
- [ ] Performance testing of refactored modules
- [ ] Update documentation

---

## Success Criteria

**Before**: 27 files >400 LOC (13 without Neo4j)
**Target**: <10 files >400 LOC (exclude Neo4j)
**Stretch**: <5 files >400 LOC

**Metrics**:
- All critical files <400 LOC ✅
- Code coverage maintained >80% ✅
- All tests passing ✅
- No performance regression ✅
