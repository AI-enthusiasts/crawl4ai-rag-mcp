# 🤖 Agentic Search Architecture

**Status**: 🎯 **HIGHEST PRIORITY** - Core Feature  
**Version**: 1.0  
**Last Updated**: 2025-11-05

---

## 📋 Executive Summary

Agentic Search is an intelligent, iterative search system that combines local knowledge (Qdrant), web search (SearXNG), selective crawling (Crawl4AI), and LLM-based decision making to provide comprehensive, high-quality answers to user queries.

**Key Innovation**: Unlike traditional search-then-crawl approaches, Agentic Search uses LLM evaluation at each stage to determine:
- Is local knowledge sufficient?
- Which URLs are worth crawling?
- What information gaps remain?
- How to refine queries for better results?

---

## 🎯 Goals

### Primary Goals
1. **Maximize Answer Quality**: Provide complete, accurate answers by iteratively filling knowledge gaps
2. **Minimize Costs**: Only crawl URLs that LLM deems relevant (selective crawling)
3. **Leverage Existing Knowledge**: Check Qdrant first before hitting the web
4. **Iterative Refinement**: Automatically refine queries when initial results are insufficient

### Success Metrics
- **Completeness Score**: >80% answer completeness (LLM-evaluated)
- **Crawl Efficiency**: <30% of search results crawled (vs 100% in traditional approach)
- **Cost Reduction**: 50-70% fewer crawled pages vs exhaustive crawling
- **User Satisfaction**: Comprehensive answers in 1-3 iterations

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGENTIC SEARCH PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query                                                     │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: Local Knowledge Check                           │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ 1. Query Qdrant (with all RAG enhancements)          │ │  │
│  │ │    - Contextual embeddings (if enabled)              │ │  │
│  │ │    - Hybrid search (if enabled)                      │ │  │
│  │ │    - Reranking (if enabled)                          │ │  │
│  │ │                                                       │ │  │
│  │ │ 2. LLM Evaluation: Completeness Assessment           │ │  │
│  │ │    Input: Query + RAG results                        │ │  │
│  │ │    Output: {                                         │ │  │
│  │ │      score: 0.0-1.0,                                 │ │  │
│  │ │      reasoning: "...",                               │ │  │
│  │ │      gaps: ["missing X", "unclear Y"]                │ │  │
│  │ │    }                                                 │ │  │
│  │ │                                                       │ │  │
│  │ │ 3. Decision:                                         │ │  │
│  │ │    if score >= threshold (default 0.8):              │ │  │
│  │ │      → Return results ✅                             │ │  │
│  │ │    else:                                             │ │  │
│  │ │      → Go to STAGE 2                                 │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: Web Search                                      │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ 1. SearXNG Search                                    │ │  │
│  │ │    → Get URLs with titles/snippets                   │ │  │
│  │ │                                                       │ │  │
│  │ │ 2. LLM Ranking: URL Relevance Assessment             │ │  │
│  │ │    Input: Query + gaps + URL metadata               │ │  │
│  │ │    Output: [                                         │ │  │
│  │ │      {url, title, score: 0.0-1.0, reasoning},       │ │  │
│  │ │      ...                                             │ │  │
│  │ │    ]                                                 │ │  │
│  │ │                                                       │ │  │
│  │ │ 3. Filter Promising URLs                             │ │  │
│  │ │    → Keep only score > 0.7                           │ │  │
│  │ │    → Limit to top N (default 3)                      │ │  │
│  │ │                                                       │ │  │
│  │ │ 4. Decision:                                         │ │  │
│  │ │    if no promising URLs:                             │ │  │
│  │ │      → Go to STAGE 4 (query refinement)             │ │  │
│  │ │    else:                                             │ │  │
│  │ │      → Go to STAGE 3                                 │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 3: Selective Crawling & Indexing                   │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ 1. Crawl Promising URLs                              │ │  │
│  │ │    scrape_urls(promising_urls)                       │ │  │
│  │ │    → Crawl4AI extracts content                       │ │  │
│  │ │                                                       │ │  │
│  │ │ 2. Full Indexing Pipeline                            │ │  │
│  │ │    ┌────────────────────────────────────────────┐    │ │  │
│  │ │    │ a. Smart Chunking                          │    │ │  │
│  │ │    │    - Respect code blocks                   │    │ │  │
│  │ │    │    - Respect paragraphs                    │    │ │  │
│  │ │    │    - Configurable chunk_size               │    │ │  │
│  │ │    │                                            │    │ │  │
│  │ │    │ b. Contextual Embeddings (if enabled)      │    │ │  │
│  │ │    │    - LLM generates context for each chunk  │    │ │  │
│  │ │    │    - Parallel processing (ThreadPool)      │    │ │  │
│  │ │    │    - Fallback to standard on error         │    │ │  │
│  │ │    │                                            │    │ │  │
│  │ │    │ c. Embedding Generation                    │    │ │  │
│  │ │    │    - OpenAI text-embedding-3-small         │    │ │  │
│  │ │    │    - Batch processing (20 chunks/batch)    │    │ │  │
│  │ │    │                                            │    │ │  │
│  │ │    │ d. Qdrant Storage                          │    │ │  │
│  │ │    │    - Store chunks with embeddings          │    │ │  │
│  │ │    │    - Add metadata (url, chunk_number, etc) │    │ │  │
│  │ │    │    - Update sources table                  │    │ │  │
│  │ │    └────────────────────────────────────────────┘    │ │  │
│  │ │                                                       │ │  │
│  │ │ 3. Extract Crawl Metadata (RESEARCH NEEDED)          │ │  │
│  │ │    ⚠️ TODO: Investigate Crawl4AI capabilities        │ │  │
│  │ │    - Check if Crawl4AI returns content summaries     │ │  │
│  │ │    - Check if metadata extraction is available       │ │  │
│  │ │    - Explore CrawlResult object structure            │ │  │
│  │ │                                                       │ │  │
│  │ │    Potential metadata to extract:                    │ │  │
│  │ │    - Page title, description                         │ │  │
│  │ │    - Main topics/keywords                            │ │  │
│  │ │    - Content structure (headers)                     │ │  │
│  │ │    - Code blocks count/languages                     │ │  │
│  │ │                                                       │ │  │
│  │ │    Use case: Pass to LLM for smarter Qdrant queries  │ │  │
│  │ │                                                       │ │  │
│  │ │ 4. Generate Search Hints (Optional Enhancement)      │ │  │
│  │ │    If metadata available:                            │ │  │
│  │ │      LLM: Generate optimal Qdrant queries            │ │  │
│  │ │      Input: Original query + crawled metadata        │ │  │
│  │ │      Output: [                                       │ │  │
│  │ │        "specific query 1",                           │ │  │
│  │ │        "specific query 2"                            │ │  │
│  │ │      ]                                               │ │  │
│  │ │                                                       │ │  │
│  │ │ 5. Re-query Qdrant                                   │ │  │
│  │ │    perform_rag_query(original_query)                 │ │  │
│  │ │    OR (if hints available):                          │ │  │
│  │ │    for hint in hints:                                │ │  │
│  │ │      perform_rag_query(hint)                         │ │  │
│  │ │                                                       │ │  │
│  │ │ 6. LLM Evaluation: Re-assess Completeness            │ │  │
│  │ │    (Same as STAGE 1)                                 │ │  │
│  │ │                                                       │ │  │
│  │ │ 7. Decision:                                         │ │  │
│  │ │    if score >= threshold:                            │ │  │
│  │ │      → Return results ✅                             │ │  │
│  │ │    else:                                             │ │  │
│  │ │      → Go to STAGE 4                                 │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 4: Query Refinement & Iteration                    │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ 1. LLM: Generate Refined Queries                     │ │  │
│  │ │    Input: Original query + current query + gaps      │ │  │
│  │ │    Output: [                                         │ │  │
│  │ │      "refined query 1",                              │ │  │
│  │ │      "refined query 2"                               │ │  │
│  │ │    ]                                                 │ │  │
│  │ │                                                       │ │  │
│  │ │ 2. Select Next Query                                 │ │  │
│  │ │    current_query = refined_queries[0]                │ │  │
│  │ │                                                       │ │  │
│  │ │ 3. Iteration Control                                 │ │  │
│  │ │    iteration++                                       │ │  │
│  │ │    if iteration < max_iterations (default 3):        │ │  │
│  │ │      → Go back to STAGE 2                            │ │  │
│  │ │    else:                                             │ │  │
│  │ │      → Return best available results                 │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Details

### 1. Local Knowledge Check (STAGE 1)

**Purpose**: Leverage existing indexed content before hitting the web

**Components**:
- **Qdrant Query**: Uses existing `perform_rag_query` with all enabled enhancements
  - Respects `USE_CONTEXTUAL_EMBEDDINGS` flag
  - Respects `USE_HYBRID_SEARCH` flag
  - Respects `USE_RERANKING` flag
  - Respects `USE_AGENTIC_RAG` flag (for code search)

- **LLM Completeness Evaluator**:
  - Model: Configurable via `MODEL_CHOICE` env var (default: `gpt-4o-mini`)
  - Input: User query + RAG results (top 5 chunks)
  - Output: JSON with score, reasoning, gaps
  - Temperature: 0.3 (deterministic)

**Configuration**:
```bash
# Existing RAG enhancements (all respected)
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_AGENTIC_RAG=true

# New agentic search settings
AGENTIC_SEARCH_COMPLETENESS_THRESHOLD=0.8  # 0.0-1.0
```

---

### 2. Web Search (STAGE 2)

**Purpose**: Find promising URLs when local knowledge is insufficient

**Components**:
- **SearXNG Integration**: Uses existing `search()` tool
  - Returns URLs with titles and snippets
  - Configurable result count

- **LLM URL Ranker**:
  - Model: Same as completeness evaluator
  - Input: Query + information gaps + URL metadata (title, snippet)
  - Output: Ranked list with relevance scores (0.0-1.0)
  - Temperature: 0.3

- **URL Filter**:
  - Keep only URLs with score > threshold (default 0.7)
  - Limit to top N URLs (default 3)

**Configuration**:
```bash
AGENTIC_SEARCH_URL_SCORE_THRESHOLD=0.7     # Min score to crawl
AGENTIC_SEARCH_MAX_URLS_PER_ITERATION=3    # Max URLs to crawl
```

---

### 3. Selective Crawling & Indexing (STAGE 3)

**Purpose**: Crawl only promising URLs and index with full pipeline

**Components**:

#### 3.1 Crawling
- Uses existing `scrape_urls()` tool
- Parallel crawling (configurable concurrency)
- Respects all Crawl4AI settings

#### 3.2 Indexing Pipeline
**Uses existing `add_documents_to_database()` function** - no changes needed!

Pipeline stages (all automatic):
1. **Smart Chunking** (`smart_chunk_markdown`)
   - Respects code blocks (```)
   - Respects paragraphs (\n\n)
   - Respects sentences (.)
   - Configurable chunk size

2. **Contextual Embeddings** (if `USE_CONTEXTUAL_EMBEDDINGS=true`)
   - LLM generates context for each chunk
   - Parallel processing via ThreadPoolExecutor
   - Fallback to standard embeddings on error
   - Model: `CONTEXTUAL_EMBEDDING_MODEL` (default: gpt-4o-mini)

3. **Embedding Generation**
   - OpenAI `text-embedding-3-small`
   - Batch processing (20 chunks/batch)
   - Retry logic with exponential backoff

4. **Qdrant Storage**
   - Deterministic IDs (URL + chunk_number)
   - Automatic deduplication by URL
   - Metadata storage (url, chunk_number, source_id)
   - Sources table update

#### 3.3 Crawl Metadata Extraction (RESEARCH NEEDED)

**⚠️ TODO: Research Crawl4AI capabilities**

Questions to investigate:
1. Does `CrawlResult` object contain content summaries?
2. Can we extract structured metadata (topics, keywords)?
3. Is there a way to get page structure (headers, sections)?
4. Can we detect code blocks and their languages?

**Research sources**:
- Crawl4AI documentation: https://crawl4ai.com/docs
- `CrawlResult` class definition in codebase
- Existing usage in `services/crawling.py`

**Potential metadata to extract**:
```python
# If available in CrawlResult:
metadata = {
    "title": result.title,
    "description": result.description,
    "main_topics": result.topics,  # if available
    "headers": result.headers,     # if available
    "code_blocks": [
        {"language": "python", "count": 5},
        {"language": "javascript", "count": 3}
    ],
    "content_summary": result.summary  # if available
}
```

**Use case**: Pass metadata to LLM for generating smarter Qdrant queries

#### 3.4 Search Hints Generation (Optional Enhancement)

If metadata is available:
```python
# LLM generates optimal Qdrant queries based on crawled content
hints = llm.generate_search_hints(
    original_query="How to implement OAuth2 in FastAPI?",
    crawled_metadata={
        "title": "FastAPI Security Tutorial",
        "topics": ["OAuth2", "JWT", "authentication"],
        "code_blocks": [{"language": "python", "count": 8}]
    }
)

# Output:
# [
#   "FastAPI OAuth2PasswordBearer implementation",
#   "JWT token generation in FastAPI",
#   "FastAPI security dependencies"
# ]

# Then query Qdrant with each hint
for hint in hints:
    results = perform_rag_query(hint)
```

**Configuration**:
```bash
AGENTIC_SEARCH_USE_SEARCH_HINTS=true  # Enable smart query generation
```

---

### 4. Query Refinement (STAGE 4)

**Purpose**: Generate better queries when results are still incomplete

**Components**:
- **LLM Query Refiner**:
  - Model: Same as other LLM calls
  - Input: Original query + current query + information gaps
  - Output: 2-3 refined queries
  - Temperature: 0.5 (slightly more creative)

- **Iteration Controller**:
  - Tracks iteration count
  - Enforces max iterations limit
  - Prevents infinite loops

**Configuration**:
```bash
AGENTIC_SEARCH_MAX_ITERATIONS=3  # Max search-crawl cycles
```

---

## 📝 MCP Tool Interface

```python
@mcp.tool()
async def agentic_search(
    ctx: Context,
    query: str,
    completeness_threshold: float = 0.8,
    max_iterations: int = 3,
    max_urls_per_iteration: int = 3,
    url_score_threshold: float = 0.7,
    use_search_hints: bool = False,
) -> str:
    """
    Intelligent iterative search with automatic refinement.
    
    Workflow:
    1. Check Qdrant for existing knowledge
    2. If incomplete, search web and rank URLs with LLM
    3. Crawl promising URLs selectively
    4. Re-evaluate completeness
    5. If still incomplete, refine query and repeat
    
    Args:
        query: User's search query
        completeness_threshold: Min score for answer completeness (0-1)
        max_iterations: Max search-crawl cycles (default: 3)
        max_urls_per_iteration: Max URLs to crawl per cycle (default: 3)
        url_score_threshold: Min relevance score to crawl URL (default: 0.7)
        use_search_hints: Generate smart Qdrant queries from metadata (default: False)
    
    Returns:
        JSON with:
        - success: bool
        - query: original query
        - iterations: number of cycles performed
        - completeness: final completeness score
        - results: RAG results from Qdrant
        - search_history: detailed log of actions taken
        - status: "complete" | "max_iterations_reached"
    """
```

**Response Format**:
```json
{
  "success": true,
  "query": "How to implement OAuth2 in FastAPI?",
  "iterations": 2,
  "completeness": 0.92,
  "results": [
    {
      "content": "...",
      "url": "https://fastapi.tiangolo.com/tutorial/security/",
      "similarity_score": 0.89,
      "chunk_index": 0
    }
  ],
  "search_history": [
    {
      "iteration": 1,
      "query": "How to implement OAuth2 in FastAPI?",
      "action": "local_check",
      "completeness": 0.45,
      "gaps": ["JWT token generation", "refresh tokens"]
    },
    {
      "iteration": 1,
      "action": "web_search",
      "urls_found": 10,
      "urls_ranked": 10,
      "promising_urls": 3
    },
    {
      "iteration": 1,
      "action": "crawl",
      "urls": [
        "https://fastapi.tiangolo.com/tutorial/security/",
        "https://realpython.com/fastapi-oauth2/",
        "https://auth0.com/blog/fastapi-authentication/"
      ],
      "urls_stored": 3,
      "chunks_stored": 45
    },
    {
      "iteration": 2,
      "query": "FastAPI JWT token generation and refresh",
      "action": "local_check",
      "completeness": 0.92,
      "gaps": []
    }
  ],
  "status": "complete"
}
```

---

## 🔬 Research Tasks

### Priority 1: Crawl4AI Metadata Extraction

**Goal**: Determine what metadata Crawl4AI can provide

**Tasks**:
1. Read Crawl4AI documentation
   - Focus on `CrawlResult` object
   - Look for content analysis features
   - Check for summarization capabilities

2. Examine existing code
   - Review `services/crawling.py`
   - Check what fields are currently used from `CrawlResult`
   - Look for unused fields that might contain metadata

3. Experiment with Crawl4AI
   - Test crawling a sample page
   - Inspect full `CrawlResult` object
   - Document available fields

**Deliverable**: Document listing available metadata fields and their use cases

---

### Priority 2: Search Hints Effectiveness

**Goal**: Determine if LLM-generated search hints improve results

**Tasks**:
1. Implement basic version without hints
2. Implement version with hints
3. Compare results on test queries
4. Measure:
   - Completeness scores
   - Number of iterations needed
   - User satisfaction (if possible)

**Deliverable**: Decision on whether to include search hints feature

---

## 🚀 Implementation Plan

### Phase 1: Core Pipeline (Week 1)
**Priority**: 🔥 Critical

**Tasks**:
1. Implement STAGE 1 (Local Knowledge Check)
   - LLM completeness evaluator
   - Integration with existing `perform_rag_query`
   - Unit tests

2. Implement STAGE 2 (Web Search)
   - LLM URL ranker
   - Integration with existing `search()` tool
   - URL filtering logic
   - Unit tests

3. Implement STAGE 3 (Selective Crawling)
   - Integration with existing `scrape_urls()`
   - Integration with existing indexing pipeline
   - Re-query logic
   - Unit tests

4. Implement STAGE 4 (Query Refinement)
   - LLM query refiner
   - Iteration controller
   - Unit tests

5. Integration Testing
   - End-to-end test with real queries
   - Test iteration limits
   - Test completeness thresholds

**Deliverable**: Working `agentic_search` tool with basic functionality

---

### Phase 2: Metadata Enhancement (Week 2)
**Priority**: ⚠️ High

**Tasks**:
1. Research Crawl4AI metadata capabilities (see Research Tasks)
2. Implement metadata extraction (if available)
3. Implement search hints generation (if metadata available)
4. A/B testing: with vs without hints
5. Documentation

**Deliverable**: Enhanced `agentic_search` with metadata-driven queries

---

### Phase 3: Optimization & Monitoring (Week 3)
**Priority**: ⚠️ Medium

**Tasks**:
1. Add detailed logging
2. Add performance metrics
   - Completeness scores over time
   - Crawl efficiency (URLs crawled vs total)
   - Cost tracking (LLM calls, embeddings)
3. Add configuration validation
4. Add error recovery
5. Documentation

**Deliverable**: Production-ready `agentic_search` with monitoring

---

## 📊 Configuration Reference

### Environment Variables

```bash
# ============================================
# AGENTIC SEARCH CONFIGURATION
# ============================================

# Core Settings
AGENTIC_SEARCH_ENABLED=true                      # Enable agentic search tool
AGENTIC_SEARCH_COMPLETENESS_THRESHOLD=0.8        # Min completeness score (0.0-1.0)
AGENTIC_SEARCH_MAX_ITERATIONS=3                  # Max search-crawl cycles
AGENTIC_SEARCH_MAX_URLS_PER_ITERATION=3          # Max URLs to crawl per cycle
AGENTIC_SEARCH_URL_SCORE_THRESHOLD=0.7           # Min URL relevance score

# Advanced Settings
AGENTIC_SEARCH_USE_SEARCH_HINTS=false            # Generate smart Qdrant queries
AGENTIC_SEARCH_LLM_TEMPERATURE=0.3               # LLM temperature for evaluation
AGENTIC_SEARCH_MAX_QDRANT_RESULTS=10             # Max results from Qdrant

# Existing RAG Settings (all respected)
USE_CONTEXTUAL_EMBEDDINGS=true                   # Use contextual embeddings
USE_HYBRID_SEARCH=true                           # Use hybrid search
USE_RERANKING=true                               # Use reranking
USE_AGENTIC_RAG=true                             # Use agentic RAG for code

# LLM Settings (shared)
MODEL_CHOICE=gpt-4o-mini                         # LLM model for all evaluations
OPENAI_API_KEY=sk-...                            # OpenAI API key
```

---

## 🎯 Success Criteria

### Functional Requirements
- ✅ Check Qdrant before web search
- ✅ LLM evaluates answer completeness
- ✅ LLM ranks URLs by relevance
- ✅ Selective crawling (only promising URLs)
- ✅ Full indexing pipeline (all RAG enhancements)
- ✅ Iterative query refinement
- ✅ Configurable thresholds and limits

### Performance Requirements
- ✅ Completeness score >80% in 90% of queries
- ✅ <30% of search results crawled (vs 100% baseline)
- ✅ 50-70% cost reduction vs exhaustive crawling
- ✅ <60 seconds per iteration (average)

### Quality Requirements
- ✅ 80%+ test coverage
- ✅ Comprehensive error handling
- ✅ Detailed logging and monitoring
- ✅ Clear documentation

---

## 📚 Related Documentation

- [Contextual Embeddings](CONTEXTUAL_EMBEDDINGS.md) - Enhanced RAG with LLM context
- [Multi-Language Parsing](MULTI_LANGUAGE_PARSING.md) - Code analysis across languages
- [Project Cleanup Plan](PROJECT_CLEANUP_PLAN.md) - Overall project roadmap

---

## 🔄 Integration with Existing Features

### Respects All RAG Enhancements
- **Contextual Embeddings**: Automatically used if `USE_CONTEXTUAL_EMBEDDINGS=true`
- **Hybrid Search**: Automatically used if `USE_HYBRID_SEARCH=true`
- **Reranking**: Automatically used if `USE_RERANKING=true`
- **Agentic RAG**: Automatically used if `USE_AGENTIC_RAG=true`

### Uses Existing Tools
- `perform_rag_query()` - For Qdrant queries
- `search()` - For SearXNG integration
- `scrape_urls()` - For crawling
- `add_documents_to_database()` - For indexing

### No Breaking Changes
- All existing tools continue to work
- Agentic search is a new, optional tool
- Existing configurations are respected

---

## 🎯 Priority in Project Roadmap

**STATUS: 🔥 HIGHEST PRIORITY**

This feature is designated as the **most important development direction** for the project because:

1. **Unique Value Proposition**: No other MCP server offers intelligent, iterative search
2. **Cost Efficiency**: Dramatically reduces crawling costs through selective crawling
3. **Quality Improvement**: LLM-driven decisions ensure high-quality results
4. **User Experience**: Comprehensive answers without manual iteration
5. **Competitive Advantage**: Positions this project as the most advanced RAG-MCP solution

**Recommendation**: Prioritize this over all other features in the backlog.

---

## 📝 Notes

- Implementation details intentionally omitted (as requested)
- Focus on architecture and data flow
- Research tasks clearly marked
- Integration points with existing code identified
- No assumptions about Crawl4AI capabilities (marked for research)

---

**Last Updated**: 2025-11-05  
**Next Review**: After Phase 1 completion
