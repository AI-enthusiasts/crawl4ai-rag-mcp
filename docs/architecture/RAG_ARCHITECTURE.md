# 🏗️ RAG Architecture - Current Implementation

**Last Updated**: 2025-11-05  
**Status**: Production  
**Purpose**: Complete documentation of current RAG pipeline architecture

---

## 📋 Overview

This document describes the **current** RAG (Retrieval Augmented Generation) architecture in crawl4ai-rag-mcp, covering the complete pipeline from URL crawling to search results.

---

## 🔄 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: URL or Search Query                                     │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: Content Acquisition                             │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ Option A: Direct Crawling                            │ │  │
│  │ │   scrape_urls(url) → Crawl4AI                        │ │  │
│  │ │                                                       │ │  │
│  │ │ Option B: Search + Crawl                             │ │  │
│  │ │   search(query) → SearXNG → URLs → Crawl4AI         │ │  │
│  │ │                                                       │ │  │
│  │ │ Option C: Smart Crawl                                │ │  │
│  │ │   smart_crawl_url(url) → Auto-detect type           │ │  │
│  │ │   - Sitemap: Parse all URLs                         │ │  │
│  │ │   - Text file: Direct download                      │ │  │
│  │ │   - Regular page: Recursive crawl                   │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │ OUTPUT: Markdown content                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: Smart Chunking                                  │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ smart_chunk_markdown(content, chunk_size=2000)       │ │  │
│  │ │                                                       │ │  │
│  │ │ Algorithm:                                           │ │  │
│  │ │ 1. Respect code blocks (```) - never split          │ │  │
│  │ │ 2. Respect paragraphs (\n\n) - split between        │ │  │
│  │ │ 3. Respect sentences (.) - split at periods         │ │  │
│  │ │ 4. Hard break if no boundary found                  │ │  │
│  │ │                                                       │ │  │
│  │ │ Thresholds:                                          │ │  │
│  │ │ - Boundary must be >30% into chunk                  │ │  │
│  │ │ - Prevents tiny chunks at boundaries                │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │ OUTPUT: List of chunks (each ~2000 chars)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 3: Embedding Generation                            │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ Option A: Standard Embeddings (default)              │ │  │
│  │ │   chunk → OpenAI API → embedding [1536 dims]        │ │  │
│  │ │                                                       │ │  │
│  │ │ Option B: Contextual Embeddings                      │ │  │
│  │ │   (if USE_CONTEXTUAL_EMBEDDINGS=true)                │ │  │
│  │ │                                                       │ │  │
│  │ │   For each chunk:                                    │ │  │
│  │ │   1. LLM generates context (gpt-4o-mini)            │ │  │
│  │ │      Input: full document + chunk                   │ │  │
│  │ │      Output: 200 token context                      │ │  │
│  │ │                                                       │ │  │
│  │ │   2. Combine: context + "---" + chunk               │ │  │
│  │ │                                                       │ │  │
│  │ │   3. Generate embedding for enhanced chunk          │ │  │
│  │ │      → OpenAI API → embedding [1536 dims]           │ │  │
│  │ │                                                       │ │  │
│  │ │   Parallel processing: ThreadPoolExecutor           │ │  │
│  │ │   Fallback: Standard embedding on error             │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │ OUTPUT: List of embeddings                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 4: Vector Storage (Qdrant/Supabase)               │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ Deduplication:                                       │ │  │
│  │ │   delete_documents_by_url(url)                       │ │  │
│  │ │   → Removes old chunks for same URL                 │ │  │
│  │ │                                                       │ │  │
│  │ │ Storage:                                             │ │  │
│  │ │   For each chunk:                                    │ │  │
│  │ │     point_id = uuid5(url + chunk_number)            │ │  │
│  │ │     payload = {                                      │ │  │
│  │ │       url: "...",                                    │ │  │
│  │ │       chunk_number: 0,                               │ │  │
│  │ │       content: "original chunk",                     │ │  │
│  │ │       source_id: "example.com",                      │ │  │
│  │ │       metadata: {...}                                │ │  │
│  │ │     }                                                │ │  │
│  │ │     qdrant.upsert(point_id, embedding, payload)      │ │  │
│  │ │                                                       │ │  │
│  │ │ Collections:                                         │ │  │
│  │ │   - crawled_pages: Main content                     │ │  │
│  │ │   - code_examples: Code snippets (if agentic RAG)   │ │  │
│  │ │   - sources: Source metadata                        │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │ OUTPUT: Chunks stored in vector DB                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 5: Search & Retrieval                              │  │
│  │ ┌──────────────────────────────────────────────────────┐ │  │
│  │ │ perform_rag_query(query, source_filter, match_count) │ │  │
│  │ │                                                       │ │  │
│  │ │ Search Types (configurable):                         │ │  │
│  │ │                                                       │ │  │
│  │ │ 1. Vector Search (default)                           │ │  │
│  │ │    query → embedding → cosine similarity → top-K    │ │  │
│  │ │                                                       │ │  │
│  │ │ 2. Hybrid Search (USE_HYBRID_SEARCH=true)            │ │  │
│  │ │    - Vector search (70% weight)                      │ │  │
│  │ │    - Keyword search (30% weight)                     │ │  │
│  │ │    - Merge results                                   │ │  │
│  │ │    - Boost overlapping (+0.3 score)                 │ │  │
│  │ │                                                       │ │  │
│  │ │ 3. Reranking (USE_RERANKING=true)                    │ │  │
│  │ │    - Initial search (top-20)                         │ │  │
│  │ │    - Cross-encoder model                             │ │  │
│  │ │    - Rerank by relevance                             │ │  │
│  │ │    - Return top-5                                    │ │  │
│  │ │                                                       │ │  │
│  │ │ 4. Agentic RAG (USE_AGENTIC_RAG=true)                │ │  │
│  │ │    - Search code_examples collection                │ │  │
│  │ │    - Return code + LLM summary                       │ │  │
│  │ └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │ OUTPUT: Ranked results with similarity scores            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 MCP Tools

### 1. **scrape_urls** - Direct URL Crawling

```python
scrape_urls(
    url: str | list[str],
    max_concurrent: int = 10,
    batch_size: int = 20,
    return_raw_markdown: bool = False
)
```

**Pipeline**: URL → Crawl4AI → Chunking → Embeddings → Qdrant

**Use Case**: Index specific URLs

---

### 2. **search** - Web Search + Crawl

```python
search(
    query: str,
    return_raw_markdown: bool = False,
    num_results: int = 6,
    batch_size: int = 20,
    max_concurrent: int = 10
)
```

**Pipeline**: Query → SearXNG → URLs → Crawl4AI → Chunking → Embeddings → Qdrant → RAG

**Use Case**: Discover and index new content

---

### 3. **smart_crawl_url** - Intelligent Crawling

```python
smart_crawl_url(
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    return_raw_markdown: bool = False,
    query: list[str] | None = None
)
```

**Auto-detection**:
- Sitemap.xml → Parse all URLs → Crawl in parallel
- .txt file → Direct download
- Regular page → Recursive crawl (follow internal links)

**Use Case**: Index entire websites

---

### 4. **perform_rag_query** - Search Indexed Content

```python
perform_rag_query(
    query: str,
    source: str | None = None,
    match_count: int = 5
)
```

**Pipeline**: Query → Embedding → Vector Search → Reranking (optional) → Results

**Use Case**: Retrieve relevant chunks

---

### 5. **get_available_sources** - List Sources

```python
get_available_sources()
```

**Returns**: List of indexed domains

**Use Case**: Discover what's in the database

---

### 6. **search_code_examples** - Code Search

```python
search_code_examples(
    query: str,
    source_id: str | None = None,
    match_count: int = 5
)
```

**Requires**: `USE_AGENTIC_RAG=true`

**Pipeline**: Query → Search code_examples collection → Return code + summary

**Use Case**: Find code snippets

---

## 🎛️ Configuration

### RAG Enhancement Flags

```bash
# Contextual Embeddings (+20-30% accuracy)
USE_CONTEXTUAL_EMBEDDINGS=true
CONTEXTUAL_EMBEDDING_MODEL=gpt-4o-mini
CONTEXTUAL_EMBEDDING_MAX_TOKENS=200
CONTEXTUAL_EMBEDDING_TEMPERATURE=0.3

# Hybrid Search (vector + keyword)
USE_HYBRID_SEARCH=true

# Reranking (cross-encoder)
USE_RERANKING=true

# Agentic RAG (code extraction)
USE_AGENTIC_RAG=true

# Knowledge Graph (code analysis)
USE_KNOWLEDGE_GRAPH=true
```

### Vector Database

```bash
# Qdrant (default)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # optional

# OR Supabase
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=...
```

### Embeddings

```bash
OPENAI_API_KEY=sk-...
MODEL_CHOICE=gpt-4o-mini  # for contextual embeddings
```

---

## 📊 Data Structures

### Qdrant Point (crawled_pages)

```python
{
    "id": "uuid5(url_chunk0)",  # deterministic
    "vector": [0.12, -0.45, ..., 0.34],  # 1536 dims
    "payload": {
        "url": "https://example.com/page",
        "chunk_number": 0,
        "content": "Original chunk text...",
        "source_id": "example.com",
        "metadata": {
            "url": "https://example.com/page",
            "chunk": 0,
            "title": "Page Title"
        }
    }
}
```

### Qdrant Point (code_examples)

```python
{
    "id": "code_uuid",
    "vector": [0.23, -0.12, ..., 0.56],
    "payload": {
        "code": "def authenticate(...):\n    ...",
        "summary": "Function that authenticates users...",
        "programming_language": "python",
        "source_id": "example.com",
        "url": "https://example.com/docs"
    }
}
```

### Sources Table

```python
{
    "id": "source_uuid",
    "vector": [0.45, -0.23, ..., 0.67],
    "payload": {
        "source_id": "example.com",
        "url": "https://example.com",
        "title": "example.com",
        "description": "Summary of content...",
        "metadata": {
            "type": "web_scrape",
            "chunk_count": 10,
            "total_content_length": 50000,
            "word_count": 7500
        }
    }
}
```

---

## 🔍 Search Types Explained

### 1. Vector Search (Baseline)

```python
# Query
query = "OAuth2 authentication"
query_embedding = openai.embed(query)  # [1536 dims]

# Search
results = qdrant.search(
    collection="crawled_pages",
    query_vector=query_embedding,
    limit=5,
    score_threshold=0.7
)

# Scoring: Cosine similarity
# similarity = (A · B) / (||A|| × ||B||)
# Range: 0.0 to 1.0 (higher = more similar)
```

**Pros**: Fast, simple  
**Cons**: Misses exact keyword matches

---

### 2. Hybrid Search (Vector + Keyword)

```python
# Vector search (70% weight)
vector_results = qdrant.search(
    query_vector=query_embedding,
    limit=10
)

# Keyword search (30% weight)
keyword_results = qdrant.scroll(
    scroll_filter=Filter(
        must=[FieldCondition(
            key="content",
            match=MatchValue(value="OAuth2")
        )]
    ),
    limit=10
)

# Merge with weights
for result in vector_results:
    result.score = result.score * 0.7

for result in keyword_results:
    if result.id in vector_results:
        # Found in both → boost!
        result.score += 0.3
    else:
        result.score = 0.3

# Sort by combined score
final_results = sorted(all_results, key=lambda x: x.score, reverse=True)[:5]
```

**Pros**: Catches both semantic and exact matches  
**Cons**: Slightly slower

---

### 3. Reranking (Cross-Encoder)

```python
# Step 1: Initial search (get more results)
initial_results = qdrant.search(
    query_vector=query_embedding,
    limit=20  # get more for reranking
)

# Step 2: Cross-encoder evaluation
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [[query, result.content] for result in initial_results]
relevance_scores = model.predict(pairs)  # [0.92, 0.87, 0.15, ...]

# Step 3: Rerank by relevance
for i, result in enumerate(initial_results):
    result.rerank_score = relevance_scores[i]

reranked = sorted(initial_results, key=lambda x: x.rerank_score, reverse=True)[:5]
```

**How Cross-Encoder Works**:
- Takes [query, document] pair as input
- Processes them together (not separately like bi-encoder)
- Outputs single relevance score
- More accurate but slower

**Pros**: Best relevance  
**Cons**: +50-100ms latency

---

### 4. Agentic RAG (Code Extraction)

```python
# Indexing: Extract code blocks
code_blocks = extract_code_blocks(markdown)
for code in code_blocks:
    # LLM generates summary
    summary = llm.summarize(code)
    
    # Create embedding for summary
    embedding = openai.embed(summary)
    
    # Store in separate collection
    qdrant.upsert(
        collection="code_examples",
        point={
            "vector": embedding,
            "payload": {
                "code": code,
                "summary": summary,
                "language": "python"
            }
        }
    )

# Searching: Query code collection
results = qdrant.search(
    collection="code_examples",
    query_vector=query_embedding,
    limit=5
)
```

**Pros**: Specialized for code  
**Cons**: Requires LLM for summarization

---

## 🚀 Performance Characteristics

### Indexing Speed

| Stage | Time (per page) | Bottleneck |
|-------|----------------|------------|
| Crawling | 1-3s | Network, Crawl4AI |
| Chunking | <100ms | CPU |
| Standard Embeddings | 200-500ms | OpenAI API |
| Contextual Embeddings | 2-5s | LLM calls |
| Qdrant Storage | <100ms | Network |
| **Total (standard)** | **2-4s** | Network + API |
| **Total (contextual)** | **4-9s** | LLM calls |

### Search Speed

| Search Type | Latency | Bottleneck |
|------------|---------|------------|
| Vector Search | 10-20ms | Qdrant |
| Hybrid Search | 20-50ms | Qdrant (2 queries) |
| Reranking | 60-120ms | Cross-encoder |
| Agentic RAG | 10-20ms | Qdrant |

---

## 🎯 Best Practices

### When to Use What

| Scenario | Recommended Config |
|----------|-------------------|
| **General documentation** | Hybrid + Reranking |
| **Complex technical docs** | Contextual + Hybrid + Reranking |
| **Code search** | Agentic RAG + Reranking |
| **High-volume production** | Hybrid only (fast) |
| **Maximum quality** | All enabled (slow, expensive) |

### Cost Optimization

```bash
# Low cost (baseline)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_AGENTIC_RAG=false

# Medium cost (recommended)
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_AGENTIC_RAG=false

# High cost (maximum quality)
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_AGENTIC_RAG=true
```

---

## 🔗 Related Documentation

- [AGENTIC_SEARCH_ARCHITECTURE.md](../AGENTIC_SEARCH_ARCHITECTURE.md) - Future intelligent search
- [CONTEXTUAL_EMBEDDINGS.md](../CONTEXTUAL_EMBEDDINGS.md) - Contextual embeddings details
- [NEO4J_QDRANT_INTEGRATION_GUIDE.md](NEO4J_QDRANT_INTEGRATION_GUIDE.md) - Knowledge graph integration
- [PROJECT_ROADMAP.md](../PROJECT_ROADMAP.md) - Development priorities

---

**Last Updated**: 2025-11-05  
**Maintainer**: Project Team
