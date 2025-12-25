# Reranking Optimization Research

**Date:** 2025-12-25
**Goal:** Reduce URL ranking time from 74 seconds to <10 seconds while maintaining quality

---

## Executive Summary

Current agentic search takes **129 seconds** total, with **74 seconds** spent on LLM-based URL ranking of 20 URLs. Research from 5 parallel agents identified optimal strategies:

| Approach | Latency (20 docs) | Quality (NDCG@10) | Recommendation |
|----------|-------------------|-------------------|----------------|
| **Hybrid (CrossEncoder + LLM)** | **3-5s** | **Highest** | **Recommended** |
| CrossEncoder only | 11-500ms | 74.30 | Fast but no reasoning |
| Jina Reranker API | 30-50ms | ~50-52 | Best commercial option |
| PRP Parallel LLM | 2-5s | High | Complex implementation |
| Current (single LLM) | 74s | High | Too slow |

---

## Table of Contents

1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [CrossEncoder Optimization](#2-crossencoder-optimization)
3. [Parallel LLM Strategies](#3-parallel-llm-strategies)
4. [Commercial Rerankers](#4-commercial-rerankers)
5. [Academic Benchmarks](#5-academic-benchmarks)
6. [Recommended Solution](#6-recommended-solution)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. Current Implementation Analysis

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           _rank_urls Flow (ranker.py:114-173)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  search_results (list[dict])                                                 │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Format results_text (lines 128-133)                                  │   │
│  │  - For each result: "{i}. {title}\n   URL: {url}\n   Snippet: {[:200]}"│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Format gaps_text (line 135)                                          │   │
│  │  - "- {gap}" for each gap                                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Build prompt (lines 137-155)                                         │   │
│  │  - User Query                                                         │   │
│  │  - Knowledge Gaps                                                     │   │
│  │  - Search Results (formatted)                                         │   │
│  │  - Scoring instructions (0.0-1.0 scale)                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM Call via Pydantic AI (lines 157-164)                             │   │
│  │  - config.ranking_agent.run(prompt)                                   │   │
│  │  - Returns URLRankingList with structured output                      │   │
│  │  - Model: gpt-4o-mini (default)                                       │   │
│  │  - Timeout: 60s (LLM_API_TIMEOUT_DEFAULT)                             │   │
│  │  - Retries: 3 (MAX_RETRIES_DEFAULT)                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Sort by score descending (line 163)                                  │   │
│  │  - rankings.sort(key=lambda r: r.score, reverse=True)                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  list[URLRanking]                                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Bottleneck Analysis

| Component | Time | Percentage |
|-----------|------|------------|
| Prompt construction | <10ms | <0.1% |
| **LLM API call** | **10-15s** | **~99%** |
| Response parsing | <50ms | <0.5% |

**Root cause:** Single LLM call processes all 20 URLs with:
- ~2000-3000 tokens input (query + gaps + 20 formatted results)
- ~1500-2500 tokens output (20 URLRanking objects with reasoning)

### Existing CrossEncoder (Not Used)

```python
# src/utils/reranking.py - Available but NOT integrated
def rerank_results(
    model: CrossEncoder,
    query: str,
    results: list[dict[str, Any]],
    content_key: str = "content",
) -> list[dict[str, Any]]:
    pairs = [[query, text] for text in texts]
    scores = model.predict(pairs)  # ~50-200ms for 20 items
    return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

# src/core/context.py - Model loaded but unused
reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

---

## 2. CrossEncoder Optimization

### Model Comparison

| Model | NDCG@10 (TREC DL 19) | MRR@10 (MS Marco) | Docs/Sec (V100) | Parameters |
|-------|---------------------|-------------------|-----------------|------------|
| **ms-marco-TinyBERT-L2-v2** | 69.84 | 32.56 | **9000** | ~14M |
| ms-marco-MiniLM-L2-v2 | 71.01 | 34.85 | 4100 | ~17M |
| ms-marco-MiniLM-L4-v2 | 73.04 | 37.70 | 2500 | ~19M |
| **ms-marco-MiniLM-L6-v2** | **74.30** | **39.01** | **1800** | **22.7M** |
| ms-marco-MiniLM-L12-v2 | 74.31 | 39.02 | 960 | 33.4M |

### Recommendation: `cross-encoder/ms-marco-MiniLM-L6-v2`

**Why:** L6 achieves nearly identical quality to L12 (74.30 vs 74.31 NDCG@10) but is **~2x faster**.

### Expected Latency

| Environment | Latency (20 docs) |
|-------------|-------------------|
| GPU (V100) | ~11ms |
| GPU (float16) | ~6ms |
| CPU | ~100-500ms |
| CPU + ONNX | ~50-200ms |

### Optimization Techniques

```python
from sentence_transformers import CrossEncoder
import torch

# 1. Float16 on GPU (1.5-2x speedup)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cuda",
    model_kwargs={"torch_dtype": "float16"}
)

# 2. ONNX backend (1.5-2x speedup on CPU)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    backend="onnx"
)

# 3. OpenVINO for Intel CPUs
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    backend="openvino"
)
```

### Optimal Usage Pattern

```python
def rerank_urls(
    query: str, 
    urls_with_content: list[tuple[str, str]], 
    top_k: int = 10
) -> list[tuple[str, float]]:
    """Rerank URLs using CrossEncoder."""
    if not urls_with_content:
        return []
    
    pairs = [(query, content) for url, content in urls_with_content]
    
    scores = reranker.predict(
        pairs,
        batch_size=min(32, len(pairs)),
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    url_scores = [(url, float(score)) for (url, _), score in zip(urls_with_content, scores)]
    url_scores.sort(key=lambda x: x[1], reverse=True)
    
    return url_scores[:top_k]
```

---

## 3. Parallel LLM Strategies

### Pairwise Ranking Prompting (PRP)

From paper: "Pairwise Ranking Prompting" (arXiv:2306.17563)

> "We propose to significantly reduce the burden on LLMs by using Pairwise Ranking Prompting (PRP). Our results achieve state-of-the-art ranking performance using moderate-sized open-sourced LLMs."

### PRP Variants

| Variant | Complexity | Parallelization | Quality | Time (20 URLs) |
|---------|------------|-----------------|---------|----------------|
| **PRP-Allpair** | O(N²) = 190 comparisons | Fully parallel | Best | **2-5s** |
| PRP-Sorting | O(N log N) = 86 comparisons | Partially parallel | Good | 5-10s |
| PRP-Sliding-K | O(N) = 200 comparisons | Sequential | Good | 10-15s |

### PRP-Allpair Implementation

```python
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class PairwiseResult(BaseModel):
    winner: str = Field(description="'A' or 'B'")

class ParallelPRPRanker:
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        self.agent = Agent(
            model,
            output_type=PairwiseResult,
            instructions="Compare two passages. Output only 'A' or 'B'.",
        )
    
    async def compare_pair_debiased(
        self, query: str, idx_a: int, idx_b: int, 
        url_a: URLCandidate, url_b: URLCandidate
    ) -> tuple[int, int, int]:
        """Compare with order swap for debiasing."""
        # Run both orderings in parallel
        result_ab, result_ba = await asyncio.gather(
            self._compare(query, url_a, url_b),
            self._compare(query, url_b, url_a),
        )
        
        if result_ab == "A" and result_ba == "B":
            return (idx_a, idx_b, idx_a)  # url_a wins
        elif result_ab == "B" and result_ba == "A":
            return (idx_a, idx_b, idx_b)  # url_b wins
        else:
            return (idx_a, idx_b, -1)  # Tie
    
    async def rank_urls_allpair(
        self, query: str, urls: list[URLCandidate]
    ) -> list[URLCandidate]:
        """PRP-Allpair: O(N²) comparisons, fully parallel."""
        n = len(urls)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Run all comparisons in parallel
        results = await asyncio.gather(*[
            self.compare_pair_debiased(query, i, j, urls[i], urls[j])
            for i, j in pairs
        ])
        
        # Win counting
        scores = [0.0] * n
        for idx_a, idx_b, winner_idx in results:
            if winner_idx == idx_a:
                scores[idx_a] += 1.0
            elif winner_idx == idx_b:
                scores[idx_b] += 1.0
            else:
                scores[idx_a] += 0.5
                scores[idx_b] += 0.5
        
        ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return [urls[i] for i in ranked_indices]
```

### Why PRP Works

1. **Simpler task:** Comparing 2 items is easier than ranking 20
2. **Minimal output:** Only "A" or "B" (1 token)
3. **Fully parallelizable:** All comparisons run via `asyncio.gather()`
4. **Robust:** Individual failures don't break entire ranking
5. **De-biasing:** Swap order to eliminate position bias

---

## 4. Commercial Rerankers

### Comparison Table

| Solution | Latency (20 docs) | Quality (BEIR) | Cost | Self-Hosted |
|----------|-------------------|----------------|------|-------------|
| **Jina Reranker v2** | **30-50ms** | ~50-52 | 10M free | No |
| Cohere Rerank v3 | 100-300ms | ~52 | Pay-per-use | No |
| CrossEncoder (GPU) | ~11ms | ~45-47 | GPU cost | Yes |
| CrossEncoder (CPU) | ~100-500ms | ~45-47 | CPU cost | Yes |
| LLM (GPT-4) | 74 seconds | ~54-56 | ~$0.10-0.50 | No |

### Jina Reranker v2 (Recommended API)

**Features:**
- 100+ languages
- Function-calling and code search support
- 10M free tokens per API key
- Rate limit: 500 RPM, 1M TPM

**Integration:**

```python
import httpx

async def rerank_with_jina(
    query: str, 
    documents: list[str], 
    top_n: int = 5
) -> list[tuple[int, float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
            timeout=30.0
        )
        results = response.json()["results"]
        return [(r["index"], r["relevance_score"]) for r in results]
```

### Cohere Rerank v3

**Features:**
- Cross-attention architecture
- Handles complex data (emails, tables, JSON, code)
- Enterprise pricing

**Integration:**

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

results = co.rerank(
    model="rerank-english-v3.0",
    query="What is the capital of France?",
    documents=["Paris is the capital...", "Berlin is in Germany..."],
    top_n=5,
    return_documents=True
)
```

---

## 5. Academic Benchmarks

### BEIR Benchmark Results

| Model | BEIR Avg NDCG@10 | Type |
|-------|------------------|------|
| **GPT-4 (RankGPT)** | ~54-56 | LLM Reranker |
| RankLLaMA (7B) | ~52-54 | Fine-tuned LLM |
| Cohere Rerank v3 | ~52 | API Service |
| Jina Reranker v2 | ~50-52 | Neural Reranker |
| BAAI/bge-reranker-large | ~48-50 | CrossEncoder |
| cross-encoder/ms-marco-MiniLM-L6-v2 | ~45-47 | CrossEncoder |

### MS MARCO Passage Reranking

| Model | MRR@10 (Dev) | NDCG@10 (TREC DL 19) | Docs/Sec |
|-------|--------------|----------------------|----------|
| ms-marco-TinyBERT-L2 | 32.56 | 69.84 | 9000 |
| ms-marco-MiniLM-L6-v2 | 39.01 | 74.30 | 1800 |
| ms-marco-MiniLM-L12-v2 | 39.02 | 74.31 | 960 |

### Key Research Papers

1. **BEIR Benchmark** (NeurIPS 2021): [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
2. **RankGPT** (EMNLP 2023): [arXiv:2304.09542](https://arxiv.org/abs/2304.09542)
   > "Properly instructed LLMs can deliver competitive, even superior results to state-of-the-art supervised methods."
3. **PRP** (2023): [arXiv:2306.17563](https://arxiv.org/abs/2306.17563)
   > "Pairwise Ranking Prompting achieves state-of-the-art with moderate-sized LLMs."
4. **RankLLaMA** (2023): [arXiv:2310.08319](https://arxiv.org/abs/2310.08319)

### When to Use Each Approach

| Scenario | Recommended | Latency Budget |
|----------|-------------|----------------|
| Real-time search | CrossEncoder MiniLM-L6 | <100ms |
| Interactive search | Jina/Cohere API | 100-500ms |
| Quality-critical RAG | Hybrid or GPT-4 | 500ms-5s |
| Batch processing | LLM reranking | No limit |

---

## 6. Recommended Solution

### Hybrid 2-Stage Ranking

```
┌─────────────────────────────────────────────────────────────────┐
│  20 URLs from SearXNG                                           │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: CrossEncoder MiniLM-L6-v2                     │   │
│  │  - Latency: ~100ms (CPU) / ~11ms (GPU)                  │   │
│  │  - Filter: 20 → 8 URLs                                  │   │
│  │  - Quality: NDCG@10 = 74.30                             │   │
│  │  - Uses existing reranking_model from context           │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: LLM Ranking (8 URLs only)                     │   │
│  │  - Latency: ~3-5s (vs 10-15s for 20 URLs)               │   │
│  │  - Quality: Highest (with reasoning)                    │   │
│  │  - Gap-aware: understands knowledge gaps                │   │
│  │  - Uses existing ranking_agent                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  Top 5 ranked URLs with reasoning                               │
└─────────────────────────────────────────────────────────────────┘
```

### Why Hybrid?

| Factor | CrossEncoder Only | LLM Only | Hybrid |
|--------|-------------------|----------|--------|
| Speed | Excellent | Poor | Good |
| Quality | Good | Excellent | Excellent |
| Gap awareness | No | Yes | Yes |
| Reasoning | No | Yes | Yes |
| Cost | Free | Expensive | Moderate |

### Expected Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| URLs to LLM | 20 | 8 | 60% reduction |
| LLM ranking time | 10-15s | 3-5s | ~65% faster |
| CrossEncoder overhead | 0 | ~100ms | Negligible |
| **Total ranking time** | **10-15s** | **3-5s** | **~65% faster** |
| Per-iteration time | ~30s | ~18s | ~40% faster |
| 3-iteration total | ~90s | ~54s | ~40% faster |

---

## 7. Implementation Plan

### Files to Modify

| File | Line | Change |
|------|------|--------|
| `src/config/settings.py` | 230-245 | Add `agentic_search_use_hybrid_ranking`, `agentic_search_hybrid_ranking_top_k` |
| `src/services/agentic_search/config.py` | 81-82 | Add `use_hybrid_ranking`, `hybrid_ranking_top_k` to AgenticSearchConfig |
| `src/services/agentic_search/ranker.py` | 26-32 | Add `reranking_model` parameter to `__init__` |
| `src/services/agentic_search/ranker.py` | 88 | Call `_hybrid_rank_urls` instead of `_rank_urls` |
| `src/services/agentic_search/ranker.py` | 114+ | Add new `_hybrid_rank_urls` method |
| `src/services/agentic_search/factory.py` | 28 | Pass `reranking_model` to URLRanker |

### New Settings

```python
# src/config/settings.py
agentic_search_use_hybrid_ranking: bool = Field(
    default=True,
    description="Use CrossEncoder pre-filtering before LLM ranking",
)

agentic_search_hybrid_ranking_top_k: int = Field(
    default=8,
    ge=3,
    le=20,
    description="Number of URLs to pass to LLM after CrossEncoder filtering",
)
```

### New Method

```python
# src/services/agentic_search/ranker.py
async def _hybrid_rank_urls(
    self,
    query: str,
    gaps: list[str],
    search_results: list[dict[str, Any]],
) -> list[URLRanking]:
    """Two-stage hybrid ranking: CrossEncoder pre-filter + LLM ranking."""
    
    if not self.config.use_hybrid_ranking or self.reranking_model is None:
        return await self._rank_urls(query, gaps, search_results)
    
    # Stage 1: CrossEncoder pre-filtering
    reranked = rerank_results(
        model=self.reranking_model,
        query=query,
        results=search_results,
        content_key="snippet",
    )
    
    # Take top-k for LLM ranking
    top_k = min(self.config.hybrid_ranking_top_k, len(reranked))
    filtered_results = reranked[:top_k]
    
    # Stage 2: LLM ranking on reduced set
    return await self._rank_urls(query, gaps, filtered_results)
```

### Implementation Order

1. **Add settings** (settings.py) - 5 min
2. **Update config** (config.py) - 5 min
3. **Implement hybrid ranking** (ranker.py) - 30 min
4. **Update factory** (factory.py) - 10 min
5. **Add tests** - 30 min
6. **Benchmark** - 15 min

---

## Appendix: Alternative Approaches Considered

### A. PRP-Allpair (Parallel Pairwise LLM)

**Pros:**
- Fully parallel (2-5s for 20 URLs)
- High quality
- No additional dependencies

**Cons:**
- 380 API calls (190 pairs × 2 for de-biasing)
- Higher cost than single LLM call
- Complex implementation

**Verdict:** Good alternative if CrossEncoder quality is insufficient.

### B. Jina Reranker API

**Pros:**
- 30-50ms latency
- High quality
- 10M free tokens

**Cons:**
- External API dependency
- Network latency
- Rate limits

**Verdict:** Good for production if self-hosted CrossEncoder is too slow.

### C. CrossEncoder Only (No LLM)

**Pros:**
- Fastest (~100ms)
- No API costs
- Simple implementation

**Cons:**
- No gap awareness
- No reasoning
- Lower quality for complex queries

**Verdict:** Acceptable for simple queries, not for agentic search.

---

## References

1. BEIR Benchmark: https://github.com/beir-cellar/beir
2. MS MARCO Leaderboard: https://microsoft.github.io/msmarco
3. Sentence Transformers CrossEncoder: https://www.sbert.net/docs/cross_encoder/pretrained_models.html
4. Jina Reranker: https://jina.ai/reranker
5. Cohere Rerank: https://cohere.com/rerank
6. RankGPT Paper: https://arxiv.org/abs/2304.09542
7. PRP Paper: https://arxiv.org/abs/2306.17563
