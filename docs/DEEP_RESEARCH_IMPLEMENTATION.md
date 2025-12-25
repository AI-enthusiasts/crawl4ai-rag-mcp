# Deep Research Implementation - Technical Specification

## Цель

Сделать agentic search качественным: полные, подробные ответы с первого раза, без необходимости переспрашивать. Не возвращать всю документацию, но покрывать все ключевые аспекты темы.

## Текущее состояние (что уже сделано в этой сессии)

### Исправленные баги

1. **`similarity_score = 0`** - в `src/database/rag_queries.py` было `result.get("score", 0)`, а Qdrant возвращает `result["similarity"]`. Исправлено.

2. **`url_exists` возвращал неправильные результаты** - в `src/database/qdrant/operations/documents.py` использовался `exact=False` в count query, что возвращало ~половину коллекции вместо точного count. Исправлено на `exact=True`.

3. **E2E тест не работал** - в `tests/integration/test_agentic_search_e2e.py`:
   - Добавлен `await` для `create_embeddings_batch` (async функция)
   - Исправлена проверка SearXNG: `number_of_results` всегда 0, нужно проверять `bool(results)`
   - Используется `engines=wikipedia` для стабильного результата без CAPTCHA

4. **Контент содержал навигацию сайта вместо основного текста** - добавлен `PruningContentFilter` в Crawl4AI:
   - `src/services/crawling/service.py` - CrawlerRunConfig с excluded_tags и PruningContentFilter
   - `src/services/crawling/batch.py` - то же самое
   - `src/services/crawling/recursive.py` - то же самое
   - Используется `result.markdown.fit_markdown` вместо `result.markdown`

5. **Дублирование результатов по URL** - в `src/database/rag_queries.py` добавлена группировка чанков по URL и объединение контента.

6. **Маленькие чанки (2000 символов)** - увеличено до 4000 в `src/services/crawling/service.py`.

### Текущие результаты

После исправлений:
- 7 уникальных URL вместо 10 с дублями
- 3550 символов контента вместо 300
- Примеры кода видны
- Разные источники (hashicorp.com, woodruff.dev)
- Score 0.739 вместо 0.309

### Что ещё не идеально

- Нет структурированности ответа
- Нет гарантии покрытия всех ключевых тем
- LLM оценивает completeness общим score, не по конкретным темам
- Нет multi-query для лучшего recall

---

## Исследование индустрии (ключевые находки)

### Anthropic (Claude)

1. **Orchestrator-Worker Architecture**
   > "When a user submits a query, the lead agent analyzes it, develops a strategy, and spawns subagents to explore different aspects simultaneously."
   
2. **Breadth-First → Depth Search**
   > "Search strategy should mirror expert human research: explore the landscape before drilling into specifics. Agents often default to overly long, specific queries that return few results. We counteracted this tendency by prompting agents to start with short, broad queries, evaluate what's available, then progressively narrow focus."

3. **Interleaved Thinking**
   > "Subagents also plan, then use interleaved thinking after tool results to evaluate quality, identify gaps, and refine their next query."

4. **Effort Scaling**
   > "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with 10-15 calls each, and complex research might use more than 10 subagents."

### Perplexity

1. **Trust Score Formula**
   ```
   Trust(s) = 0.4·PageRank + 0.3·FactualConsistency + 0.2·AuthorExpertise + 0.1·Recency
   ```

2. **Termination** - когда trust превышает порог или 12 итераций / 110 API calls

3. **Conflict Resolution** - majority voting с temporal decay (свежие источники важнее)

### You.com

1. **200-400 источников** на запрос (10× больше конкурентов)
2. **Marginal Utility Threshold** - остановка когда новые источники не добавляют ценности
3. **Conflict Exposure** - показывать противоречия, не скрывать

### RAG Best Practices

1. **Multi-Query Generation** - генерировать 3-4 варианта запроса ДО поиска
2. **Reciprocal Rank Fusion (RRF)** - объединять результаты:
   ```python
   RRF_score(d) = Σ 1 / (k + rank_i(d))  # k=60 typically
   ```
3. **CRAG (Corrective RAG)** - оценивать качество retrieval, fallback на web если плохо
4. **Sub-Question Decomposition** - для сложных/сравнительных запросов

---

## План реализации

### Phase 1: Query Enhancement (приоритет)

#### 1.1 Topic Decomposition

Перед поиском разбить запрос на ключевые темы которые ДОЛЖНЫ быть покрыты.

**Новая модель в `src/services/agentic_models.py`:**
```python
class TopicDecomposition(BaseModel):
    """Decomposition of query into required topics."""
    
    original_query: str
    topics: list[str]  # ["Definition", "Key concepts", "Examples", "Best practices"]
    topic_queries: dict[str, str]  # {"Definition": "What is X?", ...}
    complexity: str  # "simple" | "moderate" | "complex"
```

**Prompt для LLM:**
```
Given the query: "{query}"

Decompose into essential topics that MUST be covered for a complete answer:
- For "What is X?" → Definition, Key features, Use cases, Examples
- For "How to X?" → Prerequisites, Step-by-step, Code examples, Common pitfalls
- For "Compare X vs Y" → X overview, Y overview, Differences, When to use each

Return 3-6 topics with specific search queries for each.
```

#### 1.2 Multi-Query Generation

Для каждой темы генерировать 2-3 варианта запроса.

**Новая модель:**
```python
class MultiQueryExpansion(BaseModel):
    """Multiple query variations for better recall."""
    
    original_query: str
    variations: list[str]  # 3-4 variations
    broad_query: str  # More general version
    specific_query: str  # More specific version
```

**Стратегия:**
1. Original query as-is
2. Rephrased with synonyms
3. More general (step-back)
4. More specific (with context)

#### 1.3 Reciprocal Rank Fusion

Объединить результаты всех query variations.

**Новая функция в `src/database/rag_queries.py`:**
```python
def reciprocal_rank_fusion(
    result_lists: list[list[dict]], 
    k: int = 60
) -> list[dict]:
    """Combine results from multiple queries using RRF."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    
    for results in result_lists:
        for rank, result in enumerate(results):
            doc_id = result["url"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in docs or result["similarity_score"] > docs[doc_id]["similarity_score"]:
                docs[doc_id] = result
    
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [docs[doc_id] for doc_id in sorted_ids]
```

### Phase 2: Completeness by Topics

#### 2.1 Topic-Based Evaluation

Вместо общего score 0.0-1.0, оценивать покрытие каждой темы.

**Новая модель:**
```python
class TopicCompleteness(BaseModel):
    """Completeness evaluation per topic."""
    
    topic: str
    covered: bool
    score: float  # 0.0-1.0
    evidence: str  # Quote from results proving coverage
    gaps: list[str]  # What's missing for this topic
```

**Prompt для LLM:**
```
Query: {query}
Required topics: {topics}

For each topic, evaluate if the retrieved content covers it:

Retrieved content:
{content}

For each topic provide:
- covered: true/false
- score: 0.0-1.0
- evidence: exact quote proving coverage (or empty if not covered)
- gaps: what's missing
```

#### 2.2 Gap-Driven Iteration

После первого поиска, искать ТОЛЬКО по непокрытым темам.

**Логика в orchestrator:**
```python
# After initial search
uncovered_topics = [t for t in evaluation.topics if not t.covered]

if uncovered_topics:
    # Generate specific queries for uncovered topics
    gap_queries = [topic.gap_query for topic in uncovered_topics]
    
    # Search specifically for gaps
    gap_results = await search_for_gaps(gap_queries)
    
    # Merge with existing results
    all_results = merge_results(initial_results, gap_results)
```

### Phase 3: Result Quality

#### 3.1 Content Completeness Check

Перед возвратом проверить что контент реально отвечает на вопрос.

**Критерии:**
- Есть определение/объяснение (для "What is" вопросов)
- Есть примеры кода (для технических вопросов)
- Есть практические шаги (для "How to" вопросов)
- Нет только навигации/оглавления

#### 3.2 Source Diversity

Проверить что результаты из разных источников, не только один сайт.

```python
def check_source_diversity(results: list[RAGResult], min_sources: int = 3) -> bool:
    domains = set(extract_domain(r.url) for r in results)
    return len(domains) >= min_sources
```

---

## Файлы для изменения

### Новые файлы

1. `src/services/agentic_search/query_enhancer.py` - Multi-Query и Topic Decomposition
2. `src/services/agentic_search/topic_evaluator.py` - Topic-based completeness

### Изменения в существующих

1. `src/services/agentic_models.py` - новые модели:
   - TopicDecomposition
   - MultiQueryExpansion
   - TopicCompleteness

2. `src/services/agentic_search/orchestrator.py` - новый flow:
   - Topic decomposition перед поиском
   - Multi-query retrieval
   - RRF для объединения
   - Gap-driven iteration

3. `src/database/rag_queries.py`:
   - `reciprocal_rank_fusion()` функция
   - `perform_multi_query_search()` - поиск по нескольким queries

4. `src/services/agentic_search/evaluator.py`:
   - Topic-based evaluation вместо общего score

---

## Что осталось исследовать (приблизительно)

1. **Оптимальное количество query variations** - 3 или 4? Больше = лучше recall, но дороже
2. **Threshold для RRF k parameter** - стандарт 60, но может нужно настроить
3. **Как определять complexity запроса** - эвристики или LLM?
4. **Максимальный размер контента для возврата** - сейчас объединяем все чанки URL, может быть слишком много
5. **Caching query decomposition** - одинаковые запросы не декомпозировать повторно
6. **Streaming results** - для длинных исследований показывать прогресс

---

## Пример ожидаемого результата

**Запрос:** "What is Terraform and how to use it?"

**Topic Decomposition:**
1. Definition - "What is Terraform, Infrastructure as Code"
2. Key Concepts - "Terraform providers, resources, state, modules"
3. Installation - "How to install Terraform"
4. Basic Usage - "Terraform init, plan, apply workflow"
5. Code Examples - "Terraform HCL configuration examples"
6. Best Practices - "Terraform best practices, remote state"

**Multi-Query для каждой темы:**
- Definition: ["What is Terraform", "Terraform Infrastructure as Code explained", "Terraform overview"]
- Key Concepts: ["Terraform providers resources state", "Terraform modules explained", "Terraform core concepts"]
- ...

**Результат:**
- 5-7 уникальных источников
- Каждая тема покрыта с evidence
- Примеры кода включены
- 10000-15000 символов полезного контента

---

## Метрики успеха

1. **Topic Coverage** - 100% обязательных тем покрыто
2. **Source Diversity** - минимум 3 разных домена
3. **Code Examples** - присутствуют для технических запросов
4. **No Navigation Junk** - контент полезный, не меню сайта
5. **First-Time Success** - не нужно переспрашивать

---

## Команды для тестирования

```bash
# Запуск сервера
cd ~/Documents/repos/crawl4ai-rag-mcp
pkill -f "python.*src/main.py" 2>/dev/null
QDRANT_URL="http://localhost:6333" \
SEARXNG_URL="http://localhost:8080" \
AGENTIC_SEARCH_ENABLED="true" \
TRANSPORT="http" \
nohup uv run python src/main.py > /tmp/mcp_server.log 2>&1 &

# Тест через MCP клиент
uv run python -c "
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def test():
    async with streamablehttp_client('http://localhost:8051/mcp') as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('agentic_search', {
                'query': 'What is Terraform and how to use it?',
                'completeness_threshold': 0.7,
                'max_iterations': 2,
            })
            import json
            data = json.loads(result.content[0].text)
            print(f'SUCCESS: {data[\"success\"]}')
            print(f'COMPLETENESS: {data[\"completeness\"]}')
            print(f'RESULTS: {len(data[\"results\"])}')
            for r in data['results'][:3]:
                print(f'URL: {r[\"url\"]}')
                print(f'CONTENT: {r[\"content\"][:500]}...')
                print()

asyncio.run(test())
"
```

---

## Зависимости

- Qdrant запущен на localhost:6333
- SearXNG запущен на localhost:8080
- OPENAI_API_KEY установлен
- Python 3.13 с uv

---

## История изменений этой сессии

1. Фикс similarity_score = 0
2. Фикс url_exists exact=False
3. Фикс E2E тестов (await, SearXNG check)
4. Добавлен PruningContentFilter для чистого контента
5. Дедупликация результатов по URL
6. Увеличен размер чанков до 4000
7. Исследование индустрии (Anthropic, Perplexity, You.com, RAG best practices)
8. Создан этот план реализации
