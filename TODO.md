# ⚠️ ВАЖНО: ПРАВИЛА РАБОТЫ С TODO

**ПРИ ЗАВЕРШЕНИИ ЗАДАЧИ НА 100% - УДАЛЯЙ ЕЕ ИЗ ЭТОГО ФАЙЛА!**

**НЕ ПИШИ НИГДЕ О ЗАВЕРШЕНИИ! ПРОСТО УДАЛИ ПУНКТ!**

**ЭТО ДИНАМИЧНЫЙ ФАЙЛ - ТОЛЬКО АКТИВНЫЕ ЗАДАЧИ!**

---

## 🎯 Активный План (Недели 5-6 из 6)

### 1. 🔴 Test Coverage: Services (P3 - Week 5)
- **Файлы:** `tests/services/`
- **Что:** Services имеют <10% покрытие
- **Надо:**
  - `test_agentic_search.py` - тесты для Pydantic AI агентов
  - `test_crawling.py` - тесты для Crawl4AI интеграции
  - `test_search.py` - тесты для SearXNG
  - `test_validated_search.py` - тесты для Neo4j валидации
  - `test_smart_crawl.py` - тесты для умного краулинга
- **Цель:** Services 10% → 80% (12h)
- **Блокер:** Нет

### 2. 🔴 Test Coverage: Knowledge Graph (P3 - Week 5)
- **Файлы:** `tests/knowledge_graph/`
- **Что:** Knowledge graph имеет <5% покрытие
- **Надо:**
  - `test_parse_repo.py` - тесты для парсинга репозиториев
  - `test_git_manager.py` - тесты для Git операций
  - `test_code_extractor.py` - тесты для извлечения кода
  - `test_validators.py` - тесты для валидаторов
  - `test_analyzers/` - тесты для Python/JS/Go анализаторов
- **Цель:** Knowledge graph 5% → 80% (16h)
- **Блокер:** Нет

### 3. 🟡 Test Coverage: Database (P3 - Week 5)
- **Файлы:** `tests/database/`
- **Что:** Database адаптеры имеют ~60% покрытие
- **Надо:**
  - `test_qdrant_operations.py` - тесты для всех CRUD операций
  - `test_qdrant_search.py` - тесты для поиска и фильтров
  - `test_qdrant_code_examples.py` - тесты для code examples
  - `test_supabase_adapter.py` - тесты для legacy Supabase
- **Цель:** Database 60% → 85% (4h)
- **Блокер:** Нет

### 4. 🟡 Test Coverage: Tools (P3 - Week 6)
- **Файлы:** `tests/tools/`
- **Что:** MCP tools имеют ~10% покрытие
- **Надо:**
  - `test_search_tools.py` - search, agentic_search, analyze_code
  - `test_crawl_tools.py` - scrape_urls, smart_crawl_url
  - `test_rag_tools.py` - get_available_sources, perform_rag_query
  - `test_kg_tools.py` - query_knowledge_graph, parse_github_repository
  - `test_validation_tools.py` - check_hallucinations, extract_and_index
- **Цель:** Tools 10% → 60% (10h)
- **Блокер:** Нет

### 5. 🟡 Test Coverage: Utils (P3 - Week 6)
- **Файлы:** `tests/utils/`
- **Что:** Utils имеют ~20% покрытие
- **Надо:**
  - `test_embeddings.py` - тесты для генерации embeddings
  - `test_url_helpers.py` - тесты для URL парсинга
  - `test_text_processing.py` - тесты для chunking/processing
- **Цель:** Utils 20% → 80% (8h)
- **Блокер:** Нет

### 6. 🟢 CI/CD: pytest-cov enforcement (P3 - Week 6)
- **Файл:** `.github/workflows/tests.yml`
- **Что:** Нет CI проверки покрытия
- **Надо:**
  - GitHub Actions workflow для pytest-cov
  - Fail if coverage <80%
  - Badge в README.md
- **Цель:** Автоматическая проверка покрытия
- **Блокер:** Tests должны быть написаны

---

## 📋 Backlog (После Week 6)

### File Size Refactoring (Оставшиеся файлы >400 LOC)
- `src/knowledge_graph/knowledge_graph_validator.py` - 1020 LOC → разбить на модули
- Еще 13 файлов >400 LOC (см. PROJECT_ROADMAP.md)
- **Цель:** 27 файлов → 14 файлов >400 LOC

### Performance Optimization
- Профилирование медленных операций
- Оптимизация Neo4j запросов
- Кэширование embeddings

### Documentation
- API documentation (Sphinx/MkDocs)
- User guide для MCP tools
- Architecture diagrams

---

## 🔄 Текущий Фокус

**Week 5-6 из 6: Test Coverage**
- **Цель:** >80% coverage
- **Метод:** Real integrations с VCR.py (no mocks)
- **Стратегия:** Services → Knowledge Graph → Database → Tools → Utils
- **Прогресс:** 0% → 80% (Week 5-6)

**Roadmap Progress:**
- ✅ Week 1-2: File Refactoring (COMPLETE)
- ✅ Week 3: Type Safety (COMPLETE - 89% reduction)
- ✅ Week 4: Exception Handling (COMPLETE - 93% reduction)
- 🔴 Week 5-6: Test Coverage (IN PROGRESS - target 80%)
