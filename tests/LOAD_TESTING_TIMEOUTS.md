# Load Testing Timeouts Configuration

## Текущие настройки таймаутов

### MCPLoadTester (tests/integration/test_mcp_load_testing.py)

```python
class MCPLoadTester:
    def __init__(self, init_timeout: float = 5.0, timeout: float = 30.0, tool_timeout: float = 60.0):
        """
        Args:
            init_timeout: Connection initialization timeout (default: 5s)
            timeout: General operation timeout (default: 30s)
            tool_timeout: Tool invocation timeout (default: 60s)
        """
```

### Применение таймаутов

#### 1. Client Initialization
```python
self.client = Client(
    self.mcp_url,
    auth=BearerAuth(self.mcp_api_key) if self.mcp_api_key else None,
    timeout=self.timeout,           # 30s - общий таймаут
    init_timeout=self.init_timeout  # 5s - таймаут инициализации соединения
)
```

#### 2. Tool Invocation
```python
result = await self.client.call_tool(
    tool_name, 
    args or {}, 
    raise_on_error=False,
    timeout=self.tool_timeout  # 60s - таймаут вызова инструмента
)
```

## Рекомендуемые таймауты по типам операций

### Быстрые операции (< 5s)
- `get_available_sources` - получение списка источников
- `list_tools` - получение списка инструментов
- Простые запросы без обработки

**Рекомендуемый таймаут:** 10s

### Средние операции (5-30s)
- `perform_rag_query` - RAG запросы с небольшим match_count
- `search` с малым количеством результатов (1-3)

**Рекомендуемый таймаут:** 30s

### Долгие операции (30-120s)
- `scrape_urls` - скрапинг одного URL
- `search` с большим количеством результатов (5-10)
- `smart_crawl_url` с небольшой глубиной

**Рекомендуемый таймаут:** 60-120s

### Очень долгие операции (> 120s)
- `batch_scrape_urls` - пакетный скрапинг
- `smart_crawl_url` с большой глубиной
- Операции с большим количеством URL

**Рекомендуемый таймаут:** 300s (5 минут)

## Настройка таймаутов для разных тестов

### Throughput Tests
```python
async with MCPLoadTester(init_timeout=5.0, timeout=30.0, tool_timeout=60.0) as tester:
    # Быстрое подключение, средние операции
```

### Endurance Tests
```python
async with MCPLoadTester(init_timeout=10.0, timeout=60.0, tool_timeout=180.0) as tester:
    # Долгие операции для endurance тестов
```

## Обработка таймаутов

### В коде тестов

```python
try:
    result = await self.client.call_tool(
        tool_name, 
        args, 
        timeout=self.tool_timeout
    )
except asyncio.TimeoutError:
    return {
        "success": False,
        "error": f"Request timeout after {self.tool_timeout}s",
        "tool": tool_name,
        "args": args,
    }
```

### В Docker логах

После теста автоматически собираются логи Docker контейнера:
- Проверка на наличие "timeout" в логах
- Подсчет количества таймаутов
- Сохранение в `tests/results/docker_logs/`

## Проблемы и решения

### Проблема: 504 Gateway Timeout

**Симптомы:**
```
httpx.HTTPStatusError: Server error '504 Gateway Timeout' for url 'https://rag.melo.eu.org/mcp'
```

**Причины:**
1. Сервер перегружен
2. Холодный старт контейнера
3. Таймаут прокси/балансировщика

**Решения:**
1. Увеличить `init_timeout` до 10-30s (по умолчанию 5s)
2. Проверить логи Docker контейнера (автоматически через `docker_logs_collector`)
3. Убедиться, что сервер запущен и доступен
4. Проверить сетевое подключение

### Проблема: Tool timeout

**Симптомы:**
```
asyncio.TimeoutError: Request timeout after 60s
```

**Причины:**
1. Операция действительно долгая
2. Сервер завис
3. Deadlock в коде

**Решения:**
1. Увеличить `tool_timeout` для конкретной операции
2. Проверить Docker логи на ошибки
3. Уменьшить нагрузку (concurrency)
4. Оптимизировать запрос (меньше результатов)

## Мониторинг таймаутов

### Автоматический сбор метрик

Фикстура `docker_logs_collector` автоматически:
1. Записывает время начала теста
2. Собирает логи Docker за период теста
3. Анализирует логи на наличие:
   - ERROR
   - WARNING
   - timeout/Timeout
4. Сохраняет отчет в `tests/results/docker_logs/`

### Использование

```python
@pytest.mark.usefixtures("docker_logs_collector")
async def test_search_tool_throughput(self, load_tester, performance_thresholds):
    # Тест автоматически соберет логи
    ...
```

## Best Practices

1. **Всегда устанавливайте таймауты** - не полагайтесь на defaults
2. **Разные таймауты для разных операций** - используйте параметр `timeout` в `call_tool()`
3. **Логируйте таймауты** - используйте `docker_logs_collector`
4. **Увеличивайте постепенно** - начните с консервативных значений
5. **Мониторьте метрики** - проверяйте P95/P99 latency
6. **Warm-up** - делайте пробный запрос перед нагрузочными тестами

## Примеры конфигурации

### Для локального тестирования
```python
# Быстрые таймауты для локального Docker
MCPLoadTester(init_timeout=3.0, timeout=10.0, tool_timeout=30.0)
```

### Для удаленного сервера (по умолчанию)
```python
# Стандартные таймауты для сети
MCPLoadTester(init_timeout=5.0, timeout=30.0, tool_timeout=60.0)
```

### Для production нагрузочных тестов
```python
# Консервативные таймауты
MCPLoadTester(init_timeout=10.0, timeout=60.0, tool_timeout=120.0)
```

## Ссылки

- [FastMCP Client Docs](https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx)
- [FastMCP Timeouts](https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
