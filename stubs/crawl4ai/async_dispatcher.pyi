import abc
import asyncio
from .async_configs import CrawlerRunConfig
from .components.crawler_monitor import CrawlerMonitor as CrawlerMonitor
from .models import CrawlerTaskResult, DomainState
from .types import AsyncWebCrawler
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

class RateLimiter:
    base_delay: Incomplete
    max_delay: Incomplete
    max_retries: Incomplete
    rate_limit_codes: Incomplete
    domains: dict[str, DomainState]
    def __init__(self, base_delay: tuple[float, float] = (1.0, 3.0), max_delay: float = 60.0, max_retries: int = 3, rate_limit_codes: list[int] = None) -> None: ...
    def get_domain(self, url: str) -> str: ...
    async def wait_if_needed(self, url: str) -> None: ...
    def update_delay(self, url: str, status_code: int) -> bool: ...

class BaseDispatcher(ABC, metaclass=abc.ABCMeta):
    crawler: Incomplete
    concurrent_sessions: int
    rate_limiter: Incomplete
    monitor: Incomplete
    def __init__(self, rate_limiter: RateLimiter | None = None, monitor: CrawlerMonitor | None = None) -> None: ...
    def select_config(self, url: str, configs: CrawlerRunConfig | list[CrawlerRunConfig]) -> CrawlerRunConfig | None: ...
    @abstractmethod
    async def crawl_url(self, url: str, config: CrawlerRunConfig | list[CrawlerRunConfig], task_id: str, monitor: CrawlerMonitor | None = None) -> CrawlerTaskResult: ...
    @abstractmethod
    async def run_urls(self, urls: list[str], crawler: AsyncWebCrawler, config: CrawlerRunConfig | list[CrawlerRunConfig], monitor: CrawlerMonitor | None = None) -> list[CrawlerTaskResult]: ...

class MemoryAdaptiveDispatcher(BaseDispatcher):
    memory_threshold_percent: Incomplete
    critical_threshold_percent: Incomplete
    recovery_threshold_percent: Incomplete
    check_interval: Incomplete
    max_session_permit: Incomplete
    fairness_timeout: Incomplete
    memory_wait_timeout: Incomplete
    result_queue: Incomplete
    task_queue: Incomplete
    memory_pressure_mode: bool
    current_memory_percent: float
    def __init__(self, memory_threshold_percent: float = 90.0, critical_threshold_percent: float = 95.0, recovery_threshold_percent: float = 85.0, check_interval: float = 1.0, max_session_permit: int = 20, fairness_timeout: float = 600.0, memory_wait_timeout: float | None = 600.0, rate_limiter: RateLimiter | None = None, monitor: CrawlerMonitor | None = None) -> None: ...
    async def crawl_url(self, url: str, config: CrawlerRunConfig | list[CrawlerRunConfig], task_id: str, retry_count: int = 0) -> CrawlerTaskResult: ...
    crawler: Incomplete
    async def run_urls(self, urls: list[str], crawler: AsyncWebCrawler, config: CrawlerRunConfig | list[CrawlerRunConfig]) -> list[CrawlerTaskResult]: ...
    async def run_urls_stream(self, urls: list[str], crawler: AsyncWebCrawler, config: CrawlerRunConfig | list[CrawlerRunConfig]) -> AsyncGenerator[CrawlerTaskResult, None]: ...

class SemaphoreDispatcher(BaseDispatcher):
    semaphore_count: Incomplete
    max_session_permit: Incomplete
    def __init__(self, semaphore_count: int = 5, max_session_permit: int = 20, rate_limiter: RateLimiter | None = None, monitor: CrawlerMonitor | None = None) -> None: ...
    async def crawl_url(self, url: str, config: CrawlerRunConfig | list[CrawlerRunConfig], task_id: str, semaphore: asyncio.Semaphore = None) -> CrawlerTaskResult: ...
    crawler: Incomplete
    async def run_urls(self, crawler: AsyncWebCrawler, urls: list[str], config: CrawlerRunConfig | list[CrawlerRunConfig]) -> list[CrawlerTaskResult]: ...
