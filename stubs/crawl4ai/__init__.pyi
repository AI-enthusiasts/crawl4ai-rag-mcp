"""Type stubs for crawl4ai library."""

from collections.abc import AsyncIterator
from typing import Any

class BrowserConfig:
    """Browser configuration for crawling."""
    # Accept all kwargs - API changed
    def __init__(self, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class CrawlerRunConfig:
    """Configuration for a crawler run."""
    # Accept all kwargs - API changed
    def __init__(self, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class CacheMode:
    """Cache mode enumeration."""
    ENABLED: str
    DISABLED: str
    READ_ONLY: str
    WRITE_ONLY: str
    BYPASS: str

class MemoryAdaptiveDispatcher:
    """Memory-adaptive task dispatcher."""
    # Accept all kwargs - API changed
    def __init__(self, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class RateLimiter:
    """Rate limiter for controlling request frequency."""
    def __init__(
        self,
        *,
        base_delay: tuple[float, float] = (1.0, 3.0),
        max_delay: float = 60.0,
        **kwargs: Any,
    ) -> None: ...

class AsyncWebCrawler:
    """Asynchronous web crawler."""
    def __init__(
        self,
        *,
        browser_config: BrowserConfig | None = None,
        crawler_config: CrawlerRunConfig | None = None,
        **kwargs: Any,
    ) -> None: ...

    async def __aenter__(self) -> AsyncWebCrawler: ...
    async def __aexit__(self, *args: object) -> None: ...

    async def arun(
        self,
        url: str,
        *,
        config: CrawlerRunConfig | None = None,
        **kwargs: Any,
    ) -> CrawlResult: ...

    async def arun_many(
        self,
        urls: list[str],
        *,
        config: CrawlerRunConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CrawlResult]: ...

class CrawlResult:
    """Result of a crawl operation."""
    url: str
    html: str
    markdown: str
    cleaned_html: str
    success: bool
    error_message: str | None
    status_code: int | None

    def __init__(
        self,
        *,
        url: str,
        html: str = "",
        markdown: str = "",
        cleaned_html: str = "",
        success: bool = True,
        error_message: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None: ...
