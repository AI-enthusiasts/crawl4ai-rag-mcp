import httpx
import pathlib
from .async_configs import SeedingConfig
from .async_logger import AsyncLogger as AsyncLogger, AsyncLoggerBase
from _typeshed import Incomplete
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

LXML: bool
HAS_BROTLI: bool
HAS_BM25: bool
COLLINFO_URL: str
TTL: Incomplete

class AsyncUrlSeeder:
    ttl: Incomplete
    client: Incomplete
    logger: Incomplete
    base_directory: Incomplete
    cache_dir: Incomplete
    index_cache_path: Incomplete
    index_id: str | None
    cache_root: Incomplete
    def __init__(self, ttl: timedelta = ..., client: httpx.AsyncClient | None = None, logger: AsyncLoggerBase | None = None, base_directory: str | pathlib.Path | None = None, cache_root: str | Path | None = None) -> None: ...
    force: Incomplete
    async def urls(self, domain: str, config: SeedingConfig) -> list[dict[str, Any]]: ...
    async def many_urls(self, domains: Sequence[str], config: SeedingConfig) -> dict[str, list[dict[str, Any]]]: ...
    async def extract_head_for_urls(self, urls: list[str], config: SeedingConfig | None = None, concurrency: int = 10, timeout: int = 5) -> list[dict[str, Any]]: ...
    async def close(self) -> None: ...
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...
