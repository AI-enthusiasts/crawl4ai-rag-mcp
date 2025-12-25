from .chunking_strategy import *
from .content_filter_strategy import *
from .extraction_strategy import *
from .async_dispatcher import *
from .async_configs import BrowserConfig, CrawlerRunConfig, SeedingConfig
from .async_crawler_strategy import AsyncCrawlerStrategy
from .async_dispatcher import BaseDispatcher
from .async_logger import AsyncLoggerBase
from .async_url_seeder import AsyncUrlSeeder
from .models import CrawlResult, RunManyReturn as RunManyReturn
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any

class AsyncWebCrawler:
    browser_config: Incomplete
    logger: Incomplete
    crawler_strategy: Incomplete
    crawl4ai_folder: Incomplete
    robots_parser: Incomplete
    ready: bool
    url_seeder: AsyncUrlSeeder | None
    def __init__(self, crawler_strategy: AsyncCrawlerStrategy = None, config: BrowserConfig = None, base_directory: str = ..., thread_safe: bool = False, logger: AsyncLoggerBase = None, **kwargs) -> None: ...
    async def start(self): ...
    async def close(self) -> None: ...
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
    @asynccontextmanager
    async def nullcontext(self) -> Generator[None]: ...
    async def arun(self, url: str, config: CrawlerRunConfig = None, **kwargs) -> RunManyReturn: ...
    async def aprocess_html(self, url: str, html: str, extracted_content: str, config: CrawlerRunConfig, screenshot_data: str, pdf_data: str, verbose: bool, **kwargs) -> CrawlResult: ...
    async def arun_many(self, urls: list[str], config: Union[CrawlerRunConfig, list[CrawlerRunConfig]] | None = None, dispatcher: BaseDispatcher | None = None, **kwargs) -> RunManyReturn: ...
    async def aseed_urls(self, domain_or_domains: Union[str, list[str]], config: SeedingConfig | None = None, **kwargs) -> Union[list[str], Dict[str, list[Union[str, Dict[str, Any]]]]]: ...
