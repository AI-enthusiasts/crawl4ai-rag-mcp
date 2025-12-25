import logging
from . import DeepCrawlStrategy
from ..types import AsyncWebCrawler, CrawlResult, CrawlerRunConfig, RunManyReturn as RunManyReturn
from .filters import FilterChain
from .scorers import URLScorer
from _typeshed import Incomplete

BATCH_SIZE: int

class BestFirstCrawlingStrategy(DeepCrawlStrategy):
    max_depth: Incomplete
    filter_chain: Incomplete
    url_scorer: Incomplete
    include_external: Incomplete
    max_pages: Incomplete
    logger: Incomplete
    stats: Incomplete
    def __init__(self, max_depth: int, filter_chain: FilterChain = ..., url_scorer: URLScorer | None = None, include_external: bool = False, max_pages: int = ..., logger: logging.Logger | None = None) -> None: ...
    async def can_process_url(self, url: str, depth: int) -> bool: ...
    async def link_discovery(self, result: CrawlResult, source_url: str, current_depth: int, visited: set[str], next_links: list[tuple[str, str | None]], depths: dict[str, int]) -> None: ...
    async def arun(self, start_url: str, crawler: AsyncWebCrawler, config: CrawlerRunConfig | None = None) -> RunManyReturn: ...
    async def shutdown(self) -> None: ...
