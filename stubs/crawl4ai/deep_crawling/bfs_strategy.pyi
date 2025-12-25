import logging
from . import DeepCrawlStrategy
from ..types import CrawlResult
from ..utils import efficient_normalize_url_for_deep_crawl as efficient_normalize_url_for_deep_crawl
from .filters import FilterChain
from .scorers import URLScorer
from _typeshed import Incomplete

class BFSDeepCrawlStrategy(DeepCrawlStrategy):
    max_depth: Incomplete
    filter_chain: Incomplete
    url_scorer: Incomplete
    include_external: Incomplete
    score_threshold: Incomplete
    max_pages: Incomplete
    logger: Incomplete
    stats: Incomplete
    def __init__(self, max_depth: int, filter_chain: FilterChain = ..., url_scorer: URLScorer | None = None, include_external: bool = False, score_threshold: float = ..., max_pages: int = ..., logger: logging.Logger | None = None) -> None: ...
    async def can_process_url(self, url: str, depth: int) -> bool: ...
    async def link_discovery(self, result: CrawlResult, source_url: str, current_depth: int, visited: set[str], next_level: list[tuple[str, str | None]], depths: dict[str, int]) -> None: ...
    async def shutdown(self) -> None: ...
