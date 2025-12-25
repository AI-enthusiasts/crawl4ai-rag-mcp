import abc
from .async_logger import AsyncLogger
from .types import LLMConfig
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag

class RelevantContentFilter(ABC, metaclass=abc.ABCMeta):
    user_query: Incomplete
    included_tags: Incomplete
    excluded_tags: Incomplete
    header_tags: Incomplete
    negative_patterns: Incomplete
    min_word_count: int
    verbose: bool
    logger: Incomplete
    def __init__(self, user_query: str = None, verbose: bool = False, logger: AsyncLogger | None = None) -> None: ...
    @abstractmethod
    def filter_content(self, html: str) -> list[str]: ...
    def extract_page_query(self, soup: BeautifulSoup, body: Tag) -> str: ...
    def extract_text_chunks(self, body: Tag, min_word_threshold: int = None) -> list[tuple[str, str]]: ...
    def is_excluded(self, tag: Tag) -> bool: ...
    def clean_element(self, tag: Tag) -> str: ...

class BM25ContentFilter(RelevantContentFilter):
    bm25_threshold: Incomplete
    use_stemming: Incomplete
    priority_tags: Incomplete
    stemmer: Incomplete
    def __init__(self, user_query: str = None, bm25_threshold: float = 1.0, language: str = 'english', use_stemming: bool = True) -> None: ...
    def filter_content(self, html: str, min_word_threshold: int = None) -> list[str]: ...

class PruningContentFilter(RelevantContentFilter):
    min_word_threshold: Incomplete
    threshold_type: Incomplete
    threshold: Incomplete
    tag_importance: Incomplete
    metric_config: Incomplete
    metric_weights: Incomplete
    tag_weights: Incomplete
    def __init__(self, user_query: str = None, min_word_threshold: int = None, threshold_type: str = 'fixed', threshold: float = 0.48) -> None: ...
    def filter_content(self, html: str, min_word_threshold: int = None) -> list[str]: ...

class LLMContentFilter(RelevantContentFilter):
    provider: Incomplete
    api_token: Incomplete
    base_url: Incomplete
    llm_config: Incomplete
    instruction: Incomplete
    chunk_token_threshold: Incomplete
    overlap_rate: Incomplete
    word_token_rate: Incomplete
    token_rate: Incomplete
    extra_args: Incomplete
    ignore_cache: Incomplete
    verbose: Incomplete
    logger: Incomplete
    usages: Incomplete
    total_usage: Incomplete
    def __init__(self, llm_config: LLMConfig = None, instruction: str = None, chunk_token_threshold: int = ..., overlap_rate: float = ..., word_token_rate: float = ..., verbose: bool = False, logger: AsyncLogger | None = None, ignore_cache: bool = True, provider: str = ..., api_token: str | None = None, base_url: str | None = None, api_base: str | None = None, extra_args: dict = None) -> None: ...
    def __setattr__(self, name, value) -> None: ...
    def filter_content(self, html: str, ignore_cache: bool = True) -> list[str]: ...
    def show_usage(self) -> None: ...
