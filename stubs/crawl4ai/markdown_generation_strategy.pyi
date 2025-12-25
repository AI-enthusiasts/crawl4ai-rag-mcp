import abc
from .content_filter_strategy import RelevantContentFilter
from .models import MarkdownGenerationResult
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import Any

LINK_PATTERN: Incomplete

def fast_urljoin(base: str, url: str) -> str: ...

class MarkdownGenerationStrategy(ABC, metaclass=abc.ABCMeta):
    content_filter: Incomplete
    options: Incomplete
    verbose: Incomplete
    content_source: Incomplete
    def __init__(self, content_filter: RelevantContentFilter | None = None, options: dict[str, Any] | None = None, verbose: bool = False, content_source: str = 'cleaned_html') -> None: ...
    @abstractmethod
    def generate_markdown(self, input_html: str, base_url: str = '', html2text_options: dict[str, Any] | None = None, content_filter: RelevantContentFilter | None = None, citations: bool = True, **kwargs) -> MarkdownGenerationResult: ...

class DefaultMarkdownGenerator(MarkdownGenerationStrategy):
    def __init__(self, content_filter: RelevantContentFilter | None = None, options: dict[str, Any] | None = None, content_source: str = 'cleaned_html') -> None: ...
    def convert_links_to_citations(self, markdown: str, base_url: str = '') -> tuple[str, str]: ...
    def generate_markdown(self, input_html: str, base_url: str = '', html2text_options: dict[str, Any] | None = None, options: dict[str, Any] | None = None, content_filter: RelevantContentFilter | None = None, citations: bool = True, **kwargs) -> MarkdownGenerationResult: ...
