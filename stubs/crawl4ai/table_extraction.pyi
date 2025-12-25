import abc
from .types import LLMConfig
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from lxml import etree
from typing import Any

class TableExtractionStrategy(ABC, metaclass=abc.ABCMeta):
    verbose: Incomplete
    logger: Incomplete
    def __init__(self, **kwargs) -> None: ...
    @abstractmethod
    def extract_tables(self, element: etree.Element, **kwargs) -> list[dict[str, Any]]: ...

class DefaultTableExtraction(TableExtractionStrategy):
    table_score_threshold: Incomplete
    min_rows: Incomplete
    min_cols: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def extract_tables(self, element: etree.Element, **kwargs) -> list[dict[str, Any]]: ...
    def is_data_table(self, table: etree.Element, **kwargs) -> bool: ...
    def extract_table_data(self, table: etree.Element) -> dict[str, Any]: ...

class NoTableExtraction(TableExtractionStrategy):
    def extract_tables(self, element: etree.Element, **kwargs) -> list[dict[str, Any]]: ...

class LLMTableExtraction(TableExtractionStrategy):
    TABLE_EXTRACTION_PROMPT: str
    llm_config: Incomplete
    css_selector: Incomplete
    max_tries: Incomplete
    enable_chunking: Incomplete
    chunk_token_threshold: Incomplete
    min_rows_per_chunk: Incomplete
    max_parallel_chunks: Incomplete
    extra_args: Incomplete
    def __init__(self, llm_config: LLMConfig | None = None, css_selector: str | None = None, max_tries: int = 3, enable_chunking: bool = True, chunk_token_threshold: int = 3000, min_rows_per_chunk: int = 10, max_parallel_chunks: int = 5, verbose: bool = False, **kwargs) -> None: ...
    def extract_tables(self, element: etree.Element, **kwargs) -> list[dict[str, Any]]: ...
