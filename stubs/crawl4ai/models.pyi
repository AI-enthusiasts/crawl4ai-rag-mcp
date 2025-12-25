from .ssl_certificate import SSLCertificate
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, HttpUrl as HttpUrl
from typing import Any, AsyncGenerator, Awaitable, Callable, Generic, TypeVar

@dataclass
class DomainState:
    last_request_time: float = ...
    current_delay: float = ...
    fail_count: int = ...

@dataclass
class CrawlerTaskResult:
    task_id: str
    url: str
    result: CrawlResult
    memory_usage: float
    peak_memory: float
    start_time: datetime | float
    end_time: datetime | float
    error_message: str = ...
    retry_count: int = ...
    wait_time: float = ...
    @property
    def success(self) -> bool: ...

class CrawlStatus(Enum):
    QUEUED = 'QUEUED'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'

@dataclass
class CrawlStats:
    task_id: str
    url: str
    status: CrawlStatus
    start_time: datetime | float | None = ...
    end_time: datetime | float | None = ...
    memory_usage: float = ...
    peak_memory: float = ...
    error_message: str = ...
    wait_time: float = ...
    retry_count: int = ...
    counted_requeue: bool = ...
    @property
    def duration(self) -> str: ...

class DisplayMode(Enum):
    DETAILED = 'DETAILED'
    AGGREGATED = 'AGGREGATED'

@dataclass
class TokenUsage:
    completion_tokens: int = ...
    prompt_tokens: int = ...
    total_tokens: int = ...
    completion_tokens_details: dict | None = ...
    prompt_tokens_details: dict | None = ...

class UrlModel(BaseModel):
    url: HttpUrl
    forced: bool

@dataclass
class TraversalStats:
    start_time: datetime = ...
    urls_processed: int = ...
    urls_failed: int = ...
    urls_skipped: int = ...
    total_depth_reached: int = ...
    current_depth: int = ...

class DispatchResult(BaseModel):
    task_id: str
    memory_usage: float
    peak_memory: float
    start_time: datetime | float
    end_time: datetime | float
    error_message: str

class MarkdownGenerationResult(BaseModel):
    raw_markdown: str
    markdown_with_citations: str
    references_markdown: str
    fit_markdown: str | None
    fit_html: str | None

class CrawlResult(BaseModel):
    url: str
    html: str
    fit_html: str | None
    success: bool
    cleaned_html: str | None
    media: dict[str, list[dict]]
    links: dict[str, list[dict]]
    downloaded_files: list[str] | None
    js_execution_result: dict[str, Any] | None
    screenshot: str | None
    pdf: bytes | None
    mhtml: str | None
    extracted_content: str | None
    metadata: dict | None
    error_message: str | None
    session_id: str | None
    response_headers: dict | None
    status_code: int | None
    ssl_certificate: SSLCertificate | None
    dispatch_result: DispatchResult | None
    redirected_url: str | None
    network_requests: list[dict[str, Any]] | None
    console_messages: list[dict[str, Any]] | None
    tables: list[dict]
    class Config:
        arbitrary_types_allowed: bool
    def __init__(self, **data) -> None: ...
    @property
    def markdown(self): ...
    @markdown.setter
    def markdown(self, value) -> None: ...
    @property
    def markdown_v2(self) -> None: ...
    @property
    def fit_markdown(self) -> None: ...
    @property
    def fit_html(self) -> None: ...
    def model_dump(self, *args, **kwargs): ...

class StringCompatibleMarkdown(str):
    def __new__(cls, markdown_result): ...
    def __init__(self, markdown_result) -> None: ...
    def __getattr__(self, name): ...
CrawlResultT = TypeVar('CrawlResultT', bound=CrawlResult)

class CrawlResultContainer(Generic[CrawlResultT]):
    def __init__(self, results: CrawlResultT | list[CrawlResultT]) -> None: ...
    def __iter__(self): ...
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...
    def __getattr__(self, attr): ...
RunManyReturn = CrawlResultContainer[CrawlResultT] | AsyncGenerator[CrawlResultT, None]

class AsyncCrawlResponse(BaseModel):
    html: str
    response_headers: dict[str, str]
    js_execution_result: dict[str, Any] | None
    status_code: int
    screenshot: str | None
    pdf_data: bytes | None
    mhtml_data: str | None
    get_delayed_content: Callable[[float | None], Awaitable[str]] | None
    downloaded_files: list[str] | None
    ssl_certificate: SSLCertificate | None
    redirected_url: str | None
    network_requests: list[dict[str, Any]] | None
    console_messages: list[dict[str, Any]] | None
    class Config:
        arbitrary_types_allowed: bool

class MediaItem(BaseModel):
    src: str | None
    data: str | None
    alt: str | None
    desc: str | None
    score: int | None
    type: str
    group_id: int | None
    format: str | None
    width: int | None

class Link(BaseModel):
    href: str | None
    text: str | None
    title: str | None
    base_domain: str | None
    head_data: dict[str, Any] | None
    head_extraction_status: str | None
    head_extraction_error: str | None
    intrinsic_score: float | None
    contextual_score: float | None
    total_score: float | None

class Media(BaseModel):
    images: list[MediaItem]
    videos: list[MediaItem]
    audios: list[MediaItem]
    tables: list[dict]

class Links(BaseModel):
    internal: list[Link]
    external: list[Link]

class ScrapingResult(BaseModel):
    cleaned_html: str
    success: bool
    media: Media
    links: Links
    metadata: dict[str, Any]
