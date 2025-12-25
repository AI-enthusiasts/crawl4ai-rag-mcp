from .cache_context import CacheMode
from .chunking_strategy import ChunkingStrategy
from .config import PROVIDER_MODELS as PROVIDER_MODELS
from .content_scraping_strategy import ContentScrapingStrategy
from .deep_crawling import DeepCrawlStrategy
from .extraction_strategy import ExtractionStrategy, LLMExtractionStrategy as LLMExtractionStrategy
from .markdown_generation_strategy import MarkdownGenerationStrategy
from .proxy_strategy import ProxyRotationStrategy
from .table_extraction import TableExtractionStrategy
from _typeshed import Incomplete
from enum import Enum
from typing import Any, Callable

UrlMatcher = str | Callable[[str], bool] | list[str | Callable[[str], bool]]

class MatchMode(Enum):
    OR = 'or'
    AND = 'and'

def to_serializable_dict(obj: Any, ignore_default_value: bool = False) -> dict: ...
def from_serializable_dict(data: Any) -> Any: ...
def is_empty_value(value: Any) -> bool: ...

class GeolocationConfig:
    latitude: Incomplete
    longitude: Incomplete
    accuracy: Incomplete
    def __init__(self, latitude: float, longitude: float, accuracy: float | None = 0.0) -> None: ...
    @staticmethod
    def from_dict(geo_dict: dict) -> GeolocationConfig: ...
    def to_dict(self) -> dict: ...
    def clone(self, **kwargs) -> GeolocationConfig: ...

class ProxyConfig:
    server: Incomplete
    username: Incomplete
    password: Incomplete
    ip: Incomplete
    def __init__(self, server: str, username: str | None = None, password: str | None = None, ip: str | None = None) -> None: ...
    @staticmethod
    def from_string(proxy_str: str) -> ProxyConfig: ...
    @staticmethod
    def from_dict(proxy_dict: dict) -> ProxyConfig: ...
    @staticmethod
    def from_env(env_var: str = 'PROXIES') -> list['ProxyConfig']: ...
    def to_dict(self) -> dict: ...
    def clone(self, **kwargs) -> ProxyConfig: ...

class BrowserConfig:
    browser_type: Incomplete
    headless: Incomplete
    browser_mode: Incomplete
    use_managed_browser: Incomplete
    cdp_url: Incomplete
    use_persistent_context: Incomplete
    user_data_dir: Incomplete
    chrome_channel: Incomplete
    channel: Incomplete
    proxy: Incomplete
    proxy_config: Incomplete
    viewport_width: Incomplete
    viewport_height: Incomplete
    viewport: Incomplete
    accept_downloads: Incomplete
    downloads_path: Incomplete
    storage_state: Incomplete
    ignore_https_errors: Incomplete
    java_script_enabled: Incomplete
    cookies: Incomplete
    headers: Incomplete
    user_agent: Incomplete
    user_agent_mode: Incomplete
    user_agent_generator_config: Incomplete
    text_mode: Incomplete
    light_mode: Incomplete
    extra_args: Incomplete
    sleep_on_close: Incomplete
    verbose: Incomplete
    debugging_port: Incomplete
    host: Incomplete
    enable_stealth: Incomplete
    browser_hint: Incomplete
    def __init__(self, browser_type: str = 'chromium', headless: bool = True, browser_mode: str = 'dedicated', use_managed_browser: bool = False, cdp_url: str = None, use_persistent_context: bool = False, user_data_dir: str = None, chrome_channel: str = 'chromium', channel: str = 'chromium', proxy: str = None, proxy_config: ProxyConfig | dict | None = None, viewport_width: int = 1080, viewport_height: int = 600, viewport: dict = None, accept_downloads: bool = False, downloads_path: str = None, storage_state: str | dict | None = None, ignore_https_errors: bool = True, java_script_enabled: bool = True, sleep_on_close: bool = False, verbose: bool = True, cookies: list = None, headers: dict = None, user_agent: str = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36', user_agent_mode: str = '', user_agent_generator_config: dict = {}, text_mode: bool = False, light_mode: bool = False, extra_args: list = None, debugging_port: int = 9222, host: str = 'localhost', enable_stealth: bool = False) -> None: ...
    @staticmethod
    def from_kwargs(kwargs: dict) -> BrowserConfig: ...
    def to_dict(self): ...
    def clone(self, **kwargs): ...
    def dump(self) -> dict: ...
    @staticmethod
    def load(data: dict) -> BrowserConfig: ...

class VirtualScrollConfig:
    container_selector: Incomplete
    scroll_count: Incomplete
    scroll_by: Incomplete
    wait_after_scroll: Incomplete
    def __init__(self, container_selector: str, scroll_count: int = 10, scroll_by: str | int = 'container_height', wait_after_scroll: float = 0.5) -> None: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> VirtualScrollConfig: ...

class LinkPreviewConfig:
    include_internal: Incomplete
    include_external: Incomplete
    include_patterns: Incomplete
    exclude_patterns: Incomplete
    concurrency: Incomplete
    timeout: Incomplete
    max_links: Incomplete
    query: Incomplete
    score_threshold: Incomplete
    verbose: Incomplete
    def __init__(self, include_internal: bool = True, include_external: bool = False, include_patterns: list[str] | None = None, exclude_patterns: list[str] | None = None, concurrency: int = 10, timeout: int = 5, max_links: int = 100, query: str | None = None, score_threshold: float | None = None, verbose: bool = False) -> None: ...
    @staticmethod
    def from_dict(config_dict: dict[str, Any]) -> LinkPreviewConfig: ...
    def to_dict(self) -> dict[str, Any]: ...
    def clone(self, **kwargs) -> LinkPreviewConfig: ...

class HTTPCrawlerConfig:
    method: str
    headers: dict[str, str] | None
    data: dict[str, Any] | None
    json: dict[str, Any] | None
    follow_redirects: bool
    verify_ssl: bool
    def __init__(self, method: str = 'GET', headers: dict[str, str] | None = None, data: dict[str, Any] | None = None, json: dict[str, Any] | None = None, follow_redirects: bool = True, verify_ssl: bool = True) -> None: ...
    @staticmethod
    def from_kwargs(kwargs: dict) -> HTTPCrawlerConfig: ...
    def to_dict(self): ...
    def clone(self, **kwargs): ...
    def dump(self) -> dict: ...
    @staticmethod
    def load(data: dict) -> HTTPCrawlerConfig: ...

class CrawlerRunConfig:
    url: Incomplete
    word_count_threshold: Incomplete
    extraction_strategy: Incomplete
    chunking_strategy: Incomplete
    markdown_generator: Incomplete
    only_text: Incomplete
    css_selector: Incomplete
    target_elements: Incomplete
    excluded_tags: Incomplete
    excluded_selector: Incomplete
    keep_data_attributes: Incomplete
    keep_attrs: Incomplete
    remove_forms: Incomplete
    prettiify: Incomplete
    parser_type: Incomplete
    scraping_strategy: Incomplete
    proxy_config: Incomplete
    proxy_rotation_strategy: Incomplete
    locale: Incomplete
    timezone_id: Incomplete
    geolocation: Incomplete
    fetch_ssl_certificate: Incomplete
    cache_mode: Incomplete
    session_id: Incomplete
    bypass_cache: Incomplete
    disable_cache: Incomplete
    no_cache_read: Incomplete
    no_cache_write: Incomplete
    shared_data: Incomplete
    wait_until: Incomplete
    page_timeout: Incomplete
    wait_for: Incomplete
    wait_for_timeout: Incomplete
    wait_for_images: Incomplete
    delay_before_return_html: Incomplete
    mean_delay: Incomplete
    max_range: Incomplete
    semaphore_count: Incomplete
    js_code: Incomplete
    c4a_script: Incomplete
    js_only: Incomplete
    ignore_body_visibility: Incomplete
    scan_full_page: Incomplete
    scroll_delay: Incomplete
    max_scroll_steps: Incomplete
    process_iframes: Incomplete
    remove_overlay_elements: Incomplete
    simulate_user: Incomplete
    override_navigator: Incomplete
    magic: Incomplete
    adjust_viewport_to_content: Incomplete
    screenshot: Incomplete
    screenshot_wait_for: Incomplete
    screenshot_height_threshold: Incomplete
    pdf: Incomplete
    capture_mhtml: Incomplete
    image_description_min_word_threshold: Incomplete
    image_score_threshold: Incomplete
    exclude_external_images: Incomplete
    exclude_all_images: Incomplete
    table_score_threshold: Incomplete
    table_extraction: Incomplete
    exclude_social_media_domains: Incomplete
    exclude_external_links: Incomplete
    exclude_social_media_links: Incomplete
    exclude_domains: Incomplete
    exclude_internal_links: Incomplete
    score_links: Incomplete
    preserve_https_for_internal_links: Incomplete
    verbose: Incomplete
    log_console: Incomplete
    capture_network_requests: Incomplete
    capture_console_messages: Incomplete
    stream: Incomplete
    method: Incomplete
    check_robots_txt: Incomplete
    user_agent: Incomplete
    user_agent_mode: Incomplete
    user_agent_generator_config: Incomplete
    deep_crawl_strategy: Incomplete
    link_preview_config: Incomplete
    virtual_scroll_config: Incomplete
    url_matcher: Incomplete
    match_mode: Incomplete
    experimental: Incomplete
    def __init__(self, word_count_threshold: int = ..., extraction_strategy: ExtractionStrategy = None, chunking_strategy: ChunkingStrategy = ..., markdown_generator: MarkdownGenerationStrategy = ..., only_text: bool = False, css_selector: str = None, target_elements: list[str] = None, excluded_tags: list = None, excluded_selector: str = None, keep_data_attributes: bool = False, keep_attrs: list = None, remove_forms: bool = False, prettiify: bool = False, parser_type: str = 'lxml', scraping_strategy: ContentScrapingStrategy = None, proxy_config: ProxyConfig | dict | None = None, proxy_rotation_strategy: ProxyRotationStrategy | None = None, locale: str | None = None, timezone_id: str | None = None, geolocation: GeolocationConfig | None = None, fetch_ssl_certificate: bool = False, cache_mode: CacheMode = ..., session_id: str = None, bypass_cache: bool = False, disable_cache: bool = False, no_cache_read: bool = False, no_cache_write: bool = False, shared_data: dict = None, wait_until: str = 'domcontentloaded', page_timeout: int = ..., wait_for: str = None, wait_for_timeout: int = None, wait_for_images: bool = False, delay_before_return_html: float = 0.1, mean_delay: float = 0.1, max_range: float = 0.3, semaphore_count: int = 5, js_code: str | list[str] = None, c4a_script: str | list[str] = None, js_only: bool = False, ignore_body_visibility: bool = True, scan_full_page: bool = False, scroll_delay: float = 0.2, max_scroll_steps: int | None = None, process_iframes: bool = False, remove_overlay_elements: bool = False, simulate_user: bool = False, override_navigator: bool = False, magic: bool = False, adjust_viewport_to_content: bool = False, screenshot: bool = False, screenshot_wait_for: float = None, screenshot_height_threshold: int = ..., pdf: bool = False, capture_mhtml: bool = False, image_description_min_word_threshold: int = ..., image_score_threshold: int = ..., table_score_threshold: int = 7, table_extraction: TableExtractionStrategy = None, exclude_external_images: bool = False, exclude_all_images: bool = False, exclude_social_media_domains: list = None, exclude_external_links: bool = False, exclude_social_media_links: bool = False, exclude_domains: list = None, exclude_internal_links: bool = False, score_links: bool = False, preserve_https_for_internal_links: bool = False, verbose: bool = True, log_console: bool = False, capture_network_requests: bool = False, capture_console_messages: bool = False, method: str = 'GET', stream: bool = False, url: str = None, check_robots_txt: bool = False, user_agent: str = None, user_agent_mode: str = None, user_agent_generator_config: dict = {}, deep_crawl_strategy: DeepCrawlStrategy | None = None, link_preview_config: LinkPreviewConfig | dict[str, Any] = None, virtual_scroll_config: VirtualScrollConfig | dict[str, Any] = None, url_matcher: UrlMatcher | None = None, match_mode: MatchMode = ..., experimental: dict[str, Any] = None) -> None: ...
    def is_match(self, url: str) -> bool: ...
    def __getattr__(self, name) -> None: ...
    def __setattr__(self, name, value) -> None: ...
    @staticmethod
    def from_kwargs(kwargs: dict) -> CrawlerRunConfig: ...
    def dump(self) -> dict: ...
    @staticmethod
    def load(data: dict) -> CrawlerRunConfig: ...
    def to_dict(self): ...
    def clone(self, **kwargs): ...

class LLMConfig:
    provider: Incomplete
    api_token: Incomplete
    base_url: Incomplete
    temperature: Incomplete
    max_tokens: Incomplete
    top_p: Incomplete
    frequency_penalty: Incomplete
    presence_penalty: Incomplete
    stop: Incomplete
    n: Incomplete
    def __init__(self, provider: str = ..., api_token: str | None = None, base_url: str | None = None, temperature: float | None = None, max_tokens: int | None = None, top_p: float | None = None, frequency_penalty: float | None = None, presence_penalty: float | None = None, stop: list[str] | None = None, n: int | None = None) -> None: ...
    @staticmethod
    def from_kwargs(kwargs: dict) -> LLMConfig: ...
    def to_dict(self): ...
    def clone(self, **kwargs): ...

class SeedingConfig:
    source: Incomplete
    pattern: Incomplete
    live_check: Incomplete
    extract_head: Incomplete
    max_urls: Incomplete
    concurrency: Incomplete
    hits_per_sec: Incomplete
    force: Incomplete
    base_directory: Incomplete
    llm_config: Incomplete
    verbose: Incomplete
    query: Incomplete
    score_threshold: Incomplete
    scoring_method: Incomplete
    filter_nonsense_urls: Incomplete
    def __init__(self, source: str = 'sitemap+cc', pattern: str | None = '*', live_check: bool = False, extract_head: bool = False, max_urls: int = -1, concurrency: int = 1000, hits_per_sec: int = 5, force: bool = False, base_directory: str | None = None, llm_config: LLMConfig | None = None, verbose: bool | None = None, query: str | None = None, score_threshold: float | None = None, scoring_method: str = 'bm25', filter_nonsense_urls: bool = True) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_kwargs(kwargs: dict[str, Any]) -> SeedingConfig: ...
    def clone(self, **kwargs: Any) -> SeedingConfig: ...
