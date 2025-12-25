from _typeshed import Incomplete
from enum import Enum

class CacheMode(Enum):
    ENABLED = 'enabled'
    DISABLED = 'disabled'
    READ_ONLY = 'read_only'
    WRITE_ONLY = 'write_only'
    BYPASS = 'bypass'

class CacheContext:
    url: Incomplete
    cache_mode: Incomplete
    always_bypass: Incomplete
    is_cacheable: Incomplete
    is_web_url: Incomplete
    is_local_file: Incomplete
    is_raw_html: Incomplete
    def __init__(self, url: str, cache_mode: CacheMode, always_bypass: bool = False) -> None: ...
    def should_read(self) -> bool: ...
    def should_write(self) -> bool: ...
    @property
    def display_url(self) -> str: ...
