from _typeshed import Incomplete
from crawl4ai.hub import BaseCrawler
from crawl4ai.utils import optimize_html as optimize_html

class GoogleSearchCrawler(BaseCrawler):
    __meta__: Incomplete
    js_script: Incomplete
    def __init__(self) -> None: ...
    async def run(self, url: str = '', query: str = '', search_type: str = 'text', schema_cache_path=None, **kwargs) -> str: ...
