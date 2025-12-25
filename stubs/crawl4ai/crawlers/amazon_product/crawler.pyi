from _typeshed import Incomplete
from crawl4ai.hub import BaseCrawler

__meta__: Incomplete

class AmazonProductCrawler(BaseCrawler):
    async def run(self, url: str, **kwargs) -> str: ...
