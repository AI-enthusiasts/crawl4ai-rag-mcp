import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod

logger: Incomplete

class BaseCrawler(ABC, metaclass=abc.ABCMeta):
    logger: Incomplete
    def __init__(self) -> None: ...
    @abstractmethod
    async def run(self, url: str = '', **kwargs) -> str: ...
    def __init_subclass__(cls, **kwargs) -> None: ...

class CrawlerHub:
    @classmethod
    def get(cls, name: str) -> type[BaseCrawler] | None: ...
