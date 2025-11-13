"""Utilities for managing stdout/stderr output."""

import sys
from types import TracebackType
from typing import Literal, Self


class SuppressStdout:
    """Context manager to suppress stdout during crawl operations."""

    def __enter__(self) -> Self:
        self._stdout = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        sys.stdout = self._stdout
        return False
