"""Utilities for managing stdout/stderr output."""

import sys
from typing import Any


class SuppressStdout:
    """Context manager to suppress stdout during crawl operations."""

    def __enter__(self) -> "SuppressStdout":
        self._stdout = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        sys.stdout = self._stdout
        return False
