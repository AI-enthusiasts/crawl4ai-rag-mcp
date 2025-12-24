"""Custom Crawl4AI logger that writes to stderr instead of stdout.

This is critical for MCP stdio transport compliance. The MCP specification states:
> The server MUST NOT write anything to its stdout that is not a valid MCP message.
> The server MAY write UTF-8 strings to its standard error (stderr) for logging purposes.

Crawl4AI's default AsyncLogger uses Rich Console which writes to stdout by default,
breaking MCP protocol. This module provides a logger that redirects all output to stderr.
"""

import sys

from crawl4ai.async_logger import AsyncLogger
from rich.console import Console


class StderrAsyncLogger(AsyncLogger):
    """AsyncLogger that writes to stderr instead of stdout.

    This ensures MCP stdio transport compliance by keeping stdout clean
    for JSON-RPC messages only.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize logger with stderr-based Console.

        Args:
            **kwargs: Arguments passed to AsyncLogger (log_file, log_level, etc.)
        """
        super().__init__(**kwargs)
        # Override console to use stderr instead of stdout
        self.console = Console(file=sys.stderr, force_terminal=False)
