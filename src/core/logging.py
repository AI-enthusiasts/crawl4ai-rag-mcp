"""Logging configuration for the Crawl4AI MCP server."""

import logging
import os
import sys
from contextvars import ContextVar

# Context variable to store request_id across async calls
request_id_ctx: ContextVar[str | None] = ContextVar('request_id', default=None)


class RequestIdFilter(logging.Filter):
    """Logging filter that adds request_id to log records."""
    
    def filter(self, record):
        """Add request_id to the log record if available."""
        request_id = request_id_ctx.get()
        record.request_id = f"[{request_id}] " if request_id else ""
        return True


def configure_logging() -> logging.Logger:
    """Configure and return the logger for the application."""
    # Configure structured logging with request_id support
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(request_id)s%(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    logger = logging.getLogger("crawl4ai-mcp")
    
    # Add request_id filter to all handlers
    request_filter = RequestIdFilter()
    for handler in logger.handlers:
        handler.addFilter(request_filter)
    
    # Also add to root logger handlers
    for handler in logging.root.handlers:
        handler.addFilter(request_filter)

    # Enable debug mode from environment
    if os.getenv("MCP_DEBUG", "").lower() in ("true", "1", "yes"):
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    return logger


# Initialize logger
logger = configure_logging()
