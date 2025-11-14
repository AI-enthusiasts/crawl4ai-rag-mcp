"""Crawling services for the Crawl4AI MCP server.

This package provides comprehensive web crawling functionality with:
- Memory tracking and monitoring
- Markdown file crawling
- Batch URL processing
- Recursive internal link discovery
- Smart URL filtering
- Full integration with MCP tools and Qdrant storage
"""

# Import from modular structure
from .batch import crawl_batch
from .markdown import crawl_markdown_file
from .memory import track_memory
from .recursive import crawl_recursive_internal_links
from .service import (
    crawl_urls_for_agentic_search,
    process_urls_for_mcp,
    should_filter_url,
)

__all__ = [
    # Memory tracking
    "track_memory",
    # Basic crawling
    "crawl_markdown_file",
    "crawl_batch",
    # Recursive crawling
    "crawl_recursive_internal_links",
    # MCP service layer
    "process_urls_for_mcp",
    # Agentic search
    "crawl_urls_for_agentic_search",
    "should_filter_url",
]
