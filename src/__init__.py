"""
Crawl4AI MCP Server - A powerful RAG-enhanced web crawling and processing tool.

This package provides MCP (Model Context Protocol) tools for web crawling,
content extraction, and intelligent document processing with vector search capabilities.
"""

__version__ = "0.1.0"

# Re-export commonly used utilities for easier imports in tests
from src.core.context import (
    Crawl4AIContext,
    format_neo4j_error,
    get_app_context,
    set_app_context,
)
from src.core.decorators import track_request
from src.core.stdout_utils import SuppressStdout
from src.core.exceptions import (
    Crawl4AIError,
    DatabaseError,
    MCPToolError,
    NetworkError,
    ValidationError,
)
from src.utils.code_analysis import process_code_example
from src.utils.embeddings import create_embedding, create_embeddings_batch
from src.utils.reranking import rerank_results
from src.utils.text_processing import extract_section_info, smart_chunk_markdown
from src.utils.url_helpers import (
    is_sitemap,
    is_txt,
    parse_sitemap,
)
from src.utils.validation import (
    validate_github_url,
    validate_neo4j_connection,
    validate_script_path,
)

# Configuration
from src.config.settings import Settings, get_settings, reset_settings

# Database factory and RAG queries
from src.database.factory import (
    create_and_initialize_database,
    create_database_client,
)
from src.database.rag_queries import get_available_sources

# Service functions
from src.services.crawling import process_urls_for_mcp
from src.services.search import search_and_process
from src.services.smart_crawl import smart_crawl_url

# Knowledge graph operations
# Note: There's a naming conflict - validation.py file vs validation/ directory
# We need to import from the parent module with specific file reference
import sys
import importlib.util
from pathlib import Path

# Load validation.py module directly to avoid package conflict
# Use relative path resolution to work in both dev and Docker environments
_validation_path = Path(__file__).parent / "knowledge_graph" / "validation.py"
_validation_spec = importlib.util.spec_from_file_location(
    "src.knowledge_graph._validation_module",
    str(_validation_path)
)
if _validation_spec and _validation_spec.loader:
    _validation_module = importlib.util.module_from_spec(_validation_spec)
    sys.modules["src.knowledge_graph._validation_module"] = _validation_module
    _validation_spec.loader.exec_module(_validation_module)
    check_ai_script_hallucinations = _validation_module.check_ai_script_hallucinations
else:
    check_ai_script_hallucinations = None

# Knowledge graph repository and queries
from src.knowledge_graph.repository import parse_github_repository
from src.knowledge_graph.queries import query_knowledge_graph

# Tools registration
from src.tools import register_tools

# MCP server instance with registered tools
# Import here to make tools available (scrape_urls, perform_rag_query, etc.)
from src.main import mcp

# Export registered MCP tool functions for tests
# These are dynamically created by @mcp.tool() decorators
scrape_urls = None
perform_rag_query = None
search = None
search_code_examples = None
smart_crawl_url_tool = None

# Try to extract tool functions from mcp instance
try:
    # Tools are registered in mcp._tools dict
    if hasattr(mcp, '_tools'):
        for tool_name, tool_obj in mcp._tools.items():
            if tool_name == 'scrape_urls':
                scrape_urls = tool_obj
            elif tool_name == 'perform_rag_query':
                perform_rag_query = tool_obj
            elif tool_name == 'search':
                search = tool_obj
            elif tool_name == 'search_code_examples':
                search_code_examples = tool_obj
            elif tool_name == 'smart_crawl_url':
                smart_crawl_url_tool = tool_obj
except Exception:
    # Tools not available yet, will be None
    pass

__all__ = [
    "__version__",
    # Core
    "Crawl4AIContext",
    "SuppressStdout",
    "format_neo4j_error",
    "get_app_context",
    "set_app_context",
    "track_request",
    # Exceptions
    "Crawl4AIError",
    "DatabaseError",
    "NetworkError",
    "ValidationError",
    "MCPToolError",
    # Text processing
    "extract_section_info",
    "smart_chunk_markdown",
    # URL helpers
    "is_sitemap",
    "is_txt",
    "parse_sitemap",
    # Validation
    "validate_github_url",
    "validate_neo4j_connection",
    "validate_script_path",
    # Configuration
    "Settings",
    "get_settings",
    "reset_settings",
    # Database
    "create_database_client",
    "create_and_initialize_database",
    "get_available_sources",
    # Code analysis
    "process_code_example",
    # Reranking
    "rerank_results",
    # Embeddings
    "create_embedding",
    "create_embeddings_batch",
    # Services
    "process_urls_for_mcp",
    "search_and_process",
    "smart_crawl_url",
    # Knowledge graph
    "check_ai_script_hallucinations",
    "parse_github_repository",
    "query_knowledge_graph",
    # MCP server and tools
    "register_tools",
    "mcp",
    "scrape_urls",
    "perform_rag_query",
    "search",
    "search_code_examples",
    "smart_crawl_url_tool",
]
