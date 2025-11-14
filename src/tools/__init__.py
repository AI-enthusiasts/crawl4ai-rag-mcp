"""
MCP Tools Package.

This package contains all MCP tool definitions organized by category:
- search: Web search and agentic search tools
- crawl: URL crawling and scraping tools
- rag: RAG query and code search tools
- knowledge_graph: Neo4j knowledge graph tools
- validation: Code validation and hallucination detection tools

Each module provides a register_*_tools() function to register tools with FastMCP.
"""

from src.tools.crawl import register_crawl_tools
from src.tools.knowledge_graph import register_knowledge_graph_tools
from src.tools.rag import register_rag_tools
from src.tools.search import register_search_tools
from src.tools.validation import register_validation_tools

__all__ = [
    "register_search_tools",
    "register_crawl_tools",
    "register_rag_tools",
    "register_knowledge_graph_tools",
    "register_validation_tools",
]
