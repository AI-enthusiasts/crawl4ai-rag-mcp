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

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP

from src.tools.crawl import register_crawl_tools
from src.tools.knowledge_graph import register_knowledge_graph_tools
from src.tools.rag import register_rag_tools
from src.tools.search import register_search_tools
from src.tools.validation import register_validation_tools

logger = logging.getLogger(__name__)


def register_tools(mcp: "FastMCP") -> None:
    """
    Register all MCP tools with the FastMCP instance.

    This function delegates to modular registration functions from the tools package:
    - register_search_tools: Web search and agentic search
    - register_crawl_tools: URL scraping and smart crawling
    - register_rag_tools: RAG queries and code search
    - register_knowledge_graph_tools: Neo4j repository parsing and queries
    - register_validation_tools: Code validation and hallucination detection

    Args:
        mcp: FastMCP instance to register tools with
    """
    logger.info("Registering all MCP tools...")

    # Register tools from each module
    register_search_tools(mcp)
    register_crawl_tools(mcp)
    register_rag_tools(mcp)
    register_knowledge_graph_tools(mcp)
    register_validation_tools(mcp)

    logger.info("All MCP tools registered successfully")


__all__ = [
    "register_crawl_tools",
    "register_knowledge_graph_tools",
    "register_rag_tools",
    "register_search_tools",
    "register_tools",
    "register_validation_tools",
]
