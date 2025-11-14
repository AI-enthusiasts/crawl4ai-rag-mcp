"""
MCP Tool Definitions for Crawl4AI.

This module serves as the main entry point for registering all MCP tools.
Tools are now organized into separate modules under the tools/ package:
- tools.search: Web search and agentic search tools
- tools.crawl: URL crawling and scraping tools
- tools.rag: RAG query and code search tools
- tools.knowledge_graph: Neo4j knowledge graph tools
- tools.validation: Code validation and hallucination detection tools

Due to FastMCP's architecture, tools must be registered in the same scope
as the FastMCP instance, so this module imports the registration functions.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP

# Import registration functions from tools package
from tools.crawl import register_crawl_tools
from tools.knowledge_graph import register_knowledge_graph_tools
from tools.rag import register_rag_tools
from tools.search import register_search_tools
from tools.validation import register_validation_tools

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
