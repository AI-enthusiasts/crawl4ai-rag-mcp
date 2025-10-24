"""
Main entry point for the refactored Crawl4AI MCP server.

This demonstrates how the monolithic crawl4ai_mcp.py file can be refactored
into a modular structure following best practices.
"""

import asyncio
import sys
import traceback

from fastmcp import FastMCP
from fastmcp.server.auth import StaticTokenVerifier

from config import get_settings
from core import logger
from core.context import initialize_global_context, cleanup_global_context
from tools import register_tools

# Get settings instance
settings = get_settings()

# Initialize FastMCP server with built-in authentication
try:
    logger.info("Initializing FastMCP server...")
    # Get host and port from settings
    host = settings.host
    port = settings.port
    # Ensure port has a valid default even if empty string
    if not port:
        port = "8051"
    logger.info(f"Host: {host}, Port: {port}")

    # Use FastMCP's built-in StaticTokenVerifier for API key authentication
    if settings.mcp_api_key:
        auth = StaticTokenVerifier(
            tokens={
                settings.mcp_api_key: {
                    "client_id": "mcp-client",
                    "scopes": ["read", "write"],
                    "expires_at": None  # No expiration
                }
            }
        )
        mcp = FastMCP("Crawl4AI MCP Server", auth=auth)
        logger.info("✓ StaticTokenVerifier enabled with API key")
    else:
        mcp = FastMCP("Crawl4AI MCP Server")
        logger.info("⚠ No authentication configured")
    
    logger.info("FastMCP server initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FastMCP server: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)


# Register all MCP tools
register_tools(mcp)


def create_mcp_server():
    """
    Create and return an MCP server instance for testing purposes.
    """
    return FastMCP("Crawl4AI MCP Server Test")


async def main():
    """
    Main async function to run the MCP server.
    """
    try:
        logger.info("Main function started")
        
        # Initialize global context ONCE at startup (not per-request)
        logger.info("Initializing global application context...")
        await initialize_global_context()
        logger.info("✓ Global context initialized")
        
        transport = settings.transport.lower()
        logger.info(f"Transport mode: {transport}")

        # Flush output before starting server
        sys.stdout.flush()
        sys.stderr.flush()

        # Run server with appropriate transport
        if transport == "http":
            logger.info("Setting up HTTP server...")
            
            # Run HTTP server - authentication is handled by FastMCP's built-in auth
            # No need for custom middleware when using StaticTokenVerifier
            await mcp.run_http_async(
                transport="http", host=host, port=int(port)
            )
        elif transport == "sse":
            await mcp.run_sse_async()
        else:  # Default to stdio for Claude Desktop compatibility
            await mcp.run_stdio_async()

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup global context on shutdown
        logger.info("Shutting down - cleaning up global context...")
        await cleanup_global_context()


if __name__ == "__main__":
    try:
        logger.info("Starting main function...")
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
