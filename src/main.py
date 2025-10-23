"""
Main entry point for the refactored Crawl4AI MCP server.

This demonstrates how the monolithic crawl4ai_mcp.py file can be refactored
into a modular structure following best practices.
"""

import asyncio
import sys
import traceback

from fastmcp import FastMCP

from config import get_settings
from core import crawl4ai_lifespan, logger
from tools import register_tools

# Get settings instance
settings = get_settings()

# Initialize FastMCP server with lifespan management
try:
    logger.info("Initializing FastMCP server...")
    # Get host and port from settings
    host = settings.host
    port = settings.port
    # Ensure port has a valid default even if empty string
    if not port:
        port = "8051"
    logger.info(f"Host: {host}, Port: {port}")

    mcp = FastMCP("Crawl4AI MCP Server", lifespan=crawl4ai_lifespan)
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
        transport = settings.transport.lower()
        logger.info(f"Transport mode: {transport}")

        # Flush output before starting server
        sys.stdout.flush()
        sys.stderr.flush()

        # Run server with appropriate transport
        if transport == "http":
            # Setup authentication
            # Note: FastMCP automatically calls crawl4ai_lifespan via run_http_async()
            # No manual lifespan management needed - this fixes browser process leak
            logger.info("Setting up HTTP server...")
            
            oauth2_server = None
            if settings.use_oauth2:
                # OAuth2 + API Key dual authentication
                from auth import OAuth2Server
                from auth.setup import setup_oauth2_routes
                
                oauth2_server = OAuth2Server(
                    issuer=settings.oauth2_issuer,
                    secret_key=settings.oauth2_secret_key,
                )
                logger.info("✓ OAuth2 server initialized")
                
                setup_oauth2_routes(mcp, oauth2_server, host, port)
                logger.info("✓ OAuth2 endpoints registered")
            
            from middleware.setup import setup_middleware
            middleware = setup_middleware(
                use_oauth2=settings.use_oauth2,
                oauth2_server=oauth2_server
            )
            
            # FastMCP automatically calls crawl4ai_lifespan ONCE via run_http_async()
            await mcp.run_http_async(
                transport="http", host=host, port=int(port), middleware=middleware
            )
        elif transport == "sse":
            await mcp.run_sse_async()
        else:  # Default to stdio for Claude Desktop compatibility
            await mcp.run_stdio_async()

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting main function...")
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
