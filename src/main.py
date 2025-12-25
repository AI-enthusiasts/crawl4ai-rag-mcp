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
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions

from src.auth import PersistentOAuthProvider
from src.config import get_settings
from src.core import logger
from src.core.context import cleanup_global_context, initialize_global_context
from src.tools import (
    register_crawl_tools,
    register_knowledge_graph_tools,
    register_rag_tools,
    register_search_tools,
    register_validation_tools,
)

# Get settings instance
settings = get_settings()

# Initialize FastMCP server with flexible authentication
try:
    logger.info("Initializing FastMCP server...")
    # Get host and port from settings
    host = settings.host
    port = settings.port
    # Ensure port has a valid default even if empty string
    if not port:
        port = 8051
    logger.info(f"Host: {host}, Port: {port}")

    # Determine authentication mode based on settings
    auth: PersistentOAuthProvider | StaticTokenVerifier | None = None
    auth_mode = "none"

    if settings.use_oauth2:
        # Mode 1: OAuth Provider with DCR (for Claude Web custom connectors)
        logger.info("Configuring OAuth Provider with Dynamic Client Registration...")

        auth = PersistentOAuthProvider(
            base_url=settings.oauth2_issuer or "",
            storage_dir=settings.oauth_storage_dir,
            issuer_url=settings.oauth2_issuer,
            service_documentation_url=f"{settings.oauth2_issuer}/docs",
            client_registration_options=ClientRegistrationOptions(
                enabled=True,  # Enable DCR for Claude Web
                valid_scopes=settings.get_oauth2_scopes_list(),
            ),
            revocation_options=RevocationOptions(enabled=True),
            required_scopes=settings.get_oauth2_required_scopes_list(),
        )
        auth_mode = "oauth2"
        logger.info("✓ OAuth Provider enabled")
        logger.info(f"  - Issuer: {settings.oauth2_issuer}")
        logger.info(f"  - Valid scopes: {', '.join(settings.oauth2_scopes)}")
        logger.info(
            f"  - Required scopes: {', '.join(settings.oauth2_required_scopes)}"
        )
        logger.info("  - DCR enabled: Yes")
        logger.info("  - Endpoints:")
        logger.info("    - /.well-known/oauth-authorization-server")
        logger.info("    - /register (DCR)")
        logger.info("    - /authorize")
        logger.info("    - /token")
        logger.info("    - /revoke")

    elif settings.mcp_api_key:
        # Mode 2: Static Token Verifier (simple API key)
        logger.info("Configuring Static Token Verifier...")

        auth = StaticTokenVerifier(
            tokens={
                settings.mcp_api_key: {
                    "client_id": "mcp-client",
                    "scopes": ["read", "write"],
                    "expires_at": None,  # No expiration
                },
            },
        )
        auth_mode = "api_key"
        logger.info("✓ StaticTokenVerifier enabled with API key")

    else:
        # Mode 3: No authentication
        logger.warning("⚠ No authentication configured - server is open to all!")
        logger.warning(
            "  Set USE_OAUTH2=true for OAuth or MCP_API_KEY for API key auth"
        )
        auth_mode = "none"

    # Create FastMCP server with appropriate auth
    mcp = FastMCP("Crawl4AI MCP Server", auth=auth)
    logger.info(f"FastMCP server initialized successfully (auth mode: {auth_mode})")

except Exception as e:
    logger.error(f"Failed to initialize FastMCP server: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)


# Register all MCP tools
register_search_tools(mcp)
register_crawl_tools(mcp)
register_rag_tools(mcp)
register_knowledge_graph_tools(mcp)
register_validation_tools(mcp)


def create_mcp_server() -> FastMCP:
    """
    Create and return an MCP server instance for testing purposes.
    """
    return FastMCP("Crawl4AI MCP Server Test")


async def main() -> None:
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
        # Normalize transport names to FastMCP Transport literals
        transport_map = {
            "http": "streamable-http",
            "streamable-http": "streamable-http",
            "sse": "sse",
            "stdio": "stdio",
        }
        fastmcp_transport = transport_map.get(transport, "stdio")

        if fastmcp_transport in ("streamable-http", "sse"):
            logger.info(
                "Setting up %s server on %s:%s...", fastmcp_transport, host, port
            )
            await mcp.run_async(
                transport=fastmcp_transport,  # type: ignore[arg-type]
                host=host,
                port=int(port),
            )
        else:
            logger.info("Setting up stdio server...")
            await mcp.run_async(transport="stdio")

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
