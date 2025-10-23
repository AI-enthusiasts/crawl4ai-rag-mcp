"""
OAuth2 route registration for FastMCP server.

This module provides a clean interface to register OAuth2 endpoints
with FastMCP, using the route handlers from auth.routes module.

Architecture:
- Separates route registration (this module) from route handlers (routes.py)
- Creates closure adapters to inject oauth2_server dependency
- Maintains single responsibility principle
"""

from typing import TYPE_CHECKING

from core import logger

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from auth.oauth2_server import OAuth2Server


def setup_oauth2_routes(
    mcp: "FastMCP",
    oauth2_server: "OAuth2Server",
    host: str,
    port: str,
) -> None:
    """
    Register OAuth2 endpoints with FastMCP server.

    This function creates closure adapters around the route handlers
    from auth.routes, injecting the oauth2_server dependency.

    Registers:
    - /.well-known/oauth-authorization-server (RFC 8414)
    - /.well-known/oauth-protected-resource (RFC 9728)
    - /register (RFC 7591 - Dynamic Client Registration)
    - /authorize (GET/POST - Authorization flow)
    - /token (Token exchange)

    Args:
        mcp: FastMCP server instance
        oauth2_server: OAuth2Server instance for authentication
        host: Server host (for resource URL construction)
        port: Server port (for resource URL construction)

    Example:
        >>> from fastmcp import FastMCP
        >>> from auth import OAuth2Server
        >>> from auth.setup import setup_oauth2_routes
        >>>
        >>> mcp = FastMCP("My Server")
        >>> oauth2_server = OAuth2Server("https://example.com", "secret")
        >>> setup_oauth2_routes(mcp, oauth2_server, "0.0.0.0", "8051")
    """
    # Import route handlers
    from auth.routes import (
        authorization_server_metadata,
        protected_resource_metadata,
        register_client,
        authorize_get,
        authorize_post,
        token_endpoint,
    )
    from config import get_settings

    settings = get_settings()

    # Construct resource URL for metadata
    resource_url = (
        f"https://{host}:{port}" if host != "0.0.0.0" else settings.oauth2_issuer
    )

    # Register metadata endpoints
    @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def _authorization_server_metadata(request):
        """OAuth 2.0 Authorization Server Metadata (RFC 8414)."""
        return await authorization_server_metadata(request, oauth2_server)

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def _protected_resource_metadata(request):
        """Protected Resource Metadata (RFC 9728)."""
        return await protected_resource_metadata(request, oauth2_server, resource_url)

    # Register OAuth2 flow endpoints
    @mcp.custom_route("/register", methods=["POST"])
    async def _register_client(request):
        """Dynamic Client Registration (RFC 7591)."""
        return await register_client(request, oauth2_server)

    @mcp.custom_route("/authorize", methods=["GET"])
    async def _authorize_get(request):
        """Authorization endpoint (GET) - shows login form."""
        return await authorize_get(request, oauth2_server)

    @mcp.custom_route("/authorize", methods=["POST"])
    async def _authorize_post(request):
        """Authorization endpoint (POST) - processes login form."""
        return await authorize_post(request, oauth2_server)

    @mcp.custom_route("/token", methods=["POST"])
    async def _token_endpoint(request):
        """Token endpoint - exchanges authorization code for access token."""
        return await token_endpoint(request, oauth2_server)

    logger.info("✓ OAuth2 endpoints registered (6 routes)")
