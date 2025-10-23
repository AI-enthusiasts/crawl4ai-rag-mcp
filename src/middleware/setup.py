"""
Middleware configuration for FastMCP server.

This module provides a clean interface to configure authentication
middleware based on application settings.

Architecture:
- Separates middleware configuration from main application logic
- Supports OAuth2 + API Key dual authentication
- Supports API Key only authentication
- Single responsibility: middleware setup
"""

from typing import List, Optional, TYPE_CHECKING

from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware

from config import get_settings
from core import logger
from middleware import APIKeyMiddleware

if TYPE_CHECKING:
    from auth import OAuth2Server, DualAuthMiddleware

settings = get_settings()


def setup_middleware(
    use_oauth2: bool = False,
    oauth2_server: Optional["OAuth2Server"] = None,
) -> List[Middleware]:
    """
    Configure authentication middleware based on settings.

    This function creates a list of Starlette middleware instances
    configured for the requested authentication mode.

    Authentication modes:
    1. OAuth2 + API Key (dual): Requires oauth2_server parameter
       - SessionMiddleware (for OAuth2 authorization form)
       - DualAuthMiddleware (OAuth2 + API Key validation)

    2. API Key only: Uses settings.mcp_api_key
       - APIKeyMiddleware (API Key validation)

    3. No authentication: Returns empty list

    Args:
        use_oauth2: Enable OAuth2 + API Key dual authentication
        oauth2_server: OAuth2Server instance (required if use_oauth2=True)

    Returns:
        List of configured Middleware instances

    Raises:
        ValueError: If use_oauth2=True but oauth2_server is None

    Example:
        >>> from auth import OAuth2Server
        >>> from middleware.setup import setup_middleware
        >>>
        >>> # OAuth2 + API Key
        >>> oauth2_server = OAuth2Server("https://example.com", "secret")
        >>> middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)
        >>>
        >>> # API Key only
        >>> middleware = setup_middleware(use_oauth2=False)
    """
    middleware = []

    if use_oauth2:
        # OAuth2 + API Key dual authentication
        if not oauth2_server:
            raise ValueError(
                "oauth2_server parameter is required when use_oauth2=True"
            )

        # Import DualAuthMiddleware here to avoid circular imports
        from auth import DualAuthMiddleware

        # Session middleware (required for OAuth2 authorization form)
        middleware.append(
            Middleware(SessionMiddleware, secret_key=settings.oauth2_secret_key)
        )
        logger.info("✓ Session middleware enabled")

        # Dual authentication (OAuth2 + API Key)
        middleware.append(
            Middleware(DualAuthMiddleware, oauth2_server=oauth2_server)
        )
        logger.info("✓ OAuth2 + API Key dual authentication enabled")

    elif settings.mcp_api_key:
        # API Key only authentication
        middleware.append(Middleware(APIKeyMiddleware))
        logger.info("✓ API Key authentication enabled")

    else:
        # No authentication configured
        logger.warning("⚠ No authentication middleware configured")

    return middleware
