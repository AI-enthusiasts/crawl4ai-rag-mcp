"""
Dual authentication middleware for MCP server.

Supports both API Key and OAuth2 authentication.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.requests import Request

from config import get_settings
from auth.oauth2_server import OAuth2Server


class DualAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware supporting both API Key and OAuth2 authentication.

    Authentication methods (in order of precedence):
    1. OAuth2 Bearer token (Authorization: Bearer <token>)
    2. API Key (Authorization: Bearer <api_key>)

    If neither is valid, returns 401 Unauthorized.
    """

    def __init__(self, app, oauth2_server: OAuth2Server):
        """
        Initialize middleware.

        Args:
            app: ASGI application
            oauth2_server: OAuth2 server instance
        """
        super().__init__(app)
        self.oauth2_server = oauth2_server
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next):
        """Process request with dual authentication."""
        # Bypass health checks
        if request.url.path in ["/health", "/ping", "/healthz"]:
            return await call_next(request)

        # Bypass OAuth2 endpoints
        oauth2_paths = [
            "/.well-known/oauth-authorization-server",
            "/.well-known/oauth-protected-resource",
            "/authorize",
            "/token",
            "/register",
        ]
        if any(request.url.path.startswith(path) for path in oauth2_paths):
            return await call_next(request)

        # Get Authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return self._unauthorized_response(
                "Missing or invalid Authorization header"
            )

        # Extract token
        token = auth_header[7:]  # Remove "Bearer "

        # Try OAuth2 validation first
        resource_url = str(request.base_url).rstrip("/")
        token_payload = self.oauth2_server.validate_access_token(token, resource_url)

        if token_payload:
            # Valid OAuth2 token
            request.state.auth_type = "oauth2"
            request.state.client_id = token_payload.get("sub")
            request.state.scope = token_payload.get("scope")
            return await call_next(request)

        # Try API Key validation
        expected_key = self.settings.mcp_api_key
        if expected_key and token == expected_key:
            # Valid API Key
            request.state.auth_type = "api_key"
            return await call_next(request)

        # Neither authentication method worked
        return self._unauthorized_response("Invalid access token or API key")

    def _unauthorized_response(self, message: str) -> JSONResponse:
        """Create 401 Unauthorized response with WWW-Authenticate header."""
        # Include resource metadata URL for OAuth2 discovery
        resource_metadata_url = (
            f"{self.oauth2_server.issuer}/.well-known/oauth-protected-resource"
        )

        return JSONResponse(
            {
                "error": "Unauthorized",
                "message": message,
            },
            status_code=401,
            headers={
                "WWW-Authenticate": f'Bearer resource_metadata="{resource_metadata_url}"'
            },
        )
