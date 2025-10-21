"""Simple API Key authentication middleware."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import get_settings


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Bypass health checks
        if request.url.path in ["/health", "/ping", "/healthz"]:
            return await call_next(request)

        settings = get_settings()
        expected_key = settings.mcp_api_key

        if not expected_key:
            return await call_next(request)

        # Проверить Authorization header (MCP specification)
        auth_header = request.headers.get("Authorization")

        # Ожидаем формат: "Bearer <token>"
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                {
                    "error": "Unauthorized",
                    "message": "Missing or invalid Authorization header",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="MCP Server"'},
            )

        # Извлечь токен
        token = auth_header[7:]  # Убрать "Bearer "

        if token != expected_key:
            return JSONResponse(
                {"error": "Unauthorized", "message": "Invalid access token"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="MCP Server"'},
            )

        return await call_next(request)
