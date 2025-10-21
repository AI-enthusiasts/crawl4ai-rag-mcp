"""Middleware package for MCP server."""

from .auth import APIKeyMiddleware

__all__ = ["APIKeyMiddleware"]
