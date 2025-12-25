"""OAuth authentication module with persistent storage.

This module provides a production-ready OAuth 2.1 implementation for FastMCP
with persistent storage support for Claude Web integration.
"""

from src.auth.provider import PersistentOAuthProvider

__all__ = ["PersistentOAuthProvider"]
