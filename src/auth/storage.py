"""Pydantic models for OAuth entity storage.

These models define the structure for persistent storage of OAuth entities
including clients, authorization codes, access tokens, and refresh tokens.
"""

import time

from pydantic import BaseModel, Field


class StoredClient(BaseModel):
    """OAuth client stored in persistent storage."""

    client_id: str
    client_secret: str | None = None
    redirect_uris: list[str] = Field(default_factory=list)
    scope: str | None = None
    created_at: float = Field(default_factory=time.time)


class StoredAuthCode(BaseModel):
    """Authorization code stored in persistent storage."""

    code: str
    client_id: str
    redirect_uri: str
    scopes: list[str] = Field(default_factory=list)
    code_challenge: str | None = None
    expires_at: float


class StoredAccessToken(BaseModel):
    """Access token stored in persistent storage."""

    token: str
    client_id: str
    scopes: list[str] = Field(default_factory=list)
    expires_at: float


class StoredRefreshToken(BaseModel):
    """Refresh token stored in persistent storage."""

    token: str
    client_id: str
    scopes: list[str] = Field(default_factory=list)
    expires_at: float | None = None  # None = no expiry
