"""Persistent OAuth Provider implementation.

This module provides a production-ready OAuth 2.1 provider with persistent
storage for use with Claude Web and other MCP clients.
"""

import logging
import secrets
import time
from pathlib import Path
from typing import Any

from fastmcp.server.auth import OAuthProvider
from fastmcp.server.auth.auth import AccessToken
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
)
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl

from src.auth.storage import (
    StoredAccessToken,
    StoredAuthCode,
    StoredClient,
    StoredRefreshToken,
)

logger = logging.getLogger(__name__)


class PersistentOAuthProvider(OAuthProvider):
    """OAuth Provider with file-based persistent storage.

    This implementation stores OAuth entities (clients, tokens, auth codes)
    in JSON files on disk, providing persistence across server restarts.

    For production with multiple servers, consider using Redis or a database.
    """

    def __init__(
        self,
        *,
        base_url: str,
        storage_dir: str | Path = ".oauth_storage",
        issuer_url: str | None = None,
        service_documentation_url: str | None = None,
        client_registration_options: ClientRegistrationOptions | None = None,
        revocation_options: RevocationOptions | None = None,
        required_scopes: list[str] | None = None,
    ) -> None:
        """Initialize OAuth provider with persistent storage.

        Args:
            base_url: Base URL for OAuth endpoints
            storage_dir: Directory for storing OAuth data
            issuer_url: OAuth issuer URL (defaults to base_url)
            service_documentation_url: URL to service documentation
            client_registration_options: DCR configuration
            revocation_options: Token revocation configuration
            required_scopes: Scopes required for all requests
        """
        super().__init__(
            base_url=base_url,
            issuer_url=issuer_url,
            service_documentation_url=service_documentation_url,
            client_registration_options=client_registration_options,
            revocation_options=revocation_options,
            required_scopes=required_scopes,
        )

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each entity type
        self._clients_dir = self._storage_dir / "clients"
        self._auth_codes_dir = self._storage_dir / "auth_codes"
        self._access_tokens_dir = self._storage_dir / "access_tokens"
        self._refresh_tokens_dir = self._storage_dir / "refresh_tokens"

        for dir_path in [
            self._clients_dir,
            self._auth_codes_dir,
            self._access_tokens_dir,
            self._refresh_tokens_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

        logger.info(
            "Initialized PersistentOAuthProvider with storage at %s",
            self._storage_dir,
        )

    def _get_file_path(self, directory: Path, key: str) -> Path:
        """Get file path for a storage key (sanitized)."""
        # Sanitize key to prevent path traversal
        safe_key = key.replace("/", "_").replace("\\", "_").replace("..", "_")
        return directory / f"{safe_key}.json"

    def _read_entity(self, directory: Path, key: str, model: type[Any]) -> Any | None:
        """Read entity from file storage."""
        file_path = self._get_file_path(directory, key)
        if not file_path.exists():
            return None
        try:
            return model.model_validate_json(file_path.read_text())
        except Exception as e:
            logger.warning("Failed to read entity %s: %s", key, e)
            return None

    def _write_entity(self, directory: Path, key: str, entity: Any) -> None:
        """Write entity to file storage."""
        file_path = self._get_file_path(directory, key)
        file_path.write_text(entity.model_dump_json(indent=2))

    def _delete_entity(self, directory: Path, key: str) -> None:
        """Delete entity from file storage."""
        file_path = self._get_file_path(directory, key)
        if file_path.exists():
            file_path.unlink()

    # ========== Client Management ==========

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Retrieve client from persistent storage."""
        stored = self._read_entity(self._clients_dir, client_id, StoredClient)
        if not stored:
            return None
        return OAuthClientInformationFull(
            client_id=stored.client_id,
            client_secret=stored.client_secret,
            redirect_uris=[AnyHttpUrl(uri) for uri in stored.redirect_uris],
            scope=stored.scope,
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """Store client registration in persistent storage."""
        if not client_info.client_id:
            msg = "client_id is required"
            raise ValueError(msg)

        stored = StoredClient(
            client_id=client_info.client_id,
            client_secret=client_info.client_secret,
            redirect_uris=[str(uri) for uri in (client_info.redirect_uris or [])],
            scope=client_info.scope,
        )
        self._write_entity(self._clients_dir, client_info.client_id, stored)
        logger.info("Registered client: %s", client_info.client_id)

    # ========== Authorization Flow ==========

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        """Generate authorization code and return redirect URL."""
        code = f"authcode_{secrets.token_urlsafe(32)}"
        expires_at = time.time() + 300  # 5 minutes

        stored = StoredAuthCode(
            code=code,
            client_id=client.client_id or "",
            redirect_uri=str(params.redirect_uri),
            scopes=list(params.scopes) if params.scopes else [],
            code_challenge=params.code_challenge,
            expires_at=expires_at,
        )
        self._write_entity(self._auth_codes_dir, code, stored)

        # Build redirect URL with code
        redirect_uri = str(params.redirect_uri)
        separator = "&" if "?" in redirect_uri else "?"
        state_param = f"&state={params.state}" if params.state else ""
        return f"{redirect_uri}{separator}code={code}{state_param}"

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        """Load authorization code from storage."""
        stored = self._read_entity(
            self._auth_codes_dir,
            authorization_code,
            StoredAuthCode,
        )
        if not stored:
            return None

        # Check expiration
        if time.time() > stored.expires_at:
            self._delete_entity(self._auth_codes_dir, authorization_code)
            return None

        # Verify client
        if stored.client_id != client.client_id:
            return None

        return AuthorizationCode(
            code=stored.code,
            client_id=stored.client_id,
            redirect_uri=AnyHttpUrl(stored.redirect_uri),
            redirect_uri_provided_explicitly=True,
            scopes=stored.scopes,
            expires_at=stored.expires_at,
            code_challenge=stored.code_challenge,
        )

    # ========== Token Exchange ==========

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        """Exchange auth code for tokens."""
        # Delete used code (one-time use)
        self._delete_entity(self._auth_codes_dir, authorization_code.code)

        # Generate tokens
        access_token = f"access_{secrets.token_urlsafe(32)}"
        refresh_token = f"refresh_{secrets.token_urlsafe(32)}"
        expires_in = 3600  # 1 hour

        # Store access token
        stored_access = StoredAccessToken(
            token=access_token,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            expires_at=time.time() + expires_in,
        )
        self._write_entity(self._access_tokens_dir, access_token, stored_access)

        # Store refresh token
        stored_refresh = StoredRefreshToken(
            token=refresh_token,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            expires_at=None,  # No expiry
        )
        self._write_entity(self._refresh_tokens_dir, refresh_token, stored_refresh)

        logger.info("Issued tokens for client: %s", client.client_id)

        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=refresh_token,
            scope=" ".join(authorization_code.scopes),
        )

    # ========== Refresh Token ==========

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        """Load refresh token from storage."""
        stored = self._read_entity(
            self._refresh_tokens_dir,
            refresh_token,
            StoredRefreshToken,
        )
        if not stored:
            return None

        # Verify client
        if stored.client_id != client.client_id:
            return None

        # Check expiration if set
        if stored.expires_at and time.time() > stored.expires_at:
            self._delete_entity(self._refresh_tokens_dir, refresh_token)
            return None

        return RefreshToken(
            token=stored.token,
            client_id=stored.client_id,
            scopes=stored.scopes,
            expires_at=stored.expires_at,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Exchange refresh token for new access token."""
        # Validate scopes
        if not set(scopes).issubset(set(refresh_token.scopes)):
            msg = "Requested scopes exceed original scopes"
            raise ValueError(msg)

        # Revoke old refresh token (rotation)
        self._delete_entity(self._refresh_tokens_dir, refresh_token.token)

        # Generate new tokens
        new_access = f"access_{secrets.token_urlsafe(32)}"
        new_refresh = f"refresh_{secrets.token_urlsafe(32)}"
        expires_in = 3600

        # Store new access token
        stored_access = StoredAccessToken(
            token=new_access,
            client_id=client.client_id or "",
            scopes=scopes,
            expires_at=time.time() + expires_in,
        )
        self._write_entity(self._access_tokens_dir, new_access, stored_access)

        # Store new refresh token
        stored_refresh = StoredRefreshToken(
            token=new_refresh,
            client_id=client.client_id or "",
            scopes=scopes,
            expires_at=None,
        )
        self._write_entity(self._refresh_tokens_dir, new_refresh, stored_refresh)

        logger.info("Refreshed tokens for client: %s", client.client_id)

        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=new_refresh,
            scope=" ".join(scopes),
        )

    # ========== Token Validation ==========

    async def load_access_token(self, token: str) -> AccessToken | None:
        """Load and validate access token."""
        stored = self._read_entity(self._access_tokens_dir, token, StoredAccessToken)
        if not stored:
            return None

        # Check expiration
        if time.time() > stored.expires_at:
            self._delete_entity(self._access_tokens_dir, token)
            return None

        return AccessToken(
            token=stored.token,
            client_id=stored.client_id,
            scopes=stored.scopes,
            expires_at=int(stored.expires_at),
        )

    # ========== Revocation ==========

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        """Revoke a token."""
        if isinstance(token, RefreshToken):
            self._delete_entity(self._refresh_tokens_dir, token.token)
            logger.info("Revoked refresh token for client: %s", token.client_id)
        else:
            self._delete_entity(self._access_tokens_dir, token.token)
            logger.info("Revoked access token for client: %s", token.client_id)
