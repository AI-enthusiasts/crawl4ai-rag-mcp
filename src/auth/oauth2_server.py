"""
OAuth2 Authorization Server for MCP.

Implements OAuth 2.1 with PKCE according to MCP specification.
Supports Dynamic Client Registration (DCR) for Claude Web integration.
"""

import secrets
import hashlib
import base64
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from jose import JWTError, jwt
from passlib.context import CryptContext


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass
class OAuth2Client:
    """OAuth2 client registration."""

    client_id: str
    client_secret: str
    client_name: str
    redirect_uris: List[str]
    grant_types: List[str] = field(
        default_factory=lambda: ["authorization_code", "refresh_token"]
    )
    response_types: List[str] = field(default_factory=lambda: ["code"])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AuthorizationCode:
    """Authorization code with PKCE."""

    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    scope: str
    resource: str
    expires_at: datetime
    used: bool = False


@dataclass
class AccessToken:
    """Access token."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class OAuth2Server:
    """
    OAuth2 Authorization Server implementing MCP specification.

    Features:
    - OAuth 2.1 with PKCE (required by MCP)
    - Dynamic Client Registration (RFC 7591)
    - Authorization Server Metadata (RFC 8414)
    - Protected Resource Metadata (RFC 9728)
    """

    def __init__(
        self,
        issuer: str,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        authorization_code_expire_minutes: int = 10,
    ):
        """
        Initialize OAuth2 server.

        Args:
            issuer: OAuth2 issuer URL (e.g., "https://your-server.com")
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Access token expiration (default: 60 minutes)
            authorization_code_expire_minutes: Auth code expiration (default: 10 minutes)
        """
        self.issuer = issuer
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.authorization_code_expire_minutes = authorization_code_expire_minutes

        # In-memory storage (replace with database in production)
        self.clients: Dict[str, OAuth2Client] = {}
        self.authorization_codes: Dict[str, AuthorizationCode] = {}
        self.access_tokens: Dict[str, dict] = {}

    def get_authorization_server_metadata(self) -> dict:
        """
        Get OAuth 2.0 Authorization Server Metadata (RFC 8414).

        Returns:
            Authorization server metadata
        """
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}/authorize",
            "token_endpoint": f"{self.issuer}/token",
            "registration_endpoint": f"{self.issuer}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": [
                "client_secret_post",
                "client_secret_basic",
            ],
            "scopes_supported": ["mcp:tools", "mcp:resources", "mcp:prompts"],
        }

    def get_protected_resource_metadata(self, resource_url: str) -> dict:
        """
        Get Protected Resource Metadata (RFC 9728).

        Args:
            resource_url: MCP server URL

        Returns:
            Protected resource metadata
        """
        return {
            "resource": resource_url,
            "authorization_servers": [self.issuer],
            "scopes_supported": ["mcp:tools", "mcp:resources", "mcp:prompts"],
            "bearer_methods_supported": ["header"],
            "resource_signing_alg_values_supported": [self.algorithm],
        }

    def register_client(
        self,
        client_name: str,
        redirect_uris: List[str],
        grant_types: Optional[List[str]] = None,
        response_types: Optional[List[str]] = None,
    ) -> OAuth2Client:
        """
        Register a new OAuth2 client (Dynamic Client Registration - RFC 7591).

        Args:
            client_name: Client application name
            redirect_uris: List of allowed redirect URIs
            grant_types: Supported grant types
            response_types: Supported response types

        Returns:
            Registered OAuth2 client
        """
        # Generate client credentials
        client_id = f"mcp_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32)

        # Create client
        client = OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            client_name=client_name,
            redirect_uris=redirect_uris,
            grant_types=grant_types or ["authorization_code", "refresh_token"],
            response_types=response_types or ["code"],
        )

        # Store client
        self.clients[client_id] = client

        return client

    def create_authorization_code(
        self,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str,
        scope: str,
        resource: str,
    ) -> str:
        """
        Create authorization code with PKCE.

        Args:
            client_id: Client ID
            redirect_uri: Redirect URI
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE method (must be S256)
            scope: Requested scope
            resource: Target resource (MCP server URL)

        Returns:
            Authorization code
        """
        # Validate PKCE method
        if code_challenge_method != "S256":
            raise ValueError("Only S256 code challenge method is supported")

        # Generate authorization code
        code = secrets.token_urlsafe(32)

        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=self.authorization_code_expire_minutes
        )

        # Store authorization code
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scope=scope,
            resource=resource,
            expires_at=expires_at,
        )

        self.authorization_codes[code] = auth_code

        return code

    def verify_code_verifier(self, code: str, code_verifier: str) -> bool:
        """
        Verify PKCE code verifier.

        Args:
            code: Authorization code
            code_verifier: PKCE code verifier

        Returns:
            True if verifier is valid
        """
        auth_code = self.authorization_codes.get(code)
        if not auth_code:
            return False

        # Calculate code challenge from verifier
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        # Compare with stored challenge
        return challenge == auth_code.code_challenge

    def exchange_code_for_token(
        self,
        code: str,
        client_id: str,
        client_secret: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> AccessToken:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code
            client_id: Client ID
            client_secret: Client secret
            code_verifier: PKCE code verifier
            redirect_uri: Redirect URI

        Returns:
            Access token

        Raises:
            ValueError: If validation fails
        """
        # Validate client
        client = self.clients.get(client_id)
        if not client or client.client_secret != client_secret:
            raise ValueError("Invalid client credentials")

        # Validate authorization code
        auth_code = self.authorization_codes.get(code)
        if not auth_code:
            raise ValueError("Invalid authorization code")

        if auth_code.used:
            raise ValueError("Authorization code already used")

        if auth_code.expires_at < datetime.now(timezone.utc):
            raise ValueError("Authorization code expired")

        if auth_code.client_id != client_id:
            raise ValueError("Client ID mismatch")

        if auth_code.redirect_uri != redirect_uri:
            raise ValueError("Redirect URI mismatch")

        # Verify PKCE
        if not self.verify_code_verifier(code, code_verifier):
            raise ValueError("Invalid code verifier")

        # Mark code as used
        auth_code.used = True

        # Create access token
        access_token = self._create_access_token(
            client_id=client_id,
            scope=auth_code.scope,
            resource=auth_code.resource,
        )

        # Create refresh token
        refresh_token = self._create_refresh_token(client_id=client_id)

        return AccessToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60,
            scope=auth_code.scope,
        )

    def _create_access_token(
        self,
        client_id: str,
        scope: str,
        resource: str,
    ) -> str:
        """Create JWT access token."""
        expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        now = datetime.now(timezone.utc)
        expire = now + expires_delta

        to_encode = {
            "sub": client_id,
            "scope": scope,
            "aud": resource,  # Audience binding (RFC 8707)
            "iss": self.issuer,
            "exp": expire,
            "iat": now,
        }

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        # Store token for validation
        self.access_tokens[encoded_jwt] = to_encode

        return encoded_jwt

    def _create_refresh_token(self, client_id: str) -> str:
        """Create refresh token."""
        return secrets.token_urlsafe(32)

    def validate_access_token(self, token: str, resource: str) -> Optional[dict]:
        """
        Validate access token.

        Args:
            token: Access token
            resource: Expected resource (audience)

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience=resource,
                issuer=self.issuer,
            )
            return payload
        except JWTError:
            return None
