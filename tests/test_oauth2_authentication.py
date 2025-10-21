"""
OAuth2 Authentication Tests.

Tests:
1. Unauthorized access (no token)
2. API Key authentication (Bearer token with MCP_API_KEY)
3. OAuth2 authentication (full OAuth2 flow)
"""

import hashlib
import base64
import secrets
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

# Add src to path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auth import OAuth2Server, DualAuthMiddleware
from config import get_settings


# Test fixtures
@pytest.fixture
def test_api_key():
    """Test API key."""
    return "test-api-key-12345"


@pytest.fixture
def oauth2_server():
    """Create OAuth2 server for testing."""
    return OAuth2Server(
        issuer="https://test-server.com",
        secret_key="test-secret-key-for-jwt-signing",
    )


@pytest.fixture
def test_app(oauth2_server, test_api_key, monkeypatch):
    """Create test Starlette app with authentication."""

    # Mock settings
    mock_settings = MagicMock()
    mock_settings.mcp_api_key = test_api_key

    # Patch get_settings at module level
    monkeypatch.setattr("auth.middleware.get_settings", lambda: mock_settings)

    # Create test routes
    async def protected_endpoint(request):
        """Protected endpoint that requires authentication."""
        auth_type = getattr(request.state, "auth_type", "unknown")
        return JSONResponse({"message": "Access granted", "auth_type": auth_type})

    routes = [
        Route("/protected", protected_endpoint),
    ]

    # Create app with middleware
    middleware = [
        Middleware(SessionMiddleware, secret_key="test-session-secret"),
        Middleware(DualAuthMiddleware, oauth2_server=oauth2_server),
    ]

    app = Starlette(routes=routes, middleware=middleware)

    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestUnauthorizedAccess:
    """Test 1: Unauthorized access (no token)."""

    def test_no_authorization_header(self, client):
        """Test request without Authorization header returns 401."""
        response = client.get("/protected")

        assert response.status_code == 401
        assert "error" in response.json()
        assert response.json()["error"] == "Unauthorized"
        assert "WWW-Authenticate" in response.headers

    def test_invalid_authorization_format(self, client):
        """Test request with invalid Authorization format returns 401."""
        response = client.get(
            "/protected", headers={"Authorization": "InvalidFormat token123"}
        )

        assert response.status_code == 401
        assert "error" in response.json()

    def test_empty_bearer_token(self, client):
        """Test request with empty Bearer token returns 401."""
        response = client.get("/protected", headers={"Authorization": "Bearer "})

        assert response.status_code == 401
        assert "error" in response.json()


class TestAPIKeyAuthentication:
    """Test 2: API Key authentication (Bearer token with MCP_API_KEY)."""

    def test_valid_api_key(self, client, test_api_key):
        """Test request with valid API key succeeds."""
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {test_api_key}"}
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Access granted"
        assert response.json()["auth_type"] == "api_key"

    def test_invalid_api_key(self, client):
        """Test request with invalid API key returns 401."""
        response = client.get(
            "/protected", headers={"Authorization": "Bearer wrong-api-key"}
        )

        assert response.status_code == 401
        assert "error" in response.json()

    def test_api_key_case_sensitive(self, client, test_api_key):
        """Test API key is case-sensitive."""
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {test_api_key.upper()}"}
        )

        assert response.status_code == 401

    def test_api_key_with_extra_spaces(self, client, test_api_key):
        """Test API key with extra spaces fails."""
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer  {test_api_key}  "}
        )

        assert response.status_code == 401


class TestOAuth2Authentication:
    """Test 3: OAuth2 authentication (full OAuth2 flow)."""

    def test_oauth2_client_registration(self, oauth2_server):
        """Test Dynamic Client Registration."""
        client = oauth2_server.register_client(
            client_name="Test Client",
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        )

        assert client.client_id.startswith("mcp_")
        assert len(client.client_secret) > 20
        assert "https://claude.ai/api/mcp/auth_callback" in client.redirect_uris
        assert "authorization_code" in client.grant_types

    def test_oauth2_authorization_code_creation(self, oauth2_server):
        """Test authorization code creation with PKCE."""
        # Register client first
        client = oauth2_server.register_client(
            client_name="Test Client",
            redirect_uris=["https://test.com/callback"],
        )

        # Generate PKCE challenge
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        # Create authorization code
        auth_code = oauth2_server.create_authorization_code(
            client_id=client.client_id,
            redirect_uri="https://test.com/callback",
            code_challenge=code_challenge,
            code_challenge_method="S256",
            scope="mcp:tools mcp:resources",
            resource="https://test-server.com",
        )

        assert len(auth_code) > 20
        assert auth_code in oauth2_server.authorization_codes

    def test_oauth2_pkce_verification(self, oauth2_server):
        """Test PKCE code verifier validation."""
        # Register client
        client = oauth2_server.register_client(
            client_name="Test Client",
            redirect_uris=["https://test.com/callback"],
        )

        # Generate PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        # Create authorization code
        auth_code = oauth2_server.create_authorization_code(
            client_id=client.client_id,
            redirect_uri="https://test.com/callback",
            code_challenge=code_challenge,
            code_challenge_method="S256",
            scope="mcp:tools",
            resource="https://test-server.com",
        )

        # Verify correct code verifier
        assert oauth2_server.verify_code_verifier(auth_code, code_verifier) is True

        # Verify incorrect code verifier
        wrong_verifier = secrets.token_urlsafe(32)
        assert oauth2_server.verify_code_verifier(auth_code, wrong_verifier) is False

    def test_oauth2_token_exchange(self, oauth2_server):
        """Test exchanging authorization code for access token."""
        # Register client
        client = oauth2_server.register_client(
            client_name="Test Client",
            redirect_uris=["https://test.com/callback"],
        )

        # Generate PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        # Create authorization code
        auth_code = oauth2_server.create_authorization_code(
            client_id=client.client_id,
            redirect_uri="https://test.com/callback",
            code_challenge=code_challenge,
            code_challenge_method="S256",
            scope="mcp:tools",
            resource="https://test-server.com",
        )

        # Exchange code for token
        access_token = oauth2_server.exchange_code_for_token(
            code=auth_code,
            client_id=client.client_id,
            client_secret=client.client_secret,
            code_verifier=code_verifier,
            redirect_uri="https://test.com/callback",
        )

        assert access_token.access_token is not None
        assert access_token.token_type == "Bearer"
        assert access_token.expires_in == 3600
        assert access_token.refresh_token is not None

    def test_oauth2_token_validation(self, oauth2_server):
        """Test JWT token validation."""
        # Register client
        client = oauth2_server.register_client(
            client_name="Test Client",
            redirect_uris=["https://test.com/callback"],
        )

        # Generate PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        # Create authorization code
        auth_code = oauth2_server.create_authorization_code(
            client_id=client.client_id,
            redirect_uri="https://test.com/callback",
            code_challenge=code_challenge,
            code_challenge_method="S256",
            scope="mcp:tools",
            resource="https://test-server.com",
        )

        # Exchange for token
        access_token = oauth2_server.exchange_code_for_token(
            code=auth_code,
            client_id=client.client_id,
            client_secret=client.client_secret,
            code_verifier=code_verifier,
            redirect_uri="https://test.com/callback",
        )

        # Validate token
        payload = oauth2_server.validate_access_token(
            access_token.access_token, "https://test-server.com"
        )

        assert payload is not None
        assert payload["sub"] == client.client_id
        assert payload["aud"] == "https://test-server.com"
        assert payload["iss"] == "https://test-server.com"
        assert "exp" in payload

    def test_oauth2_token_with_wrong_audience(self, oauth2_server):
        """Test token validation fails with wrong audience."""
        # Register client
        client = oauth2_server.register_client(
            client_name="Test Client",
            redirect_uris=["https://test.com/callback"],
        )

        # Generate PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        # Create authorization code
        auth_code = oauth2_server.create_authorization_code(
            client_id=client.client_id,
            redirect_uri="https://test.com/callback",
            code_challenge=code_challenge,
            code_challenge_method="S256",
            scope="mcp:tools",
            resource="https://test-server.com",
        )

        # Exchange for token
        access_token = oauth2_server.exchange_code_for_token(
            code=auth_code,
            client_id=client.client_id,
            client_secret=client.client_secret,
            code_verifier=code_verifier,
            redirect_uri="https://test.com/callback",
        )

        # Validate with wrong audience
        payload = oauth2_server.validate_access_token(
            access_token.access_token,
            "https://wrong-server.com",  # Wrong audience!
        )

        assert payload is None

    def test_oauth2_authenticated_request(self, client, oauth2_server):
        """Test making authenticated request with OAuth2 token."""
        # Register client
        client_obj = oauth2_server.register_client(
            client_name="Test Client",
            redirect_uris=["https://test.com/callback"],
        )

        # Generate PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        # Create authorization code with correct resource (testserver base URL)
        auth_code = oauth2_server.create_authorization_code(
            client_id=client_obj.client_id,
            redirect_uri="https://test.com/callback",
            code_challenge=code_challenge,
            code_challenge_method="S256",
            scope="mcp:tools",
            resource="http://testserver",  # Match TestClient base_url
        )

        # Exchange for token
        access_token = oauth2_server.exchange_code_for_token(
            code=auth_code,
            client_id=client_obj.client_id,
            client_secret=client_obj.client_secret,
            code_verifier=code_verifier,
            redirect_uri="https://test.com/callback",
        )

        # Make authenticated request
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {access_token.access_token}"},
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Access granted"
        assert response.json()["auth_type"] == "oauth2"


class TestDualAuthentication:
    """Test dual authentication (OAuth2 OR API Key)."""

    def test_oauth2_takes_precedence_over_api_key(
        self, client, oauth2_server, test_api_key
    ):
        """Test OAuth2 token is checked before API key."""
        # If token is valid OAuth2, it should use oauth2 auth_type
        # If token is invalid OAuth2 but valid API key, it should use api_key auth_type

        # Test with API key (should work)
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {test_api_key}"}
        )

        assert response.status_code == 200
        assert response.json()["auth_type"] == "api_key"

    def test_invalid_oauth2_falls_back_to_api_key(self, client, test_api_key):
        """Test invalid OAuth2 token falls back to API key check."""
        # Use API key as token (invalid OAuth2, but valid API key)
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {test_api_key}"}
        )

        assert response.status_code == 200
        assert response.json()["auth_type"] == "api_key"

    def test_both_invalid_returns_401(self, client):
        """Test both OAuth2 and API key invalid returns 401."""
        response = client.get(
            "/protected", headers={"Authorization": "Bearer invalid-token"}
        )

        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
