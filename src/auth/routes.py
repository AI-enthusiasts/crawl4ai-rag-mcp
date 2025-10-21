"""
OAuth2 endpoints for MCP server using Starlette.

Implements:
- Authorization Server Metadata (RFC 8414)
- Protected Resource Metadata (RFC 9728)
- Dynamic Client Registration (RFC 7591)
- Authorization endpoint with Jinja2 templates
- Token endpoint
"""

from typing import List, Optional
from pathlib import Path
import json
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response, HTMLResponse
from pydantic import BaseModel

from config import get_settings
from auth.oauth2_server import OAuth2Server

# Initialize Jinja2 templates
from jinja2 import Environment, FileSystemLoader

template_dir = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))


# Pydantic models for request/response validation
class ClientRegistrationRequest(BaseModel):
    """Dynamic Client Registration request (RFC 7591)."""

    client_name: str
    redirect_uris: List[str]
    grant_types: Optional[List[str]] = None
    response_types: Optional[List[str]] = None


class ClientRegistrationResponse(BaseModel):
    """Dynamic Client Registration response."""

    client_id: str
    client_secret: str
    client_name: str
    redirect_uris: List[str]
    grant_types: List[str]
    response_types: List[str]


class TokenResponse(BaseModel):
    """Token endpoint response."""

    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


# Helper function to render Jinja2 templates
def render_template(template_name: str, context: dict) -> HTMLResponse:
    """Render Jinja2 template."""
    template = jinja_env.get_template(template_name)
    html_content = template.render(**context)
    return HTMLResponse(content=html_content)


# OAuth2 endpoint handlers
async def authorization_server_metadata(request: Request, oauth2_server: OAuth2Server):
    """OAuth 2.0 Authorization Server Metadata (RFC 8414)."""
    return JSONResponse(oauth2_server.get_authorization_server_metadata())


async def protected_resource_metadata(
    request: Request, oauth2_server: OAuth2Server, resource_url: str
):
    """Protected Resource Metadata (RFC 9728)."""
    return JSONResponse(oauth2_server.get_protected_resource_metadata(resource_url))


async def register_client(request: Request, oauth2_server: OAuth2Server):
    """Dynamic Client Registration (RFC 7591)."""
    try:
        body = await request.json()
        req = ClientRegistrationRequest(**body)

        # Validate redirect URIs
        claude_callback = "https://claude.ai/api/mcp/auth_callback"
        if claude_callback not in req.redirect_uris:
            return JSONResponse(
                {
                    "error": "invalid_redirect_uri",
                    "error_description": f"redirect_uris must include {claude_callback}",
                },
                status_code=400,
            )

        # Register client
        client = oauth2_server.register_client(
            client_name=req.client_name,
            redirect_uris=req.redirect_uris,
            grant_types=req.grant_types,
            response_types=req.response_types,
        )

        response = ClientRegistrationResponse(
            client_id=client.client_id,
            client_secret=client.client_secret,
            client_name=client.client_name,
            redirect_uris=client.redirect_uris,
            grant_types=client.grant_types,
            response_types=client.response_types,
        )

        return JSONResponse(response.dict())
    except Exception as e:
        return JSONResponse(
            {"error": "server_error", "error_description": str(e)}, status_code=500
        )


async def authorize_get(request: Request, oauth2_server: OAuth2Server):
    """Authorization endpoint (GET) - shows login form."""
    try:
        client_id = request.query_params.get("client_id")
        redirect_uri = request.query_params.get("redirect_uri")
        response_type = request.query_params.get("response_type")
        scope = request.query_params.get("scope", "")
        state = request.query_params.get("state")
        code_challenge = request.query_params.get("code_challenge")
        code_challenge_method = request.query_params.get("code_challenge_method")
        resource = request.query_params.get("resource")

        # Validate response type
        if response_type != "code":
            return JSONResponse({"error": "unsupported_response_type"}, status_code=400)

        # Validate client
        client = oauth2_server.clients.get(client_id)
        if not client:
            return JSONResponse({"error": "invalid_client"}, status_code=400)

        # Validate redirect URI
        if redirect_uri not in client.redirect_uris:
            return JSONResponse({"error": "invalid_redirect_uri"}, status_code=400)

        # Check if user is already authenticated
        if request.session.get("authenticated"):
            # Create authorization code
            code = oauth2_server.create_authorization_code(
                client_id=client_id,
                redirect_uri=redirect_uri,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
                scope=scope,
                resource=resource,
            )
            return RedirectResponse(url=f"{redirect_uri}?code={code}&state={state}")

        # Show login form
        return render_template(
            "authorize.html",
            {
                "request": request,
                "client_name": client.client_name,
                "scopes": scope.split() if scope else [],
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": response_type,
                "scope": scope,
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": code_challenge_method,
                "resource": resource,
            },
        )

    except Exception as e:
        return JSONResponse(
            {"error": "server_error", "error_description": str(e)}, status_code=500
        )


async def authorize_post(request: Request, oauth2_server: OAuth2Server):
    """Authorization endpoint (POST) - processes login form."""
    try:
        form = await request.form()
        api_key = form.get("api_key")
        client_id = form.get("client_id")
        redirect_uri = form.get("redirect_uri")
        response_type = form.get("response_type")
        scope = form.get("scope")
        state = form.get("state")
        code_challenge = form.get("code_challenge")
        code_challenge_method = form.get("code_challenge_method")
        resource = form.get("resource")

        settings = get_settings()

        # Validate API Key
        if not settings.mcp_api_key or api_key != settings.mcp_api_key:
            return render_template(
                "error.html",
                {
                    "request": request,
                    "icon": "❌",
                    "title": "Invalid API Key",
                    "message": "The API key you entered is incorrect. Please check your credentials and try again.",
                },
            )

        # Save authentication in session
        request.session["authenticated"] = True

        # Validate client
        client = oauth2_server.clients.get(client_id)
        if not client:
            return JSONResponse({"error": "invalid_client"}, status_code=400)

        # Validate redirect URI
        if redirect_uri not in client.redirect_uris:
            return JSONResponse({"error": "invalid_redirect_uri"}, status_code=400)

        # Create authorization code
        code = oauth2_server.create_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scope=scope,
            resource=resource,
        )

        return RedirectResponse(
            url=f"{redirect_uri}?code={code}&state={state}", status_code=303
        )

    except Exception as e:
        return JSONResponse(
            {"error": "server_error", "error_description": str(e)}, status_code=500
        )


async def token_endpoint(request: Request, oauth2_server: OAuth2Server):
    """Token endpoint - exchanges authorization code for access token."""
    try:
        form = await request.form()
        grant_type = form.get("grant_type")
        code = form.get("code")
        client_id = form.get("client_id")
        client_secret = form.get("client_secret")
        code_verifier = form.get("code_verifier")
        redirect_uri = form.get("redirect_uri")

        # Validate grant type
        if grant_type != "authorization_code":
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

        # Exchange code for token
        try:
            access_token = oauth2_server.exchange_code_for_token(
                code=code,
                client_id=client_id,
                client_secret=client_secret,
                code_verifier=code_verifier,
                redirect_uri=redirect_uri,
            )

            response = TokenResponse(
                access_token=access_token.access_token,
                token_type=access_token.token_type,
                expires_in=access_token.expires_in,
                refresh_token=access_token.refresh_token,
                scope=access_token.scope,
            )

            return JSONResponse(response.dict())
        except ValueError as e:
            return JSONResponse(
                {"error": "invalid_grant", "error_description": str(e)}, status_code=400
            )

    except Exception as e:
        return JSONResponse(
            {"error": "server_error", "error_description": str(e)}, status_code=500
        )
