"""
Main entry point for the refactored Crawl4AI MCP server.

This demonstrates how the monolithic crawl4ai_mcp.py file can be refactored
into a modular structure following best practices.
"""

import asyncio
import sys
import traceback

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware

from config import get_settings
from core import crawl4ai_lifespan, logger
from middleware import APIKeyMiddleware
from tools import register_tools

# Get settings instance
settings = get_settings()

# Initialize FastMCP server with lifespan management
try:
    logger.info("Initializing FastMCP server...")
    # Get host and port from settings
    host = settings.host
    port = settings.port
    # Ensure port has a valid default even if empty string
    if not port:
        port = "8051"
    logger.info(f"Host: {host}, Port: {port}")

    mcp = FastMCP("Crawl4AI MCP Server", lifespan=crawl4ai_lifespan)
    logger.info("FastMCP server initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FastMCP server: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)


# Register all MCP tools
register_tools(mcp)


def create_mcp_server():
    """
    Create and return an MCP server instance for testing purposes.
    """
    return FastMCP("Crawl4AI MCP Server Test")


async def main():
    """
    Main async function to run the MCP server.
    """
    try:
        logger.info("Main function started")
        transport = settings.transport.lower()
        logger.info(f"Transport mode: {transport}")

        # Flush output before starting server
        sys.stdout.flush()
        sys.stderr.flush()

        # Run server with appropriate transport
        if transport == "http":
            # For HTTP transport, manually initialize the lifespan context
            # because HTTP mode doesn't automatically call lifespan managers
            logger.info("Initializing application context for HTTP transport...")
            async with crawl4ai_lifespan(mcp) as context:
                logger.info("✓ Application context initialized successfully")
                logger.info(f"  - Crawler: {type(context.crawler).__name__}")
                logger.info(f"  - Database: {type(context.database_client).__name__}")
                logger.info(
                    f"  - Reranking model: {'✓' if context.reranking_model else '✗'}",
                )
                logger.info(
                    f"  - Knowledge validator: {'✓' if context.knowledge_validator else '✗'}",
                )
                logger.info(
                    f"  - Repository extractor: {'✓' if context.repo_extractor else '✗'}",
                )

                # Run the HTTP server with the context active
                # Setup authentication middleware
                middleware = []

                # Add authentication middleware
                if settings.use_oauth2:
                    # OAuth2 + API Key dual authentication
                    from pathlib import Path
                    from jinja2 import Environment, FileSystemLoader
                    from starlette.responses import JSONResponse, RedirectResponse

                    from auth import OAuth2Server, DualAuthMiddleware
                    from auth.routes import (
                        ClientRegistrationRequest,
                        ClientRegistrationResponse,
                        TokenResponse,
                    )

                    # Add session middleware (required for OAuth2 authorization form)
                    middleware.append(
                        Middleware(
                            SessionMiddleware, secret_key=settings.oauth2_secret_key
                        )
                    )
                    logger.info("✓ Session middleware enabled")

                    # Initialize OAuth2 server
                    oauth2_server = OAuth2Server(
                        issuer=settings.oauth2_issuer,
                        secret_key=settings.oauth2_secret_key,
                    )
                    logger.info("✓ OAuth2 server initialized")

                    # Setup Jinja2 templates
                    template_dir = Path(__file__).parent / "auth" / "templates"
                    jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

                    def render_template(template_name: str, context: dict):
                        """Render Jinja2 template and return HTMLResponse."""
                        from starlette.responses import HTMLResponse

                        template = jinja_env.get_template(template_name)
                        html_content = template.render(**context)
                        return HTMLResponse(content=html_content)

                    # Add OAuth2 routes using custom_route
                    resource_url = (
                        f"https://{host}:{port}"
                        if host != "0.0.0.0"
                        else settings.oauth2_issuer
                    )

                    # Register OAuth2 metadata endpoints
                    @mcp.custom_route(
                        "/.well-known/oauth-authorization-server", methods=["GET"]
                    )
                    async def authorization_server_metadata(request):
                        metadata = oauth2_server.get_authorization_server_metadata()
                        return JSONResponse(metadata)

                    @mcp.custom_route(
                        "/.well-known/oauth-protected-resource", methods=["GET"]
                    )
                    async def protected_resource_metadata(request):
                        metadata = oauth2_server.get_protected_resource_metadata(
                            resource_url
                        )
                        return JSONResponse(metadata)

                    # Register OAuth2 endpoints
                    @mcp.custom_route("/register", methods=["POST"])
                    async def register_client(request):
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
                                {"error": "server_error", "error_description": str(e)},
                                status_code=500,
                            )

                    @mcp.custom_route("/authorize", methods=["GET"])
                    async def authorize_get(request):
                        try:
                            client_id = request.query_params.get("client_id")
                            redirect_uri = request.query_params.get("redirect_uri")
                            response_type = request.query_params.get("response_type")
                            scope = request.query_params.get("scope", "")
                            state = request.query_params.get("state")
                            code_challenge = request.query_params.get("code_challenge")
                            code_challenge_method = request.query_params.get(
                                "code_challenge_method"
                            )
                            resource = request.query_params.get("resource")

                            # Validate response type
                            if response_type != "code":
                                return JSONResponse(
                                    {"error": "unsupported_response_type"},
                                    status_code=400,
                                )

                            # Validate client
                            client = oauth2_server.clients.get(client_id)
                            if not client:
                                return JSONResponse(
                                    {"error": "invalid_client"}, status_code=400
                                )

                            # Validate redirect URI
                            if redirect_uri not in client.redirect_uris:
                                return JSONResponse(
                                    {"error": "invalid_redirect_uri"}, status_code=400
                                )

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
                                return RedirectResponse(
                                    url=f"{redirect_uri}?code={code}&state={state}"
                                )

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
                                {"error": "server_error", "error_description": str(e)},
                                status_code=500,
                            )

                    @mcp.custom_route("/authorize", methods=["POST"])
                    async def authorize_post(request):
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

                            # Validate API Key
                            if (
                                not settings.mcp_api_key
                                or api_key != settings.mcp_api_key
                            ):
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
                                return JSONResponse(
                                    {"error": "invalid_client"}, status_code=400
                                )

                            # Validate redirect URI
                            if redirect_uri not in client.redirect_uris:
                                return JSONResponse(
                                    {"error": "invalid_redirect_uri"}, status_code=400
                                )

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
                                url=f"{redirect_uri}?code={code}&state={state}",
                                status_code=303,
                            )
                        except Exception as e:
                            return JSONResponse(
                                {"error": "server_error", "error_description": str(e)},
                                status_code=500,
                            )

                    @mcp.custom_route("/token", methods=["POST"])
                    async def token_endpoint(request):
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
                                return JSONResponse(
                                    {"error": "unsupported_grant_type"}, status_code=400
                                )

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
                                    {
                                        "error": "invalid_grant",
                                        "error_description": str(e),
                                    },
                                    status_code=400,
                                )
                        except Exception as e:
                            return JSONResponse(
                                {"error": "server_error", "error_description": str(e)},
                                status_code=500,
                            )

                    logger.info("✓ OAuth2 endpoints registered")

                    # Add dual authentication middleware
                    middleware.append(
                        Middleware(DualAuthMiddleware, oauth2_server=oauth2_server)
                    )
                    logger.info("✓ OAuth2 + API Key dual authentication enabled")

                elif settings.mcp_api_key:
                    # API Key only authentication
                    middleware.append(Middleware(APIKeyMiddleware))
                    logger.info("✓ API Key authentication enabled")

                await mcp.run_http_async(
                    transport="http", host=host, port=int(port), middleware=middleware
                )
        elif transport == "sse":
            await mcp.run_sse_async()
        else:  # Default to stdio for Claude Desktop compatibility
            await mcp.run_stdio_async()

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting main function...")
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
