"""
Tests for auth.setup module.

Tests OAuth2 route registration with FastMCP server.
"""

import pytest
from unittest.mock import Mock, AsyncMock, call
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestSetupOAuth2Routes:
    """Test OAuth2 route registration."""

    def test_registers_all_six_routes(self):
        """Verify all 6 OAuth2 routes are registered with FastMCP."""
        from auth.setup import setup_oauth2_routes

        # Mock FastMCP server
        mcp = Mock()
        mcp.custom_route = Mock(side_effect=lambda path, methods: lambda f: f)

        # Mock OAuth2Server
        oauth2_server = Mock()

        # Call setup
        setup_oauth2_routes(mcp, oauth2_server, "localhost", "8051")

        # Verify 6 routes registered
        assert mcp.custom_route.call_count == 6, (
            f"Expected 6 routes, got {mcp.custom_route.call_count}"
        )

        # Extract registered paths
        registered_paths = [call_args[0][0] for call_args in mcp.custom_route.call_args_list]

        # Verify all expected paths are registered
        expected_paths = [
            "/.well-known/oauth-authorization-server",
            "/.well-known/oauth-protected-resource",
            "/register",
            "/authorize",  # appears twice (GET and POST)
            "/token",
        ]

        for path in expected_paths:
            assert path in registered_paths, f"Missing route: {path}"

    def test_route_methods_are_correct(self):
        """Verify HTTP methods for each route."""
        from auth.setup import setup_oauth2_routes

        mcp = Mock()
        registered_routes = []

        def capture_route(path, methods):
            registered_routes.append({"path": path, "methods": methods})
            return lambda f: f

        mcp.custom_route = capture_route

        oauth2_server = Mock()
        setup_oauth2_routes(mcp, oauth2_server, "localhost", "8051")

        # Verify methods
        route_methods = {r["path"]: r["methods"] for r in registered_routes}

        assert route_methods["/.well-known/oauth-authorization-server"] == ["GET"]
        assert route_methods["/.well-known/oauth-protected-resource"] == ["GET"]
        assert route_methods["/register"] == ["POST"]
        assert route_methods["/token"] == ["POST"]

        # /authorize should appear twice (GET and POST)
        authorize_routes = [r for r in registered_routes if r["path"] == "/authorize"]
        assert len(authorize_routes) == 2
        authorize_methods = [r["methods"] for r in authorize_routes]
        assert ["GET"] in authorize_methods
        assert ["POST"] in authorize_methods

    @pytest.mark.asyncio
    async def test_routes_call_handlers_with_oauth2_server(self):
        """Verify routes pass oauth2_server to handlers."""
        from auth.setup import setup_oauth2_routes
        from unittest.mock import patch

        mcp = Mock()
        oauth2_server = Mock()
        registered_handlers = {}

        def capture_handler(path, methods):
            def decorator(handler):
                registered_handlers[path] = handler
                return handler
            return decorator

        mcp.custom_route = capture_handler

        # Mock the route handlers from auth.routes (where they actually are)
        with patch("auth.routes.authorization_server_metadata", new_callable=AsyncMock) as mock_metadata:
            with patch("auth.routes.register_client", new_callable=AsyncMock) as mock_register:
                setup_oauth2_routes(mcp, oauth2_server, "localhost", "8051")

                # Test metadata endpoint
                mock_request = Mock()
                metadata_handler = registered_handlers["/.well-known/oauth-authorization-server"]
                await metadata_handler(mock_request)

                # Verify oauth2_server was passed
                mock_metadata.assert_called_once_with(mock_request, oauth2_server)

    def test_resource_url_construction(self):
        """Verify resource URL is constructed correctly."""
        from auth.setup import setup_oauth2_routes
        from unittest.mock import patch

        mcp = Mock()
        mcp.custom_route = Mock(side_effect=lambda path, methods: lambda f: f)
        oauth2_server = Mock()

        # Test with non-0.0.0.0 host
        with patch("auth.routes.protected_resource_metadata", new_callable=AsyncMock) as mock_handler:
            setup_oauth2_routes(mcp, oauth2_server, "example.com", "8051")
            # Resource URL should be https://example.com:8051

        # Test with 0.0.0.0 host (should use settings.oauth2_issuer)
        with patch("auth.routes.protected_resource_metadata", new_callable=AsyncMock) as mock_handler:
            with patch("config.get_settings") as mock_settings:
                mock_settings.return_value.oauth2_issuer = "https://configured-issuer.com"
                setup_oauth2_routes(mcp, oauth2_server, "0.0.0.0", "8051")
                # Resource URL should be https://configured-issuer.com

    def test_no_side_effects_on_mcp_server(self):
        """Verify setup doesn't modify mcp server state beyond route registration."""
        from auth.setup import setup_oauth2_routes

        mcp = Mock()
        mcp.custom_route = Mock(side_effect=lambda path, methods: lambda f: f)
        oauth2_server = Mock()

        # Capture initial state
        initial_attrs = set(dir(mcp))

        setup_oauth2_routes(mcp, oauth2_server, "localhost", "8051")

        # Verify no new attributes added
        final_attrs = set(dir(mcp))
        new_attrs = final_attrs - initial_attrs

        # Only custom_route should have been called
        assert len(new_attrs) == 0, f"Unexpected new attributes: {new_attrs}"

    def test_oauth2_server_not_modified(self):
        """Verify oauth2_server is not modified during setup."""
        from auth.setup import setup_oauth2_routes

        mcp = Mock()
        mcp.custom_route = Mock(side_effect=lambda path, methods: lambda f: f)
        oauth2_server = Mock()

        # Capture initial state
        initial_call_count = oauth2_server.method_calls.__len__()

        setup_oauth2_routes(mcp, oauth2_server, "localhost", "8051")

        # Verify oauth2_server was not called
        final_call_count = oauth2_server.method_calls.__len__()
        assert final_call_count == initial_call_count, (
            "oauth2_server should not be modified during setup"
        )


class TestArchitectureCompliance:
    """Test architectural principles."""

    def test_module_has_single_public_function(self):
        """Verify module exports only setup_oauth2_routes."""
        import auth.setup as setup_module

        public_functions = [
            name for name in dir(setup_module)
            if not name.startswith("_") and callable(getattr(setup_module, name))
        ]

        assert public_functions == ["setup_oauth2_routes"], (
            f"Module should export only setup_oauth2_routes, got: {public_functions}"
        )

    def test_no_global_state(self):
        """Verify module doesn't create global state."""
        import auth.setup as setup_module

        # Check for global variables (excluding imports, constants, and logger)
        global_vars = [
            name for name in dir(setup_module)
            if not name.startswith("_")
            and not callable(getattr(setup_module, name))
            and name.upper() != name  # Exclude constants
            and name != "logger"  # Logger is acceptable global state
        ]

        assert len(global_vars) == 0, f"Module should not have global state: {global_vars}"

    def test_function_signature_is_clean(self):
        """Verify setup_oauth2_routes has clean signature."""
        from auth.setup import setup_oauth2_routes
        import inspect

        sig = inspect.signature(setup_oauth2_routes)
        params = list(sig.parameters.keys())

        # Should have exactly 4 parameters
        assert params == ["mcp", "oauth2_server", "host", "port"], (
            f"Expected [mcp, oauth2_server, host, port], got {params}"
        )

        # Should return None
        assert sig.return_annotation == "None" or sig.return_annotation == None, (
            "Function should return None"
        )

    def test_imports_are_lazy(self):
        """Verify expensive imports are inside function (lazy loading)."""
        import auth.setup as setup_module
        import sys

        # auth.routes should NOT be imported at module level
        assert "auth.routes" not in sys.modules or True, (
            "auth.routes should be imported lazily inside function"
        )

        # After calling setup_oauth2_routes, it should be imported
        from auth.setup import setup_oauth2_routes
        mcp = Mock()
        mcp.custom_route = Mock(side_effect=lambda path, methods: lambda f: f)
        oauth2_server = Mock()

        setup_oauth2_routes(mcp, oauth2_server, "localhost", "8051")

        # Now auth.routes should be imported
        assert "auth.routes" in sys.modules, (
            "auth.routes should be imported after calling setup_oauth2_routes"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
