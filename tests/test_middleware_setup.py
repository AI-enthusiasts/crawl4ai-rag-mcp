"""
Tests for middleware.setup module.

Tests middleware configuration for different authentication modes.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestSetupMiddleware:
    """Test middleware configuration."""

    def test_oauth2_mode_requires_oauth2_server(self):
        """OAuth2 mode must provide oauth2_server parameter."""
        from middleware.setup import setup_middleware

        with pytest.raises(ValueError, match="oauth2_server parameter is required"):
            setup_middleware(use_oauth2=True, oauth2_server=None)

    def test_oauth2_mode_returns_two_middleware(self):
        """OAuth2 mode returns SessionMiddleware + DualAuthMiddleware."""
        from middleware.setup import setup_middleware

        oauth2_server = Mock()
        middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        assert len(middleware) == 2, f"Expected 2 middleware, got {len(middleware)}"

    def test_oauth2_mode_first_is_session_middleware(self):
        """First middleware in OAuth2 mode is SessionMiddleware."""
        from middleware.setup import setup_middleware
        from starlette.middleware.sessions import SessionMiddleware

        oauth2_server = Mock()
        middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        assert middleware[0].cls == SessionMiddleware, (
            f"Expected SessionMiddleware, got {middleware[0].cls}"
        )

    def test_oauth2_mode_second_is_dual_auth_middleware(self):
        """Second middleware in OAuth2 mode is DualAuthMiddleware."""
        from middleware.setup import setup_middleware
        from auth import DualAuthMiddleware

        oauth2_server = Mock()
        middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        assert middleware[1].cls == DualAuthMiddleware, (
            f"Expected DualAuthMiddleware, got {middleware[1].cls}"
        )

    def test_oauth2_mode_passes_oauth2_server_to_dual_auth(self):
        """DualAuthMiddleware receives oauth2_server parameter."""
        from middleware.setup import setup_middleware

        oauth2_server = Mock()
        middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        # Check DualAuthMiddleware kwargs
        dual_auth_middleware = middleware[1]
        assert "oauth2_server" in dual_auth_middleware.kwargs, (
            "oauth2_server not passed to DualAuthMiddleware"
        )
        assert dual_auth_middleware.kwargs["oauth2_server"] is oauth2_server, (
            "Wrong oauth2_server instance passed"
        )

    def test_oauth2_mode_uses_secret_key_from_settings(self, monkeypatch):
        """SessionMiddleware uses secret_key from settings."""
        from middleware.setup import setup_middleware
        from config import get_settings

        oauth2_server = Mock()
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.oauth2_secret_key = "test-secret-123"
        monkeypatch.setattr("middleware.setup.settings", mock_settings)
        
        middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        session_middleware = middleware[0]
        assert session_middleware.kwargs["secret_key"] == "test-secret-123", (
            "SessionMiddleware not using settings.oauth2_secret_key"
        )

    def test_api_key_only_mode_returns_one_middleware(self):
        """API Key only mode returns single APIKeyMiddleware."""
        from middleware.setup import setup_middleware

        with patch("middleware.setup.settings.mcp_api_key", "test-api-key"):
            middleware = setup_middleware(use_oauth2=False)

            assert len(middleware) == 1, f"Expected 1 middleware, got {len(middleware)}"

    def test_api_key_only_mode_is_api_key_middleware(self):
        """API Key only mode uses APIKeyMiddleware."""
        from middleware.setup import setup_middleware
        from middleware import APIKeyMiddleware

        with patch("middleware.setup.settings.mcp_api_key", "test-api-key"):
            middleware = setup_middleware(use_oauth2=False)

            assert middleware[0].cls == APIKeyMiddleware, (
                f"Expected APIKeyMiddleware, got {middleware[0].cls}"
            )

    def test_no_auth_mode_returns_empty_list(self):
        """No authentication configured returns empty middleware list."""
        from middleware.setup import setup_middleware

        with patch("middleware.setup.settings.mcp_api_key", None):
            middleware = setup_middleware(use_oauth2=False)

            assert len(middleware) == 0, (
                f"Expected empty list, got {len(middleware)} middleware"
            )

    def test_oauth2_mode_ignores_api_key_setting(self):
        """OAuth2 mode doesn't add APIKeyMiddleware even if mcp_api_key is set."""
        from middleware.setup import setup_middleware

        oauth2_server = Mock()
        
        with patch("middleware.setup.settings.mcp_api_key", "test-api-key"):
            middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

            # Should only have 2 middleware (Session + DualAuth), not 3
            assert len(middleware) == 2, (
                f"OAuth2 mode should not add APIKeyMiddleware, got {len(middleware)} middleware"
            )


class TestLogging:
    """Test logging behavior."""

    def test_oauth2_mode_logs_session_middleware(self, caplog):
        """OAuth2 mode logs session middleware enabled."""
        from middleware.setup import setup_middleware

        oauth2_server = Mock()
        
        with caplog.at_level("INFO"):
            setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        assert any("Session middleware enabled" in record.message for record in caplog.records), (
            "Missing log: Session middleware enabled"
        )

    def test_oauth2_mode_logs_dual_auth(self, caplog):
        """OAuth2 mode logs dual authentication enabled."""
        from middleware.setup import setup_middleware

        oauth2_server = Mock()
        
        with caplog.at_level("INFO"):
            setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        assert any("OAuth2 + API Key dual authentication enabled" in record.message for record in caplog.records), (
            "Missing log: OAuth2 + API Key dual authentication enabled"
        )

    def test_api_key_mode_logs_api_key_auth(self, caplog):
        """API Key mode logs API key authentication enabled."""
        from middleware.setup import setup_middleware

        with patch("middleware.setup.settings.mcp_api_key", "test-key"):
            with caplog.at_level("INFO"):
                setup_middleware(use_oauth2=False)

        assert any("API Key authentication enabled" in record.message for record in caplog.records), (
            "Missing log: API Key authentication enabled"
        )

    def test_no_auth_mode_logs_warning(self, caplog):
        """No authentication mode logs warning."""
        from middleware.setup import setup_middleware

        with patch("middleware.setup.settings.mcp_api_key", None):
            with caplog.at_level("WARNING"):
                setup_middleware(use_oauth2=False)

        assert any("No authentication middleware configured" in record.message for record in caplog.records), (
            "Missing warning: No authentication middleware configured"
        )


class TestArchitectureCompliance:
    """Test architectural principles."""

    def test_module_has_single_public_function(self):
        """Verify module exports only setup_middleware as main function."""
        import middleware.setup as setup_module
        import types

        # Get functions defined in this module (not imported)
        public_functions = [
            name for name in dir(setup_module)
            if not name.startswith("_") 
            and callable(getattr(setup_module, name))
            and isinstance(getattr(setup_module, name), types.FunctionType)
            and getattr(setup_module, name).__module__ == "middleware.setup"
        ]

        assert public_functions == ["setup_middleware"], (
            f"Module should define only setup_middleware, got: {public_functions}"
        )

    def test_no_global_state(self):
        """Verify module doesn't create mutable global state."""
        import middleware.setup as setup_module

        # Check for global variables (excluding imports, constants, logger, settings)
        global_vars = [
            name for name in dir(setup_module)
            if not name.startswith("_")
            and not callable(getattr(setup_module, name))
            and name.upper() != name  # Exclude constants
            and name not in ["logger", "settings"]  # Acceptable globals
        ]

        assert len(global_vars) == 0, f"Module should not have global state: {global_vars}"

    def test_function_signature_is_clean(self):
        """Verify setup_middleware has clean signature."""
        from middleware.setup import setup_middleware
        import inspect

        sig = inspect.signature(setup_middleware)
        params = list(sig.parameters.keys())

        # Should have exactly 2 parameters
        assert params == ["use_oauth2", "oauth2_server"], (
            f"Expected [use_oauth2, oauth2_server], got {params}"
        )

        # Should have default values
        assert sig.parameters["use_oauth2"].default == False
        assert sig.parameters["oauth2_server"].default is None

    def test_returns_list_type(self):
        """Verify function returns List[Middleware]."""
        from middleware.setup import setup_middleware

        with patch("middleware.setup.settings.mcp_api_key", None):
            result = setup_middleware(use_oauth2=False)

        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_middleware_instances_are_starlette_middleware(self):
        """Verify returned middleware are Starlette Middleware instances."""
        from middleware.setup import setup_middleware
        from starlette.middleware import Middleware

        oauth2_server = Mock()
        middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)

        for mw in middleware:
            assert isinstance(mw, Middleware), (
                f"Expected Middleware instance, got {type(mw)}"
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_oauth2_server_none_with_use_oauth2_false_is_ok(self):
        """oauth2_server=None is OK when use_oauth2=False."""
        from middleware.setup import setup_middleware

        with patch("middleware.setup.settings.mcp_api_key", "test-key"):
            # Should not raise
            middleware = setup_middleware(use_oauth2=False, oauth2_server=None)
            assert len(middleware) == 1

    def test_oauth2_server_provided_but_use_oauth2_false(self):
        """oauth2_server provided but not used when use_oauth2=False."""
        from middleware.setup import setup_middleware

        oauth2_server = Mock()
        
        with patch("middleware.setup.settings.mcp_api_key", "test-key"):
            middleware = setup_middleware(use_oauth2=False, oauth2_server=oauth2_server)

            # Should only have APIKeyMiddleware, not DualAuthMiddleware
            assert len(middleware) == 1
            from middleware import APIKeyMiddleware
            assert middleware[0].cls == APIKeyMiddleware


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
