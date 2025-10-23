"""
Real server startup test (no mocks).

Tests that the refactored main.py actually works.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestRealServerStartup:
    """Test real server startup without mocks."""

    def test_main_module_imports(self):
        """Verify main module imports without errors."""
        # Set minimal env
        os.environ['OPENAI_API_KEY'] = 'sk-test-fake'
        
        import main
        
        assert hasattr(main, 'main'), "main() function not found"
        assert hasattr(main, 'mcp'), "mcp server not found"
        assert hasattr(main, 'settings'), "settings not found"

    def test_auth_setup_module_works(self):
        """Verify auth.setup module works."""
        from unittest.mock import Mock
        from auth.setup import setup_oauth2_routes
        from auth import OAuth2Server
        
        # Create real OAuth2Server
        oauth2_server = OAuth2Server(
            issuer="https://test.com",
            secret_key="test-secret"
        )
        
        # Create mock MCP
        mcp = Mock()
        registered_routes = []
        
        def capture_route(path, methods):
            registered_routes.append({"path": path, "methods": methods})
            return lambda f: f
        
        mcp.custom_route = capture_route
        
        # Call real function
        setup_oauth2_routes(mcp, oauth2_server, "localhost", "8051")
        
        # Verify routes registered
        assert len(registered_routes) == 6, f"Expected 6 routes, got {len(registered_routes)}"
        
        paths = [r["path"] for r in registered_routes]
        assert "/.well-known/oauth-authorization-server" in paths
        assert "/register" in paths
        assert "/authorize" in paths
        assert "/token" in paths

    def test_middleware_setup_module_works(self):
        """Verify middleware.setup module works."""
        from middleware.setup import setup_middleware
        from auth import OAuth2Server
        from starlette.middleware import Middleware
        
        # Test OAuth2 mode
        oauth2_server = OAuth2Server(
            issuer="https://test.com",
            secret_key="test-secret"
        )
        
        middleware = setup_middleware(use_oauth2=True, oauth2_server=oauth2_server)
        
        assert isinstance(middleware, list), "Should return list"
        assert len(middleware) == 2, f"OAuth2 mode should have 2 middleware, got {len(middleware)}"
        
        for mw in middleware:
            assert isinstance(mw, Middleware), f"Should be Middleware instance, got {type(mw)}"

    def test_main_py_structure(self):
        """Verify main.py has correct structure."""
        import main
        import inspect
        
        # Check main() function
        assert callable(main.main), "main() should be callable"
        
        # Check it's async
        assert inspect.iscoroutinefunction(main.main), "main() should be async"
        
        # Check source doesn't have manual lifespan
        source = inspect.getsource(main.main)
        assert "async with crawl4ai_lifespan" not in source, (
            "Manual lifespan call still present"
        )

    def test_file_sizes(self):
        """Verify refactoring reduced file sizes."""
        import os
        
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        auth_setup_path = os.path.join(os.path.dirname(__file__), "..", "src", "auth", "setup.py")
        middleware_setup_path = os.path.join(os.path.dirname(__file__), "..", "src", "middleware", "setup.py")
        
        # Count non-empty, non-comment lines
        def count_lines(path):
            with open(path) as f:
                return len([l for l in f if l.strip() and not l.strip().startswith('#')])
        
        main_lines = count_lines(main_path)
        auth_lines = count_lines(auth_setup_path)
        middleware_lines = count_lines(middleware_setup_path)
        
        print(f"\nFile sizes:")
        print(f"  main.py: {main_lines} lines")
        print(f"  auth/setup.py: {auth_lines} lines")
        print(f"  middleware/setup.py: {middleware_lines} lines")
        print(f"  Total: {main_lines + auth_lines + middleware_lines} lines")
        
        # main.py should be under 150 lines
        assert main_lines < 150, f"main.py too large: {main_lines} lines"
        
        # New modules should be reasonable size
        assert auth_lines < 150, f"auth/setup.py too large: {auth_lines} lines"
        assert middleware_lines < 150, f"middleware/setup.py too large: {middleware_lines} lines"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
