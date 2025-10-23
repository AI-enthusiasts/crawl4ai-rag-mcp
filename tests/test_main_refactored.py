"""
Tests for refactored main.py.

Verifies that Phase 7 refactoring maintains functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestMainRefactoring:
    """Test main.py refactoring."""

    def test_main_py_line_count(self):
        """Verify main.py is under 150 lines (was 419)."""
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        with open(main_path, 'r') as f:
            lines = [l for l in f.readlines() if l.strip() and not l.strip().startswith('#')]
        
        assert len(lines) < 150, f"main.py has {len(lines)} lines (should be < 150)"

    def test_no_manual_lifespan_call(self):
        """Verify manual lifespan call is removed (fixes browser leak)."""
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Should NOT contain manual lifespan call
        assert "async with crawl4ai_lifespan" not in content, (
            "Manual lifespan call still present (causes browser leak)"
        )
        assert "as context:" not in content, (
            "Context manager still present"
        )

    def test_no_nested_route_definitions(self):
        """Verify OAuth2 routes are not defined in main() (moved to auth.setup)."""
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Should NOT contain route definitions
        assert "@mcp.custom_route" not in content, (
            "Route definitions still in main.py (should be in auth.setup)"
        )
        assert "async def authorize_get" not in content
        assert "async def token_endpoint" not in content

    def test_uses_setup_modules(self):
        """Verify main.py uses new setup modules."""
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Should import and use setup modules
        assert "from auth.setup import setup_oauth2_routes" in content, (
            "Missing import: auth.setup.setup_oauth2_routes"
        )
        assert "from middleware.setup import setup_middleware" in content, (
            "Missing import: middleware.setup.setup_middleware"
        )
        assert "setup_oauth2_routes(mcp, oauth2_server, host, port)" in content, (
            "Missing call: setup_oauth2_routes()"
        )
        assert "setup_middleware(" in content, (
            "Missing call: setup_middleware()"
        )

    def test_main_function_structure(self):
        """Verify main() function has clean structure."""
        import inspect
        from main import main
        
        source = inspect.getsource(main)
        lines = [l for l in source.split('\n') if l.strip() and not l.strip().startswith('#')]
        
        # main() should be under 60 lines
        assert len(lines) < 60, f"main() has {len(lines)} lines (should be < 60)"

    @pytest.mark.asyncio
    async def test_http_mode_calls_setup_modules(self):
        """Verify HTTP mode calls setup modules correctly."""
        with patch("main.settings") as mock_settings:
            mock_settings.transport = "http"
            mock_settings.use_oauth2 = True
            mock_settings.oauth2_issuer = "https://test.com"
            mock_settings.oauth2_secret_key = "test-secret"
            mock_settings.host = "localhost"
            mock_settings.port = "8051"
            
            with patch("main.OAuth2Server") as mock_oauth2_class:
                with patch("main.setup_oauth2_routes") as mock_setup_routes:
                    with patch("main.setup_middleware") as mock_setup_middleware:
                        with patch("main.mcp.run_http_async", new_callable=AsyncMock) as mock_run:
                            from main import main
                            
                            await main()
                            
                            # Verify OAuth2Server was created
                            mock_oauth2_class.assert_called_once()
                            
                            # Verify setup_oauth2_routes was called
                            mock_setup_routes.assert_called_once()
                            
                            # Verify setup_middleware was called
                            mock_setup_middleware.assert_called_once()
                            
                            # Verify run_http_async was called
                            mock_run.assert_called_once()

    def test_no_unused_imports(self):
        """Verify unused imports were removed."""
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # These imports should be removed (not used anymore)
        unused_imports = [
            "from pathlib import Path",
            "from jinja2 import",
            "from starlette.middleware.sessions import SessionMiddleware",
            "from starlette.responses import JSONResponse",
            "from auth.routes import ClientRegistrationRequest",
        ]
        
        for imp in unused_imports:
            assert imp not in content, f"Unused import still present: {imp}"


class TestBrowserLeakFix:
    """Test that browser leak is fixed."""

    def test_single_lifespan_registration(self):
        """Verify lifespan is only registered once (at FastMCP init)."""
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Count lifespan references
        lifespan_count = content.count("lifespan=crawl4ai_lifespan")
        
        assert lifespan_count == 1, (
            f"Expected 1 lifespan registration, found {lifespan_count}"
        )

    def test_comment_explains_fix(self):
        """Verify comment explains the browser leak fix."""
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Should have comment explaining the fix
        assert "FastMCP automatically calls crawl4ai_lifespan" in content, (
            "Missing comment explaining automatic lifespan management"
        )
        assert "fixes browser process leak" in content or "No manual lifespan management" in content, (
            "Missing comment about browser leak fix"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
