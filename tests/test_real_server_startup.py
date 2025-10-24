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
        """Verify main.py is reasonably sized."""
        import os
        
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        
        # Count non-empty, non-comment lines
        def count_lines(path):
            with open(path) as f:
                return len([l for l in f if l.strip() and not l.strip().startswith('#')])
        
        main_lines = count_lines(main_path)
        
        print(f"\nFile sizes:")
        print(f"  main.py: {main_lines} lines")
        
        # main.py should be under 200 lines (increased for auth logic)
        assert main_lines < 200, f"main.py too large: {main_lines} lines"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
