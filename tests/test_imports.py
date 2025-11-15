"""Test that all modules can be imported without errors.

This test verifies the module structure is correct and there are no import-time errors.
It's critical for catching refactoring issues and should run in pre-commit hooks.
"""

import importlib
import pkgutil
from pathlib import Path

import pytest


def get_all_modules(package_name: str = "src") -> list[str]:
    """Recursively discover all Python modules in the package."""
    modules = []
    package_path = Path(__file__).parent.parent / package_name
    
    for root, dirs, files in package_path.walk():
        # Skip __pycache__ and other non-code directories
        dirs[:] = [d for d in dirs if not d.startswith("__") and not d.startswith(".")]
        
        for file in files:
            if file.endswith(".py") and not file.startswith("_"):
                # Convert file path to module path
                rel_path = Path(root).relative_to(package_path.parent)
                module_path = str(rel_path / file[:-3]).replace("/", ".")
                modules.append(module_path)
    
    return sorted(modules)


class TestImports:
    """Test all modules can be imported."""

    @pytest.mark.parametrize("module_name", get_all_modules())
    def test_module_imports(self, module_name: str):
        """Test that each module can be imported without errors."""
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing {module_name}: {e}")

    def test_all_core_modules_import(self):
        """Test critical core modules import successfully."""
        core_modules = [
            "src.config.settings",
            "src.core.context",
            "src.core.decorators",
            "src.core.exceptions",
            "src.database.factory",
            "src.database.qdrant_adapter",
            "src.services.crawling",
            "src.services.search",
            "src.utils.url_helpers",
            "src.utils.validation",
            "src.main",
        ]
        
        for module in core_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Critical module {module} failed to import: {e}")

    def test_no_circular_imports(self):
        """Test that importing main module doesn't cause circular imports."""
        # This will fail if there are circular dependencies
        try:
            import src.main
            assert src.main is not None
        except ImportError as e:
            pytest.fail(f"Circular import or missing dependency detected: {e}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v"])
