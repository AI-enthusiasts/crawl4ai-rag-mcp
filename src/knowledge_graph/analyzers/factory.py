"""
Code analyzer factory.

Provides a factory for creating language-specific code analyzers.
"""

import logging
from pathlib import Path
from typing import Optional

from .base import CodeAnalyzer
from .javascript import JavaScriptAnalyzer
from .go import GoAnalyzer

logger = logging.getLogger(__name__)


class AnalyzerFactory:
    """Factory for creating language-specific code analyzers."""

    def __init__(self):
        """Initialize the analyzer factory."""
        self.analyzers = {
            "python": None,  # Will use existing PythonAnalyzer
            "javascript": JavaScriptAnalyzer(),
            "typescript": JavaScriptAnalyzer(),
            "go": GoAnalyzer(),
        }
        
        # Map file extensions to analyzers
        self.extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript", 
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".go": "go",
        }

    def get_analyzer(self, file_path: str) -> Optional[CodeAnalyzer]:
        """
        Get the appropriate analyzer for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Appropriate analyzer instance or None
        """
        ext = Path(file_path).suffix.lower()
        
        # Get analyzer type from extension
        analyzer_type = self.extension_map.get(ext)
        
        if analyzer_type:
            analyzer = self.analyzers.get(analyzer_type)
            if analyzer:
                return analyzer
            elif analyzer_type == "python":
                # Return None for Python - will use existing analyzer
                return None
                
        logger.debug(f"No analyzer found for file extension: {ext}")
        return None

    def can_analyze(self, file_path: str) -> bool:
        """
        Check if any analyzer can handle the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file can be analyzed
        """
        analyzer = self.get_analyzer(file_path)
        if analyzer:
            return analyzer.can_analyze(file_path)
        
        # Check if it's a Python file (handled by existing analyzer)
        ext = Path(file_path).suffix.lower()
        return ext == ".py"

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(self.extension_map.keys())

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported programming languages.
        
        Returns:
            List of supported languages
        """
        return ["Python", "JavaScript", "TypeScript", "Go"]