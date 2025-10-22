"""
Base analyzer for code extraction.

Provides abstract base class for language-specific code analyzers.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CodeAnalyzer(ABC):
    """Abstract base class for language-specific code analyzers."""

    def __init__(self):
        """Initialize the code analyzer."""
        self.logger = logger
        self.supported_extensions: List[str] = []

    @abstractmethod
    async def analyze_file(
        self,
        file_path: str,
        repo_path: str,
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a code file and extract structural information.

        Args:
            file_path: Path to the file to analyze
            repo_path: Root path of the repository
            content: Optional file content (if already loaded)

        Returns:
            Dictionary containing extracted code structure:
            {
                "file_path": str,
                "module_name": str,
                "imports": List[Dict],
                "classes": List[Dict],
                "functions": List[Dict],
                "variables": List[Dict],
                "exports": List[str],
                "dependencies": List[str],
            }
        """
        pass

    @abstractmethod
    def can_analyze(self, file_path: str) -> bool:
        """
        Check if this analyzer can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if the analyzer can handle this file type
        """
        pass

    def get_module_name(self, file_path: str, repo_path: str) -> str:
        """
        Generate module name from file path.

        Args:
            file_path: Path to the file
            repo_path: Root path of the repository

        Returns:
            Module name derived from the file path
        """
        try:
            # Get relative path from repo root
            rel_path = Path(file_path).relative_to(Path(repo_path))

            # Remove file extension
            module_path = rel_path.with_suffix("")

            # Convert path to module notation
            module_name = str(module_path).replace("/", ".").replace("\\", ".")

            # Remove index/main suffixes for cleaner names
            if module_name.endswith(".index"):
                module_name = module_name[:-6]
            elif module_name.endswith(".main"):
                module_name = module_name[:-5]

            return module_name

        except Exception:
            # Fallback to filename without extension
            return Path(file_path).stem

    async def read_file_content(self, file_path: str) -> Optional[str]:
        """
        Read file content with proper encoding handling.

        Args:
            file_path: Path to the file

        Returns:
            File content as string, or None if reading fails
        """
        try:
            # Try UTF-8 first
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Fallback to Latin-1
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Failed to read file {file_path}: {e}")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to read file {file_path}: {e}")
            return None

    def extract_docstring(self, lines: List[str], start_line: int) -> Optional[str]:
        """
        Extract docstring from code lines.

        Args:
            lines: List of code lines
            start_line: Starting line number

        Returns:
            Extracted docstring or None
        """
        if start_line >= len(lines):
            return None

        line = lines[start_line].strip()

        # Check for various docstring formats
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            if line.endswith(quote) and len(line) > 6:
                # Single-line docstring
                return line[3:-3].strip()
            else:
                # Multi-line docstring
                docstring_lines = [line[3:]] if len(line) > 3 else []
                for i in range(start_line + 1, len(lines)):
                    if quote in lines[i]:
                        docstring_lines.append(lines[i].split(quote)[0])
                        break
                    docstring_lines.append(lines[i])
                return "\n".join(docstring_lines).strip()

        return None

    def extract_line_range(
        self,
        lines: List[str],
        start_line: int,
        end_markers: Optional[List[str]] = None,
    ) -> tuple[int, int]:
        """
        Extract the line range for a code block.

        Args:
            lines: List of code lines
            start_line: Starting line number
            end_markers: Optional markers that indicate end of block

        Returns:
            Tuple of (start_line, end_line)
        """
        if not end_markers:
            end_markers = []

        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = start_line

        for i in range(start_line + 1, len(lines)):
            line = lines[i]

            # Check for end markers
            if any(marker in line for marker in end_markers):
                break

            # Check indentation
            if line.strip() and not line.startswith(" " * (indent_level + 1)):
                # Line with same or less indentation means end of block
                if len(line) - len(line.lstrip()) <= indent_level:
                    break

            end_line = i

        return start_line, end_line

    def sanitize_string(self, s: Optional[str]) -> str:
        """
        Sanitize string for storage.

        Args:
            s: String to sanitize

        Returns:
            Sanitized string
        """
        if not s:
            return ""
        # Remove null bytes and control characters
        return "".join(ch for ch in s if ch.isprintable() or ch.isspace())


# Create javascript.py file content
