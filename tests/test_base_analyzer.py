"""
Unit tests for the base code analyzer.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.knowledge_graph.analyzers.base import CodeAnalyzer


class MockAnalyzer(CodeAnalyzer):
    """Concrete implementation of CodeAnalyzer for testing."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = [".py", ".js"]

    async def analyze_file(self, file_path: str, repo_path: str, content=None):
        return {"file_path": file_path}

    def can_analyze(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions


class TestBaseAnalyzer:
    """Test the base code analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MockAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.logger is not None
        assert self.analyzer.supported_extensions == [".py", ".js"]

    def test_can_analyze(self):
        """Test file analysis capability detection."""
        # Test supported extensions
        assert self.analyzer.can_analyze("test.py") is True
        assert self.analyzer.can_analyze("script.js") is True

        # Test unsupported extensions
        assert self.analyzer.can_analyze("test.go") is False
        assert self.analyzer.can_analyze("doc.md") is False

        # Test case insensitivity
        assert self.analyzer.can_analyze("Test.PY") is True
        assert self.analyzer.can_analyze("Script.JS") is True

    def test_get_module_name_basic(self):
        """Test basic module name generation."""
        # Basic path conversion
        result = self.analyzer.get_module_name("/repo/src/utils/helpers.py", "/repo")
        assert result == "src.utils.helpers"

    def test_get_module_name_index_suffix(self):
        """Test module name generation with index suffix removal."""
        result = self.analyzer.get_module_name("/repo/src/components/index.js", "/repo")
        assert result == "src.components"

    def test_get_module_name_main_suffix(self):
        """Test module name generation with main suffix removal."""
        result = self.analyzer.get_module_name("/repo/src/main.py", "/repo")
        assert result == "src"

    def test_get_module_name_windows_paths(self):
        """Test module name generation with Windows-style paths."""
        result = self.analyzer.get_module_name(
            "C:\\repo\\src\\utils\\helpers.py", "C:\\repo"
        )
        assert result == "src.utils.helpers"

    def test_get_module_name_exception_fallback(self):
        """Test module name generation falls back to stem on exception."""
        # Invalid relative path should trigger exception
        result = self.analyzer.get_module_name("/different/path/file.py", "/repo")
        assert result == "file"

    @pytest.mark.asyncio
    async def test_read_file_content_success_utf8(self):
        """Test successful file reading with UTF-8 encoding."""
        content = "def hello():\n    return 'world'"

        with patch("builtins.open", mock_open(read_data=content)):
            result = await self.analyzer.read_file_content("/test/file.py")

        assert result == content

    @pytest.mark.asyncio
    async def test_read_file_content_fallback_latin1(self):
        """Test file reading fallback to Latin-1 encoding."""
        content = "def hello():\n    return 'world'"

        # Mock UTF-8 to fail, Latin-1 to succeed
        with patch("builtins.open") as mock_file:
            mock_file.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
                mock_open(read_data=content).return_value,
            ]

            result = await self.analyzer.read_file_content("/test/file.py")

        assert result == content

    @pytest.mark.asyncio
    async def test_read_file_content_complete_failure(self):
        """Test file reading when both encodings fail."""
        with patch("builtins.open") as mock_file:
            mock_file.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
                Exception("File not found"),
            ]

            result = await self.analyzer.read_file_content("/test/file.py")

        assert result is None

    @pytest.mark.asyncio
    async def test_read_file_content_file_not_found(self):
        """Test file reading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError("No such file")):
            result = await self.analyzer.read_file_content("/nonexistent/file.py")

        assert result is None

    def test_extract_docstring_single_line_triple_quotes(self):
        """Test single-line docstring extraction with triple quotes."""
        lines = [
            "def function():",
            '    """This is a single-line docstring."""',
            "    pass",
        ]

        result = self.analyzer.extract_docstring(lines, 1)
        assert result == "This is a single-line docstring."

    def test_extract_docstring_single_line_single_quotes(self):
        """Test single-line docstring extraction with single quotes."""
        lines = [
            "def function():",
            "    '''This is a single-line docstring.'''",
            "    pass",
        ]

        result = self.analyzer.extract_docstring(lines, 1)
        assert result == "This is a single-line docstring."

    def test_extract_docstring_multi_line(self):
        """Test multi-line docstring extraction."""
        lines = [
            "def function():",
            '    """',
            "    This is a multi-line docstring.",
            "    It spans multiple lines.",
            '    """',
            "    pass",
        ]

        result = self.analyzer.extract_docstring(lines, 1)
        expected = "This is a multi-line docstring.\n    It spans multiple lines.\n    "
        assert result == expected

    def test_extract_docstring_multi_line_with_content_on_first_line(self):
        """Test multi-line docstring with content on first line."""
        lines = [
            "def function():",
            '    """This starts on the same line.',
            "    And continues here.",
            '    """',
            "    pass",
        ]

        result = self.analyzer.extract_docstring(lines, 1)
        expected = "This starts on the same line.\n    And continues here.\n    "
        assert result == expected

    def test_extract_docstring_no_docstring(self):
        """Test docstring extraction when no docstring is present."""
        lines = ["def function():", "    pass"]

        result = self.analyzer.extract_docstring(lines, 1)
        assert result is None

    def test_extract_docstring_out_of_bounds(self):
        """Test docstring extraction with invalid line number."""
        lines = ["def function():"]

        result = self.analyzer.extract_docstring(lines, 5)
        assert result is None

    def test_extract_line_range_basic(self):
        """Test basic line range extraction."""
        lines = [
            "def function():",
            "    if True:",
            "        pass",
            "    return",
            "next_function()",
        ]

        start, end = self.analyzer.extract_line_range(lines, 0)
        assert start == 0
        assert end == 3  # Should include the 'return' line

    def test_extract_line_range_with_end_markers(self):
        """Test line range extraction with end markers."""
        lines = ["def function():", "    code_here", "    # END", "    more_code"]

        start, end = self.analyzer.extract_line_range(lines, 0, ["# END"])
        assert start == 0
        assert end == 1  # Should stop before the END marker

    def test_extract_line_range_same_indentation_break(self):
        """Test line range extraction stops at same indentation level."""
        lines = [
            "if condition:",
            "    nested_code",
            "    more_nested",
            "same_level_code",
        ]

        start, end = self.analyzer.extract_line_range(lines, 0)
        assert start == 0
        assert end == 2  # Should stop before same_level_code

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        test_string = "Hello\x00World\x01Test"
        result = self.analyzer.sanitize_string(test_string)
        assert result == "HelloWorldTest"

    def test_sanitize_string_preserve_whitespace(self):
        """Test string sanitization preserves normal whitespace."""
        test_string = "Hello\tWorld\nTest\r\nEnd"
        result = self.analyzer.sanitize_string(test_string)
        assert result == "Hello\tWorld\nTest\r\nEnd"

    def test_sanitize_string_empty_input(self):
        """Test string sanitization with empty input."""
        result = self.analyzer.sanitize_string("")
        assert result == ""

    def test_sanitize_string_none_input(self):
        """Test string sanitization with None input."""
        result = self.analyzer.sanitize_string(None)
        assert result == ""

    def test_sanitize_string_only_printable(self):
        """Test string sanitization with only printable characters."""
        test_string = "Hello World 123!@#"
        result = self.analyzer.sanitize_string(test_string)
        assert result == test_string

    @pytest.mark.asyncio
    async def test_analyze_file_abstract_method(self):
        """Test that analyze_file is abstract and implemented in subclass."""
        result = await self.analyzer.analyze_file("/test/file.py", "/test")
        assert result == {"file_path": "/test/file.py"}
