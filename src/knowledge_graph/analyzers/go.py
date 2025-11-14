"""
Go code analyzer.

Analyzes Go files to extract structs, functions, interfaces, and imports.
"""

import logging
import re
from pathlib import Path
from typing import Any

from .base import CodeAnalyzer

logger = logging.getLogger(__name__)


class GoAnalyzer(CodeAnalyzer):
    """Analyzer for Go source files."""

    def __init__(self) -> None:
        """Initialize the Go analyzer."""
        super().__init__()
        self.supported_extensions = [".go"]

        # Regex patterns for Go constructs
        self.patterns = {
            # Package declaration
            "package": re.compile(r"^package\s+(\w+)", re.MULTILINE),
            # Import statements
            "import": re.compile(r"import\s+(?:\(\s*((?:[^)]+))\s*\)|\"([^\"]+)\")"),
            # Struct definitions
            "struct": re.compile(r"type\s+(\w+)\s+struct\s*\{"),
            # Interface definitions
            "interface": re.compile(r"type\s+(\w+)\s+interface\s*\{"),
            # Type aliases
            "type_alias": re.compile(r"type\s+(\w+)\s+(?!struct|interface)(\w+)"),
            # Functions and methods
            "function": re.compile(
                r"func\s+(?:\((?:[^)]*)\)\s+)?(\w+)\s*\([^)]*\)(?:\s*(?:\([^)]*\)|[\w\[\]\*]+))?",
            ),
            # Method receivers
            "method": re.compile(
                r"func\s+\((\w+)\s+([\*]?\w+(?:\[.*?\])?)\)\s+(\w+)\s*\([^)]*\)",
            ),
            # Constants
            "const": re.compile(
                r"const\s+(?:\(\s*((?:[^)]+))\s*\)|(\w+)(?:\s+[\w\[\]\*]+)?\s*=)",
            ),
            # Variables
            "var": re.compile(
                r"var\s+(?:\(\s*((?:[^)]+)\s*\)|(\w+)(?:\s+[\w\[\]\*]+)?(?:\s*=)?))",
            ),
            # Go doc comments
            "godoc": re.compile(r"^//\s*(\w+)\s+(.*)$", re.MULTILINE),
            # Struct fields
            "field": re.compile(
                r"^\s*(\w+)\s+((?:[\*\[\]]*\w+)+)(?:\s+`[^`]+`)?", re.MULTILINE,
            ),
            # Interface methods
            "interface_method": re.compile(
                r"^\s*(\w+)\s*\([^)]*\)(?:\s*(?:\([^)]*\)|[\w\[\]\*]+))?", re.MULTILINE,
            ),
        }

    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can handle the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    async def analyze_file(
        self,
        file_path: str,
        repo_path: str,
        content: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a Go file.

        Args:
            file_path: Path to the file
            repo_path: Root path of the repository
            content: Optional file content

        Returns:
            Extracted code structure
        """
        try:
            # Read content if not provided
            if content is None:
                content = await self.read_file_content(file_path)
                if content is None:
                    return self._empty_result(file_path, repo_path)

            # Extract package name
            package_name = self._extract_package(content)

            # Extract various constructs
            imports = self._extract_imports(content)
            structs = self._extract_structs(content)
            interfaces = self._extract_interfaces(content)
            functions = self._extract_functions(content)
            types = self._extract_types(content)
            constants = self._extract_constants(content)
            variables = self._extract_variables(content)

            # Extract dependencies from imports
            dependencies = self._extract_dependencies(imports)

            return {
                "file_path": file_path,
                "module_name": self.get_module_name(file_path, repo_path),
                "package": package_name,
                "language": "Go",
                "imports": imports,
                "structs": structs,
                "interfaces": interfaces,
                "functions": functions,
                "types": types,
                "constants": constants,
                "variables": variables,
                "exports": self._extract_exports(structs, interfaces, functions, types),
                "dependencies": dependencies,
            }

        except Exception as e:
            logger.exception(f"Error analyzing {file_path}: {e}")
            return self._empty_result(file_path, repo_path)

    def _empty_result(self, file_path: str, repo_path: str) -> dict[str, Any]:
        """Return empty analysis result."""
        return {
            "file_path": file_path,
            "module_name": self.get_module_name(file_path, repo_path),
            "package": "",
            "language": "Go",
            "imports": [],
            "structs": [],
            "interfaces": [],
            "functions": [],
            "types": [],
            "constants": [],
            "variables": [],
            "exports": [],
            "dependencies": [],
        }

    def _extract_package(self, content: str) -> str:
        """Extract package name."""
        match = self.patterns["package"].search(content)
        return match.group(1) if match else ""

    def _extract_imports(self, content: str) -> list[dict[str, Any]]:
        """Extract import statements."""
        imports = []

        for match in self.patterns["import"].finditer(content):
            if match.group(1):  # Group import
                import_lines = match.group(1).strip().split("\n")
                for line in import_lines:
                    line = line.strip()
                    if line and not line.startswith("//"):
                        # Handle aliased imports
                        parts = line.split()
                        if len(parts) == 2:
                            alias = parts[0]
                            path = parts[1].strip('"')
                            imports.append(
                                {
                                    "path": path,
                                    "alias": alias,
                                    "line": content[: match.start()].count("\n") + 1,
                                },
                            )
                        else:
                            path = line.strip('"')
                            imports.append(
                                {
                                    "path": path,
                                    "alias": None,
                                    "line": content[: match.start()].count("\n") + 1,
                                },
                            )
            else:  # Single import
                path = match.group(2)
                imports.append(
                    {
                        "path": path,
                        "alias": None,
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        return imports

    def _extract_structs(self, content: str) -> list[dict[str, Any]]:
        """Extract struct definitions."""
        structs = []

        for match in self.patterns["struct"].finditer(content):
            struct_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            # Extract struct body
            struct_start = match.end()
            struct_body = self._extract_block(content[struct_start - 1 :])

            # Extract fields
            fields = self._extract_struct_fields(struct_body)

            # Check if exported (starts with uppercase)
            exported = struct_name[0].isupper() if struct_name else False

            structs.append(
                {
                    "name": struct_name,
                    "fields": fields,
                    "exported": exported,
                    "line": line_num,
                    "type": "struct",
                },
            )

        return structs

    def _extract_struct_fields(self, struct_body: str) -> list[dict[str, Any]]:
        """Extract fields from a struct body."""
        fields: list[dict[str, Any]] = []

        for match in self.patterns["field"].finditer(struct_body):
            field_name = match.group(1)
            field_type = match.group(2)

            # Skip if it looks like a method or comment
            if field_name in ["func", "type", "var", "const", "//"]:
                continue

            fields.append(
                {
                    "name": field_name,
                    "type": field_type,
                    "exported": field_name[0].isupper() if field_name else False,
                },
            )

        return fields

    def _extract_interfaces(self, content: str) -> list[dict[str, Any]]:
        """Extract interface definitions."""
        interfaces = []

        for match in self.patterns["interface"].finditer(content):
            interface_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            # Extract interface body
            interface_start = match.end()
            interface_body = self._extract_block(content[interface_start - 1 :])

            # Extract methods
            methods = self._extract_interface_methods(interface_body)

            # Check if exported
            exported = interface_name[0].isupper() if interface_name else False

            interfaces.append(
                {
                    "name": interface_name,
                    "methods": methods,
                    "exported": exported,
                    "line": line_num,
                    "type": "interface",
                },
            )

        return interfaces

    def _extract_interface_methods(self, interface_body: str) -> list[dict[str, str]]:
        """Extract methods from an interface body."""
        methods = []

        for match in self.patterns["interface_method"].finditer(interface_body):
            method_name = match.group(1)

            # Skip empty lines and comments
            if method_name and not method_name.startswith("//"):
                methods.append(
                    {
                        "name": method_name,
                        "type": "method",
                    },
                )

        return methods

    def _extract_functions(self, content: str) -> list[dict[str, Any]]:
        """Extract function and method definitions."""
        functions = []
        seen = set()

        # Extract methods (functions with receivers)
        for match in self.patterns["method"].finditer(content):
            receiver_name = match.group(1)
            receiver_type = match.group(2)
            method_name = match.group(3)

            key = f"{receiver_type}.{method_name}"
            if key not in seen:
                seen.add(key)
                functions.append(
                    {
                        "name": method_name,
                        "receiver": receiver_type,
                        "receiver_name": receiver_name,
                        "type": "method",
                        "exported": method_name[0].isupper() if method_name else False,
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        # Extract regular functions
        for match in self.patterns["function"].finditer(content):
            func_name = match.group(1)

            # Skip if already captured as method
            if func_name and func_name not in [f["name"] for f in functions]:
                functions.append(
                    {
                        "name": func_name,
                        "type": "function",
                        "exported": func_name[0].isupper() if func_name else False,
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        return functions

    def _extract_types(self, content: str) -> list[dict[str, Any]]:
        """Extract type definitions."""
        types = []

        for match in self.patterns["type_alias"].finditer(content):
            type_name = match.group(1)
            base_type = match.group(2)

            types.append(
                {
                    "name": type_name,
                    "base": base_type,
                    "exported": type_name[0].isupper() if type_name else False,
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        return types

    def _extract_constants(self, content: str) -> list[dict[str, Any]]:
        """Extract constant definitions."""
        constants = []

        for match in self.patterns["const"].finditer(content):
            if match.group(1):  # Group const
                const_lines = match.group(1).strip().split("\n")
                for line in const_lines:
                    line = line.strip()
                    if line and not line.startswith("//"):
                        # Extract name from line
                        name_match = re.match(r"(\w+)", line)
                        if name_match:
                            const_name = name_match.group(1)
                            constants.append(
                                {
                                    "name": const_name,
                                    "exported": const_name[0].isupper(),
                                    "line": content[: match.start()].count("\n") + 1,
                                },
                            )
            elif match.group(2):  # Single const
                const_name = match.group(2)
                constants.append(
                    {
                        "name": const_name,
                        "exported": const_name[0].isupper(),
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        return constants

    def _extract_variables(self, content: str) -> list[dict[str, Any]]:
        """Extract variable definitions."""
        variables = []

        for match in self.patterns["var"].finditer(content):
            if match.group(1):  # Group var
                var_lines = match.group(1).strip().split("\n")
                for line in var_lines:
                    line = line.strip()
                    if line and not line.startswith("//"):
                        # Extract name from line
                        name_match = re.match(r"(\w+)", line)
                        if name_match:
                            var_name = name_match.group(1)
                            variables.append(
                                {
                                    "name": var_name,
                                    "exported": var_name[0].isupper(),
                                    "line": content[: match.start()].count("\n") + 1,
                                },
                            )
            elif match.group(2):  # Single var
                var_name = match.group(2)
                variables.append(
                    {
                        "name": var_name,
                        "exported": var_name[0].isupper(),
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        return variables

    def _extract_exports(
        self,
        structs: list[dict[str, Any]],
        interfaces: list[dict[str, Any]],
        functions: list[dict[str, Any]],
        types: list[dict[str, Any]],
    ) -> list[str]:
        """
        Extract exported symbols (capitalized names in Go).

        In Go, exported symbols start with an uppercase letter.
        """
        exports = []

        # Add exported structs
        for struct in structs:
            if struct.get("exported"):
                exports.append(struct["name"])

        # Add exported interfaces
        for interface in interfaces:
            if interface.get("exported"):
                exports.append(interface["name"])

        # Add exported functions
        for func in functions:
            if func.get("exported"):
                exports.append(func["name"])

        # Add exported types
        for type_def in types:
            if type_def.get("exported"):
                exports.append(type_def["name"])

        return exports

    def _extract_dependencies(self, imports: list[dict[str, Any]]) -> list[str]:
        """Extract unique dependencies from imports."""
        deps = set()

        for imp in imports:
            path = imp.get("path", "")
            if path:
                # Filter out standard library imports (no dots in path)
                if "." in path or path.startswith("github.com/"):
                    # Extract the main package name
                    parts = path.split("/")
                    if path.startswith("github.com/") and len(parts) >= 3:
                        # GitHub packages: github.com/owner/repo
                        deps.add("/".join(parts[:3]))
                    elif "." in parts[0]:
                        # Other external packages
                        deps.add(parts[0])

        return sorted(deps)

    def _extract_block(self, content: str) -> str:
        """Extract a code block starting with { and ending with matching }."""
        if not content:
            return ""

        # Find the opening brace
        start = content.find("{")
        if start == -1:
            return ""

        # Count braces to find matching closing brace
        count = 1
        pos = start + 1
        in_string = False
        escape_next = False

        while pos < len(content) and count > 0:
            char = content[pos]

            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1

            pos += 1

        return content[start:pos]
