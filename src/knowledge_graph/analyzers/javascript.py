"""
JavaScript/TypeScript code analyzer.

Analyzes JS/TS files to extract classes, functions, imports, and exports.
"""

import logging
import re
from pathlib import Path
from typing import Any

from src.core.exceptions import AnalysisError, ParsingError

from .base import CodeAnalyzer

logger = logging.getLogger(__name__)


class JavaScriptAnalyzer(CodeAnalyzer):
    """Analyzer for JavaScript and TypeScript files."""

    def __init__(self) -> None:
        """Initialize the JavaScript/TypeScript analyzer."""
        super().__init__()
        self.supported_extensions = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]

        # Regex patterns for JavaScript/TypeScript constructs
        self.patterns = {
            # ES6 Classes
            "class": re.compile(
                r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?",
            ),
            # Methods inside classes
            "method": re.compile(
                r"(?:(?:public|private|protected|static|async|get|set)\s+)*(\w+)\s*\([^)]*\)\s*(?::\s*[\w<>\[\]|]+)?\s*\{",
            ),
            # Functions (regular, async, generator)
            "function": re.compile(
                r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+(\w+)\s*\([^)]*\)",
            ),
            # Arrow functions assigned to variables
            "arrow_function": re.compile(
                r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>",
            ),
            # ES6 imports - improved to handle mixed imports
            "import": re.compile(
                r"import\s+(?:type\s+)?(?:(\*\s+as\s+\w+)|(\w+)|(\{[^}]+\}))\s+from\s+['\"]([^'\"]+)['\"]",
            ),
            # Mixed imports: import Default, { named } from 'module'
            "mixed_import": re.compile(
                r"import\s+(?:type\s+)?(\w+)\s*,\s*(\{[^}]+\})\s+from\s+['\"]([^'\"]+)['\"]",
            ),
            # Dynamic imports
            "dynamic_import": re.compile(
                r"(?:await\s+)?import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            ),
            # CommonJS require
            "require": re.compile(
                r"(?:const|let|var)\s+(?:(\w+)|\{([^}]+)\})\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            ),
            # ES6 exports
            "export": re.compile(
                r"export\s+(?:default\s+)?(?:(class|function|const|let|var|interface|type|enum)\s+)?(\w+)?",
            ),
            # Export from
            "export_from": re.compile(
                r"export\s+(?:(\*)|(\{[^}]+\}))\s+from\s+['\"]([^'\"]+)['\"]",
            ),
            # module.exports
            "module_exports": re.compile(
                r"module\.exports\s*=\s*(?:(\w+)|\{([^}]+)\})",
            ),
            # TypeScript interfaces
            "interface": re.compile(
                r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*\{",
            ),
            # TypeScript types
            "type": re.compile(r"(?:export\s+)?type\s+(\w+)\s*=\s*"),
            # TypeScript enums
            "enum": re.compile(r"(?:export\s+)?enum\s+(\w+)\s*\{"),
            # Variables and constants
            "variable": re.compile(
                r"(?:export\s+)?(?:const|let|var)\s+(?:(\w+)|\{([^}]+)\}|\[([^\]]+)\])\s*(?::\s*[\w<>\[\]|]+)?\s*=",
            ),
            # React functional components
            "react_component": re.compile(
                r"(?:export\s+)?(?:default\s+)?(?:const|function)\s+(\w+)\s*(?::\s*(?:React\.)?FC)?\s*=?\s*(?:\([^)]*\))?\s*(?:=>\s*)?(?:\(|\{)",
            ),
            # JSDoc comments
            "jsdoc": re.compile(r"/\*\*((?:[^*]|\*(?!/))*)\*/"),
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
        Analyze a JavaScript/TypeScript file.

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

            # Remove comments to avoid false matches (but preserve JSDoc)
            jsdoc_comments = self._extract_jsdoc(content)
            clean_content = self._remove_comments(content)

            # Extract various constructs
            classes = self._extract_classes(clean_content)
            functions = self._extract_functions(clean_content)
            imports = self._extract_imports(clean_content)
            exports = self._extract_exports(clean_content)
            interfaces = self._extract_interfaces(clean_content)
            types = self._extract_types(clean_content)
            variables = self._extract_variables(clean_content)

            # Add JSDoc to relevant items
            self._attach_jsdoc(classes, jsdoc_comments)
            self._attach_jsdoc(functions, jsdoc_comments)

            # Extract dependencies from imports
            dependencies = self._extract_dependencies(imports)

            return {
                "file_path": file_path,
                "module_name": self.get_module_name(file_path, repo_path),
                "language": self._detect_language(file_path),
                "imports": imports,
                "classes": classes,
                "functions": functions,
                "interfaces": interfaces,
                "types": types,
                "variables": variables,
                "exports": exports,
                "dependencies": dependencies,
            }

        except (ParsingError, AnalysisError) as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            return self._empty_result(file_path, repo_path)
        except Exception as e:
            logger.exception(f"Unexpected error analyzing {file_path}: {e}")
            return self._empty_result(file_path, repo_path)

    def _empty_result(self, file_path: str, repo_path: str) -> dict[str, Any]:
        """Return empty analysis result."""
        return {
            "file_path": file_path,
            "module_name": self.get_module_name(file_path, repo_path),
            "language": self._detect_language(file_path),
            "imports": [],
            "classes": [],
            "functions": [],
            "interfaces": [],
            "types": [],
            "variables": [],
            "exports": [],
            "dependencies": [],
        }

    def _detect_language(self, file_path: str) -> str:
        """Detect if file is JavaScript or TypeScript."""
        ext = Path(file_path).suffix.lower()
        if ext in [".ts", ".tsx"]:
            return "TypeScript"
        return "JavaScript"

    def _remove_comments(self, content: str) -> str:
        """Remove comments from code while preserving JSDoc."""
        # Remove single-line comments
        content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)

        # Remove multi-line comments (but not JSDoc)
        return re.sub(r"/\*(?!\*)[^*]*\*+(?:[^/*][^*]*\*+)*/", "", content)


    def _extract_jsdoc(self, content: str) -> list[dict[str, Any]]:
        """Extract JSDoc comments."""
        jsdocs = []
        for match in self.patterns["jsdoc"].finditer(content):
            jsdocs.append(
                {
                    "content": match.group(1).strip(),
                    "position": match.start(),
                },
            )
        return jsdocs

    def _extract_classes(self, content: str) -> list[dict[str, Any]]:
        """Extract class definitions."""
        classes = []
        content.split("\n")

        for match in self.patterns["class"].finditer(content):
            class_name = match.group(1)
            extends = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            # Find class body and extract methods
            class_start = match.end()
            class_body = self._extract_block(content[class_start:])
            methods = self._extract_class_methods(class_body)

            classes.append(
                {
                    "name": class_name,
                    "extends": extends,
                    "methods": methods,
                    "line": line_num,
                    "type": "class",
                },
            )

        return classes

    def _extract_class_methods(self, class_body: str) -> list[dict[str, str]]:
        """Extract methods from a class body."""
        methods = []

        # Match methods including constructor
        method_pattern = re.compile(
            r"(?:(?:public|private|protected|static|async|get|set)\s+)*"
            r"(?:constructor|(\w+))\s*\([^)]*\)",
        )

        for match in method_pattern.finditer(class_body):
            method_name = match.group(1) if match.group(1) else "constructor"
            methods.append(
                {
                    "name": method_name,
                    "type": "method",
                },
            )

        return methods

    def _extract_functions(self, content: str) -> list[dict[str, Any]]:
        """Extract function definitions."""
        functions = []

        # Regular functions
        for match in self.patterns["function"].finditer(content):
            functions.append(
                {
                    "name": match.group(1),
                    "type": "function",
                    "async": "async" in match.group(0),
                    "generator": "*" in match.group(0),
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        # Arrow functions
        for match in self.patterns["arrow_function"].finditer(content):
            functions.append(
                {
                    "name": match.group(1),
                    "type": "arrow_function",
                    "async": "async" in match.group(0),
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        # React components (that look like functions)
        for match in self.patterns["react_component"].finditer(content):
            name = match.group(1)
            # Check if it's likely a React component (PascalCase)
            if name and name[0].isupper():
                functions.append(
                    {
                        "name": name,
                        "type": "react_component",
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        return functions

    def _extract_imports(self, content: str) -> list[dict[str, Any]]:
        """Extract import statements."""
        imports = []

        # ES6 imports (simple cases)
        for match in self.patterns["import"].finditer(content):
            source = match.group(4)
            imported = []

            if match.group(1):  # import * as name
                imported.append(match.group(1).split()[-1])
            elif match.group(2):  # import defaultName
                imported.append(match.group(2))
            elif match.group(3):  # import { named }
                imported.extend(self._parse_named_imports(match.group(3)))

            imports.append(
                {
                    "type": "es6",
                    "source": source,
                    "imported": imported,
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        # Mixed imports: import Default, { named } from 'module'
        for match in self.patterns["mixed_import"].finditer(content):
            source = match.group(3)
            imported = []

            # Add default import
            imported.append(match.group(1))

            # Add named imports with proper alias handling
            imported.extend(self._parse_named_imports(match.group(2)))

            imports.append(
                {
                    "type": "es6",
                    "source": source,
                    "imported": imported,
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        # CommonJS require
        for match in self.patterns["require"].finditer(content):
            source = match.group(3)
            imported = []

            if match.group(1):  # const name = require()
                imported.append(match.group(1))
            elif match.group(2):  # const { destructured } = require()
                # Use the same parsing logic for CommonJS destructuring
                imported.extend(self._parse_named_imports(match.group(2)))

            imports.append(
                {
                    "type": "commonjs",
                    "source": source,
                    "imported": imported,
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        # Dynamic imports
        for match in self.patterns["dynamic_import"].finditer(content):
            imports.append(
                {
                    "type": "dynamic",
                    "source": match.group(1),
                    "imported": [],
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        return imports

    def _extract_exports(self, content: str) -> list[dict[str, Any]]:
        """Extract export statements."""
        exports = []

        # ES6 exports
        for match in self.patterns["export"].finditer(content):
            export_type = match.group(1)
            name = match.group(2)

            if name:
                exports.append(
                    {
                        "type": "named",
                        "name": name,
                        "kind": export_type or "value",
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )
            elif "default" in match.group(0):
                exports.append(
                    {
                        "type": "default",
                        "name": "default",
                        "kind": export_type or "value",
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        # Export from
        for match in self.patterns["export_from"].finditer(content):
            source = match.group(3)
            if match.group(1):  # export *
                exports.append(
                    {
                        "type": "all",
                        "source": source,
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )
            elif match.group(2):  # export { named }
                names = match.group(2).strip("{}").split(",")
                for name in names:
                    exports.append(
                        {
                            "type": "named",
                            "name": name.strip().split()[-1],
                            "source": source,
                            "line": content[: match.start()].count("\n") + 1,
                        },
                    )

        # module.exports
        for match in self.patterns["module_exports"].finditer(content):
            if match.group(1):  # module.exports = name
                exports.append(
                    {
                        "type": "commonjs",
                        "name": match.group(1),
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )
            elif match.group(2):  # module.exports = { names }
                names = match.group(2).split(",")
                for name in names:
                    exports.append(
                        {
                            "type": "commonjs",
                            "name": name.strip(),
                            "line": content[: match.start()].count("\n") + 1,
                        },
                    )

        return exports

    def _extract_interfaces(self, content: str) -> list[dict[str, Any]]:
        """Extract TypeScript interfaces."""
        interfaces = []

        for match in self.patterns["interface"].finditer(content):
            interfaces.append(
                {
                    "name": match.group(1),
                    "extends": match.group(2).strip() if match.group(2) else None,
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        return interfaces

    def _extract_types(self, content: str) -> list[dict[str, Any]]:
        """Extract TypeScript type definitions."""
        types = []

        for match in self.patterns["type"].finditer(content):
            types.append(
                {
                    "name": match.group(1),
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        for match in self.patterns["enum"].finditer(content):
            types.append(
                {
                    "name": match.group(1),
                    "kind": "enum",
                    "line": content[: match.start()].count("\n") + 1,
                },
            )

        return types

    def _extract_variables(self, content: str) -> list[dict[str, Any]]:
        """Extract variable declarations."""
        variables = []
        seen = set()

        for match in self.patterns["variable"].finditer(content):
            # Get variable name(s)
            if match.group(1):  # Simple variable
                name = match.group(1)
            elif match.group(2):  # Object destructuring
                names = [n.strip() for n in match.group(2).split(",")]
                name = names[0] if names else None
            elif match.group(3):  # Array destructuring
                names = [n.strip() for n in match.group(3).split(",")]
                name = names[0] if names else None
            else:
                continue

            # Avoid duplicates and function names
            if name and name not in seen:
                seen.add(name)
                variables.append(
                    {
                        "name": name,
                        "kind": "const"
                        if "const" in match.group(0)
                        else "let"
                        if "let" in match.group(0)
                        else "var",
                        "line": content[: match.start()].count("\n") + 1,
                    },
                )

        return variables

    def _extract_dependencies(self, imports: list[dict[str, Any]]) -> list[str]:
        """Extract unique dependencies from imports."""
        deps = set()

        for imp in imports:
            source = imp.get("source", "")
            # Filter out relative imports and Node built-ins
            if source and not source.startswith(".") and not source.startswith("/"):
                # Extract package name (handle scoped packages)
                if source.startswith("@"):
                    parts = source.split("/")[:2]
                    deps.add("/".join(parts))
                else:
                    deps.add(source.split("/")[0])

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

        while pos < len(content) and count > 0:
            if content[pos] == "{":
                count += 1
            elif content[pos] == "}":
                count -= 1
            pos += 1

        return content[start:pos]

    def _parse_named_imports(self, import_string: str) -> list[str]:
        """
        Parse named imports handling 'as' aliases properly.

        Args:
            import_string: String containing named imports like "Component as Comp, useState, useEffect as Effect"

        Returns:
            List of imported names (using aliases where present)
        """
        imports: list[str] = []
        if not import_string:
            return imports

        # Clean up the string - remove braces and extra whitespace
        clean_string = import_string.strip("{}")

        # Split by comma and process each import
        for item in clean_string.split(","):
            item = item.strip()
            if not item:
                continue

            # Check if this item has an alias (contains 'as')
            if " as " in item:
                # Split by 'as' and take the alias (second part)
                parts = item.split(" as ")
                if len(parts) == 2:
                    alias = parts[1].strip()
                    imports.append(alias)
                else:
                    # Fallback if parsing fails
                    imports.append(item.split()[-1])
            else:
                # No alias, use the original name
                imports.append(item.strip())

        return imports

    def _attach_jsdoc(
        self, items: list[dict[str, Any]], jsdocs: list[dict[str, Any]],
    ) -> None:
        """Attach JSDoc comments to code items."""
        for item in items:
            if "line" in item:
                # Find the closest JSDoc before this item
                for jsdoc in jsdocs:
                    # Simple proximity check - JSDoc should be right before the item
                    if jsdoc["position"] < item.get("position", float("inf")):
                        item["doc"] = jsdoc["content"]
                        break
