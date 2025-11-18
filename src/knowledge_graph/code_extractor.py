"""
Code extraction from Neo4j knowledge graph for Qdrant indexing.

This module extracts structured code examples from Neo4j and prepares them
for embedding generation and storage in Qdrant vector database.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.core.exceptions import QueryError

logger = logging.getLogger(__name__)


@dataclass
class CodeExample:
    """Structured code example for embedding and indexing."""

    repository_name: str
    file_path: str
    module_name: str
    code_type: str  # 'class', 'method', 'function'
    name: str
    full_name: str
    code_text: str  # Generated code representation
    parameters: list[str] | None = None
    return_type: str | None = None
    class_name: str | None = None  # For methods
    method_count: int | None = None  # For classes
    language: str = "python"
    validation_status: str = "extracted"  # extracted, validated, verified

    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata dictionary for Qdrant storage."""
        metadata: dict[str, Any] = {
            "repository_name": self.repository_name,
            "file_path": self.file_path,
            "module_name": self.module_name,
            "code_type": self.code_type,
            "name": self.name,
            "full_name": self.full_name,
            "language": self.language,
            "validation_status": self.validation_status,
        }

        if self.parameters:
            metadata["parameters"] = self.parameters
        if self.return_type:
            metadata["return_type"] = self.return_type
        if self.class_name:
            metadata["class_name"] = self.class_name
        if self.method_count is not None:
            metadata["method_count"] = self.method_count

        return metadata

    def generate_embedding_text(self) -> str:
        """Generate text representation for embedding generation."""
        if self.code_type == "class":
            text = f"Python class {self.name} in {self.module_name}\n"
            text += f"Full name: {self.full_name}\n"
            if self.method_count:
                text += f"Contains {self.method_count} methods\n"
            text += f"Code: {self.code_text}"

        elif self.code_type == "method":
            text = f"Python method {self.name}"
            if self.class_name:
                text += f" in class {self.class_name}"
            text += f" from {self.module_name}\n"
            if self.parameters:
                text += f"Parameters: {', '.join(self.parameters)}\n"
            if self.return_type:
                text += f"Returns: {self.return_type}\n"
            text += f"Code: {self.code_text}"

        elif self.code_type == "function":
            text = f"Python function {self.name} from {self.module_name}\n"
            if self.parameters:
                text += f"Parameters: {', '.join(self.parameters)}\n"
            if self.return_type:
                text += f"Returns: {self.return_type}\n"
            text += f"Code: {self.code_text}"

        else:
            text = f"Python {self.code_type} {self.name}: {self.code_text}"

        return text


# Constants for validation
SUPPORTED_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Go"}
SUPPORTED_CODE_TYPES = {
    "class", "function", "method", "interface", "struct",
    "type", "variable", "constant",
}
VALIDATION_STATUSES = {"extracted", "validated", "verified"}


@dataclass
class UniversalCodeExample:
    """
    Enhanced code example supporting multiple programming languages.

    Supports Python, JavaScript, TypeScript, and Go with language-specific
    metadata and multi-context embedding generation.
    """

    # Core fields (required)
    repository_name: str
    file_path: str
    module_name: str
    language: str  # Python, JavaScript, TypeScript, Go
    # Code type: class, function, method, interface, struct, type, variable, constant
    code_type: str
    name: str
    full_name: str

    # Language-agnostic properties
    signature: str | None = None
    documentation: str | None = None
    line_number: int | None = None
    visibility: str | None = None  # public, private, protected
    is_async: bool = False
    is_static: bool = False

    # Relationship fields
    parent_name: str | None = None  # For methods in classes, nested types
    # Methods in class, fields in struct
    child_elements: list[str] = field(default_factory=list)

    # Language-specific metadata as JSON
    language_specific: dict[str, Any] = field(default_factory=dict)

    # Enhanced embedding context support
    embedding_context: dict[str, str] = field(default_factory=dict)

    # Validation metadata
    validation_status: str = "extracted"  # extracted, validated, verified
    confidence_score: float | None = None

    def __post_init__(self) -> None:
        """Initialize default values and validate inputs."""
        self.validate()

    def validate(self) -> None:
        """Validate the code example data."""
        if self.language not in SUPPORTED_LANGUAGES:
            msg = (
                f"Unsupported language: {self.language}. "
                f"Supported: {SUPPORTED_LANGUAGES}"
            )
            raise ValueError(msg)

        if self.code_type not in SUPPORTED_CODE_TYPES:
            msg = (
                f"Unsupported code_type: {self.code_type}. "
                f"Supported: {SUPPORTED_CODE_TYPES}"
            )
            raise ValueError(msg)

        if self.validation_status not in VALIDATION_STATUSES:
            msg = (
                f"Invalid validation_status: {self.validation_status}. "
                f"Supported: {VALIDATION_STATUSES}"
            )
            raise ValueError(msg)

        if (
            self.confidence_score is not None
            and not (0.0 <= self.confidence_score <= 1.0)
        ):
            msg = (
                f"confidence_score must be between 0.0 and 1.0, "
                f"got: {self.confidence_score}"
            )
            raise ValueError(msg)

    def to_metadata(self) -> dict[str, Any]:
        """Convert to comprehensive metadata dictionary for Qdrant storage."""
        metadata: dict[str, Any] = {
            "repository_name": self.repository_name,
            "file_path": self.file_path,
            "module_name": self.module_name,
            "language": self.language,
            "code_type": self.code_type,
            "name": self.name,
            "full_name": self.full_name,
            "validation_status": self.validation_status,
        }

        # Add optional fields
        optional_fields = [
            "signature", "documentation", "line_number", "visibility",
            "is_async", "is_static", "parent_name", "confidence_score",
        ]
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                metadata[field_name] = value

        if self.child_elements:
            metadata["child_elements"] = self.child_elements

        if self.language_specific:
            metadata["language_specific"] = self.language_specific

        return metadata

    def generate_embedding_contexts(self) -> dict[str, str]:
        """Generate multiple embedding contexts for different use cases."""
        contexts = {}

        # Signature context - just the declaration/signature
        contexts["signature"] = self._generate_signature_context()

        # Semantic context - natural language description
        contexts["semantic"] = self._generate_semantic_context()

        # Usage context - how it's typically used
        contexts["usage"] = self._generate_usage_context()

        # Full context - complete representation with documentation
        contexts["full"] = self._generate_full_context()

        # Store contexts for future use
        self.embedding_context = contexts
        return contexts

    def generate_embedding_text(self, context_type: str = "full") -> str:
        """
        Generate text representation for embedding generation.

        Args:
            context_type: Type of context ("signature", "semantic", "usage", "full")
        """
        if not self.embedding_context:
            self.generate_embedding_contexts()

        return self.embedding_context.get(
            context_type, self.embedding_context.get("full", ""),
        )

    def _generate_signature_context(self) -> str:
        """Generate signature-only context."""
        if self.signature:
            return f"{self.language} {self.code_type} {self.name}: {self.signature}"

        # Generate basic signature based on language and type
        if self.language == "Python":
            return self._generate_python_signature()
        if self.language in ["JavaScript", "TypeScript"]:
            return self._generate_js_ts_signature()
        if self.language == "Go":
            return self._generate_go_signature()
        return f"{self.language} {self.code_type} {self.name}"

    def _generate_semantic_context(self) -> str:
        """Generate semantic description context."""
        base = f"{self.language} {self.code_type} '{self.name}'"

        if self.parent_name:
            base += f" in {self.parent_name}"

        base += f" from {self.module_name or 'module'}"

        if self.documentation:
            base += f"\nDescription: {self.documentation}"

        if self.child_elements:
            # Limit to first 5
            base += f"\nContains: {', '.join(self.child_elements[:5])}"

        return base

    def _generate_usage_context(self) -> str:
        """Generate usage example context."""
        if self.language == "Python":
            return self._generate_python_usage()
        if self.language in ["JavaScript", "TypeScript"]:
            return self._generate_js_ts_usage()
        if self.language == "Go":
            return self._generate_go_usage()
        return f"Usage example for {self.language} {self.code_type} {self.name}"

    def _generate_full_context(self) -> str:
        """Generate complete context with all information."""
        parts = []

        # Header
        header = f"{self.language} {self.code_type} {self.name}"
        if self.parent_name:
            header += f" in {self.parent_name}"
        header += f" from {self.module_name}"
        parts.append(header)

        # Signature
        if self.signature:
            parts.append(f"Signature: {self.signature}")

        # Documentation
        if self.documentation:
            parts.append(f"Documentation: {self.documentation}")

        # Properties
        properties = []
        if self.visibility:
            properties.append(f"visibility: {self.visibility}")
        if self.is_async:
            properties.append("async")
        if self.is_static:
            properties.append("static")
        if properties:
            parts.append(f"Properties: {', '.join(properties)}")

        # Child elements
        if self.child_elements:
            parts.append(f"Contains: {', '.join(self.child_elements)}")

        # Language-specific information
        if self.language_specific:
            lang_info = []
            for key, value in self.language_specific.items():
                if isinstance(value, list):
                    lang_info.append(f"{key}: {', '.join(value)}")
                else:
                    lang_info.append(f"{key}: {value}")
            if lang_info:
                parts.append(f"Language-specific: {'; '.join(lang_info)}")

        return "\n".join(parts)

    def _generate_python_signature(self) -> str:
        """Generate Python-style signature."""
        if self.code_type == "class":
            return f"class {self.name}:"
        if self.code_type in ["function", "method"]:
            params = self.language_specific.get("parameters", [])
            return_type = self.language_specific.get("return_type", "")
            param_str = ", ".join(params) if params else ""
            return_str = f" -> {return_type}" if return_type else ""
            return f"def {self.name}({param_str}){return_str}:"
        return f"{self.name}: {self.language_specific.get('type_hint', 'Any')}"

    def _generate_js_ts_signature(self) -> str:
        """Generate JavaScript/TypeScript-style signature."""
        if self.code_type == "class":
            return f"class {self.name} {{"
        if self.code_type == "interface" and self.language == "TypeScript":
            return f"interface {self.name} {{"
        if self.code_type == "type" and self.language == "TypeScript":
            type_def = self.language_specific.get('type_definition', 'any')
            return f"type {self.name} = {type_def}"
        if self.code_type in ["function", "method"]:
            params = self.language_specific.get("parameters", [])
            return_type = self.language_specific.get("return_type", "")
            param_str = ", ".join(params) if params else ""
            is_typescript = self.language == "TypeScript"
            return_str = f": {return_type}" if return_type and is_typescript else ""
            prefix = "async " if self.is_async else ""
            return f"{prefix}function {self.name}({param_str}){return_str} {{"
        return f"const {self.name} = {self.language_specific.get('value', 'undefined')}"

    def _generate_go_signature(self) -> str:
        """Generate Go-style signature."""
        if self.code_type == "struct":
            return f"type {self.name} struct {{"
        if self.code_type == "interface":
            return f"type {self.name} interface {{"
        if self.code_type == "function":
            params = self.language_specific.get("parameters", [])
            return_type = self.language_specific.get("return_type", "")
            param_str = ", ".join(params) if params else ""
            return_str = f" {return_type}" if return_type else ""
            return f"func {self.name}({param_str}){return_str} {{"
        if self.code_type == "method":
            receiver = self.language_specific.get("receiver", "")
            params = self.language_specific.get("parameters", [])
            return_type = self.language_specific.get("return_type", "")
            receiver_str = f"({receiver}) " if receiver else ""
            param_str = ", ".join(params) if params else ""
            return_str = f" {return_type}" if return_type else ""
            return f"func {receiver_str}{self.name}({param_str}){return_str} {{"
        return f"var {self.name} {self.language_specific.get('type', 'interface{}')}"

    def _generate_python_usage(self) -> str:
        """Generate Python usage example."""
        if self.code_type == "class":
            return (
                f"# Usage:\ninstance = {self.name}()\n"
                f"# Access methods: instance.method_name()"
            )
        if self.code_type == "function":
            params = self.language_specific.get("parameters", [])
            if params:
                example_params = ", ".join(["arg" + str(i) for i in range(len(params))])
                return f"# Usage:\nresult = {self.name}({example_params})"
            return f"# Usage:\nresult = {self.name}()"
        if self.code_type == "method":
            return f"# Usage:\ninstance.{self.name}(args)"
        return f"# Usage: {self.name}"

    def _generate_js_ts_usage(self) -> str:
        """Generate JavaScript/TypeScript usage example."""
        if self.code_type == "class":
            return (
                f"// Usage:\nconst instance = new {self.name}();\n"
                f"// Access methods: instance.methodName()"
            )
        if self.code_type == "interface" and self.language == "TypeScript":
            return (
                f"// Usage:\nconst obj: {self.name} = "
                f"{{ /* implement interface */ }};"
            )
        if self.code_type == "function":
            return f"// Usage:\nconst result = {self.name}(args);"
        return f"// Usage: {self.name}"

    def _generate_go_usage(self) -> str:
        """Generate Go usage example."""
        if self.code_type == "struct":
            return (
                f"// Usage:\ninstance := {self.name}{{}}\n"
                f"// Access fields: instance.FieldName"
            )
        if self.code_type == "function":
            return f"// Usage:\nresult := {self.name}(args)"
        if self.code_type == "method":
            return f"// Usage:\ninstance.{self.name}(args)"
        return f"// Usage: {self.name}"

    def to_code_example(self) -> "CodeExample":
        """
        Convert to legacy CodeExample for backward compatibility.

        Returns:
            CodeExample instance with compatible fields
        """
        # Extract Python-specific fields from language_specific
        parameters = None
        return_type = None
        class_name = None
        method_count = None

        if self.language == "Python" and self.language_specific:
            parameters = self.language_specific.get("parameters")
            return_type = self.language_specific.get("return_type")
            class_name = self.parent_name
            method_count = len(self.child_elements) if self.child_elements else None

        # Generate code text from signature if available
        code_text = self.signature or self.generate_embedding_text("signature")

        return CodeExample(
            repository_name=self.repository_name,
            file_path=self.file_path,
            module_name=self.module_name,
            code_type=self.code_type,
            name=self.name,
            full_name=self.full_name,
            code_text=code_text,
            parameters=parameters,
            return_type=return_type,
            class_name=class_name,
            method_count=method_count,
            language=self.language,
            validation_status=self.validation_status,
        )


class Neo4jCodeExtractor:
    """Extracts code examples from Neo4j knowledge graph."""

    def __init__(
        self,
        neo4j_session: Any,
        use_universal: bool = False,
        language: str = "Python",
    ):
        """
        Initialize with Neo4j session.

        Args:
            neo4j_session: Neo4j database session
            use_universal: If True, returns UniversalCodeExample instances
            language: Target language for extraction (default: Python)
        """
        self.session = neo4j_session
        self.use_universal = use_universal
        self.language = language

    async def extract_repository_code(
        self, repo_name: str
    ) -> list[CodeExample | UniversalCodeExample]:
        """
        Extract all code examples from a repository in Neo4j.

        Args:
            repo_name: Name of the repository to extract from

        Returns:
            List of CodeExample or UniversalCodeExample objects ready for embedding
        """
        logger.info("Extracting code from repository: %s", repo_name)

        # Check if repository exists
        if not await self._repository_exists(repo_name):
            msg = f"Repository '{repo_name}' not found in knowledge graph"
            raise ValueError(msg)

        code_examples = []

        # Extract classes with their methods
        classes = await self._extract_classes(repo_name)
        code_examples.extend(classes)

        # Extract standalone functions
        functions = await self._extract_functions(repo_name)
        code_examples.extend(functions)

        logger.info("Extracted %s code examples from %s", len(code_examples), repo_name)
        return code_examples

    async def extract_repository_code_universal(
        self, repo_name: str
    ) -> list[CodeExample | UniversalCodeExample]:
        """
        Extract repository code using UniversalCodeExample format.

        Args:
            repo_name: Name of the repository to extract from

        Returns:
            List of UniversalCodeExample objects
        """
        original_setting = self.use_universal
        self.use_universal = True
        try:
            return await self.extract_repository_code(repo_name)
        finally:
            self.use_universal = original_setting

    async def _repository_exists(self, repo_name: str) -> bool:
        """Check if repository exists in Neo4j."""
        query = """
        MATCH (r:Repository {name: $repo_name})
        RETURN r.name as name
        LIMIT 1
        """
        result = await self.session.run(query, repo_name=repo_name)
        record = await result.single()
        return record is not None

    async def _extract_classes(
        self, repo_name: str
    ) -> list[CodeExample | UniversalCodeExample]:
        """Extract class definitions with their methods."""
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
        -[:DEFINES]->(c:Class)
        OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
        RETURN
            c.name as class_name,
            c.full_name as class_full_name,
            f.path as file_path,
            f.module_name as module_name,
            count(m) as method_count,
            collect({
                name: m.name,
                params_list: m.params_list,
                params_detailed: m.params_detailed,
                return_type: m.return_type,
                args: m.args
            }) as methods
        ORDER BY c.name
        """

        result = await self.session.run(query, repo_name=repo_name)
        classes: list[CodeExample | UniversalCodeExample] = []

        async for record in result:
            class_name = record["class_name"]
            class_full_name = record["class_full_name"]
            file_path = record["file_path"]
            module_name = record["module_name"] or ""
            method_count = record["method_count"]
            methods = record["methods"]

            # Create class example
            class_example: CodeExample | UniversalCodeExample
            if self.use_universal:
                class_example = self._create_universal_class_example(
                    repo_name, file_path, module_name, class_name,
                    class_full_name, methods, method_count,
                )
            else:
                # Generate class code representation (legacy)
                class_code = self._generate_class_code(class_name, methods)
                class_example = CodeExample(
                    repository_name=repo_name,
                    file_path=file_path,
                    module_name=module_name,
                    code_type="class",
                    name=class_name,
                    full_name=class_full_name,
                    code_text=class_code,
                    method_count=method_count,
                    language=self.language,
                )

            classes.append(class_example)

            # Create individual method examples for important methods
            for method in methods:
                # Public methods
                if method["name"] and not method["name"].startswith("_"):
                    method_example: CodeExample | UniversalCodeExample
                    if self.use_universal:
                        method_example = self._create_universal_method_example(
                            repo_name, file_path, module_name, method,
                            class_name, class_full_name,
                        )
                    else:
                        method_code = self._generate_method_code(method)
                        method_example = CodeExample(
                            repository_name=repo_name,
                            file_path=file_path,
                            module_name=module_name,
                            code_type="method",
                            name=method["name"],
                            full_name=f"{class_full_name}.{method['name']}",
                            code_text=method_code,
                            parameters=method["params_list"],
                            return_type=method["return_type"],
                            class_name=class_name,
                            language=self.language,
                        )
                    classes.append(method_example)

        return classes

    async def _extract_functions(
        self, repo_name: str
    ) -> list[CodeExample | UniversalCodeExample]:
        """Extract standalone function definitions."""
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
        -[:DEFINES]->(func:Function)
        RETURN
            func.name as function_name,
            func.params_list as params_list,
            func.params_detailed as params_detailed,
            func.return_type as return_type,
            func.args as args,
            f.path as file_path,
            f.module_name as module_name
        ORDER BY func.name
        """

        result = await self.session.run(query, repo_name=repo_name)
        functions: list[CodeExample | UniversalCodeExample] = []

        async for record in result:
            function_name = record["function_name"]
            # Skip private functions
            if not function_name or function_name.startswith("_"):
                continue

            file_path = record["file_path"]
            module_name = record["module_name"] or ""
            params_list = record["params_list"]
            return_type = record["return_type"]

            if module_name:
                full_name = f"{module_name}.{function_name}"
            else:
                full_name = function_name

            function_example: CodeExample | UniversalCodeExample
            if self.use_universal:
                function_example = self._create_universal_function_example(
                    repo_name, file_path, module_name, function_name,
                    full_name, params_list, return_type, record,
                )
            else:
                # Generate function code representation (legacy)
                function_code = self._generate_function_code({
                    "name": function_name,
                    "params_list": params_list,
                    "params_detailed": record["params_detailed"],
                    "return_type": return_type,
                    "args": record["args"],
                })

                function_example = CodeExample(
                    repository_name=repo_name,
                    file_path=file_path,
                    module_name=module_name,
                    code_type="function",
                    name=function_name,
                    full_name=full_name,
                    code_text=function_code,
                    parameters=params_list,
                    return_type=return_type,
                    language=self.language,
                )

            functions.append(function_example)

        return functions

    def _create_universal_class_example(
        self, repo_name: str, file_path: str, module_name: str,
        class_name: str, class_full_name: str, methods: list[dict[str, Any]],
        method_count: int,
    ) -> UniversalCodeExample:
        """Create UniversalCodeExample for a class."""
        # Extract method names for child_elements
        public_methods = [
            m["name"] for m in methods
            if m["name"] and not m["name"].startswith("_")
        ]

        # Generate signature
        if self.language == "Python":
            signature = f"class {class_name}:"
        elif self.language in ["JavaScript", "TypeScript"]:
            signature = f"class {class_name} {{"
        elif self.language == "Go":
            signature = f"type {class_name} struct {{"
        else:
            signature = f"class {class_name}"

        # Language-specific metadata
        language_specific = {
            "method_count": method_count,
            "public_methods": public_methods[:10],  # Limit to first 10
        }

        if self.language == "Python":
            language_specific.update({
                "decorators": [],  # Could be extracted from Neo4j if available
                "base_classes": [],  # Could be extracted from Neo4j if available
            })

        return UniversalCodeExample(
            repository_name=repo_name,
            file_path=file_path,
            module_name=module_name,
            language=self.language,
            code_type="class",
            name=class_name,
            full_name=class_full_name,
            signature=signature,
            visibility="public",  # Default assumption
            child_elements=public_methods,
            language_specific=language_specific,
        )

    def _create_universal_method_example(
        self, repo_name: str, file_path: str, module_name: str,
        method: dict[str, Any], class_name: str, class_full_name: str,
    ) -> UniversalCodeExample:
        """Create UniversalCodeExample for a method."""
        method_name = method["name"]
        params_list = method["params_list"] or []
        return_type = method["return_type"]

        # Generate signature
        if self.language == "Python":
            param_str = ", ".join(params_list)
            return_str = f" -> {return_type}" if return_type else ""
            signature = f"def {method_name}({param_str}){return_str}:"
        elif self.language in ["JavaScript", "TypeScript"]:
            param_str = ", ".join(params_list)
            is_typescript = self.language == "TypeScript"
            return_str = f": {return_type}" if return_type and is_typescript else ""
            signature = f"function {method_name}({param_str}){return_str} {{"
        elif self.language == "Go":
            param_str = ", ".join(params_list)
            return_str = f" {return_type}" if return_type else ""
            signature = f"func (receiver) {method_name}({param_str}){return_str} {{"
        else:
            signature = f"def {method_name}()"

        # Language-specific metadata
        language_specific = {
            "parameters": params_list,
            "return_type": return_type,
            "detailed_params": method.get("params_detailed", []),
            "args": method.get("args", []),
        }

        # Determine visibility
        visibility = "private" if method_name.startswith("_") else "public"
        if method_name.startswith("__") and not method_name.endswith("__"):
            visibility = "private"  # Python name mangling

        return UniversalCodeExample(
            repository_name=repo_name,
            file_path=file_path,
            module_name=module_name,
            language=self.language,
            code_type="method",
            name=method_name,
            full_name=f"{class_full_name}.{method_name}",
            signature=signature,
            visibility=visibility,
            parent_name=class_name,
            language_specific=language_specific,
        )

    def _create_universal_function_example(
        self, repo_name: str, file_path: str, module_name: str,
        function_name: str, full_name: str, params_list: list[str],
        return_type: str, record: dict[str, Any],
    ) -> UniversalCodeExample:
        """Create UniversalCodeExample for a function."""
        # Generate signature
        if self.language == "Python":
            param_str = ", ".join(params_list) if params_list else ""
            return_str = f" -> {return_type}" if return_type else ""
            signature = f"def {function_name}({param_str}){return_str}:"
        elif self.language in ["JavaScript", "TypeScript"]:
            param_str = ", ".join(params_list) if params_list else ""
            is_ts = self.language == "TypeScript"
            return_str = f": {return_type}" if return_type and is_ts else ""
            signature = f"function {function_name}({param_str}){return_str} {{"
        elif self.language == "Go":
            param_str = ", ".join(params_list) if params_list else ""
            return_str = f" {return_type}" if return_type else ""
            signature = f"func {function_name}({param_str}){return_str} {{"
        else:
            signature = f"function {function_name}()"

        # Language-specific metadata
        language_specific = {
            "parameters": params_list,
            "return_type": return_type,
            "detailed_params": record.get("params_detailed", []),
            "args": record.get("args", []),
        }

        # Determine visibility
        visibility = "private" if function_name.startswith("_") else "public"

        return UniversalCodeExample(
            repository_name=repo_name,
            file_path=file_path,
            module_name=module_name,
            language=self.language,
            code_type="function",
            name=function_name,
            full_name=full_name,
            signature=signature,
            visibility=visibility,
            language_specific=language_specific,
        )

    def _generate_class_code(
        self, class_name: str, methods: list[dict[str, Any]]
    ) -> str:
        """Generate a code representation for a class (legacy method)."""
        code = f"class {class_name}:\n"
        code += '    """Class with the following public methods:"""\n'

        public_methods = [
            m for m in methods if m["name"] and not m["name"].startswith("_")
        ]
        if public_methods:
            for method in public_methods[:5]:  # Limit to first 5 methods
                params = ", ".join(method["params_list"] or [])
                return_type = method["return_type"] or "Any"
                code += f"    def {method['name']}({params}) -> {return_type}: ...\n"
        else:
            code += "    pass\n"

        return code

    def _generate_method_code(self, method: dict[str, Any]) -> str:
        """Generate a code representation for a method (legacy method)."""
        name = method["name"]
        params = ", ".join(method["params_list"] or [])
        return_type = method["return_type"] or "Any"

        code = f"def {name}({params}) -> {return_type}:\n"
        code += '    """Method implementation"""\n'
        code += "    pass"

        return code

    def _generate_function_code(self, function: dict[str, Any]) -> str:
        """Generate a code representation for a function (legacy method)."""
        name = function["name"]
        params = ", ".join(function["params_list"] or [])
        return_type = function["return_type"] or "Any"

        code = f"def {name}({params}) -> {return_type}:\n"
        code += '    """Function implementation"""\n'
        code += "    pass"

        return code


async def extract_repository_code(
    repo_extractor: Any, repo_name: str, use_universal: bool = False,
) -> dict[str, Any]:
    """
    Main function to extract code from a repository using the repository extractor.

    Args:
        repo_extractor: Repository extractor instance with Neo4j connection
        repo_name: Name of the repository to extract from
        use_universal: If True, use UniversalCodeExample format

    Returns:
        Dictionary with extraction results
    """
    try:
        # Get Neo4j session from the repository extractor
        async with repo_extractor.driver.session() as session:
            extractor = Neo4jCodeExtractor(session, use_universal=use_universal)
            code_examples = await extractor.extract_repository_code(repo_name)

            # Convert to serializable format
            examples_data = []
            for example in code_examples:
                if use_universal and isinstance(example, UniversalCodeExample):
                    # Enhanced data for UniversalCodeExample
                    examples_data.append({
                        "repository_name": example.repository_name,
                        "file_path": example.file_path,
                        "module_name": example.module_name,
                        "language": example.language,
                        "code_type": example.code_type,
                        "name": example.name,
                        "full_name": example.full_name,
                        "signature": example.signature,
                        "visibility": example.visibility,
                        "parent_name": example.parent_name,
                        "child_elements": example.child_elements,
                        "embedding_contexts": example.generate_embedding_contexts(),
                        "metadata": example.to_metadata(),
                        "validation_status": example.validation_status,
                        "confidence_score": example.confidence_score,
                        "language_specific": example.language_specific,
                    })
                else:
                    # Legacy format for CodeExample
                    assert isinstance(example, CodeExample), (
                        "Expected CodeExample in legacy format"
                    )
                    examples_data.append({
                        "repository_name": example.repository_name,
                        "file_path": example.file_path,
                        "module_name": example.module_name,
                        "code_type": example.code_type,
                        "name": example.name,
                        "full_name": example.full_name,
                        "code_text": example.code_text,
                        "embedding_text": example.generate_embedding_text(),
                        "metadata": example.to_metadata(),
                    })

            # Generate summary statistics
            if use_universal:
                class_count = len([e for e in code_examples
                                   if e.code_type == "class"])
                method_count = len([e for e in code_examples
                                    if e.code_type == "method"])
                function_count = len([e for e in code_examples
                                      if e.code_type == "function"])
                interface_count = len([e for e in code_examples
                                       if e.code_type == "interface"])
                struct_count = len([e for e in code_examples
                                    if e.code_type == "struct"])
                type_count = len([e for e in code_examples
                                  if e.code_type == "type"])
                variable_count = len([e for e in code_examples
                                      if e.code_type == "variable"])
                constant_count = len([e for e in code_examples
                                      if e.code_type == "constant"])
                languages = list({e.language for e in code_examples
                                  if hasattr(e, "language")})
                summary = {
                    "classes": class_count,
                    "methods": method_count,
                    "functions": function_count,
                    "interfaces": interface_count,
                    "structs": struct_count,
                    "types": type_count,
                    "variables": variable_count,
                    "constants": constant_count,
                    "languages": languages,
                }
            else:
                class_count = len([e for e in code_examples
                                   if e.code_type == "class"])
                method_count = len([e for e in code_examples
                                    if e.code_type == "method"])
                function_count = len([e for e in code_examples
                                      if e.code_type == "function"])
                summary = {
                    "classes": class_count,
                    "methods": method_count,
                    "functions": function_count,
                }

            return {
                "success": True,
                "repository_name": repo_name,
                "use_universal": use_universal,
                "code_examples_count": len(code_examples),
                "code_examples": examples_data,
                "extraction_summary": summary,
            }

    except QueryError as e:
        logger.exception(
            "Neo4j query failed extracting code from repository %s",
            repo_name
        )
        return {
            "success": False,
            "repository_name": repo_name,
            "use_universal": use_universal,
            "error": str(e),
        }
    except Exception as e:
        logger.exception(
            "Unexpected error extracting code from repository %s", repo_name
        )
        return {
            "success": False,
            "repository_name": repo_name,
            "use_universal": use_universal,
            "error": str(e),
        }


# Example usage of the enhanced multi-language code extraction:
#
# # Legacy mode (backward compatible)
# extractor = Neo4jCodeExtractor(session)
# examples = await extractor.extract_repository_code("my-repo")
#
# # Universal mode with multi-language support
# extractor = Neo4jCodeExtractor(session, use_universal=True, language="TypeScript")
# examples = await extractor.extract_repository_code("my-ts-repo")
#
# # Generate different embedding contexts for enhanced search
# for example in examples:
#     if isinstance(example, UniversalCodeExample):
#         signature_text = example.generate_embedding_text("signature")
#         semantic_text = example.generate_embedding_text("semantic")
#         usage_text = example.generate_embedding_text("usage")
#         full_text = example.generate_embedding_text("full")
