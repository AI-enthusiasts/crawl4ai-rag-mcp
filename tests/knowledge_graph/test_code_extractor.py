"""
Comprehensive unit tests for code_extractor.py.

Tests cover:
- CodeExample dataclass and methods
- UniversalCodeExample dataclass with validation
- Neo4jCodeExtractor extraction logic
- Module-level extraction function
- Error handling and edge cases
- Multi-language support (Python, JavaScript, TypeScript, Go)
"""

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.exceptions import QueryError
from src.knowledge_graph.code_extractor import (
    SUPPORTED_CODE_TYPES,
    SUPPORTED_LANGUAGES,
    VALIDATION_STATUSES,
    CodeExample,
    Neo4jCodeExtractor,
    UniversalCodeExample,
    extract_repository_code,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_code_example() -> CodeExample:
    """Create a sample CodeExample instance for testing."""
    return CodeExample(
        repository_name="test-repo",
        file_path="/src/test.py",
        module_name="test_module",
        code_type="function",
        name="test_function",
        full_name="test_module.test_function",
        code_text="def test_function(arg1, arg2):\n    pass",
        parameters=["arg1", "arg2"],
        return_type="str",
        language="python",
    )


@pytest.fixture
def sample_class_example() -> CodeExample:
    """Create a sample class CodeExample."""
    return CodeExample(
        repository_name="test-repo",
        file_path="/src/myclass.py",
        module_name="mymodule",
        code_type="class",
        name="MyClass",
        full_name="mymodule.MyClass",
        code_text="class MyClass:\n    pass",
        method_count=5,
        language="python",
    )


@pytest.fixture
def sample_method_example() -> CodeExample:
    """Create a sample method CodeExample."""
    return CodeExample(
        repository_name="test-repo",
        file_path="/src/myclass.py",
        module_name="mymodule",
        code_type="method",
        name="my_method",
        full_name="mymodule.MyClass.my_method",
        code_text="def my_method(self, arg1):\n    pass",
        parameters=["self", "arg1"],
        return_type="bool",
        class_name="MyClass",
        language="python",
    )


@pytest.fixture
def sample_universal_example() -> UniversalCodeExample:
    """Create a sample UniversalCodeExample."""
    return UniversalCodeExample(
        repository_name="test-repo",
        file_path="/src/test.py",
        module_name="test_module",
        language="Python",
        code_type="function",
        name="test_func",
        full_name="test_module.test_func",
        signature="def test_func(arg1: str) -> int:",
        documentation="Test function documentation",
        line_number=10,
        visibility="public",
        is_async=False,
        language_specific={"parameters": ["arg1: str"], "return_type": "int"},
    )


@pytest.fixture
def mock_neo4j_session() -> AsyncMock:
    """Create a mock Neo4j session."""
    session = AsyncMock()
    session.run = AsyncMock()
    return session


@pytest.fixture
def mock_neo4j_driver() -> MagicMock:
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    driver.session = MagicMock()
    return driver


# ============================================================================
# CODEEXAMPLE TESTS
# ============================================================================


class TestCodeExample:
    """Test CodeExample dataclass and methods."""

    def test_code_example_creation(self, sample_code_example: CodeExample) -> None:
        """Test basic CodeExample creation."""
        assert sample_code_example.repository_name == "test-repo"
        assert sample_code_example.name == "test_function"
        assert sample_code_example.code_type == "function"
        assert sample_code_example.parameters == ["arg1", "arg2"]

    def test_to_metadata_basic(self, sample_code_example: CodeExample) -> None:
        """Test to_metadata method with basic fields."""
        metadata = sample_code_example.to_metadata()

        assert metadata["repository_name"] == "test-repo"
        assert metadata["file_path"] == "/src/test.py"
        assert metadata["module_name"] == "test_module"
        assert metadata["code_type"] == "function"
        assert metadata["name"] == "test_function"
        assert metadata["full_name"] == "test_module.test_function"
        assert metadata["language"] == "python"
        assert metadata["validation_status"] == "extracted"

    def test_to_metadata_with_parameters(self, sample_code_example: CodeExample) -> None:
        """Test to_metadata includes parameters."""
        metadata = sample_code_example.to_metadata()
        assert "parameters" in metadata
        assert metadata["parameters"] == ["arg1", "arg2"]

    def test_to_metadata_with_return_type(self, sample_code_example: CodeExample) -> None:
        """Test to_metadata includes return type."""
        metadata = sample_code_example.to_metadata()
        assert "return_type" in metadata
        assert metadata["return_type"] == "str"

    def test_to_metadata_with_class_name(self, sample_method_example: CodeExample) -> None:
        """Test to_metadata includes class name for methods."""
        metadata = sample_method_example.to_metadata()
        assert "class_name" in metadata
        assert metadata["class_name"] == "MyClass"

    def test_to_metadata_with_method_count(self, sample_class_example: CodeExample) -> None:
        """Test to_metadata includes method count for classes."""
        metadata = sample_class_example.to_metadata()
        assert "method_count" in metadata
        assert metadata["method_count"] == 5

    def test_to_metadata_excludes_none_values(self) -> None:
        """Test to_metadata excludes None optional fields."""
        example = CodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            code_type="function",
            name="func",
            full_name="test.func",
            code_text="def func(): pass",
        )
        metadata = example.to_metadata()

        assert "parameters" not in metadata
        assert "return_type" not in metadata
        assert "class_name" not in metadata

    def test_generate_embedding_text_function(self, sample_code_example: CodeExample) -> None:
        """Test generate_embedding_text for function type."""
        text = sample_code_example.generate_embedding_text()

        assert "Python function test_function" in text
        assert "test_module" in text
        assert "Parameters: arg1, arg2" in text
        assert "Returns: str" in text
        assert "def test_function(arg1, arg2):" in text

    def test_generate_embedding_text_class(self, sample_class_example: CodeExample) -> None:
        """Test generate_embedding_text for class type."""
        text = sample_class_example.generate_embedding_text()

        assert "Python class MyClass" in text
        assert "mymodule" in text
        assert "mymodule.MyClass" in text
        assert "Contains 5 methods" in text

    def test_generate_embedding_text_method(self, sample_method_example: CodeExample) -> None:
        """Test generate_embedding_text for method type."""
        text = sample_method_example.generate_embedding_text()

        assert "Python method my_method" in text
        assert "in class MyClass" in text
        assert "mymodule" in text
        assert "Parameters: self, arg1" in text
        assert "Returns: bool" in text

    def test_generate_embedding_text_unknown_type(self) -> None:
        """Test generate_embedding_text with unknown code type."""
        example = CodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            code_type="unknown",
            name="something",
            full_name="test.something",
            code_text="unknown code",
        )
        text = example.generate_embedding_text()

        assert "Python unknown something" in text
        assert "unknown code" in text


# ============================================================================
# UNIVERSALCODEEXAMPLE TESTS
# ============================================================================


class TestUniversalCodeExample:
    """Test UniversalCodeExample dataclass and validation."""

    def test_universal_example_creation(self, sample_universal_example: UniversalCodeExample) -> None:
        """Test basic UniversalCodeExample creation."""
        assert sample_universal_example.repository_name == "test-repo"
        assert sample_universal_example.language == "Python"
        assert sample_universal_example.code_type == "function"
        assert sample_universal_example.name == "test_func"
        assert sample_universal_example.visibility == "public"

    def test_validation_success(self) -> None:
        """Test validation passes with valid data."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="function",
            name="func",
            full_name="test.func",
            confidence_score=0.95,
        )
        assert example.validation_status == "extracted"
        assert example.confidence_score == 0.95

    def test_validation_invalid_language(self) -> None:
        """Test validation raises ValueError for unsupported language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            UniversalCodeExample(
                repository_name="test-repo",
                file_path="/src/test.py",
                module_name="test",
                language="Ruby",  # Not in SUPPORTED_LANGUAGES
                code_type="function",
                name="func",
                full_name="test.func",
            )

    def test_validation_invalid_code_type(self) -> None:
        """Test validation raises ValueError for unsupported code type."""
        with pytest.raises(ValueError, match="Unsupported code_type"):
            UniversalCodeExample(
                repository_name="test-repo",
                file_path="/src/test.py",
                module_name="test",
                language="Python",
                code_type="procedure",  # Not in SUPPORTED_CODE_TYPES
                name="proc",
                full_name="test.proc",
            )

    def test_validation_invalid_status(self) -> None:
        """Test validation raises ValueError for invalid validation status."""
        with pytest.raises(ValueError, match="Invalid validation_status"):
            UniversalCodeExample(
                repository_name="test-repo",
                file_path="/src/test.py",
                module_name="test",
                language="Python",
                code_type="function",
                name="func",
                full_name="test.func",
                validation_status="invalid_status",
            )

    def test_validation_confidence_score_too_low(self) -> None:
        """Test validation raises ValueError for confidence score < 0."""
        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            UniversalCodeExample(
                repository_name="test-repo",
                file_path="/src/test.py",
                module_name="test",
                language="Python",
                code_type="function",
                name="func",
                full_name="test.func",
                confidence_score=-0.1,
            )

    def test_validation_confidence_score_too_high(self) -> None:
        """Test validation raises ValueError for confidence score > 1."""
        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            UniversalCodeExample(
                repository_name="test-repo",
                file_path="/src/test.py",
                module_name="test",
                language="Python",
                code_type="function",
                name="func",
                full_name="test.func",
                confidence_score=1.5,
            )

    def test_to_metadata_comprehensive(self, sample_universal_example: UniversalCodeExample) -> None:
        """Test to_metadata includes all fields."""
        metadata = sample_universal_example.to_metadata()

        assert metadata["repository_name"] == "test-repo"
        assert metadata["file_path"] == "/src/test.py"
        assert metadata["module_name"] == "test_module"
        assert metadata["language"] == "Python"
        assert metadata["code_type"] == "function"
        assert metadata["name"] == "test_func"
        assert metadata["full_name"] == "test_module.test_func"
        assert metadata["signature"] == "def test_func(arg1: str) -> int:"
        assert metadata["visibility"] == "public"
        assert metadata["line_number"] == 10
        assert metadata["documentation"] == "Test function documentation"

    def test_to_metadata_with_child_elements(self) -> None:
        """Test to_metadata includes child_elements."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="class",
            name="MyClass",
            full_name="test.MyClass",
            child_elements=["method1", "method2", "method3"],
        )
        metadata = example.to_metadata()

        assert "child_elements" in metadata
        assert len(metadata["child_elements"]) == 3

    def test_generate_embedding_contexts(self, sample_universal_example: UniversalCodeExample) -> None:
        """Test generate_embedding_contexts creates all context types."""
        contexts = sample_universal_example.generate_embedding_contexts()

        assert "signature" in contexts
        assert "semantic" in contexts
        assert "usage" in contexts
        assert "full" in contexts

    def test_generate_embedding_text_signature(self, sample_universal_example: UniversalCodeExample) -> None:
        """Test generate_embedding_text with signature context."""
        text = sample_universal_example.generate_embedding_text("signature")

        assert "Python" in text
        assert "function" in text
        assert "test_func" in text

    def test_generate_embedding_text_default_full(self, sample_universal_example: UniversalCodeExample) -> None:
        """Test generate_embedding_text defaults to full context."""
        text = sample_universal_example.generate_embedding_text()

        assert "Python" in text
        assert "function" in text
        assert "test_func" in text
        assert "test_module" in text

    def test_python_signature_generation_function(self) -> None:
        """Test Python function signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="function",
            name="my_func",
            full_name="test.my_func",
            language_specific={"parameters": ["x: int", "y: str"], "return_type": "bool"},
        )
        contexts = example.generate_embedding_contexts()

        assert "def my_func(x: int, y: str) -> bool:" in contexts["signature"]

    def test_python_signature_generation_class(self) -> None:
        """Test Python class signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="class",
            name="MyClass",
            full_name="test.MyClass",
        )
        contexts = example.generate_embedding_contexts()

        assert "class MyClass:" in contexts["signature"]

    def test_javascript_signature_generation_function(self) -> None:
        """Test JavaScript function signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.js",
            module_name="test",
            language="JavaScript",
            code_type="function",
            name="myFunc",
            full_name="test.myFunc",
            language_specific={"parameters": ["x", "y"], "return_type": ""},
        )
        contexts = example.generate_embedding_contexts()

        assert "function myFunc(x, y)" in contexts["signature"]

    def test_javascript_signature_generation_async(self) -> None:
        """Test JavaScript async function signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.js",
            module_name="test",
            language="JavaScript",
            code_type="function",
            name="asyncFunc",
            full_name="test.asyncFunc",
            is_async=True,
            language_specific={"parameters": []},
        )
        contexts = example.generate_embedding_contexts()

        assert "async function asyncFunc()" in contexts["signature"]

    def test_typescript_signature_generation_with_types(self) -> None:
        """Test TypeScript function with return type."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.ts",
            module_name="test",
            language="TypeScript",
            code_type="function",
            name="typedFunc",
            full_name="test.typedFunc",
            language_specific={"parameters": ["x: number"], "return_type": "string"},
        )
        contexts = example.generate_embedding_contexts()

        assert "function typedFunc(x: number): string" in contexts["signature"]

    def test_typescript_interface_generation(self) -> None:
        """Test TypeScript interface signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.ts",
            module_name="test",
            language="TypeScript",
            code_type="interface",
            name="MyInterface",
            full_name="test.MyInterface",
        )
        contexts = example.generate_embedding_contexts()

        assert "interface MyInterface {" in contexts["signature"]

    def test_go_signature_generation_function(self) -> None:
        """Test Go function signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.go",
            module_name="test",
            language="Go",
            code_type="function",
            name="MyFunc",
            full_name="test.MyFunc",
            language_specific={"parameters": ["x int", "y string"], "return_type": "bool"},
        )
        contexts = example.generate_embedding_contexts()

        assert "func MyFunc(x int, y string) bool" in contexts["signature"]

    def test_go_signature_generation_method_with_receiver(self) -> None:
        """Test Go method with receiver signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.go",
            module_name="test",
            language="Go",
            code_type="method",
            name="MyMethod",
            full_name="test.MyStruct.MyMethod",
            language_specific={"receiver": "s *MyStruct", "parameters": ["x int"], "return_type": "error"},
        )
        contexts = example.generate_embedding_contexts()

        assert "func (s *MyStruct) MyMethod(x int) error" in contexts["signature"]

    def test_go_struct_generation(self) -> None:
        """Test Go struct signature generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.go",
            module_name="test",
            language="Go",
            code_type="struct",
            name="MyStruct",
            full_name="test.MyStruct",
        )
        contexts = example.generate_embedding_contexts()

        assert "type MyStruct struct {" in contexts["signature"]

    def test_semantic_context_with_parent(self) -> None:
        """Test semantic context includes parent name."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="method",
            name="my_method",
            full_name="test.MyClass.my_method",
            parent_name="MyClass",
            documentation="A test method",
        )
        contexts = example.generate_embedding_contexts()

        assert "my_method" in contexts["semantic"]
        assert "in MyClass" in contexts["semantic"]
        assert "A test method" in contexts["semantic"]

    def test_semantic_context_with_children(self) -> None:
        """Test semantic context includes child elements."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="class",
            name="MyClass",
            full_name="test.MyClass",
            child_elements=["method1", "method2", "method3", "method4", "method5", "method6"],
        )
        contexts = example.generate_embedding_contexts()

        assert "Contains:" in contexts["semantic"]
        # Should limit to first 5
        assert "method1" in contexts["semantic"]
        assert "method5" in contexts["semantic"]

    def test_python_usage_context_class(self) -> None:
        """Test Python class usage generation."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="class",
            name="MyClass",
            full_name="test.MyClass",
        )
        contexts = example.generate_embedding_contexts()

        assert "instance = MyClass()" in contexts["usage"]

    def test_python_usage_context_function_with_params(self) -> None:
        """Test Python function usage with parameters."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="function",
            name="my_func",
            full_name="test.my_func",
            language_specific={"parameters": ["x", "y", "z"]},
        )
        contexts = example.generate_embedding_contexts()

        assert "my_func(arg0, arg1, arg2)" in contexts["usage"]

    def test_full_context_comprehensive(self) -> None:
        """Test full context includes all information."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="function",
            name="complex_func",
            full_name="test.complex_func",
            signature="def complex_func(x: int) -> str:",
            documentation="A complex function",
            visibility="public",
            is_async=True,
            is_static=True,
            child_elements=["helper1", "helper2"],
            language_specific={"decorator": "@staticmethod", "complexity": "high"},
        )
        contexts = example.generate_embedding_contexts()

        full = contexts["full"]
        assert "complex_func" in full
        assert "def complex_func(x: int) -> str:" in full
        assert "A complex function" in full
        assert "async" in full
        assert "static" in full
        assert "helper1" in full
        assert "decorator" in full

    def test_to_code_example_conversion(self) -> None:
        """Test conversion from UniversalCodeExample to CodeExample."""
        universal = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="function",
            name="my_func",
            full_name="test.my_func",
            signature="def my_func(x: int) -> str:",
            parent_name="MyClass",
            child_elements=["helper1", "helper2"],
            language_specific={"parameters": ["x: int"], "return_type": "str"},
        )

        code_example = universal.to_code_example()

        assert isinstance(code_example, CodeExample)
        assert code_example.repository_name == "test-repo"
        assert code_example.file_path == "/src/test.py"
        assert code_example.module_name == "test"
        assert code_example.name == "my_func"
        assert code_example.full_name == "test.my_func"
        assert code_example.parameters == ["x: int"]
        assert code_example.return_type == "str"

    def test_to_code_example_non_python(self) -> None:
        """Test conversion from UniversalCodeExample with non-Python language."""
        universal = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.go",
            module_name="test",
            language="Go",
            code_type="function",
            name="MyFunc",
            full_name="test.MyFunc",
            signature="func MyFunc(x int) bool",
        )

        code_example = universal.to_code_example()

        assert code_example.language == "Go"
        assert code_example.parameters is None
        assert code_example.return_type is None


# ============================================================================
# NEO4JCODEEXTRACTOR TESTS
# ============================================================================


class TestNeo4jCodeExtractor:
    """Test Neo4jCodeExtractor class."""

    def test_extractor_initialization(self, mock_neo4j_session: AsyncMock) -> None:
        """Test Neo4jCodeExtractor initialization."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session)

        assert extractor.session == mock_neo4j_session
        assert extractor.use_universal is False
        assert extractor.language == "Python"

    def test_extractor_initialization_with_universal(self, mock_neo4j_session: AsyncMock) -> None:
        """Test Neo4jCodeExtractor initialization with universal mode."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=True, language="TypeScript")

        assert extractor.use_universal is True
        assert extractor.language == "TypeScript"

    @pytest.mark.asyncio
    async def test_repository_exists_true(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _repository_exists returns True when repository exists."""
        mock_result = AsyncMock()
        mock_record = MagicMock()
        mock_record.get.return_value = "test-repo"
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_neo4j_session.run = AsyncMock(return_value=mock_result)

        extractor = Neo4jCodeExtractor(mock_neo4j_session)
        exists = await extractor._repository_exists("test-repo")

        assert exists is True
        mock_neo4j_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_repository_exists_false(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _repository_exists returns False when repository doesn't exist."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_neo4j_session.run = AsyncMock(return_value=mock_result)

        extractor = Neo4jCodeExtractor(mock_neo4j_session)
        exists = await extractor._repository_exists("nonexistent-repo")

        assert exists is False

    @pytest.mark.asyncio
    async def test_extract_repository_code_not_found(self, mock_neo4j_session: AsyncMock) -> None:
        """Test extract_repository_code raises ValueError for nonexistent repo."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_neo4j_session.run = AsyncMock(return_value=mock_result)

        extractor = Neo4jCodeExtractor(mock_neo4j_session)

        with pytest.raises(ValueError, match="Repository 'nonexistent' not found"):
            await extractor.extract_repository_code("nonexistent")

    @pytest.mark.asyncio
    async def test_extract_classes_legacy_mode(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _extract_classes in legacy mode."""
        # Mock repository exists check
        mock_exists_result = AsyncMock()
        mock_exists_record = MagicMock()
        mock_exists_result.single = AsyncMock(return_value=mock_exists_record)

        # Mock classes query result with proper async iterator
        class_records = [
            {
                "class_name": "TestClass",
                "class_full_name": "module.TestClass",
                "file_path": "/src/test.py",
                "module_name": "module",
                "method_count": 2,
                "methods": [
                    {
                        "name": "public_method",
                        "params_list": ["self", "arg1"],
                        "params_detailed": [],
                        "return_type": "str",
                        "args": [],
                    },
                    {
                        "name": "_private_method",
                        "params_list": ["self"],
                        "params_detailed": [],
                        "return_type": None,
                        "args": [],
                    },
                ],
            }
        ]

        async def async_iter_classes():
            for record in class_records:
                yield record

        mock_classes_result = AsyncMock()
        mock_classes_result.__aiter__ = lambda self: async_iter_classes()

        # Mock functions query result (empty)
        async def async_iter_functions():
            for record in []:
                yield record

        mock_functions_result = AsyncMock()
        mock_functions_result.__aiter__ = lambda self: async_iter_functions()

        mock_neo4j_session.run = AsyncMock(side_effect=[
            mock_exists_result,  # Repository exists check
            mock_classes_result,  # Classes query
            mock_functions_result,  # Functions query
        ])

        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=False)
        examples = await extractor.extract_repository_code("test-repo")

        # Should have class + 1 public method (private method excluded)
        assert len(examples) == 2
        assert examples[0].code_type == "class"
        assert examples[0].name == "TestClass"
        assert examples[1].code_type == "method"
        assert examples[1].name == "public_method"

    @pytest.mark.asyncio
    async def test_extract_functions_legacy_mode(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _extract_functions in legacy mode."""
        # Mock repository exists
        mock_exists_result = AsyncMock()
        mock_exists_record = MagicMock()
        mock_exists_result.single = AsyncMock(return_value=mock_exists_record)

        # Mock empty classes result
        async def async_iter_classes():
            for record in []:
                yield record

        mock_classes_result = AsyncMock()
        mock_classes_result.__aiter__ = lambda self: async_iter_classes()

        # Mock functions query result
        function_records = [
            {
                "function_name": "test_function",
                "params_list": ["arg1", "arg2"],
                "params_detailed": [],
                "return_type": "bool",
                "args": [],
                "file_path": "/src/test.py",
                "module_name": "test_module",
            },
            {
                "function_name": "_private_function",
                "params_list": [],
                "params_detailed": [],
                "return_type": None,
                "args": [],
                "file_path": "/src/test.py",
                "module_name": "test_module",
            },
        ]

        async def async_iter_functions():
            for record in function_records:
                yield record

        mock_functions_result = AsyncMock()
        mock_functions_result.__aiter__ = lambda self: async_iter_functions()

        mock_neo4j_session.run = AsyncMock(side_effect=[
            mock_exists_result,
            mock_classes_result,
            mock_functions_result,
        ])

        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=False)
        examples = await extractor.extract_repository_code("test-repo")

        # Should only have public function (private function excluded)
        assert len(examples) == 1
        assert examples[0].code_type == "function"
        assert examples[0].name == "test_function"
        assert examples[0].parameters == ["arg1", "arg2"]

    @pytest.mark.asyncio
    async def test_extract_repository_code_universal_mode(self, mock_neo4j_session: AsyncMock) -> None:
        """Test extract_repository_code_universal wrapper."""
        # Mock repository exists
        mock_exists_result = AsyncMock()
        mock_exists_record = MagicMock()
        mock_exists_result.single = AsyncMock(return_value=mock_exists_record)

        # Mock empty results
        async def async_iter_empty():
            for record in []:
                yield record

        mock_classes_result = AsyncMock()
        mock_classes_result.__aiter__ = lambda self: async_iter_empty()
        mock_functions_result = AsyncMock()
        mock_functions_result.__aiter__ = lambda self: async_iter_empty()

        mock_neo4j_session.run = AsyncMock(side_effect=[
            mock_exists_result,
            mock_classes_result,
            mock_functions_result,
        ])

        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=False)
        examples = await extractor.extract_repository_code_universal("test-repo")

        # Should be empty but use_universal should be reset
        assert len(examples) == 0
        assert extractor.use_universal is False

    @pytest.mark.asyncio
    async def test_extract_classes_universal_mode(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _extract_classes in universal mode."""
        # Mock repository exists
        mock_exists_result = AsyncMock()
        mock_exists_record = MagicMock()
        mock_exists_result.single = AsyncMock(return_value=mock_exists_record)

        # Mock classes query
        class_records = [
            {
                "class_name": "UniversalClass",
                "class_full_name": "module.UniversalClass",
                "file_path": "/src/universal.py",
                "module_name": "module",
                "method_count": 1,
                "methods": [
                    {
                        "name": "method",
                        "params_list": ["self"],
                        "params_detailed": [],
                        "return_type": "None",
                        "args": [],
                    }
                ],
            }
        ]

        async def async_iter_classes():
            for record in class_records:
                yield record

        mock_classes_result = AsyncMock()
        mock_classes_result.__aiter__ = lambda self: async_iter_classes()

        # Mock empty functions
        async def async_iter_empty():
            for record in []:
                yield record

        mock_functions_result = AsyncMock()
        mock_functions_result.__aiter__ = lambda self: async_iter_empty()

        mock_neo4j_session.run = AsyncMock(side_effect=[
            mock_exists_result,
            mock_classes_result,
            mock_functions_result,
        ])

        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=True, language="Python")
        examples = await extractor.extract_repository_code("test-repo")

        assert len(examples) == 2
        assert isinstance(examples[0], UniversalCodeExample)
        assert examples[0].language == "Python"
        assert examples[0].code_type == "class"

    def test_generate_class_code(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _generate_class_code method."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session)
        methods = [
            {"name": "method1", "params_list": ["self", "arg1"], "return_type": "str"},
            {"name": "method2", "params_list": ["self"], "return_type": "int"},
        ]

        code = extractor._generate_class_code("TestClass", methods)

        assert "class TestClass:" in code
        assert "def method1(self, arg1) -> str" in code
        assert "def method2(self) -> int" in code

    def test_generate_method_code(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _generate_method_code method."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session)
        method = {
            "name": "test_method",
            "params_list": ["self", "arg1", "arg2"],
            "return_type": "bool",
        }

        code = extractor._generate_method_code(method)

        assert "def test_method(self, arg1, arg2) -> bool:" in code
        assert "Method implementation" in code

    def test_generate_function_code(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _generate_function_code method."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session)
        function = {
            "name": "test_func",
            "params_list": ["x", "y"],
            "return_type": "str",
        }

        code = extractor._generate_function_code(function)

        assert "def test_func(x, y) -> str:" in code
        assert "Function implementation" in code

    def test_create_universal_class_example(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _create_universal_class_example method."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=True, language="Python")
        methods = [
            {"name": "method1", "params_list": ["self"]},
            {"name": "_private", "params_list": ["self"]},
            {"name": "method2", "params_list": ["self", "arg"]},
        ]

        example = extractor._create_universal_class_example(
            repo_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            class_name="MyClass",
            class_full_name="test.MyClass",
            methods=methods,
            method_count=3,
        )

        assert isinstance(example, UniversalCodeExample)
        assert example.code_type == "class"
        assert example.name == "MyClass"
        assert "class MyClass:" in example.signature
        assert len(example.child_elements) == 2  # Excludes private method

    def test_create_universal_method_example(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _create_universal_method_example method."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=True, language="Python")
        method = {
            "name": "test_method",
            "params_list": ["self", "arg1"],
            "return_type": "str",
            "params_detailed": [],
            "args": [],
        }

        example = extractor._create_universal_method_example(
            repo_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            method=method,
            class_name="MyClass",
            class_full_name="test.MyClass",
        )

        assert isinstance(example, UniversalCodeExample)
        assert example.code_type == "method"
        assert example.name == "test_method"
        assert example.parent_name == "MyClass"
        assert example.visibility == "public"

    def test_create_universal_method_example_private(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _create_universal_method_example with private method."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=True)
        method = {
            "name": "_private_method",
            "params_list": ["self"],
            "return_type": None,
            "params_detailed": [],
            "args": [],
        }

        example = extractor._create_universal_method_example(
            repo_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            method=method,
            class_name="MyClass",
            class_full_name="test.MyClass",
        )

        assert example.visibility == "private"

    def test_create_universal_function_example(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _create_universal_function_example method."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session, use_universal=True, language="Python")
        record = {
            "params_detailed": [],
            "args": [],
        }

        example = extractor._create_universal_function_example(
            repo_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            function_name="my_func",
            full_name="test.my_func",
            params_list=["x", "y"],
            return_type="bool",
            record=record,
        )

        assert isinstance(example, UniversalCodeExample)
        assert example.code_type == "function"
        assert example.name == "my_func"
        assert "def my_func(x, y) -> bool:" in example.signature


# ============================================================================
# MODULE-LEVEL FUNCTION TESTS
# ============================================================================


class TestExtractRepositoryCodeFunction:
    """Test module-level extract_repository_code function."""

    @pytest.mark.asyncio
    async def test_extract_repository_code_success(self, mock_neo4j_driver: MagicMock) -> None:
        """Test extract_repository_code function with successful extraction."""
        # Create mock session with proper async context manager
        mock_session = AsyncMock()

        # Mock repository exists
        mock_exists_result = AsyncMock()
        mock_exists_record = MagicMock()
        mock_exists_result.single = AsyncMock(return_value=mock_exists_record)

        # Mock classes query (one class)
        class_records = [
            {
                "class_name": "TestClass",
                "class_full_name": "test.TestClass",
                "file_path": "/src/test.py",
                "module_name": "test",
                "method_count": 1,
                "methods": [{"name": "method", "params_list": [], "params_detailed": [], "return_type": None, "args": []}],
            }
        ]

        async def async_iter_classes():
            for record in class_records:
                yield record

        mock_classes_result = AsyncMock()
        mock_classes_result.__aiter__ = lambda self: async_iter_classes()

        # Mock functions query (one function)
        function_records = [
            {
                "function_name": "test_func",
                "params_list": ["arg"],
                "params_detailed": [],
                "return_type": "str",
                "args": [],
                "file_path": "/src/test.py",
                "module_name": "test",
            }
        ]

        async def async_iter_functions():
            for record in function_records:
                yield record

        mock_functions_result = AsyncMock()
        mock_functions_result.__aiter__ = lambda self: async_iter_functions()

        mock_session.run = AsyncMock(side_effect=[
            mock_exists_result,
            mock_classes_result,
            mock_functions_result,
        ])

        # Create async context manager for session
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        # Create mock repo_extractor with driver
        mock_repo_extractor = MagicMock()
        mock_repo_extractor.driver.session.return_value = AsyncContextManager()

        result = await extract_repository_code(mock_repo_extractor, "test-repo", use_universal=False)

        assert result["success"] is True
        assert result["repository_name"] == "test-repo"
        assert result["use_universal"] is False
        assert result["code_examples_count"] == 3  # class + method + function
        assert "extraction_summary" in result
        assert result["extraction_summary"]["classes"] == 1
        assert result["extraction_summary"]["methods"] == 1
        assert result["extraction_summary"]["functions"] == 1

    @pytest.mark.asyncio
    async def test_extract_repository_code_query_error(self, mock_neo4j_driver: MagicMock) -> None:
        """Test extract_repository_code handles QueryError."""
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=QueryError("Query failed"))

        class AsyncContextManager:
            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        mock_repo_extractor = MagicMock()
        mock_repo_extractor.driver.session.return_value = AsyncContextManager()

        result = await extract_repository_code(mock_repo_extractor, "test-repo")

        assert result["success"] is False
        assert "error" in result
        assert "Query failed" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_repository_code_generic_exception(self, mock_neo4j_driver: MagicMock) -> None:
        """Test extract_repository_code handles generic exceptions."""
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        class AsyncContextManager:
            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        mock_repo_extractor = MagicMock()
        mock_repo_extractor.driver.session.return_value = AsyncContextManager()

        result = await extract_repository_code(mock_repo_extractor, "test-repo")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_repository_code_universal_mode(self, mock_neo4j_driver: MagicMock) -> None:
        """Test extract_repository_code with universal mode."""
        mock_session = AsyncMock()

        # Mock repository exists
        mock_exists_result = AsyncMock()
        mock_exists_record = MagicMock()
        mock_exists_result.single = AsyncMock(return_value=mock_exists_record)

        # Mock empty results
        async def async_iter_empty():
            for record in []:
                yield record

        mock_classes_result = AsyncMock()
        mock_classes_result.__aiter__ = lambda self: async_iter_empty()
        mock_functions_result = AsyncMock()
        mock_functions_result.__aiter__ = lambda self: async_iter_empty()

        mock_session.run = AsyncMock(side_effect=[
            mock_exists_result,
            mock_classes_result,
            mock_functions_result,
        ])

        class AsyncContextManager:
            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        mock_repo_extractor = MagicMock()
        mock_repo_extractor.driver.session.return_value = AsyncContextManager()

        result = await extract_repository_code(mock_repo_extractor, "test-repo", use_universal=True)

        assert result["success"] is True
        assert result["use_universal"] is True
        # Universal mode includes more code types in summary
        assert "interfaces" in result["extraction_summary"]
        assert "structs" in result["extraction_summary"]


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_code_example_with_empty_parameters(self) -> None:
        """Test CodeExample with empty parameter list excludes parameters from metadata."""
        example = CodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            code_type="function",
            name="func",
            full_name="test.func",
            code_text="def func(): pass",
            parameters=[],
        )

        assert example.parameters == []
        metadata = example.to_metadata()
        # Empty list is falsy, so parameters should not be included
        assert "parameters" not in metadata

    def test_universal_example_with_empty_strings(self) -> None:
        """Test UniversalCodeExample with empty string fields."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="",
            module_name="",
            language="Python",
            code_type="function",
            name="func",
            full_name="func",
        )

        assert example.file_path == ""
        assert example.module_name == ""

    def test_universal_example_confidence_score_boundaries(self) -> None:
        """Test UniversalCodeExample with boundary confidence scores."""
        # Test 0.0
        example1 = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="function",
            name="func",
            full_name="test.func",
            confidence_score=0.0,
        )
        assert example1.confidence_score == 0.0

        # Test 1.0
        example2 = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="function",
            name="func",
            full_name="test.func",
            confidence_score=1.0,
        )
        assert example2.confidence_score == 1.0

    def test_supported_languages_constant(self) -> None:
        """Test SUPPORTED_LANGUAGES constant."""
        assert "Python" in SUPPORTED_LANGUAGES
        assert "JavaScript" in SUPPORTED_LANGUAGES
        assert "TypeScript" in SUPPORTED_LANGUAGES
        assert "Go" in SUPPORTED_LANGUAGES

    def test_supported_code_types_constant(self) -> None:
        """Test SUPPORTED_CODE_TYPES constant."""
        assert "class" in SUPPORTED_CODE_TYPES
        assert "function" in SUPPORTED_CODE_TYPES
        assert "method" in SUPPORTED_CODE_TYPES
        assert "interface" in SUPPORTED_CODE_TYPES
        assert "struct" in SUPPORTED_CODE_TYPES

    def test_validation_statuses_constant(self) -> None:
        """Test VALIDATION_STATUSES constant."""
        assert "extracted" in VALIDATION_STATUSES
        assert "validated" in VALIDATION_STATUSES
        assert "verified" in VALIDATION_STATUSES

    def test_generate_class_code_with_no_public_methods(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _generate_class_code with no public methods."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session)
        methods = [
            {"name": "_private1", "params_list": ["self"], "return_type": None},
            {"name": "__private2", "params_list": ["self"], "return_type": None},
        ]

        code = extractor._generate_class_code("EmptyClass", methods)

        assert "class EmptyClass:" in code
        assert "pass" in code

    def test_generate_class_code_with_many_methods(self, mock_neo4j_session: AsyncMock) -> None:
        """Test _generate_class_code limits to first 5 methods."""
        extractor = Neo4jCodeExtractor(mock_neo4j_session)
        methods = [
            {"name": f"method{i}", "params_list": ["self"], "return_type": "None"}
            for i in range(10)
        ]

        code = extractor._generate_class_code("BigClass", methods)

        # Should only include first 5 methods
        assert "method0" in code
        assert "method4" in code
        assert "method5" not in code

    def test_universal_example_with_no_signature(self) -> None:
        """Test UniversalCodeExample generates signature when not provided."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="class",
            name="MyClass",
            full_name="test.MyClass",
        )

        contexts = example.generate_embedding_contexts()
        assert "class MyClass:" in contexts["signature"]

    def test_universal_example_language_specific_empty(self) -> None:
        """Test UniversalCodeExample with empty language_specific dict."""
        example = UniversalCodeExample(
            repository_name="test-repo",
            file_path="/src/test.py",
            module_name="test",
            language="Python",
            code_type="variable",
            name="my_var",
            full_name="test.my_var",
            language_specific={},
        )

        contexts = example.generate_embedding_contexts()
        assert "my_var" in contexts["signature"]
