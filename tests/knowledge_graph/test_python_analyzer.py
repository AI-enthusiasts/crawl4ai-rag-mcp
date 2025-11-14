"""
Comprehensive unit tests for the Python code analyzer.

Tests cover:
- Neo4jCodeAnalyzer class initialization and methods
- AST parsing for Python files
- Class extraction with methods and attributes
- Function extraction with parameters
- Import extraction (internal vs external)
- Error handling with AnalysisError and ParsingError
- Edge cases and invalid syntax
"""

import ast
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.knowledge_graph.analyzers.python_analyzer import Neo4jCodeAnalyzer
from src.core.exceptions import ParsingError, AnalysisError


class TestNeo4jCodeAnalyzer:
    """Test suite for Python code analyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = Neo4jCodeAnalyzer()
        self.repo_root = Path("/test/repo")
        self.project_modules = {"myproject", "src"}

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer is not None
        assert len(self.analyzer.external_modules) > 0
        # Verify some known external modules
        assert "os" in self.analyzer.external_modules
        assert "sys" in self.analyzer.external_modules
        assert "requests" in self.analyzer.external_modules
        assert "pydantic" in self.analyzer.external_modules

    def test_analyze_python_file_simple_class(self):
        """Test analyzing a simple Python class."""
        python_content = '''
"""Module docstring."""

class SimpleClass:
    """A simple class."""

    def __init__(self, name: str):
        """Initialize the class."""
        self.name = name

    def get_name(self) -> str:
        """Get the name."""
        return self.name
'''
        file_path = self.repo_root / "simple.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        assert result["module_name"] == "simple"
        assert result["file_path"] == "simple.py"
        assert len(result["classes"]) == 1

        # Verify class structure
        cls = result["classes"][0]
        assert cls["name"] == "SimpleClass"
        assert cls["full_name"] == "simple.SimpleClass"
        assert len(cls["methods"]) == 1  # Only public methods
        assert cls["methods"][0]["name"] == "get_name"
        assert cls["methods"][0]["return_type"] == "str"

    def test_analyze_python_file_with_functions(self):
        """Test analyzing Python functions."""
        python_content = '''
def public_function(param1: int, param2: str = "default") -> bool:
    """A public function."""
    return True

def _private_function():
    """Should be ignored."""
    pass

async def async_function(data: dict) -> None:
    """An async function."""
    await process(data)
'''
        file_path = self.repo_root / "functions.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        assert len(result["functions"]) == 2  # Excludes private functions

        # Check public_function
        func = next(f for f in result["functions"] if f["name"] == "public_function")
        assert func["return_type"] == "bool"
        assert len(func["params"]) == 2
        assert func["params"][0]["name"] == "param1"
        assert func["params"][0]["type"] == "int"
        assert func["params"][1]["name"] == "param2"
        assert func["params"][1]["type"] == "str"
        assert func["params"][1]["optional"] is True
        assert func["params"][1]["default"] == "'default'"

    def test_analyze_python_file_with_imports(self):
        """Test extracting imports (internal vs external)."""
        python_content = '''
import os
import sys
from pathlib import Path
from typing import List, Dict

# Internal imports
from myproject.utils import helper
from src.services import ServiceClass
import myproject.models
from .relative import RelativeImport
'''
        file_path = self.repo_root / "src/myproject/module.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        imports = result["imports"]

        # Should only include internal imports
        assert "myproject.utils" in imports or "myproject" in imports
        assert "src.services" in imports or "src" in imports

        # Should NOT include external modules
        assert "os" not in imports
        assert "sys" not in imports
        assert "pathlib" not in imports
        assert "typing" not in imports

    def test_extract_class_attributes_dataclass(self):
        """Test extracting attributes from dataclass."""
        python_content = '''
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataModel:
    """A dataclass model."""
    name: str
    age: int = 0
    email: Optional[str] = None
    active: bool = True
'''
        file_path = self.repo_root / "models.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        assert len(result["classes"]) == 1

        cls = result["classes"][0]
        assert cls["name"] == "DataModel"
        assert len(cls["attributes"]) >= 4

        # Check attribute details
        name_attr = next(a for a in cls["attributes"] if a["name"] == "name")
        assert name_attr["type"] == "str"
        assert name_attr["has_type_hint"] is True
        assert name_attr["is_instance"] is True

    def test_extract_class_attributes_init(self):
        """Test extracting attributes from __init__ method."""
        python_content = '''
class User:
    """User class with __init__."""

    class_var = "constant"

    def __init__(self, username: str, password: str):
        """Initialize user."""
        self.username = username
        self.password = password
        self.created_at = None
        self._private = "hidden"  # Should be excluded
'''
        file_path = self.repo_root / "user.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        cls = result["classes"][0]

        # Should have attributes from __init__ (excluding private ones)
        public_attrs = [a for a in cls["attributes"] if not a["name"].startswith("_")]
        assert len(public_attrs) >= 3  # username, password, created_at, class_var

        # Verify attribute extraction
        usernames = [a["name"] for a in public_attrs]
        assert "username" in usernames
        assert "password" in usernames
        assert "created_at" in usernames
        assert "_private" not in usernames

    def test_extract_class_with_properties(self):
        """Test extracting class properties."""
        python_content = '''
class PropertyClass:
    """Class with properties."""

    def __init__(self):
        self._value = 0

    @property
    def value(self) -> int:
        """Get the value."""
        return self._value

    @value.setter
    def value(self, val: int):
        """Set the value."""
        self._value = val
'''
        file_path = self.repo_root / "props.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        cls = result["classes"][0]

        # Should detect property
        prop_attrs = [a for a in cls["attributes"] if a.get("is_property")]
        assert len(prop_attrs) >= 1
        assert prop_attrs[0]["name"] == "value"
        assert prop_attrs[0]["type"] == "int"

    def test_extract_function_parameters_comprehensive(self):
        """Test comprehensive parameter extraction."""
        python_content = '''
def complex_function(
    pos_arg: str,
    optional_arg: int = 42,
    *args: tuple,
    keyword_only: bool = False,
    **kwargs: dict
) -> dict:
    """Function with various parameter types."""
    return {}
'''
        file_path = self.repo_root / "params.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        func = result["functions"][0]
        params = func["params"]

        # Verify parameter details
        assert len(params) == 5

        # Positional arg
        assert params[0]["name"] == "pos_arg"
        assert params[0]["type"] == "str"
        assert params[0]["kind"] == "positional"
        assert params[0]["optional"] is False

        # Optional positional arg
        assert params[1]["name"] == "optional_arg"
        assert params[1]["type"] == "int"
        assert params[1]["optional"] is True
        assert params[1]["default"] == "42"

        # *args
        assert params[2]["name"] == "*args"
        assert params[2]["kind"] == "var_positional"

        # Keyword-only arg
        assert params[3]["name"] == "keyword_only"
        assert params[3]["kind"] == "keyword_only"

        # **kwargs
        assert params[4]["name"] == "**kwargs"
        assert params[4]["kind"] == "var_keyword"

    def test_analyze_python_file_syntax_error(self):
        """Test handling of syntax errors in Python files."""
        invalid_python = '''
def broken_function(
    missing closing parenthesis
'''
        file_path = self.repo_root / "broken.py"

        with patch("builtins.open", mock_open(read_data=invalid_python)):
            with pytest.raises(ParsingError) as exc_info:
                self.analyzer.analyze_python_file(
                    file_path, self.repo_root, self.project_modules
                )

        assert "Python parsing failed" in str(exc_info.value)

    def test_analyze_python_file_value_error(self):
        """Test handling of ValueError during parsing."""
        python_content = "# Valid Python but might trigger issues"
        file_path = self.repo_root / "test.py"

        # Mock ast.parse to raise ValueError
        with patch("builtins.open", mock_open(read_data=python_content)):
            with patch("ast.parse", side_effect=ValueError("Parse error")):
                with pytest.raises(ParsingError):
                    self.analyzer.analyze_python_file(
                        file_path, self.repo_root, self.project_modules
                    )

    def test_analyze_python_file_file_not_found(self):
        """Test handling of missing files."""
        file_path = self.repo_root / "nonexistent.py"

        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        # Should return None for unexpected errors
        assert result is None

    def test_is_likely_internal_relative_imports(self):
        """Test detection of relative imports."""
        assert self.analyzer._is_likely_internal(".module", self.project_modules) is True
        assert self.analyzer._is_likely_internal("..parent", self.project_modules) is True
        assert self.analyzer._is_likely_internal("...grandparent", self.project_modules) is True

    def test_is_likely_internal_external_modules(self):
        """Test detection of external modules."""
        # Standard library
        assert self.analyzer._is_likely_internal("os", self.project_modules) is False
        assert self.analyzer._is_likely_internal("sys", self.project_modules) is False
        assert self.analyzer._is_likely_internal("json", self.project_modules) is False

        # Third-party libraries
        assert self.analyzer._is_likely_internal("requests", self.project_modules) is False
        assert self.analyzer._is_likely_internal("django", self.project_modules) is False
        assert self.analyzer._is_likely_internal("pydantic", self.project_modules) is False

    def test_is_likely_internal_project_modules(self):
        """Test detection of project modules."""
        assert self.analyzer._is_likely_internal("myproject", self.project_modules) is True
        assert self.analyzer._is_likely_internal("myproject.utils", self.project_modules) is True
        assert self.analyzer._is_likely_internal("src", self.project_modules) is True
        assert self.analyzer._is_likely_internal("src.services", self.project_modules) is True

    def test_get_importable_module_name_simple(self):
        """Test module name extraction for simple files."""
        file_path = self.repo_root / "module.py"
        relative_path = "module.py"

        module_name = self.analyzer._get_importable_module_name(
            file_path, self.repo_root, relative_path
        )

        assert module_name == "module"

    def test_get_importable_module_name_nested(self):
        """Test module name extraction for nested files."""
        file_path = self.repo_root / "src/myproject/services/api.py"
        relative_path = "src/myproject/services/api.py"

        # Mock __init__.py existence
        with patch.object(Path, "exists", return_value=True):
            module_name = self.analyzer._get_importable_module_name(
                file_path, self.repo_root, relative_path
            )

        # Should skip "src" and start from first package with __init__.py
        assert "myproject" in module_name or "src" in module_name

    def test_infer_type_from_value_constants(self):
        """Test type inference from constant values."""
        # Create AST nodes for different constant types
        bool_node = ast.Constant(value=True)
        assert self.analyzer._infer_type_from_value(bool_node) == "bool"

        int_node = ast.Constant(value=42)
        assert self.analyzer._infer_type_from_value(int_node) == "int"

        float_node = ast.Constant(value=3.14)
        assert self.analyzer._infer_type_from_value(float_node) == "float"

        str_node = ast.Constant(value="hello")
        assert self.analyzer._infer_type_from_value(str_node) == "str"

        none_node = ast.Constant(value=None)
        assert self.analyzer._infer_type_from_value(none_node) == "Optional[Any]"

    def test_infer_type_from_value_collections(self):
        """Test type inference from collection literals."""
        list_node = ast.List(elts=[], ctx=ast.Load())
        assert self.analyzer._infer_type_from_value(list_node) == "List[Any]"

        dict_node = ast.Dict(keys=[], values=[])
        assert self.analyzer._infer_type_from_value(dict_node) == "Dict[Any, Any]"

        set_node = ast.Set(elts=[])
        assert self.analyzer._infer_type_from_value(set_node) == "Set[Any]"

        tuple_node = ast.Tuple(elts=[], ctx=ast.Load())
        assert self.analyzer._infer_type_from_value(tuple_node) == "Tuple[Any, ...]"

    def test_infer_type_from_value_function_calls(self):
        """Test type inference from function calls."""
        # list() call
        list_call = ast.Call(
            func=ast.Name(id="list", ctx=ast.Load()),
            args=[],
            keywords=[]
        )
        assert self.analyzer._infer_type_from_value(list_call) == "list"

        # Path() call
        path_call = ast.Call(
            func=ast.Name(id="Path", ctx=ast.Load()),
            args=[],
            keywords=[]
        )
        assert self.analyzer._infer_type_from_value(path_call) == "pathlib.Path"

    def test_has_dataclass_decorator(self):
        """Test dataclass decorator detection."""
        # Simple @dataclass
        cls_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            body=[],
            decorator_list=[ast.Name(id="dataclass", ctx=ast.Load())]
        )
        assert self.analyzer._has_dataclass_decorator(cls_node) is True

        # No decorator
        cls_node_no_dec = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            body=[],
            decorator_list=[]
        )
        assert self.analyzer._has_dataclass_decorator(cls_node_no_dec) is False

    def test_has_attrs_decorator(self):
        """Test attrs decorator detection."""
        # @attrs decorator
        cls_node = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            body=[],
            decorator_list=[ast.Name(id="attrs", ctx=ast.Load())]
        )
        assert self.analyzer._has_attrs_decorator(cls_node) is True

    def test_extract_slots(self):
        """Test __slots__ extraction."""
        # List of slots
        slots_list = ast.List(
            elts=[
                ast.Constant(value="name"),
                ast.Constant(value="age"),
                ast.Constant(value="email")
            ],
            ctx=ast.Load()
        )
        slots = self.analyzer._extract_slots(slots_list)
        assert len(slots) == 3
        assert "name" in slots
        assert "age" in slots
        assert "email" in slots

        # Single slot
        slots_str = ast.Constant(value="value")
        slots = self.analyzer._extract_slots(slots_str)
        assert len(slots) == 1
        assert "value" in slots

    def test_get_name_simple(self):
        """Test name extraction from simple AST nodes."""
        # Name node
        name_node = ast.Name(id="MyClass", ctx=ast.Load())
        assert self.analyzer._get_name(name_node) == "MyClass"

        # None node
        assert self.analyzer._get_name(None) == "Any"

    def test_get_name_attribute(self):
        """Test name extraction from attribute nodes."""
        # module.Class
        attr_node = ast.Attribute(
            value=ast.Name(id="module", ctx=ast.Load()),
            attr="Class",
            ctx=ast.Load()
        )
        assert self.analyzer._get_name(attr_node) == "module.Class"

    def test_get_name_subscript(self):
        """Test name extraction from subscript nodes (generics)."""
        # List[str]
        subscript_node = ast.Subscript(
            value=ast.Name(id="List", ctx=ast.Load()),
            slice=ast.Name(id="str", ctx=ast.Load()),
            ctx=ast.Load()
        )
        assert self.analyzer._get_name(subscript_node) == "List[str]"

    def test_get_default_value(self):
        """Test default value extraction."""
        # String constant
        str_const = ast.Constant(value="default")
        assert self.analyzer._get_default_value(str_const) == "'default'"

        # Integer constant
        int_const = ast.Constant(value=42)
        assert self.analyzer._get_default_value(int_const) == "42"

        # Name node (like None or True)
        name_node = ast.Name(id="None", ctx=ast.Load())
        assert self.analyzer._get_default_value(name_node) == "None"

        # Empty list
        list_node = ast.List(elts=[], ctx=ast.Load())
        assert self.analyzer._get_default_value(list_node) == "[]"

        # Empty dict
        dict_node = ast.Dict(keys=[], values=[])
        assert self.analyzer._get_default_value(dict_node) == "{}"

    def test_analyze_complex_class_with_all_features(self):
        """Test analyzing a complex class with all features."""
        python_content = '''
from dataclasses import dataclass, field
from typing import List, Optional, ClassVar

@dataclass
class ComplexModel:
    """A complex model with all features."""

    # Class variable
    class_constant: ClassVar[str] = "constant"

    # Instance attributes
    name: str
    age: int = 0
    tags: List[str] = field(default_factory=list)

    # Property
    @property
    def display_name(self) -> str:
        """Get display name."""
        return f"{self.name} ({self.age})"

    # Public method
    def save(self) -> bool:
        """Save the model."""
        return True

    # Private method (should be excluded)
    def _internal_method(self):
        """Internal method."""
        pass
'''
        file_path = self.repo_root / "complex.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        assert len(result["classes"]) == 1

        cls = result["classes"][0]
        assert cls["name"] == "ComplexModel"

        # Should have multiple attributes
        assert len(cls["attributes"]) >= 4

        # Should have public methods only
        method_names = [m["name"] for m in cls["methods"]]
        assert "save" in method_names
        assert "_internal_method" not in method_names

        # Should have property
        props = [a for a in cls["attributes"] if a.get("is_property")]
        assert len(props) >= 1
        assert props[0]["name"] == "display_name"

    def test_empty_file(self):
        """Test analyzing an empty Python file."""
        python_content = ""
        file_path = self.repo_root / "empty.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        assert len(result["classes"]) == 0
        assert len(result["functions"]) == 0
        assert len(result["imports"]) == 0

    def test_file_with_only_comments(self):
        """Test analyzing a file with only comments."""
        python_content = '''
# This is a comment
# Another comment

"""
This is a module docstring
that spans multiple lines.
"""

# More comments
'''
        file_path = self.repo_root / "comments.py"

        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.analyzer.analyze_python_file(
                file_path, self.repo_root, self.project_modules
            )

        assert result is not None
        assert len(result["classes"]) == 0
        assert len(result["functions"]) == 0
