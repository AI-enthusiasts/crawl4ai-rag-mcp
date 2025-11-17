"""
Python code analyzer for Neo4j graph extraction.

Analyzes Python source files using AST to extract:
- Classes with attributes and methods
- Functions with parameters and return types
- Import relationships
"""

import ast
import logging
from pathlib import Path
from typing import Any

from src.core.exceptions import AnalysisError, ParsingError

# Configure logging
logger = logging.getLogger(__name__)


class Neo4jCodeAnalyzer:
    """Analyzes code for direct Neo4j insertion"""

    def __init__(self) -> None:
        # External modules to ignore
        self.external_modules = {
            # Python standard library
            "os", "sys", "json", "logging", "datetime", "pathlib", "typing", "collections",
            "asyncio", "subprocess", "ast", "re", "string", "urllib", "http", "email",
            "time", "uuid", "hashlib", "base64", "itertools", "functools", "operator",
            "contextlib", "copy", "pickle", "tempfile", "shutil", "glob", "fnmatch",
            "io", "codecs", "locale", "platform", "socket", "ssl", "threading", "queue",
            "multiprocessing", "concurrent", "warnings", "traceback", "inspect",
            "importlib", "pkgutil", "types", "weakref", "gc", "dataclasses", "enum",
            "abc", "numbers", "decimal", "fractions", "math", "cmath", "random", "statistics",

            # Common third-party libraries
            "requests", "urllib3", "httpx", "aiohttp", "flask", "django", "fastapi",
            "pydantic", "sqlalchemy", "alembic", "psycopg2", "pymongo", "redis",
            "celery", "pytest", "unittest", "mock", "faker", "factory", "hypothesis",
            "numpy", "pandas", "matplotlib", "seaborn", "scipy", "sklearn", "torch",
            "tensorflow", "keras", "opencv", "pillow", "boto3", "botocore", "azure",
            "google", "openai", "anthropic", "langchain", "transformers", "huggingface_hub",
            "click", "typer", "rich", "colorama", "tqdm", "python-dotenv", "pyyaml",
            "toml", "configargparse", "marshmallow", "attrs", "dataclasses-json",
            "jsonschema", "cerberus", "voluptuous", "schema", "jinja2", "mako",
            "cryptography", "bcrypt", "passlib", "jwt", "authlib", "oauthlib",
        }

    def analyze_python_file(self, file_path: Path, repo_root: Path, project_modules: set[str]) -> dict[str, Any] | None:
        """Extract structure for direct Neo4j insertion"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            relative_path = str(file_path.relative_to(repo_root))
            module_name = self._get_importable_module_name(file_path, repo_root, relative_path)

            # Extract structure
            classes = []
            functions = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract class with its methods and comprehensive attributes
                    methods = []

                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not item.name.startswith("_"):  # Public methods only
                                # Extract comprehensive parameter info
                                params = self._extract_function_parameters(item)

                                # Get return type annotation
                                return_type = self._get_name(item.returns) if item.returns else "Any"

                                # Create detailed parameter list for Neo4j storage
                                params_detailed = []
                                for p in params:
                                    param_str = f"{p['name']}:{p['type']}"
                                    if p["optional"] and p["default"] is not None:
                                        param_str += f"={p['default']}"
                                    elif p["optional"]:
                                        param_str += "=None"
                                    if p["kind"] != "positional":
                                        param_str = f"[{p['kind']}] {param_str}"
                                    params_detailed.append(param_str)

                                methods.append({
                                    "name": item.name,
                                    "params": params,  # Full parameter objects
                                    "params_detailed": params_detailed,  # Detailed string format
                                    "return_type": return_type,
                                    "args": [arg.arg for arg in item.args.args if arg.arg != "self"],  # Keep for backwards compatibility
                                })

                    # Use comprehensive attribute extraction
                    attributes = self._extract_class_attributes(node)

                    classes.append({
                        "name": node.name,
                        "full_name": f"{module_name}.{node.name}",
                        "methods": methods,
                        "attributes": attributes,
                    })

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Only top-level functions
                    if not any(node in cls_node.body for cls_node in ast.walk(tree) if isinstance(cls_node, ast.ClassDef)):
                        if not node.name.startswith("_"):
                            # Extract comprehensive parameter info
                            params = self._extract_function_parameters(node)

                            # Get return type annotation
                            return_type = self._get_name(node.returns) if node.returns else "Any"

                            # Create detailed parameter list for Neo4j storage
                            params_detailed = []
                            for p in params:
                                param_str = f"{p['name']}:{p['type']}"
                                if p["optional"] and p["default"] is not None:
                                    param_str += f"={p['default']}"
                                elif p["optional"]:
                                    param_str += "=None"
                                if p["kind"] != "positional":
                                    param_str = f"[{p['kind']}] {param_str}"
                                params_detailed.append(param_str)

                            # Simple format for backwards compatibility
                            params_list = [f"{p['name']}:{p['type']}" for p in params]

                            functions.append({
                                "name": node.name,
                                "full_name": f"{module_name}.{node.name}",
                                "params": params,  # Full parameter objects
                                "params_detailed": params_detailed,  # Detailed string format
                                "params_list": params_list,  # Simple string format for backwards compatibility
                                "return_type": return_type,
                                "args": [arg.arg for arg in node.args.args],  # Keep for backwards compatibility
                            })

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Track internal imports only
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if self._is_likely_internal(alias.name, project_modules):
                                imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        if (node.module.startswith(".") or self._is_likely_internal(node.module, project_modules)):
                            imports.append(node.module)

            return {
                "module_name": module_name,
                "file_path": relative_path,
                "classes": classes,
                "functions": functions,
                "imports": list(set(imports)),  # Remove duplicates
                "line_count": len(content.splitlines()),
            }

        except (SyntaxError, ValueError) as e:
            logger.error(f"Failed to parse Python file {file_path}: {e}")
            raise ParsingError(f"Python parsing failed for {file_path}: {e}") from e
        except ParsingError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error analyzing {file_path}: {e}")
            return None

    def _extract_class_attributes(self, class_node: ast.ClassDef) -> list[dict[str, Any]]:
        """
        Comprehensively extract all class attributes including:
        - Instance attributes from __init__ methods (self.attr = value)
        - Type annotated attributes in __init__ (self.attr: Type = value)
        - Property decorators (@property def attr)
        - Class-level attributes (both annotated and non-annotated)
        - __slots__ definitions
        - Dataclass and attrs field definitions
        """
        attributes = []
        attribute_stats = {"total": 0, "dataclass": 0, "attrs": 0, "class_vars": 0, "properties": 0, "slots": 0}

        try:
            # Check if class has dataclass or attrs decorators
            is_dataclass = self._has_dataclass_decorator(class_node)
            is_attrs_class = self._has_attrs_decorator(class_node)

            # Extract class-level attributes
            for item in class_node.body:
                try:
                    # Type annotated class attributes
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        if not item.target.id.startswith("_"):
                            # FIXED: Check for ClassVar annotations before assuming dataclass/attrs semantics
                            is_class_var = self._is_class_var_annotation(item.annotation)

                            # Determine attribute classification based on ClassVar and framework
                            if is_class_var:
                                # ClassVar attributes are always class attributes, regardless of framework
                                is_instance_attr = False
                                is_class_attr = True
                                attribute_stats["class_vars"] += 1
                            elif is_dataclass or is_attrs_class:
                                # In dataclass/attrs, non-ClassVar annotations are instance variables
                                is_instance_attr = True
                                is_class_attr = False
                                if is_dataclass:
                                    attribute_stats["dataclass"] += 1
                                if is_attrs_class:
                                    attribute_stats["attrs"] += 1
                            else:
                                # Regular classes: annotations without assignment are typically class-level type hints
                                is_instance_attr = False
                                is_class_attr = True

                            attr_info = {
                                "name": item.target.id,
                                "type": self._get_name(item.annotation) if item.annotation else "Any",
                                "is_instance": is_instance_attr,
                                "is_class": is_class_attr,
                                "is_property": False,
                                "has_type_hint": True,
                                "default_value": self._get_default_value(item.value) if item.value else None,
                                "line_number": item.lineno,
                                "from_dataclass": is_dataclass,
                                "from_attrs": is_attrs_class,
                                "is_class_var": is_class_var,
                            }
                            attributes.append(attr_info)
                            attribute_stats["total"] += 1

                    # Non-annotated class attributes
                    elif isinstance(item, ast.Assign):
                        # Check for __slots__
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                if target.id == "__slots__":
                                    slots = self._extract_slots(item.value)
                                    for slot_name in slots:
                                        if not slot_name.startswith("_"):
                                            attributes.append({
                                                "name": slot_name,
                                                "type": "Any",
                                                "is_instance": True,  # slots are instance attributes
                                                "is_class": False,
                                                "is_property": False,
                                                "has_type_hint": False,
                                                "default_value": None,
                                                "line_number": item.lineno,
                                                "from_slots": True,
                                                "from_dataclass": False,
                                                "from_attrs": False,
                                                "is_class_var": False,
                                            })
                                            attribute_stats["slots"] += 1
                                            attribute_stats["total"] += 1
                                elif not target.id.startswith("_"):
                                    # Regular class attribute
                                    attributes.append({
                                        "name": target.id,
                                        "type": self._infer_type_from_value(item.value) if item.value else "Any",
                                        "is_instance": False,
                                        "is_class": True,
                                        "is_property": False,
                                        "has_type_hint": False,
                                        "default_value": self._get_default_value(item.value) if item.value else None,
                                        "line_number": item.lineno,
                                        "from_dataclass": False,
                                        "from_attrs": False,
                                        "is_class_var": False,
                                    })
                                    attribute_stats["total"] += 1

                    # Properties with @property decorator
                    elif isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                        if any(isinstance(dec, ast.Name) and dec.id == "property"
                               for dec in item.decorator_list):
                            return_type = self._get_name(item.returns) if item.returns else "Any"
                            attributes.append({
                                "name": item.name,
                                "type": return_type,
                                "is_instance": False,  # properties are accessed on instances but defined at class level
                                "is_class": False,
                                "is_property": True,
                                "has_type_hint": item.returns is not None,
                                "default_value": None,
                                "line_number": item.lineno,
                                "from_dataclass": False,
                                "from_attrs": False,
                                "is_class_var": False,
                            })
                            attribute_stats["properties"] += 1
                            attribute_stats["total"] += 1

                except AnalysisError as e:
                    logger.debug(f"Failed to extract attribute from class body: {e}")
                    continue
                except Exception as e:
                    logger.exception(f"Unexpected error extracting attribute: {e}")
                    continue

            # Extract attributes from __init__ method (unless it's a dataclass/attrs class with no __init__)
            init_attributes = self._extract_init_attributes(class_node)
            for init_attr in init_attributes:
                # Ensure init attributes have framework metadata
                init_attr.setdefault("from_dataclass", False)
                init_attr.setdefault("from_attrs", False)
                init_attr.setdefault("is_class_var", False)
            attributes.extend(init_attributes)
            attribute_stats["total"] += len(init_attributes)

            # IMPROVED: Enhanced deduplication logic that respects dataclass semantics
            unique_attributes = {}
            for attr in attributes:
                name = attr["name"]
                if name not in unique_attributes:
                    unique_attributes[name] = attr
                else:
                    existing = unique_attributes[name]
                    should_replace = False

                    # Priority 1: Dataclass/attrs fields take precedence over regular attributes
                    if (attr.get("from_dataclass") or attr.get("from_attrs")) and not (existing.get("from_dataclass") or existing.get("from_attrs")):
                        should_replace = True
                    # Priority 2: Type-hinted attributes over non-hinted (within same framework)
                    elif attr["has_type_hint"] and not existing["has_type_hint"] and not should_replace:
                        # Only if not already prioritizing dataclass/attrs
                        if not ((existing.get("from_dataclass") or existing.get("from_attrs")) and not (attr.get("from_dataclass") or attr.get("from_attrs"))):
                            should_replace = True
                    # Priority 3: Instance attributes over class attributes (within same framework and type hint status)
                    elif (attr["is_instance"] and not existing["is_instance"] and
                          attr["has_type_hint"] == existing["has_type_hint"] and
                          not should_replace):
                        # Only if not already prioritizing by framework or type hints
                        existing_is_framework = existing.get("from_dataclass") or existing.get("from_attrs")
                        attr_is_framework = attr.get("from_dataclass") or attr.get("from_attrs")
                        if existing_is_framework == attr_is_framework:
                            should_replace = True
                    # Priority 4: Properties are always kept (they're unique)
                    elif attr.get("is_property") and not existing.get("is_property"):
                        should_replace = True

                    if should_replace:
                        unique_attributes[name] = attr

            # Log attribute extraction statistics
            final_count = len(unique_attributes)
            if attribute_stats["total"] > 0:
                logger.debug(f"Extracted {final_count} unique attributes from {class_node.name}: "
                           f"dataclass={attribute_stats['dataclass']}, attrs={attribute_stats['attrs']}, "
                           f"class_vars={attribute_stats['class_vars']}, properties={attribute_stats['properties']}, "
                           f"slots={attribute_stats['slots']}, total_processed={attribute_stats['total']}")

            return list(unique_attributes.values())

        except AnalysisError as e:
            logger.error(f"Failed to extract class attributes from {class_node.name}: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error extracting class attributes from {class_node.name}: {e}")
            return []

    def _has_dataclass_decorator(self, class_node: ast.ClassDef) -> bool:
        """Check if class has @dataclass decorator"""
        try:
            for decorator in class_node.decorator_list:
                if isinstance(decorator, ast.Name):
                    if decorator.id in ["dataclass", "dataclasses"]:
                        return True
                elif isinstance(decorator, ast.Attribute):
                    # Handle dataclasses.dataclass
                    attr_name = self._get_name(decorator)
                    if "dataclass" in attr_name.lower():
                        return True
                elif isinstance(decorator, ast.Call):
                    # Handle @dataclass() with parameters
                    func_name = self._get_name(decorator.func)
                    if "dataclass" in func_name.lower():
                        return True
        except AnalysisError as e:
            logger.debug(f"Failed to check dataclass decorator: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error checking dataclass decorator: {e}")
        return False

    def _has_attrs_decorator(self, class_node: ast.ClassDef) -> bool:
        """Check if class has @attrs decorator"""
        try:
            for decorator in class_node.decorator_list:
                if isinstance(decorator, ast.Name):
                    if decorator.id in ["attrs", "attr"]:
                        return True
                elif isinstance(decorator, ast.Attribute):
                    # Handle attr.s, attrs.define, etc.
                    attr_name = self._get_name(decorator)
                    if any(x in attr_name.lower() for x in ["attr.s", "attr.define", "attrs.define", "attrs.frozen"]):
                        return True
                elif isinstance(decorator, ast.Call):
                    # Handle @attr.s() with parameters
                    func_name = self._get_name(decorator.func)
                    if any(x in func_name.lower() for x in ["attr.s", "attr.define", "attrs.define", "attrs.frozen"]):
                        return True
        except AnalysisError as e:
            logger.debug(f"Failed to check attrs decorator: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error checking attrs decorator: {e}")
        return False

    def _is_class_var_annotation(self, annotation_node: Any) -> bool:
        """
        Check if an annotation is a ClassVar type.
        Handles patterns like ClassVar[int], typing.ClassVar[str], etc.
        """
        if annotation_node is None:
            return False

        try:
            annotation_str = self._get_name(annotation_node)
            return "ClassVar" in annotation_str
        except AnalysisError as e:
            logger.debug(f"Failed to check ClassVar annotation: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error checking ClassVar annotation: {e}")
            return False

    def _extract_init_attributes(self, class_node: ast.ClassDef) -> list[dict[str, Any]]:
        """Extract attributes from __init__ method"""
        attributes: list[dict[str, Any]] = []

        # Find __init__ method
        init_method = None
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break

        if not init_method:
            return attributes

        try:
            for node in ast.walk(init_method):
                try:
                    # Handle annotated assignments: self.attr: Type = value
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
                        if (isinstance(node.target.value, ast.Name) and
                            node.target.value.id == "self" and
                            not node.target.attr.startswith("_")):

                            attributes.append({
                                "name": node.target.attr,
                                "type": self._get_name(node.annotation) if node.annotation else "Any",
                                "is_instance": True,
                                "is_class": False,
                                "is_property": False,
                                "has_type_hint": True,
                                "default_value": self._get_default_value(node.value) if node.value else None,
                                "line_number": node.lineno,
                            })

                    # Handle regular assignments: self.attr = value
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Attribute):
                                if (isinstance(target.value, ast.Name) and
                                    target.value.id == "self" and
                                    not target.attr.startswith("_")):

                                    # Try to infer type from assignment value
                                    inferred_type = self._infer_type_from_value(node.value)

                                    attributes.append({
                                        "name": target.attr,
                                        "type": inferred_type,
                                        "is_instance": True,
                                        "is_class": False,
                                        "is_property": False,
                                        "has_type_hint": False,
                                        "default_value": self._get_default_value(node.value),
                                        "line_number": node.lineno,
                                    })

                            # Handle multiple assignments: self.x = self.y = value
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if (isinstance(elt, ast.Attribute) and
                                        isinstance(elt.value, ast.Name) and
                                        elt.value.id == "self" and
                                        not elt.attr.startswith("_")):

                                        inferred_type = self._infer_type_from_value(node.value)
                                        attributes.append({
                                            "name": elt.attr,
                                            "type": inferred_type,
                                            "is_instance": True,
                                            "is_class": False,
                                            "is_property": False,
                                            "has_type_hint": False,
                                            "default_value": self._get_default_value(node.value),
                                            "line_number": node.lineno,
                                        })

                except AnalysisError as e:
                    logger.debug(f"Failed to extract __init__ attribute: {e}")
                    continue
                except Exception as e:
                    logger.exception(f"Unexpected error extracting __init__ attribute: {e}")
                    continue

        except AnalysisError as e:
            logger.debug(f"Failed to walk __init__ method: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error walking __init__ method: {e}")

        return attributes

    def _extract_slots(self, slots_node: Any) -> list[str]:
        """Extract slot names from __slots__ definition"""
        slots: list[str] = []

        try:
            if isinstance(slots_node, (ast.List, ast.Tuple)):
                for elt in slots_node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        slots.append(elt.value)
                    elif isinstance(elt, ast.Str) and isinstance(elt.s, str):  # Python < 3.8 compatibility
                        slots.append(elt.s)
            elif isinstance(slots_node, ast.Constant) and isinstance(slots_node.value, str):
                slots.append(slots_node.value)
            elif isinstance(slots_node, ast.Str) and isinstance(slots_node.s, str):  # Python < 3.8 compatibility
                slots.append(slots_node.s)

        except AnalysisError as e:
            logger.debug(f"Failed to extract slots: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error extracting slots: {e}")

        return slots

    def _infer_type_from_value(self, value_node: Any) -> str:
        """Attempt to infer type from assignment value with enhanced patterns"""
        try:
            if isinstance(value_node, ast.Constant):
                if isinstance(value_node.value, bool):
                    return "bool"
                if isinstance(value_node.value, int):
                    return "int"
                if isinstance(value_node.value, float):
                    return "float"
                if isinstance(value_node.value, str):
                    return "str"
                if isinstance(value_node.value, bytes):
                    return "bytes"
                if value_node.value is None:
                    return "Optional[Any]"
            elif isinstance(value_node, (ast.List, ast.ListComp)):
                return "List[Any]"
            elif isinstance(value_node, (ast.Dict, ast.DictComp)):
                return "Dict[Any, Any]"
            elif isinstance(value_node, (ast.Set, ast.SetComp)):
                return "Set[Any]"
            elif isinstance(value_node, ast.Tuple):
                return "Tuple[Any, ...]"
            elif isinstance(value_node, ast.Call):
                # Try to get type from function call
                func_name = self._get_name(value_node.func)
                if func_name in ["list", "dict", "set", "tuple", "str", "int", "float", "bool"]:
                    return func_name
                if func_name in ["defaultdict", "Counter", "OrderedDict"]:
                    return f"collections.{func_name}"
                if func_name in ["deque"]:
                    return "collections.deque"
                if func_name in ["Path"]:
                    return "pathlib.Path"
                if func_name in ["datetime", "date", "time"]:
                    return f"datetime.{func_name}"
                if func_name in ["UUID"]:
                    return "uuid.UUID"
                if func_name in ["re.compile", "compile"]:
                    return "re.Pattern"
                # Handle dataclass/attrs field calls
                if "field" in func_name.lower():
                    return "Any"  # Field type should be inferred from annotation
                return "Any"  # Unknown function call
            elif isinstance(value_node, ast.Attribute):
                # Handle attribute access like self.other_attr, module.CONSTANT
                attr_name = self._get_name(value_node)
                if "CONSTANT" in attr_name.upper() or attr_name.isupper():
                    return "Any"  # Constants could be anything
                return "Any"
            elif isinstance(value_node, ast.Name):
                # Handle variable references
                if value_node.id in ["True", "False"]:
                    return "bool"
                if value_node.id in ["None"]:
                    return "Optional[Any]"
                return "Any"  # Could be any variable
            elif isinstance(value_node, ast.BinOp):
                # Handle binary operations - try to infer from operands
                return "Any"  # Could be various types depending on operation
        except AnalysisError as e:
            logger.debug(f"Failed to infer type: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in type inference: {e}")

        return "Any"

    def _is_likely_internal(self, import_name: str, project_modules: set[str]) -> bool:
        """Check if an import is likely internal to the project"""
        if not import_name:
            return False

        # Relative imports are definitely internal
        if import_name.startswith("."):
            return True

        # Check if it's a known external module
        base_module = import_name.split(".")[0]
        if base_module in self.external_modules:
            return False

        # Check if it matches any project module
        for project_module in project_modules:
            if import_name.startswith(project_module):
                return True

        # If it's not obviously external, consider it internal
        return bool(not any(ext in base_module.lower() for ext in ["test", "mock", "fake"]) and not base_module.startswith("_") and len(base_module) > 2)

    def _get_importable_module_name(self, file_path: Path, repo_root: Path, relative_path: str) -> str:
        """Determine the actual importable module name for a Python file"""
        # Start with the default: convert file path to module path
        default_module = relative_path.replace("/", ".").replace("\\", ".").replace(".py", "")

        # Common patterns to detect the actual package root
        path_parts = Path(relative_path).parts

        # Look for common package indicators
        package_roots = []

        # Check each directory level for __init__.py to find package boundaries
        current_path = repo_root
        for i, part in enumerate(path_parts[:-1]):  # Exclude the .py file itself
            current_path = current_path / part
            if (current_path / "__init__.py").exists():
                # This is a package directory, mark it as a potential root
                package_roots.append(i)

        if package_roots:
            # Use the first (outermost) package as the root
            package_start = package_roots[0]
            module_parts = path_parts[package_start:]
            return ".".join(module_parts).replace(".py", "")

        # Fallback: look for common Python project structures
        # Skip common non-package directories
        skip_dirs = {"src", "lib", "source", "python", "pkg", "packages"}

        # Find the first directory that's not in skip_dirs
        filtered_parts: list[str] = []
        for part in path_parts:
            if part.lower() not in skip_dirs or filtered_parts:  # Once we start including, include everything
                filtered_parts.append(part)

        if filtered_parts:
            return ".".join(filtered_parts).replace(".py", "")

        # Final fallback: use the default
        return default_module

    def _extract_function_parameters(self, func_node: Any) -> list[dict[str, Any]]:
        """Comprehensive parameter extraction from function definition"""
        params: list[dict[str, Any]] = []

        # Regular positional arguments
        for i, arg in enumerate(func_node.args.args):
            if arg.arg == "self":
                continue

            param_info = {
                "name": arg.arg,
                "type": self._get_name(arg.annotation) if arg.annotation else "Any",
                "kind": "positional",
                "optional": False,
                "default": None,
            }

            # Check if this argument has a default value
            defaults_start = len(func_node.args.args) - len(func_node.args.defaults)
            if i >= defaults_start:
                default_idx = i - defaults_start
                if default_idx < len(func_node.args.defaults):
                    param_info["optional"] = True
                    param_info["default"] = self._get_default_value(func_node.args.defaults[default_idx])

            params.append(param_info)

        # *args parameter
        if func_node.args.vararg:
            params.append({
                "name": f"*{func_node.args.vararg.arg}",
                "type": self._get_name(func_node.args.vararg.annotation) if func_node.args.vararg.annotation else "Any",
                "kind": "var_positional",
                "optional": True,
                "default": None,
            })

        # Keyword-only arguments (after *)
        for i, arg in enumerate(func_node.args.kwonlyargs):
            param_info = {
                "name": arg.arg,
                "type": self._get_name(arg.annotation) if arg.annotation else "Any",
                "kind": "keyword_only",
                "optional": True,  # All kwonly args are optional unless explicitly required
                "default": None,
            }

            # Check for default value
            if i < len(func_node.args.kw_defaults) and func_node.args.kw_defaults[i] is not None:
                param_info["default"] = self._get_default_value(func_node.args.kw_defaults[i])
            else:
                param_info["optional"] = False  # No default = required kwonly arg

            params.append(param_info)

        # **kwargs parameter
        if func_node.args.kwarg:
            params.append({
                "name": f"**{func_node.args.kwarg.arg}",
                "type": self._get_name(func_node.args.kwarg.annotation) if func_node.args.kwarg.annotation else "Dict[str, Any]",
                "kind": "var_keyword",
                "optional": True,
                "default": None,
            })

        return params

    def _get_default_value(self, default_node: Any) -> str:
        """Extract default value from AST node"""
        try:
            if isinstance(default_node, ast.Constant):
                return repr(default_node.value)
            if isinstance(default_node, ast.Name):
                return default_node.id
            if isinstance(default_node, ast.Attribute):
                return self._get_name(default_node)
            if isinstance(default_node, ast.List):
                return "[]"
            if isinstance(default_node, ast.Dict):
                return "{}"
            return "..."
        except AnalysisError:
            return "..."
        except Exception as e:
            logger.exception(f"Unexpected error extracting default value: {e}")
            return "..."

    def _get_name(self, node: Any) -> str:
        """Extract name from AST node, handling complex types safely"""
        if node is None:
            return "Any"

        try:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                if hasattr(node, "value"):
                    return f"{self._get_name(node.value)}.{node.attr}"
                return node.attr
            if isinstance(node, ast.Subscript):
                # Handle List[Type], Dict[K,V], etc.
                base = self._get_name(node.value)
                if hasattr(node, "slice"):
                    if isinstance(node.slice, ast.Name):
                        return f"{base}[{node.slice.id}]"
                    if isinstance(node.slice, ast.Tuple):
                        elts = [self._get_name(elt) for elt in node.slice.elts]
                        return f"{base}[{', '.join(elts)}]"
                    if isinstance(node.slice, ast.Constant):
                        return f"{base}[{node.slice.value!r}]"
                    if isinstance(node.slice, (ast.Attribute, ast.Subscript)):
                        return f"{base}[{self._get_name(node.slice)}]"
                    # Try to get the name of the slice, fallback to Any if it fails
                    try:
                        slice_name = self._get_name(node.slice)
                        return f"{base}[{slice_name}]"
                    except:
                        return f"{base}[Any]"
                return base
            if isinstance(node, ast.Constant):
                return str(node.value)
            if isinstance(node, ast.Str):  # Python < 3.8
                return f'"{node.s!r}"'
            if isinstance(node, ast.Tuple):
                elts = [self._get_name(elt) for elt in node.elts]
                return f"({', '.join(elts)})"
            if isinstance(node, ast.List):
                elts = [self._get_name(elt) for elt in node.elts]
                return f"[{', '.join(elts)}]"
            # Fallback for complex types - return a simple string representation
            return "Any"
        except AnalysisError:
            return "Any"
        except Exception as e:
            logger.exception(f"Unexpected error extracting name from AST node: {e}")
            return "Any"
