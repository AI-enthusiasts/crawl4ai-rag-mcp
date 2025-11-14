"""
Import Validator

Validates import statements against Neo4j knowledge graph containing
repository information.
"""

import logging
from dataclasses import dataclass, field

from ..ai_script_analyzer import ImportInfo
from . import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


__all__ = ["ImportValidation", "validate_imports", "validate_single_import"]


@dataclass
class ImportValidation:
    """Validation result for an import"""
    import_info: ImportInfo
    validation: ValidationResult
    available_classes: list[str] = field(default_factory=list)
    available_functions: list[str] = field(default_factory=list)


async def validate_imports(
    imports: list[ImportInfo],
    find_modules,
    get_module_contents,
    module_cache: dict[str, list[str]],
    knowledge_graph_modules: set[str],
) -> list[ImportValidation]:
    """
    Validate all imports against knowledge graph.

    Args:
        imports: List of import statements to validate
        find_modules: Async function to find modules in knowledge graph
        get_module_contents: Async function to get module classes and functions
        module_cache: Cache for module lookups
        knowledge_graph_modules: Set of modules found in knowledge graph

    Returns:
        List of import validation results
    """
    validations = []

    for import_info in imports:
        validation = await validate_single_import(
            import_info,
            find_modules,
            get_module_contents,
            module_cache,
            knowledge_graph_modules,
        )
        validations.append(validation)

    return validations


async def validate_single_import(
    import_info: ImportInfo,
    find_modules,
    get_module_contents,
    module_cache: dict[str, list[str]],
    knowledge_graph_modules: set[str],
) -> ImportValidation:
    """
    Validate a single import statement.

    Args:
        import_info: The import to validate
        find_modules: Async function to find modules in knowledge graph
        get_module_contents: Async function to get module classes and functions
        module_cache: Cache for module lookups
        knowledge_graph_modules: Set of modules found in knowledge graph

    Returns:
        Validation result for the import
    """
    # Determine module to search for
    search_module = import_info.module if import_info.is_from_import else import_info.name

    # Check cache first
    if search_module in module_cache:
        available_files = module_cache[search_module]
    else:
        # Query Neo4j for matching modules
        available_files = await find_modules(search_module)
        module_cache[search_module] = available_files

    if available_files:
        # Get available classes and functions from the module
        classes, functions = await get_module_contents(search_module)

        # Track this module as being in the knowledge graph
        knowledge_graph_modules.add(search_module)

        # Also track the base module for "from X.Y.Z import ..." patterns
        if "." in search_module:
            base_module = search_module.split(".")[0]
            knowledge_graph_modules.add(base_module)

        validation = ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.9,
            message=f"Module '{search_module}' found in knowledge graph",
            details={"matched_files": available_files, "in_knowledge_graph": True},
        )

        return ImportValidation(
            import_info=import_info,
            validation=validation,
            available_classes=classes,
            available_functions=functions,
        )

    # External library - mark as such but don't treat as error
    validation = ValidationResult(
        status=ValidationStatus.UNCERTAIN,
        confidence=0.8,  # High confidence it's external, not an error
        message=f"Module '{search_module}' is external (not in knowledge graph)",
        details={"could_be_external": True, "in_knowledge_graph": False},
    )

    return ImportValidation(
        import_info=import_info,
        validation=validation,
    )
