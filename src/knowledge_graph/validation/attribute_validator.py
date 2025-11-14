"""
Attribute Access Validator

Validates attribute accesses against Neo4j knowledge graph containing
repository attribute information.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..ai_script_analyzer import AttributeAccess
from . import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


__all__ = [
    "AttributeValidation",
    "validate_attribute_accesses",
    "validate_single_attribute_access",
]


@dataclass
class AttributeValidation:
    """Validation result for attribute access"""
    attribute_access: AttributeAccess
    validation: ValidationResult
    expected_type: str | None = None


async def validate_attribute_accesses(
    attribute_accesses: list[AttributeAccess],
    find_attribute: Callable[..., Any],
    find_method: Callable[..., Any],
    is_from_knowledge_graph: Callable[..., Any],
    knowledge_graph_modules: set[str],
) -> list[AttributeValidation]:
    """
    Validate attribute accesses against knowledge graph.

    Args:
        attribute_accesses: List of attribute accesses to validate
        find_attribute: Async function to find attribute on a class
        find_method: Async function to find method on a class
        is_from_knowledge_graph: Function to check if class is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        List of attribute validation results
    """
    validations = []

    for attr_access in attribute_accesses:
        validation = await validate_single_attribute_access(
            attr_access,
            find_attribute,
            find_method,
            is_from_knowledge_graph,
            knowledge_graph_modules,
        )
        validations.append(validation)

    return validations


async def validate_single_attribute_access(
    attr_access: AttributeAccess,
    find_attribute: Callable[..., Any],
    find_method: Callable[..., Any],
    is_from_knowledge_graph: Callable[..., Any],
    knowledge_graph_modules: set[str],
) -> AttributeValidation:
    """
    Validate a single attribute access.

    Args:
        attr_access: The attribute access to validate
        find_attribute: Async function to find attribute on a class
        find_method: Async function to find method on a class
        is_from_knowledge_graph: Function to check if class is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        Validation result for the attribute access
    """
    class_type = attr_access.object_type

    if not class_type:
        validation = ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.3,
            message=f"Cannot determine object type for '{attr_access.object_name}'",
        )
        return AttributeValidation(
            attribute_access=attr_access,
            validation=validation,
        )

    # Skip validation for classes not from knowledge graph
    if not is_from_knowledge_graph(class_type, knowledge_graph_modules):
        validation = ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.8,
            message=f"Skipping validation: '{class_type}' is not from knowledge graph",
        )
        return AttributeValidation(
            attribute_access=attr_access,
            validation=validation,
        )

    # Find attribute in knowledge graph
    attr_info = await find_attribute(class_type, attr_access.attribute_name)

    if not attr_info:
        # If not found as attribute, check if it's a method (for decorators like @agent.tool)
        method_info = await find_method(class_type, attr_access.attribute_name)

        if method_info:
            validation = ValidationResult(
                status=ValidationStatus.VALID,
                confidence=0.8,
                message=f"'{attr_access.attribute_name}' found as method on class '{class_type}' (likely used as decorator)",
            )
            return AttributeValidation(
                attribute_access=attr_access,
                validation=validation,
                expected_type="method",
            )

        validation = ValidationResult(
            status=ValidationStatus.NOT_FOUND,
            confidence=0.2,
            message=f"'{attr_access.attribute_name}' not found on class '{class_type}'",
        )
        return AttributeValidation(
            attribute_access=attr_access,
            validation=validation,
        )

    validation = ValidationResult(
        status=ValidationStatus.VALID,
        confidence=0.8,
        message=f"Attribute '{attr_access.attribute_name}' found on class '{class_type}'",
    )

    return AttributeValidation(
        attribute_access=attr_access,
        validation=validation,
        expected_type=attr_info.get("type"),
    )
