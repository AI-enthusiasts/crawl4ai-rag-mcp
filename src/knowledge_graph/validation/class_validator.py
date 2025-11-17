"""
Class Instantiation Validator

Validates class instantiations against Neo4j knowledge graph containing
repository class information.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.knowledge_graph.ai_script_analyzer import ClassInstantiation
from src.knowledge_graph.validation import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


__all__ = [
    "ClassValidation",
    "validate_class_instantiations",
    "validate_single_class_instantiation",
]


@dataclass
class ClassValidation:
    """Validation result for class instantiation"""
    class_instantiation: ClassInstantiation
    validation: ValidationResult
    constructor_params: list[str] = field(default_factory=list)
    parameter_validation: ValidationResult | None = None


async def validate_class_instantiations(
    instantiations: list[ClassInstantiation],
    find_class: Callable[..., Any],
    find_method: Callable[..., Any],
    validate_parameters: Callable[..., Any],
    is_from_knowledge_graph: Callable[..., Any],
    knowledge_graph_modules: set[str],
) -> list[ClassValidation]:
    """
    Validate class instantiations against knowledge graph.

    Args:
        instantiations: List of class instantiations to validate
        find_class: Async function to find class in knowledge graph
        find_method: Async function to find method on a class
        validate_parameters: Function to validate function/method parameters
        is_from_knowledge_graph: Function to check if class is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        List of class validation results
    """
    validations = []

    for instantiation in instantiations:
        validation = await validate_single_class_instantiation(
            instantiation,
            find_class,
            find_method,
            validate_parameters,
            is_from_knowledge_graph,
            knowledge_graph_modules,
        )
        validations.append(validation)

    return validations


async def validate_single_class_instantiation(
    instantiation: ClassInstantiation,
    find_class: Callable[..., Any],
    find_method: Callable[..., Any],
    validate_parameters: Callable[..., Any],
    is_from_knowledge_graph: Callable[..., Any],
    knowledge_graph_modules: set[str],
) -> ClassValidation:
    """
    Validate a single class instantiation.

    Args:
        instantiation: The class instantiation to validate
        find_class: Async function to find class in knowledge graph
        find_method: Async function to find method on a class
        validate_parameters: Function to validate function/method parameters
        is_from_knowledge_graph: Function to check if class is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        Validation result for the class instantiation
    """
    class_name = instantiation.full_class_name or instantiation.class_name

    # Skip validation for classes not from knowledge graph
    if not is_from_knowledge_graph(class_name, knowledge_graph_modules):
        validation = ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.8,
            message=f"Skipping validation: '{class_name}' is not from knowledge graph",
        )
        return ClassValidation(
            class_instantiation=instantiation,
            validation=validation,
        )

    # Find class in knowledge graph
    class_info = await find_class(class_name)

    if not class_info:
        validation = ValidationResult(
            status=ValidationStatus.NOT_FOUND,
            confidence=0.2,
            message=f"Class '{class_name}' not found in knowledge graph",
        )
        return ClassValidation(
            class_instantiation=instantiation,
            validation=validation,
        )

    # Check constructor parameters (look for __init__ method)
    init_method = await find_method(class_name, "__init__")

    if init_method:
        param_validation = validate_parameters(
            expected_params=init_method.get("params_list", []),
            provided_args=instantiation.args,
            provided_kwargs=instantiation.kwargs,
        )
    else:
        param_validation = ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.5,
            message="Constructor parameters not found",
        )

    # Use parameter validation result if it failed
    if param_validation.status == ValidationStatus.INVALID:
        validation = ValidationResult(
            status=ValidationStatus.INVALID,
            confidence=param_validation.confidence,
            message=f"Class '{class_name}' found but has invalid constructor parameters: {param_validation.message}",
            suggestions=param_validation.suggestions,
        )
    else:
        validation = ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.8,
            message=f"Class '{class_name}' found in knowledge graph",
        )

    return ClassValidation(
        class_instantiation=instantiation,
        validation=validation,
        parameter_validation=param_validation,
    )
