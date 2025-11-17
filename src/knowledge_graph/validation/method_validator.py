"""
Method Call Validator

Validates method calls against Neo4j knowledge graph containing
repository method information.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.knowledge_graph.ai_script_analyzer import MethodCall
from src.knowledge_graph.validation import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


__all__ = [
    "MethodValidation",
    "validate_method_calls",
    "validate_single_method_call",
]


@dataclass
class MethodValidation:
    """Validation result for a method call"""
    method_call: MethodCall
    validation: ValidationResult
    expected_params: list[str] = field(default_factory=list)
    actual_params: list[str] = field(default_factory=list)
    parameter_validation: ValidationResult | None = None


async def validate_method_calls(
    method_calls: list[MethodCall],
    find_method: Callable[..., Any],
    find_similar_methods: Callable[..., Any],
    validate_parameters: Callable[..., Any],
    is_from_knowledge_graph: Callable[..., Any],
    knowledge_graph_modules: set[str],
) -> list[MethodValidation]:
    """
    Validate method calls against knowledge graph.

    Args:
        method_calls: List of method calls to validate
        find_method: Async function to find method on a class
        find_similar_methods: Async function to find similar method names
        validate_parameters: Function to validate function/method parameters
        is_from_knowledge_graph: Function to check if class is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        List of method validation results
    """
    validations = []

    for method_call in method_calls:
        validation = await validate_single_method_call(
            method_call,
            find_method,
            find_similar_methods,
            validate_parameters,
            is_from_knowledge_graph,
            knowledge_graph_modules,
        )
        validations.append(validation)

    return validations


async def validate_single_method_call(
    method_call: MethodCall,
    find_method: Callable[..., Any],
    find_similar_methods: Callable[..., Any],
    validate_parameters: Callable[..., Any],
    is_from_knowledge_graph: Callable[..., Any],
    knowledge_graph_modules: set[str],
) -> MethodValidation:
    """
    Validate a single method call.

    Args:
        method_call: The method call to validate
        find_method: Async function to find method on a class
        find_similar_methods: Async function to find similar method names
        validate_parameters: Function to validate function/method parameters
        is_from_knowledge_graph: Function to check if class is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        Validation result for the method call
    """
    class_type = method_call.object_type

    if not class_type:
        validation = ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.3,
            message=f"Cannot determine object type for '{method_call.object_name}'",
        )
        return MethodValidation(
            method_call=method_call,
            validation=validation,
        )

    # Skip validation for classes not from knowledge graph
    if not is_from_knowledge_graph(class_type, knowledge_graph_modules):
        validation = ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.8,
            message=f"Skipping validation: '{class_type}' is not from knowledge graph",
        )
        return MethodValidation(
            method_call=method_call,
            validation=validation,
        )

    # Find method in knowledge graph
    method_info = await find_method(class_type, method_call.method_name)

    if not method_info:
        # Check for similar method names
        similar_methods = await find_similar_methods(class_type, method_call.method_name)

        validation = ValidationResult(
            status=ValidationStatus.NOT_FOUND,
            confidence=0.1,
            message=f"Method '{method_call.method_name}' not found on class '{class_type}'",
            suggestions=similar_methods,
        )
        return MethodValidation(
            method_call=method_call,
            validation=validation,
        )

    # Validate parameters
    expected_params = method_info.get("params_list", [])
    param_validation = validate_parameters(
        expected_params=expected_params,
        provided_args=method_call.args,
        provided_kwargs=method_call.kwargs,
    )

    # Use parameter validation result if it failed
    if param_validation.status == ValidationStatus.INVALID:
        validation = ValidationResult(
            status=ValidationStatus.INVALID,
            confidence=param_validation.confidence,
            message=f"Method '{method_call.method_name}' found but has invalid parameters: {param_validation.message}",
            suggestions=param_validation.suggestions,
        )
    else:
        validation = ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.9,
            message=f"Method '{method_call.method_name}' found on class '{class_type}'",
        )

    return MethodValidation(
        method_call=method_call,
        validation=validation,
        expected_params=expected_params,
        actual_params=method_call.args + list(method_call.kwargs.keys()),
        parameter_validation=param_validation,
    )
