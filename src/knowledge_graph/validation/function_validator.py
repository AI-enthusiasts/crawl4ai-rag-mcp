"""
Function Call Validator

Validates function calls against Neo4j knowledge graph containing
repository function information.
"""

import logging
from dataclasses import dataclass, field

from ..ai_script_analyzer import FunctionCall
from . import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


__all__ = [
    "FunctionValidation",
    "validate_function_calls",
    "validate_single_function_call",
]


@dataclass
class FunctionValidation:
    """Validation result for function call"""
    function_call: FunctionCall
    validation: ValidationResult
    expected_params: list[str] = field(default_factory=list)
    actual_params: list[str] = field(default_factory=list)
    parameter_validation: ValidationResult = None


async def validate_function_calls(
    function_calls: list[FunctionCall],
    find_function,
    validate_parameters,
    is_from_knowledge_graph,
    knowledge_graph_modules: set[str],
) -> list[FunctionValidation]:
    """
    Validate function calls against knowledge graph.

    Args:
        function_calls: List of function calls to validate
        find_function: Async function to find function in knowledge graph
        validate_parameters: Function to validate function/method parameters
        is_from_knowledge_graph: Function to check if function is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        List of function validation results
    """
    validations = []

    for func_call in function_calls:
        validation = await validate_single_function_call(
            func_call,
            find_function,
            validate_parameters,
            is_from_knowledge_graph,
            knowledge_graph_modules,
        )
        validations.append(validation)

    return validations


async def validate_single_function_call(
    func_call: FunctionCall,
    find_function,
    validate_parameters,
    is_from_knowledge_graph,
    knowledge_graph_modules: set[str],
) -> FunctionValidation:
    """
    Validate a single function call.

    Args:
        func_call: The function call to validate
        find_function: Async function to find function in knowledge graph
        validate_parameters: Function to validate function/method parameters
        is_from_knowledge_graph: Function to check if function is from knowledge graph
        knowledge_graph_modules: Set of modules in knowledge graph

    Returns:
        Validation result for the function call
    """
    func_name = func_call.full_name or func_call.function_name

    # Skip validation for functions not from knowledge graph
    if func_call.full_name and not is_from_knowledge_graph(func_call.full_name, knowledge_graph_modules):
        validation = ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.8,
            message=f"Skipping validation: '{func_name}' is not from knowledge graph",
        )
        return FunctionValidation(
            function_call=func_call,
            validation=validation,
        )

    # Find function in knowledge graph
    func_info = await find_function(func_name)

    if not func_info:
        validation = ValidationResult(
            status=ValidationStatus.NOT_FOUND,
            confidence=0.2,
            message=f"Function '{func_name}' not found in knowledge graph",
        )
        return FunctionValidation(
            function_call=func_call,
            validation=validation,
        )

    # Validate parameters
    expected_params = func_info.get("params_list", [])
    param_validation = validate_parameters(
        expected_params=expected_params,
        provided_args=func_call.args,
        provided_kwargs=func_call.kwargs,
    )

    # Use parameter validation result if it failed
    if param_validation.status == ValidationStatus.INVALID:
        validation = ValidationResult(
            status=ValidationStatus.INVALID,
            confidence=param_validation.confidence,
            message=f"Function '{func_name}' found but has invalid parameters: {param_validation.message}",
            suggestions=param_validation.suggestions,
        )
    else:
        validation = ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.8,
            message=f"Function '{func_name}' found in knowledge graph",
        )

    return FunctionValidation(
        function_call=func_call,
        validation=validation,
        expected_params=expected_params,
        actual_params=func_call.args + list(func_call.kwargs.keys()),
        parameter_validation=param_validation,
    )
