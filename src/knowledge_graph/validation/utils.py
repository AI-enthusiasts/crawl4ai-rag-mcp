"""
Validation utility functions for knowledge graph validator.

Provides standalone utility functions for validating parameters,
calculating confidence scores, detecting hallucinations, and checking
knowledge graph membership.
"""

from dataclasses import dataclass, field
from typing import Any

from . import ValidationResult, ValidationStatus


@dataclass
class ScriptValidationResult:
    """Complete validation results for a script."""
    script_path: str
    analysis_result: Any
    import_validations: list[Any] = field(default_factory=list)
    class_validations: list[Any] = field(default_factory=list)
    method_validations: list[Any] = field(default_factory=list)
    attribute_validations: list[Any] = field(default_factory=list)
    function_validations: list[Any] = field(default_factory=list)
    overall_confidence: float = 0.0
    hallucinations_detected: list[dict[str, Any]] = field(default_factory=list)


def validate_parameters(
    expected_params: list[str],
    provided_args: list[str],
    provided_kwargs: dict[str, str],
) -> ValidationResult:
    """
    Validate function/method parameters with comprehensive support.

    Validates positional and keyword arguments against expected parameters,
    supporting varargs, varkwargs, and keyword-only parameters.

    Args:
        expected_params: List of expected parameter definitions.
            Format: "name:type" or "[keyword_only] name:type=default"
        provided_args: List of provided positional arguments.
        provided_kwargs: Dictionary of provided keyword arguments.

    Returns:
        ValidationResult indicating if parameters are valid, with confidence
        score and detailed message.
    """
    if not expected_params:
        return ValidationResult(
            status=ValidationStatus.UNCERTAIN,
            confidence=0.5,
            message="Parameter information not available",
        )

    # Parse expected parameters - handle detailed format
    required_positional = []
    optional_positional = []
    keyword_only_required = []
    keyword_only_optional = []
    has_varargs = False
    has_varkwargs = False

    for param in expected_params:
        # Handle detailed format: "[keyword_only] name:type=default" or "name:type"
        param_clean = param.strip()

        # Check for parameter kind prefix
        kind = "positional"
        if param_clean.startswith("["):
            end_bracket = param_clean.find("]")
            if end_bracket > 0:
                kind = param_clean[1:end_bracket]
                param_clean = param_clean[end_bracket+1:].strip()

        # Check for varargs/varkwargs
        if param_clean.startswith("*") and not param_clean.startswith("**"):
            has_varargs = True
            continue
        if param_clean.startswith("**"):
            has_varkwargs = True
            continue

        # Parse name and check if optional
        if ":" in param_clean:
            param_name = param_clean.split(":")[0]
            is_optional = "=" in param_clean

            if kind == "keyword_only":
                if is_optional:
                    keyword_only_optional.append(param_name)
                else:
                    keyword_only_required.append(param_name)
            elif is_optional:
                optional_positional.append(param_name)
            else:
                required_positional.append(param_name)

    # Count provided parameters
    provided_positional_count = len(provided_args)
    provided_keyword_names = set(provided_kwargs.keys())

    # Validate positional arguments
    min_required_positional = len(required_positional)
    max_allowed_positional = len(required_positional) + len(optional_positional)

    if not has_varargs and provided_positional_count > max_allowed_positional:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            confidence=0.8,
            message=f"Too many positional arguments: provided {provided_positional_count}, max allowed {max_allowed_positional}",
        )

    if provided_positional_count < min_required_positional:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            confidence=0.8,
            message=f"Too few positional arguments: provided {provided_positional_count}, required {min_required_positional}",
        )

    # Validate keyword arguments
    all_valid_kwarg_names = set(required_positional + optional_positional + keyword_only_required + keyword_only_optional)
    invalid_kwargs = provided_keyword_names - all_valid_kwarg_names

    if invalid_kwargs and not has_varkwargs:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            confidence=0.7,
            message=f"Invalid keyword arguments: {list(invalid_kwargs)}",
            suggestions=[f"Valid parameters: {list(all_valid_kwarg_names)}"],
        )

    # Check required keyword-only arguments
    missing_required_kwargs = set(keyword_only_required) - provided_keyword_names
    if missing_required_kwargs:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            confidence=0.8,
            message=f"Missing required keyword arguments: {list(missing_required_kwargs)}",
        )

    return ValidationResult(
        status=ValidationStatus.VALID,
        confidence=0.9,
        message="Parameters are valid",
    )


def calculate_overall_confidence(
    result: ScriptValidationResult,
    knowledge_graph_modules: set[str],
) -> float:
    """
    Calculate overall confidence score for the validation (knowledge graph items only).

    Computes the average confidence of validations for items originating from
    the knowledge graph. Returns perfect confidence (1.0) if no knowledge graph
    items were validated.

    Args:
        result: ScriptValidationResult containing all validation results.
        knowledge_graph_modules: Set of module names in the knowledge graph.

    Returns:
        Float between 0.0 and 1.0 representing overall confidence score.
    """
    kg_validations = []

    # Only count validations from knowledge graph imports
    for val in result.import_validations:
        if val.validation.details.get("in_knowledge_graph", False):
            kg_validations.append(val.validation.confidence)

    # Only count validations from knowledge graph classes
    for val in result.class_validations:
        class_name = val.class_instantiation.full_class_name or val.class_instantiation.class_name
        if is_from_knowledge_graph(class_name, knowledge_graph_modules):
            kg_validations.append(val.validation.confidence)

    # Only count validations from knowledge graph methods
    for val in result.method_validations:
        if val.method_call.object_type and is_from_knowledge_graph(val.method_call.object_type, knowledge_graph_modules):
            kg_validations.append(val.validation.confidence)

    # Only count validations from knowledge graph attributes
    for val in result.attribute_validations:
        if val.attribute_access.object_type and is_from_knowledge_graph(val.attribute_access.object_type, knowledge_graph_modules):
            kg_validations.append(val.validation.confidence)

    # Only count validations from knowledge graph functions
    for val in result.function_validations:
        if val.function_call.full_name and is_from_knowledge_graph(val.function_call.full_name, knowledge_graph_modules):
            kg_validations.append(val.validation.confidence)

    if not kg_validations:
        return 1.0  # No knowledge graph items to validate = perfect confidence

    return sum(kg_validations) / len(kg_validations)


def is_from_knowledge_graph(
    class_type: str,
    knowledge_graph_modules: set[str],
) -> bool:
    """
    Check if a class type comes from a module in the knowledge graph.

    For dotted names like "pydantic_ai.Agent", extracts and checks the base
    module. For simple names, checks for exact matches.

    Args:
        class_type: The class name or dotted class path to check.
        knowledge_graph_modules: Set of module names in the knowledge graph.

    Returns:
        True if the class type is from a knowledge graph module, False otherwise.
    """
    if not class_type:
        return False

    # For dotted names like "pydantic_ai.Agent" or "pydantic_ai.StreamedRunResult", check the base module
    if "." in class_type:
        base_module = class_type.split(".")[0]
        # Exact match only - "pydantic" should not match "pydantic_ai"
        return base_module in knowledge_graph_modules

    # For simple names, check if any knowledge graph module matches exactly
    # Don't use substring matching to avoid "pydantic" matching "pydantic_ai"
    return class_type in knowledge_graph_modules


def detect_hallucinations(
    result: ScriptValidationResult,
    knowledge_graph_modules: set[str],
) -> list[dict[str, Any]]:
    """
    Detect and categorize hallucinations in the validation result.

    Identifies invalid methods, attributes, and parameters that only exist
    on classes from the knowledge graph to avoid false positives for external
    classes.

    Args:
        result: ScriptValidationResult containing all validation results.
        knowledge_graph_modules: Set of module names in the knowledge graph.

    Returns:
        List of dictionaries describing detected hallucinations, each containing:
            - type: Hallucination type (METHOD_NOT_FOUND, ATTRIBUTE_NOT_FOUND, INVALID_PARAMETERS)
            - location: Line number where the hallucination was found
            - description: Human-readable description of the issue
            - suggestion: Optional suggested correction
    """
    hallucinations = []
    reported_items = set()  # Track reported items to avoid duplicates

    # Check method calls (only for knowledge graph classes)
    for val in result.method_validations:
        if (val.validation.status == ValidationStatus.NOT_FOUND and
            val.method_call.object_type and
            is_from_knowledge_graph(val.method_call.object_type, knowledge_graph_modules)):

            # Create unique key to avoid duplicates
            key = (val.method_call.line_number, val.method_call.method_name, val.method_call.object_type)
            if key not in reported_items:
                reported_items.add(key)
                hallucinations.append({
                    "type": "METHOD_NOT_FOUND",
                    "location": f"line {val.method_call.line_number}",
                    "description": f"Method '{val.method_call.method_name}' not found on class '{val.method_call.object_type}'",
                    "suggestion": val.validation.suggestions[0] if val.validation.suggestions else None,
                })

    # Check attributes (only for knowledge graph classes) - but skip if already reported as method
    for val in result.attribute_validations:
        if (val.validation.status == ValidationStatus.NOT_FOUND and
            val.attribute_access.object_type and
            is_from_knowledge_graph(val.attribute_access.object_type, knowledge_graph_modules)):

            # Create unique key - if this was already reported as a method, skip it
            key = (val.attribute_access.line_number, val.attribute_access.attribute_name, val.attribute_access.object_type)
            if key not in reported_items:
                reported_items.add(key)
                hallucinations.append({
                    "type": "ATTRIBUTE_NOT_FOUND",
                    "location": f"line {val.attribute_access.line_number}",
                    "description": f"Attribute '{val.attribute_access.attribute_name}' not found on class '{val.attribute_access.object_type}'",
                })

    # Check parameter issues (only for knowledge graph methods)
    for val in result.method_validations:
        if (val.parameter_validation and
            val.parameter_validation.status == ValidationStatus.INVALID and
            val.method_call.object_type and
            is_from_knowledge_graph(val.method_call.object_type, knowledge_graph_modules)):
            hallucinations.append({
                "type": "INVALID_PARAMETERS",
                "location": f"line {val.method_call.line_number}",
                "description": f"Invalid parameters for method '{val.method_call.method_name}': {val.parameter_validation.message}",
            })

    return hallucinations
