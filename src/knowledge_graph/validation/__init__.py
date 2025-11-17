"""
Validation module for knowledge graph operations.

Provides standalone query functions and validation utilities for the Neo4j knowledge graph.
"""

from .neo4j_queries import (
    find_attribute,
    find_class,
    find_function,
    find_method,
    find_modules,
    find_pydantic_ai_result_method,
    find_repository_for_module,
    find_similar_methods,
    find_similar_modules,
    get_module_contents,
)
from .utils import (
    ScriptValidationResult,
    ValidationResult,
    ValidationStatus,
    calculate_overall_confidence,
    detect_hallucinations,
    is_from_knowledge_graph,
    validate_parameters,
)

__all__ = [
    # Query functions
    "find_modules",
    "get_module_contents",
    "find_repository_for_module",
    "find_class",
    "find_method",
    "find_attribute",
    "find_function",
    "find_pydantic_ai_result_method",
    "find_similar_modules",
    "find_similar_methods",
    # Validation classes
    "ValidationStatus",
    "ValidationResult",
    "ScriptValidationResult",
    # Validation utility functions
    "calculate_overall_confidence",
    "detect_hallucinations",
    "is_from_knowledge_graph",
    "validate_parameters",
]
