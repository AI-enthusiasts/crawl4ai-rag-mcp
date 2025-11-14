"""
Validation module for knowledge graph operations.

Provides standalone query functions and validation utilities for the Neo4j knowledge graph.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .neo4j_queries import (
    find_attribute,
    find_class,
    find_function,
    find_method,
    find_modules,
    find_pydantic_ai_result_method,
    find_similar_methods,
    find_similar_modules,
    find_repository_for_module,
    get_module_contents,
)
from .utils import (
    ScriptValidationResult,
    calculate_overall_confidence,
    detect_hallucinations,
    is_from_knowledge_graph,
    validate_parameters,
)


class ValidationStatus(Enum):
    """Status of a validation result"""
    VALID = "VALID"
    INVALID = "INVALID"
    UNCERTAIN = "UNCERTAIN"
    NOT_FOUND = "NOT_FOUND"


@dataclass
class ValidationResult:
    """Result of validating a single element"""
    status: ValidationStatus
    confidence: float  # 0.0 to 1.0
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


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
