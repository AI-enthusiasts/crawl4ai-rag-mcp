"""Data models for validated code search.

This module defines the core data structures used in validated search operations.
"""

from typing import Any


class ValidationResult:
    """Container for validation results with confidence scoring."""

    def __init__(self) -> None:
        self.is_valid = False
        self.confidence_score = 0.0
        self.validation_details: dict[str, Any] = {}
        self.suggestions: list[str] = []
        self.metadata: dict[str, Any] = {}
