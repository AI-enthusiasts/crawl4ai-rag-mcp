"""Validation logic and scoring for code search results.

This module performs Neo4j-based structural validation and generates
confidence scores and suggestions for search results.
"""

import asyncio
import logging
from typing import Any

from src.core.exceptions import DatabaseError
from src.utils.integration_helpers import create_cache_key

from .models import ValidationResult
from .neo4j_client import Neo4jValidationClient

logger = logging.getLogger(__name__)


class CodeValidator:
    """Validates code search results against Neo4j knowledge graph."""

    def __init__(
        self,
        neo4j_client: Neo4jValidationClient,
        performance_optimizer: Any,
        min_confidence: float = 0.6,
        high_confidence: float = 0.8,
    ) -> None:
        """Initialize code validator.

        Args:
            neo4j_client: Neo4j client for structural queries
            performance_optimizer: Performance optimization service
            min_confidence: Minimum confidence threshold
            high_confidence: High confidence threshold
        """
        self.neo4j_client = neo4j_client
        self.performance_optimizer = performance_optimizer
        self.MIN_CONFIDENCE_THRESHOLD = min_confidence
        self.HIGH_CONFIDENCE_THRESHOLD = high_confidence

    async def validate_results_parallel(
        self,
        results: list[dict[str, Any]],
        include_suggestions: bool,
    ) -> list[dict[str, Any]]:
        """Validate search results in parallel for better performance.

        Args:
            results: List of search results to validate
            include_suggestions: Whether to include correction suggestions

        Returns:
            List of validated results with validation metadata
        """
        if not results or not self.neo4j_client.neo4j_enabled:
            # Add empty validation for non-Neo4j mode
            return [self._add_empty_validation(result) for result in results]

        # Create validation tasks
        validation_tasks = [
            self.validate_single_result(result, include_suggestions)
            for result in results
        ]

        # Execute validations in parallel
        try:
            validated_results = await asyncio.gather(
                *validation_tasks, return_exceptions=True,
            )

            # Handle any exceptions in individual validations
            final_results = []
            for i, result in enumerate(validated_results):
                if isinstance(result, Exception):
                    logger.warning("Validation failed for result %s: %s", i, result)
                    # Add the original result with empty validation
                    final_results.append(self._add_empty_validation(results[i]))
                else:
                    final_results.append(result)  # type: ignore[arg-type]

            return final_results

        except DatabaseError as e:
            logger.error("Database error in parallel validation: %s", e)
            return [self._add_empty_validation(result) for result in results]
        except Exception as e:
            logger.exception("Unexpected error in parallel validation: %s", e)
            return [self._add_empty_validation(result) for result in results]

    async def validate_results_sequential(
        self,
        results: list[dict[str, Any]],
        include_suggestions: bool,
    ) -> list[dict[str, Any]]:
        """Validate search results sequentially.

        Args:
            results: List of search results to validate
            include_suggestions: Whether to include correction suggestions

        Returns:
            List of validated results with validation metadata
        """
        validated_results = []

        for result in results:
            try:
                if self.neo4j_client.neo4j_enabled:
                    validated_result = await self.validate_single_result(
                        result, include_suggestions,
                    )
                else:
                    validated_result = self._add_empty_validation(result)
                validated_results.append(validated_result)
            except DatabaseError as e:
                logger.warning("Database error during validation: %s", e)
                validated_results.append(self._add_empty_validation(result))
            except Exception as e:
                logger.warning("Validation failed for single result: %s", e)
                validated_results.append(self._add_empty_validation(result))

        return validated_results

    async def validate_single_result(
        self,
        result: dict[str, Any],
        include_suggestions: bool,
    ) -> dict[str, Any]:
        """Validate a single search result against Neo4j knowledge graph.

        Args:
            result: Search result to validate
            include_suggestions: Whether to include correction suggestions

        Returns:
            Enhanced result with validation metadata
        """
        # Create cache key for this result
        cache_key = create_cache_key(
            "validation", result.get("source_id", ""), result.get("metadata", {}),
        )

        # Check performance cache first
        cached_validation = await self.performance_optimizer.cache.get(cache_key)
        if cached_validation:
            validation = cached_validation
        else:
            # Perform validation
            validation = await self._perform_neo4j_validation(
                result, include_suggestions,
            )
            # Cache the result for 1 hour
            await self.performance_optimizer.cache.set(cache_key, validation, ttl=3600)

        # Add validation to result
        enhanced_result = result.copy()
        enhanced_result["validation"] = validation

        return enhanced_result

    async def _perform_neo4j_validation(
        self,
        result: dict[str, Any],
        include_suggestions: bool,
    ) -> dict[str, Any]:
        """Perform the actual Neo4j validation logic.

        Args:
            result: Search result to validate
            include_suggestions: Whether to include correction suggestions

        Returns:
            Validation result dictionary
        """
        session = await self.neo4j_client.get_session()
        if not session:
            return self._create_empty_validation()

        try:
            # Extract code metadata from result
            metadata = result.get("metadata", {})
            code_type = metadata.get("code_type", "unknown")
            class_name = metadata.get("class_name")
            method_name = metadata.get("method_name") or metadata.get("name")

            ValidationResult()
            validation_checks = []

            # Validation 1: Check if the repository exists
            repo_exists = await self.neo4j_client.check_repository_exists(
                session, result.get("source_id") or "",
            )
            validation_checks.append(
                {
                    "check": "repository_exists",
                    "passed": repo_exists,
                    "weight": 0.3,
                },
            )

            if code_type == "class" and class_name:
                # Validation 2: Check if class exists
                class_exists = await self.neo4j_client.check_class_exists(
                    session, class_name, result.get("source_id") or "",
                )
                validation_checks.append(
                    {
                        "check": "class_exists",
                        "passed": class_exists,
                        "weight": 0.4,
                    },
                )

                # Validation 3: Check class attributes/methods if available
                if class_exists:
                    structure_valid = await self.neo4j_client.validate_class_structure(
                        session,
                        class_name,
                        metadata,
                        result.get("source_id") or "",
                    )
                    validation_checks.append(
                        {
                            "check": "structure_valid",
                            "passed": structure_valid,
                            "weight": 0.3,
                        },
                    )

            elif code_type == "method" and method_name:
                # Validation 2: Check if method exists
                method_exists = await self.neo4j_client.check_method_exists(
                    session,
                    method_name,
                    class_name or "",
                    result.get("source_id") or "",
                )
                validation_checks.append(
                    {
                        "check": "method_exists",
                        "passed": method_exists,
                        "weight": 0.4,
                    },
                )

                # Validation 3: Check method signature if available
                if method_exists:
                    signature_valid = await self.neo4j_client.validate_method_signature(
                        session,
                        method_name,
                        class_name or "",
                        metadata,
                        result.get("source_id") or "",
                    )
                    validation_checks.append(
                        {
                            "check": "signature_valid",
                            "passed": signature_valid,
                            "weight": 0.3,
                        },
                    )

            elif code_type == "function" and method_name:
                # Validation for standalone functions
                function_exists = await self.neo4j_client.check_function_exists(
                    session,
                    method_name,
                    result.get("source_id") or "",
                )
                validation_checks.append(
                    {
                        "check": "function_exists",
                        "passed": function_exists,
                        "weight": 0.7,
                    },
                )

            # Calculate overall confidence score
            confidence_score = self.calculate_confidence_score(validation_checks)

            # Generate suggestions if requested and confidence is low
            suggestions = []
            if (
                include_suggestions
                and confidence_score < self.HIGH_CONFIDENCE_THRESHOLD
            ):
                suggestions = self._generate_suggestions(
                    result,
                    validation_checks,
                )

            return {
                "is_valid": confidence_score >= self.MIN_CONFIDENCE_THRESHOLD,
                "confidence_score": confidence_score,
                "validation_checks": validation_checks,
                "suggestions": suggestions,
                "neo4j_validated": True,
            }

        except DatabaseError as e:
            logger.error("Neo4j validation error: %s", e)
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "validation_checks": [],
                "suggestions": [],
                "neo4j_validated": False,
                "error": str(e),
            }
        except Exception as e:
            logger.exception("Unexpected error in Neo4j validation: %s", e)
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "validation_checks": [],
                "suggestions": [],
                "neo4j_validated": False,
                "error": str(e),
            }
        finally:
            await session.close()

    def _create_empty_validation(self) -> dict[str, Any]:
        """Create an empty validation result when Neo4j is not available.

        Returns:
            Empty validation dictionary
        """
        return {
            "is_valid": True,  # Assume valid when we can't validate
            "confidence_score": 0.5,  # Neutral confidence
            "validation_checks": [],
            "suggestions": [],
            "neo4j_validated": False,
        }

    def _add_empty_validation(self, result: dict[str, Any]) -> dict[str, Any]:
        """Add empty validation to a result.

        Args:
            result: Search result

        Returns:
            Result with empty validation metadata
        """
        enhanced_result = result.copy()
        enhanced_result["validation"] = self._create_empty_validation()
        return enhanced_result

    def _generate_suggestions(
        self,
        result: dict[str, Any],
        validation_checks: list[dict[str, Any]],
    ) -> list[str]:
        """Generate suggestions for improving low-confidence results.

        Args:
            result: Search result
            validation_checks: List of validation check results

        Returns:
            List of suggestion strings
        """
        suggestions = []

        for check in validation_checks:
            if not check["passed"]:
                if check["check"] == "repository_exists":
                    suggestions.append(
                        f"Repository '{result.get('source_id')}' not found in knowledge graph. Consider parsing this repository first.",
                    )
                elif check["check"] == "class_exists":
                    metadata = result.get("metadata", {})
                    class_name = metadata.get("class_name")
                    suggestions.append(
                        f"Class '{class_name}' not found. Check class name spelling or repository parsing completeness.",
                    )
                elif check["check"] == "method_exists":
                    metadata = result.get("metadata", {})
                    method_name = metadata.get("method_name") or metadata.get("name")
                    suggestions.append(
                        f"Method '{method_name}' not found. Verify method name or check if it's inherited.",
                    )
                elif check["check"] == "function_exists":
                    metadata = result.get("metadata", {})
                    function_name = metadata.get("method_name") or metadata.get("name")
                    suggestions.append(
                        f"Function '{function_name}' not found. Check function name or module location.",
                    )

        return suggestions

    @staticmethod
    def calculate_confidence_score(validation_checks: list[dict[str, Any]]) -> float:
        """Calculate weighted confidence score from validation checks.

        Args:
            validation_checks: List of validation check results

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not validation_checks:
            return 0.5  # Neutral when no checks available

        weighted_sum = 0.0
        total_weight = 0.0

        for check in validation_checks:
            weight = check.get("weight", 1.0)
            passed = check.get("passed", False)
            weighted_sum += weight * (1.0 if passed else 0.0)
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def calculate_combined_score(result: dict[str, Any]) -> float:
        """Calculate combined score from semantic similarity and validation confidence.

        Args:
            result: Search result with validation metadata

        Returns:
            Combined score between 0.0 and 1.0
        """
        semantic_score = float(result.get("similarity", 0.0))
        validation = result.get("validation", {})
        confidence_score = float(validation.get("confidence_score", 0.0))

        # Weight semantic similarity and validation confidence
        # Higher weight on validation for more reliable results
        return (semantic_score * 0.4) + (confidence_score * 0.6)
