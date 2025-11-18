"""Main validated code search service.

This module provides the high-level service that combines Qdrant semantic search
with Neo4j structural validation for high-confidence code search results.
"""

import logging
from typing import Any

from src.core.exceptions import DatabaseError, SearchError
from src.utils import create_embeddings_batch
from src.utils.integration_helpers import (
    get_performance_optimizer,
    performance_monitor,
    validate_integration_health,
)

from .neo4j_client import Neo4jValidationClient
from .validator import CodeValidator

logger = logging.getLogger(__name__)


class ValidatedCodeSearchService:
    """Service that combines Qdrant semantic search with Neo4j structural validation."""

    def __init__(self, database_client: Any, neo4j_driver: Any = None):
        """Initialize the validated search service.

        Args:
            database_client: Qdrant database client for semantic search
            neo4j_driver: Neo4j driver for structural validation (optional)
        """
        self.database_client = database_client

        # Performance optimization
        self.performance_optimizer = get_performance_optimizer()

        # Initialize Neo4j client
        self.neo4j_client = Neo4jValidationClient(neo4j_driver)

        # Initialize validator
        self.validator = CodeValidator(
            neo4j_client=self.neo4j_client,
            performance_optimizer=self.performance_optimizer,
            min_confidence=0.6,
            high_confidence=0.8,
        )

        # Expose thresholds for backward compatibility
        self.MIN_CONFIDENCE_THRESHOLD = self.validator.MIN_CONFIDENCE_THRESHOLD
        self.HIGH_CONFIDENCE_THRESHOLD = self.validator.HIGH_CONFIDENCE_THRESHOLD

        # Deprecated cache (kept for compatibility)
        self._validation_cache: dict[str, Any] = {}

        # Expose neo4j_enabled for backward compatibility
        self.neo4j_enabled = self.neo4j_client.neo4j_enabled

    @performance_monitor
    async def search_and_validate_code(
        self,
        query: str,
        match_count: int = 10,
        source_filter: str | None = None,
        min_confidence: float | None = None,
        include_suggestions: bool = True,
        parallel_validation: bool = True,
    ) -> dict[str, Any]:
        """Search for code examples and validate them against the knowledge graph.

        Args:
            query: Search query for semantic matching
            match_count: Maximum number of results to return
            source_filter: Optional source repository filter
            min_confidence: Minimum confidence threshold (defaults to service threshold)
            include_suggestions: Whether to include correction suggestions
            parallel_validation: Whether to validate results in parallel

        Returns:
            Dictionary with validated search results and metadata
        """
        if min_confidence is None:
            min_confidence = self.MIN_CONFIDENCE_THRESHOLD

        logger.info("Starting validated code search for query: %s", query)

        try:
            # Step 1: Perform semantic search in Qdrant
            semantic_results = await self._perform_semantic_search(
                query,
                match_count * 2,
                source_filter,  # Get more results to filter
            )

            if not semantic_results:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "validation_summary": {
                        "total_found": 0,
                        "validated": 0,
                        "high_confidence": 0,
                        "neo4j_available": self.neo4j_enabled,
                    },
                }

            # Step 2: Validate results against Neo4j knowledge graph
            if parallel_validation and self.neo4j_enabled:
                validated_results = await self.validator.validate_results_parallel(
                    semantic_results,
                    include_suggestions,
                )
            else:
                validated_results = await self.validator.validate_results_sequential(
                    semantic_results,
                    include_suggestions,
                )

            # Step 3: Filter and rank by confidence
            filtered_results = [
                result
                for result in validated_results
                if result.get("validation", {}).get("confidence_score", 0)
                >= min_confidence
            ]

            # Sort by combined score (semantic similarity + validation confidence)
            filtered_results.sort(
                key=lambda x: self.validator.calculate_combined_score(x),
                reverse=True,
            )

            # Limit to requested count
            final_results = filtered_results[:match_count]

            # Step 4: Generate summary statistics
            validation_summary = self._generate_validation_summary(
                semantic_results,
                validated_results,
                final_results,
            )

            return {
                "success": True,
                "query": query,
                "results": final_results,
                "validation_summary": validation_summary,
                "search_metadata": {
                    "semantic_search_count": len(semantic_results),
                    "post_validation_count": len(validated_results),
                    "final_result_count": len(final_results),
                    "min_confidence_threshold": min_confidence,
                    "parallel_validation": parallel_validation and self.neo4j_enabled,
                },
            }

        except (DatabaseError, SearchError) as e:
            logger.error("Database or search error in validated code search: %s", e)
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "fallback_available": not self.neo4j_enabled,
            }
        except Exception as e:
            logger.exception("Unexpected error in validated code search: %s", e)
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "fallback_available": not self.neo4j_enabled,
            }

    async def _perform_semantic_search(
        self,
        query: str,
        match_count: int,
        source_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Perform semantic search in Qdrant.

        Args:
            query: Search query
            match_count: Number of results to return
            source_filter: Optional source repository filter

        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embeddings = create_embeddings_batch([query])
            if not query_embeddings:
                return []

            # Prepare metadata filter
            filter_metadata = None
            if source_filter:
                filter_metadata = {"source_id": source_filter}

            # Search code examples
            # Note: Using query parameter instead of query_embedding for newer interface
            return await self.database_client.search_code_examples(  # type: ignore[no-any-return]
                query=query,  # Pass the query string, the adapter will create embeddings
                match_count=match_count,
                filter_metadata=filter_metadata,
            )

        except SearchError as e:
            logger.error("Search operation failed: %s", e)
            return []
        except Exception as e:
            logger.exception("Unexpected error in semantic search: %s", e)
            return []

    def _generate_validation_summary(
        self,
        semantic_results: list[dict[str, Any]],
        validated_results: list[dict[str, Any]],
        final_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate summary statistics for the validation process.

        Args:
            semantic_results: Results from semantic search
            validated_results: Results after validation
            final_results: Final filtered results

        Returns:
            Validation summary statistics
        """
        high_confidence_count = sum(
            1
            for result in final_results
            if result.get("validation", {}).get("confidence_score", 0)
            >= self.HIGH_CONFIDENCE_THRESHOLD
        )

        return {
            "total_found": len(semantic_results),
            "validated": len(validated_results),
            "final_count": len(final_results),
            "high_confidence": high_confidence_count,
            "validation_rate": len(validated_results) / len(semantic_results)
            if semantic_results
            else 0,
            "high_confidence_rate": high_confidence_count / len(final_results)
            if final_results
            else 0,
            "neo4j_available": self.neo4j_enabled,
            "cache_hits": len(self._validation_cache),
            "confidence_thresholds": {
                "minimum": self.MIN_CONFIDENCE_THRESHOLD,
                "high": self.HIGH_CONFIDENCE_THRESHOLD,
            },
        }

    async def clear_validation_cache(self) -> None:
        """Clear the validation cache."""
        await self.performance_optimizer.cache.clear()
        logger.info("Validation cache cleared")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get validation cache statistics.

        Returns:
            Cache statistics and thresholds
        """
        cache_stats = self.performance_optimizer.cache.get_stats()
        return {
            "cache_stats": cache_stats,
            "neo4j_enabled": self.neo4j_enabled,
            "thresholds": {
                "min_confidence": self.MIN_CONFIDENCE_THRESHOLD,
                "high_confidence": self.HIGH_CONFIDENCE_THRESHOLD,
            },
        }

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of the validated search service.

        Returns:
            Health status dictionary
        """
        return await validate_integration_health(
            database_client=self.database_client,
            neo4j_driver=self.neo4j_client.neo4j_driver,
        )
