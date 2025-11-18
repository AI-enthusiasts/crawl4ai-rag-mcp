"""Performance optimization utilities for integration layer.

Provides batch processing, circuit breaker pattern, performance monitoring
decorator, and integrated performance optimizer.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.config import get_settings
from src.core.exceptions import DatabaseError, ValidationError

from .cache import PerformanceCache, create_cache_key
from .health import IntegrationHealthMonitor

logger = logging.getLogger(__name__)
settings = get_settings()


class BatchProcessor:
    """Utility for batching and parallelizing validation operations.

    Uses global MAX_CONCURRENT_SESSIONS limit.
    """

    def __init__(self, batch_size: int = 20):
        """Initialize the batch processor.

        Args:
            batch_size: Size of each processing batch
        """
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_sessions)

    async def process_batch(
        self,
        items: list[Any],
        processor_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Process items in batches with concurrency control.

        Args:
            items: List of items to process
            processor_func: Async function to process each item
            *args, **kwargs: Additional arguments for processor function

        Returns:
            List of processing results
        """
        results = []

        # Process in chunks
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]

            # Create tasks for this batch
            tasks = [
                self._process_with_semaphore(processor_func, item, *args, **kwargs)
                for item in batch
            ]

            # Execute batch and collect results
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)

        return results

    async def _process_with_semaphore(
        self, processor_func: Callable[..., Any], item: Any, *args: Any, **kwargs: Any,
    ) -> Any:
        """Process an item with semaphore control.

        Args:
            processor_func: Function to process the item
            item: Item to process
            *args, **kwargs: Additional arguments

        Returns:
            Processing result or exception
        """
        async with self._semaphore:
            try:
                return await processor_func(item, *args, **kwargs)
            except ValidationError as e:
                logger.warning("Validation error processing item: %s", e)
                return e
            except DatabaseError as e:
                logger.warning("Database error processing item: %s", e)
                return e
            except Exception as e:
                logger.warning("Unexpected error processing item: %s", e)
                return e


class CircuitBreaker:
    """Circuit breaker pattern for handling service failures gracefully."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type[BaseException] = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before attempting to close circuit
            expected_exception: Type of exception that triggers the circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call a function through the circuit breaker.

        Args:
            func: Function to call
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit breaker is open
        """
        # Check if circuit should be half-open
        if self.state == "open" and self._should_attempt_reset():
            self.state = "half-open"

        # If circuit is open, fail fast
        if self.state == "open":
            msg = "Circuit breaker is OPEN"
            raise Exception(msg)

        try:
            result = await func(*args, **kwargs)
            # Success - reset failure count
            if self.state == "half-open":
                self.state = "closed"
            self.failure_count = 0
            return result

        except self.expected_exception:
            self._record_failure()
            raise

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset.

        Returns:
            True if circuit should attempt reset
        """
        return (
            self.last_failure_time is not None
            and time.time() - self.last_failure_time >= self.timeout
        )

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state.

        Returns:
            Dictionary with circuit breaker state
        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
        }


def performance_monitor(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to monitor function performance.

    Logs execution time and catches exceptions.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with performance monitoring
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        function_name = func.__name__

        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.debug("%s executed in %.3fs", function_name, execution_time)
            return result

        except ValidationError as e:
            execution_time = time.time() - start_time
            logger.error("%s validation error after %.3fs: %s", function_name, execution_time, e)
            raise
        except DatabaseError as e:
            execution_time = time.time() - start_time
            logger.error("%s database error after %.3fs: %s", function_name, execution_time, e)
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception("%s failed after %.3fs: %s", function_name, execution_time, e)
            raise

    return wrapper


class PerformanceOptimizer:
    """Performance optimization utilities for the integration layer."""

    def __init__(self) -> None:
        """Initialize performance optimizer with all components."""
        self.cache = PerformanceCache()
        self.batch_processor = BatchProcessor()
        self.circuit_breaker = CircuitBreaker()
        self.health_monitor = IntegrationHealthMonitor()

    @performance_monitor
    async def optimize_search_query(self, query: str, context: dict[str, Any]) -> str:
        """Optimize search queries for better performance and accuracy.

        Args:
            query: Original search query
            context: Additional context for optimization

        Returns:
            Optimized search query
        """
        # Cache key for query optimization
        cache_key = create_cache_key("query_opt", query, context)

        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result  # type: ignore[no-any-return]

        # Perform optimization
        optimized_query = self._apply_query_optimizations(query, context)

        # Cache the result
        await self.cache.set(cache_key, optimized_query, ttl=1800)  # 30 minutes

        return optimized_query

    def _apply_query_optimizations(self, query: str, context: dict[str, Any]) -> str:
        """Apply various query optimization techniques.

        Args:
            query: Original query
            context: Optimization context

        Returns:
            Optimized query string
        """
        optimized = query.strip()

        # Add context-specific terms
        if context.get("code_type"):
            optimized = f"{context['code_type']} {optimized}"

        # Add programming language context
        if context.get("language"):
            optimized = f"{optimized} {context['language']}"

        # Remove redundant words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = optimized.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]

        if len(filtered_words) > 0:
            optimized = " ".join(filtered_words)

        return optimized

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        return {
            "cache_stats": self.cache.get_stats(),
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "batch_processor": {
                "batch_size": self.batch_processor.batch_size,
            },
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.cache.clear()


# Global performance optimizer instance
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance.

    Returns:
        Singleton PerformanceOptimizer instance
    """
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer
