"""Memory tracking utilities for crawling operations.

This module provides context managers for monitoring memory usage
during crawling operations using Crawl4AI's memory statistics.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from crawl4ai.utils import get_memory_stats

logger = logging.getLogger(__name__)


@asynccontextmanager
async def track_memory(operation_name: str) -> AsyncIterator[dict[str, Any]]:
    """Context manager to track memory usage before and after an operation.

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        dict: Dictionary to store results for memory analysis
    """
    start_memory_percent, start_available_gb, total_gb = get_memory_stats()
    logger.info(
        "[%s] Memory before: %.1f%% used, %.2f/%.2f GB available",
        operation_name,
        start_memory_percent,
        start_available_gb,
        total_gb,
    )

    # Yield a dict to collect results
    context = {"results": None}

    try:
        yield context
    finally:
        end_memory_percent, end_available_gb, _ = get_memory_stats()
        memory_delta = end_memory_percent - start_memory_percent

        logger.info(
            "[%s] Memory after: %.1f%% used (Δ %+.1f%%), %.2f GB available",
            operation_name,
            end_memory_percent,
            memory_delta,
            end_available_gb,
        )

        # Log dispatch stats if results are available
        if context["results"]:
            dispatch_stats = []
            for r in context["results"]:
                if hasattr(r, "dispatch_result") and r.dispatch_result:
                    dispatch_stats.append(
                        {
                            "memory_usage": r.dispatch_result.memory_usage,
                            "peak_memory": r.dispatch_result.peak_memory,
                        },
                    )

            # Explicit length check to prevent division by zero (defensive programming)
            if dispatch_stats and len(dispatch_stats) > 0:
                avg_memory = sum(s["memory_usage"] for s in dispatch_stats) / len(
                    dispatch_stats,
                )
                peak_memory = max(s["peak_memory"] for s in dispatch_stats)
                logger.info(
                    "[%s] Dispatch stats: avg %.1f MB, peak %.1f MB",
                    operation_name,
                    avg_memory,
                    peak_memory,
                )
