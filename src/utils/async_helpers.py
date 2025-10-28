"""
Async execution helpers to prevent blocking the main event loop.

This module provides utilities for running async operations in separate threads
to keep FastMCP server responsive during long-running operations like web crawling.
"""

import asyncio
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar('T')


async def run_async_in_executor(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Run an async function in a separate thread to avoid blocking the main event loop.
    
    This is crucial for FastMCP to remain responsive during long-running operations.
    Without this, crawler operations block the event loop and FastMCP returns 504 errors.
    
    The function creates a new event loop in a thread pool worker and executes
    the async function there, allowing the main event loop to continue processing
    other requests (like health checks, new API calls, etc.).
    
    Args:
        async_func: The async function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the async function
        
    Example:
        ```python
        # ❌ Blocking - blocks event loop
        results = await crawler.arun_many(urls=urls, config=config)
        
        # ✅ Non-blocking - runs in separate thread
        results = await run_async_in_executor(
            crawler.arun_many,
            urls=urls,
            config=config,
            dispatcher=dispatcher
        )
        
        # Also works for single URL crawls
        result = await run_async_in_executor(crawler.arun, url=url, config=config)
        ```
    
    Note:
        This pattern is similar to how QdrantClient (synchronous) is wrapped
        with run_in_executor in qdrant_adapter.py
    """
    func_name = getattr(async_func, '__name__', str(async_func))
    logger.info(f"[run_async_in_executor] Starting {func_name} in thread pool")
    
    loop = asyncio.get_event_loop()
    
    def _run_in_thread() -> Any:
        """Execute async function in a new event loop within a thread"""
        logger.debug(f"[run_async_in_executor] Thread started for {func_name}")
        # Create new event loop for this thread
        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        try:
            # Run the async function in the thread's event loop
            result = thread_loop.run_until_complete(async_func(*args, **kwargs))
            logger.info(f"[run_async_in_executor] Completed {func_name}")
            return result
        except Exception as e:
            logger.error(f"[run_async_in_executor] Error in {func_name}: {e}")
            raise
        finally:
            # Always clean up the event loop
            thread_loop.close()
    
    # Execute in thread pool to prevent blocking main event loop
    return await loop.run_in_executor(None, _run_in_thread)
