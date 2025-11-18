"""Decorators for the Crawl4AI MCP server."""

import functools
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import ParamSpec, TypeVar

from .logging import logger, request_id_ctx

P = ParamSpec("P")
R = TypeVar("R")


def track_request(
    tool_name: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator to track MCP tool requests with timing and error handling.

    Args:
        tool_name: Name of the tool being tracked

    Returns:
        Decorated function with request tracking
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            request_id = str(uuid.uuid4())[:8]
            start_time = datetime.now(UTC).timestamp()

            # Store request_id in ContextVar for automatic logging
            request_id_ctx.set(request_id)

            logger.info("Starting %s request", tool_name)
            if logger.isEnabledFor(10):  # DEBUG level
                logger.debug("Arguments: %s", kwargs)

            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                duration = datetime.now(UTC).timestamp() - start_time
                logger.error("Failed %s after %.2fs: %s", tool_name, duration, str(e))
                logger.debug("Traceback: %s", traceback.format_exc())
                raise
            else:
                duration = datetime.now(UTC).timestamp() - start_time
                logger.info("Completed %s in %.2fs", tool_name, duration)
            finally:
                # Clean up context variable
                request_id_ctx.set(None)

            return result

        return wrapper

    return decorator
