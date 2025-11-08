"""Decorators for the Crawl4AI MCP server."""

import functools
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, TypeVar

from fastmcp import Context

from .logging import logger, request_id_ctx

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def track_request(tool_name: str) -> Callable[[F], F]:
    """Decorator to track MCP tool requests with timing and error handling."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(ctx: Context, *args: Any, **kwargs: Any) -> Any:
            request_id = str(uuid.uuid4())[:8]
            start_time = datetime.now().timestamp()
            
            # Store request_id in ContextVar for automatic logging
            request_id_ctx.set(request_id)

            logger.info(f"Starting {tool_name} request")
            logger.debug(f"Arguments: {kwargs}")

            try:
                result = await func(ctx, *args, **kwargs)
                duration = datetime.now().timestamp() - start_time
                logger.info(f"Completed {tool_name} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = datetime.now().timestamp() - start_time
                logger.error(
                    f"Failed {tool_name} after {duration:.2f}s: {e!s}",
                )
                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
            finally:
                # Clean up context variable
                request_id_ctx.set(None)

        return wrapper  # type: ignore[return-value]

    return decorator
