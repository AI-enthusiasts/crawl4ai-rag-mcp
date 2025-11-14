"""Core functionality for the Crawl4AI MCP server."""

from .constants import (
    COMPLETENESS_THRESHOLD_DEFAULT,
    HTTP_OK,
    LLM_TEMPERATURE_DETERMINISTIC,
    MAX_INPUT_SIZE,
    MAX_ITERATIONS_DEFAULT,
    MAX_QDRANT_RESULTS_DEFAULT,
    MAX_RETRIES_DEFAULT,
    MAX_URLS_PER_ITERATION_DEFAULT,
    OPENAI_EMBEDDING_DIMENSION,
    QDRANT_BATCH_SIZE,
    TEST_MODEL_CHEAP,
    URL_SCORE_THRESHOLD_DEFAULT,
)
from .context import (
    Crawl4AIContext,
    cleanup_global_context,
    crawl4ai_lifespan,
    get_app_context,
    initialize_global_context,
)
from .decorators import track_request
from .exceptions import MCPToolError
from .logging import configure_logging, logger
from .stdout_utils import SuppressStdout

__all__ = [
    # Core
    "Crawl4AIContext",
    "MCPToolError",
    "SuppressStdout",
    "cleanup_global_context",
    "configure_logging",
    "crawl4ai_lifespan",
    "get_app_context",
    "initialize_global_context",
    "logger",
    "track_request",
    # Constants - most commonly used
    "COMPLETENESS_THRESHOLD_DEFAULT",
    "HTTP_OK",
    "LLM_TEMPERATURE_DETERMINISTIC",
    "MAX_INPUT_SIZE",
    "MAX_ITERATIONS_DEFAULT",
    "MAX_QDRANT_RESULTS_DEFAULT",
    "MAX_RETRIES_DEFAULT",
    "MAX_URLS_PER_ITERATION_DEFAULT",
    "OPENAI_EMBEDDING_DIMENSION",
    "QDRANT_BATCH_SIZE",
    "TEST_MODEL_CHEAP",
    "URL_SCORE_THRESHOLD_DEFAULT",
]
