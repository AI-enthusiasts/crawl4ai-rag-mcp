"""Application context and lifecycle management for the Crawl4AI MCP server."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from crawl4ai import BrowserConfig, MemoryAdaptiveDispatcher, RateLimiter
from fastmcp import FastMCP
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.core.exceptions import (
    DatabaseError,
    KnowledgeGraphError,
)
from src.database.base import VectorDatabase
from src.database.factory import create_and_initialize_database

from .logging import logger

# Get settings instance
settings = get_settings()

# Global context storage
_app_context: Optional["Crawl4AIContext"] = None
_context_lock = None  # Will be initialized in async context


def set_app_context(context: "Crawl4AIContext") -> None:
    """Store the application context globally."""
    global _app_context
    _app_context = context


def get_app_context() -> Optional["Crawl4AIContext"]:
    """Get the stored application context."""
    return _app_context


async def initialize_global_context() -> "Crawl4AIContext":
    """Initialize the global application context once.

    This should be called at application startup, not per-request.

    Returns:
        Crawl4AIContext: The initialized context
    """
    global _app_context, _context_lock

    # Initialize lock if needed
    if _context_lock is None:
        _context_lock = asyncio.Lock()

    async with _context_lock:
        # Return existing context if already initialized
        if _app_context is not None:
            logger.info("Using existing application context (singleton)")
            return _app_context

        logger.info("Initializing global application context (first time)...")

        # Create browser configuration with resource optimization
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            # Resource optimization flags to prevent memory leaks
            extra_args=[
                "--disable-extensions",
                "--disable-sync",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        logger.info(
            f"✓ BrowserConfig created for per-request crawler instances "
            f"(headless={browser_config.headless}, "
            f"browser_type={browser_config.browser_type})",
        )

        # Initialize database client
        database_client = await create_and_initialize_database()

        # Initialize cross-encoder model for reranking if enabled
        reranking_model = None
        if settings.use_reranking:
            try:
                reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except ImportError as e:
                logger.error(f"Reranking model dependencies not available: {e}")
                reranking_model = None
            except Exception as e:
                logger.error(f"Unexpected error loading reranking model: {e}")
                reranking_model = None

        # Initialize Neo4j components if configured and enabled
        knowledge_validator = None
        repo_extractor = None

        if settings.use_knowledge_graph:
            try:
                from src.knowledge_graph.knowledge_graph_validator import (
                    KnowledgeGraphValidator,
                )
                from src.knowledge_graph.parse_repo_into_neo4j import (
                    DirectNeo4jExtractor,
                )

                neo4j_uri = settings.neo4j_uri
                neo4j_user = settings.neo4j_username
                neo4j_password = settings.neo4j_password

                if neo4j_uri and neo4j_user and neo4j_password:
                    try:
                        logger.info("Initializing knowledge graph components...")

                        knowledge_validator = KnowledgeGraphValidator(
                            neo4j_uri,
                            neo4j_user,
                            neo4j_password,
                        )
                        await knowledge_validator.initialize()
                        logger.info("✓ Knowledge graph validator initialized")

                        repo_extractor = DirectNeo4jExtractor(
                            neo4j_uri,
                            neo4j_user,
                            neo4j_password,
                        )
                        await repo_extractor.initialize()
                        logger.info("✓ Repository extractor initialized")

                    except KnowledgeGraphError as e:
                        error_msg = (
                            f"Knowledge graph error initializing Neo4j: "
                            f"{format_neo4j_error(e)}"
                        )
                        logger.error(error_msg)
                        knowledge_validator = None
                        repo_extractor = None
                    except DatabaseError as e:
                        error_msg = (
                            f"Database error initializing Neo4j: "
                            f"{format_neo4j_error(e)}"
                        )
                        logger.error(error_msg)
                        knowledge_validator = None
                        repo_extractor = None
                    except Exception as e:
                        error_msg = (
                            f"Unexpected error initializing Neo4j components: "
                            f"{format_neo4j_error(e)}"
                        )
                        logger.error(error_msg)
                        knowledge_validator = None
                        repo_extractor = None
                else:
                    warning_msg = (
                        "Neo4j credentials not configured - knowledge graph "
                        "tools will be unavailable"
                    )
                    logger.warning(warning_msg)
            except ImportError as e:
                logger.warning(f"Knowledge graph dependencies not available: {e}")
        else:
            disabled_msg = (
                "Knowledge graph functionality disabled - set "
                "USE_KNOWLEDGE_GRAPH=true to enable"
            )
            logger.info(disabled_msg)

        # Initialize shared dispatcher for global concurrency control
        # This ensures max_session_permit applies across ALL tool calls, not per-call
        rate_limiter = RateLimiter(
            base_delay=(0.5, 1.5),
            max_delay=30.0,
            max_retries=3,
            rate_limit_codes=[429, 503],
        )

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=settings.max_concurrent_sessions,
            rate_limiter=rate_limiter,
        )
        dispatcher_msg = (
            f"✓ Shared dispatcher initialized "
            f"(max_session_permit={settings.max_concurrent_sessions})"
        )
        logger.info(dispatcher_msg)

        context = Crawl4AIContext(
            browser_config=browser_config,
            database_client=database_client,
            dispatcher=dispatcher,
            reranking_model=reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor,
        )

        _app_context = context
        logger.info("✓ Global application context initialized successfully")
        return context


async def cleanup_global_context() -> None:
    """Clean up the global application context.

    This should be called at application shutdown.
    """
    global _app_context

    if _app_context is None:
        logger.info("No global context to clean up")
        return

    logger.info("Starting cleanup of global application context...")

    # No crawler to close - crawlers are created per-request with context managers

    if _app_context.knowledge_validator:
        try:
            await _app_context.knowledge_validator.close()
            logger.info("✓ Knowledge graph validator closed")
        except KnowledgeGraphError as e:
            logger.error(
                f"Knowledge graph error closing validator: {e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error closing knowledge validator: {e}",
                exc_info=True,
            )

    if _app_context.repo_extractor:
        try:
            await _app_context.repo_extractor.close()
            logger.info("✓ Repository extractor closed")
        except KnowledgeGraphError as e:
            logger.error(
                f"Knowledge graph error closing extractor: {e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error closing repository extractor: {e}",
                exc_info=True,
            )

    _app_context = None
    logger.info("✓ Global application context cleanup completed")


@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""

    # Shared config for creating crawlers per-request
    browser_config: BrowserConfig
    database_client: VectorDatabase
    # Shared dispatcher for global concurrency control
    dispatcher: MemoryAdaptiveDispatcher
    reranking_model: CrossEncoder | None = None
    # KnowledgeGraphValidator when available
    knowledge_validator: Any | None = None
    # DirectNeo4jExtractor when available
    repo_extractor: Any | None = None


def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j errors for user-friendly display."""
    error_str = str(error).lower()

    if "authentication" in error_str or "unauthorized" in error_str:
        return "Authentication failed. Please check your Neo4j username and password."
    if (
        "failed to establish connection" in error_str
        or "connection refused" in error_str
    ):
        return (
            "Connection failed. Please ensure Neo4j is running and accessible "
            "at the specified URI."
        )
    if "unable to retrieve routing information" in error_str:
        return (
            "Connection failed. The Neo4j URI may be incorrect or the database "
            "may not be accessible."
        )
    return f"Neo4j error: {error!s}"


@asynccontextmanager
async def crawl4ai_lifespan(_server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Lifespan context manager for FastMCP.

    NOTE: FastMCP HTTP mode calls this on EVERY request, not once at startup.
    Therefore, we use a singleton pattern to ensure only one crawler instance exists.

    Args:
        _server: The FastMCP server instance (required by FastMCP interface)

    Yields:
        Crawl4AIContext: The singleton context containing the Crawl4AI crawler
    """
    # Get or create the singleton context
    context = await initialize_global_context()

    try:
        yield context
    finally:
        # Don't cleanup here - FastMCP calls this per-request!
        # Cleanup will happen at application shutdown via cleanup_global_context()
        pass
