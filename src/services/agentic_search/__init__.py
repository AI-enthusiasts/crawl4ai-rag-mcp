"""Agentic search service - modular implementation.

This package implements the complete agentic search pipeline with modular design:
- config: Agent initialization and configuration
- evaluator: Local knowledge evaluation (Stage 1)
- ranker: Web search and URL ranking (Stage 2)
- crawler: Selective crawling (Stage 3)
- orchestrator: Main pipeline orchestration and query refinement (Stage 4)
"""

from .config import AgenticSearchConfig
from .crawler import SelectiveCrawler
from .evaluator import LocalKnowledgeEvaluator
from .factory import get_agentic_search_service
from .mcp_wrapper import agentic_search_impl
from .orchestrator import AgenticSearchService
from .ranker import URLRanker

# Re-export main service and singleton for backward compatibility
__all__ = [
    "AgenticSearchConfig",
    "AgenticSearchService",
    "agentic_search_impl",
    "get_agentic_search_service",
]
