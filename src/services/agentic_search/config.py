"""Configuration and agent initialization for agentic search.

This module handles:
- Pydantic AI agent creation and configuration
- OpenAI model setup with API keys
- Model settings (temperature, timeout, retries)
- Service configuration parameters
"""

import logging

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from src.config import get_settings
from src.core.constants import LLM_API_TIMEOUT_DEFAULT, MAX_RETRIES_DEFAULT
from src.services.agentic_models import CompletenessEvaluation, URLRankingList

logger = logging.getLogger(__name__)
settings = get_settings()


class AgenticSearchConfig:
    """Configuration container for agentic search agents and parameters."""

    def __init__(self) -> None:
        """Initialize Pydantic AI agents and configuration parameters."""
        # Create OpenAI model instance
        model = OpenAIModel(
            model_name=settings.model_choice,
        )

        # Shared model settings for all agents
        # Per Pydantic AI docs: timeout, temperature configured via ModelSettings
        self.base_model_settings = ModelSettings(
            temperature=settings.agentic_search_llm_temperature,
            timeout=LLM_API_TIMEOUT_DEFAULT,  # 60s timeout
        )

        # Create specialized agents for each LLM task
        # Per Pydantic AI docs: Agent with output_type for structured outputs

        # Agent for evaluating knowledge completeness (Stage 1)
        self.completeness_agent = Agent(
            model=model,
            output_type=CompletenessEvaluation,
            output_retries=MAX_RETRIES_DEFAULT,  # Retry 3 times for validation errors
            model_settings=self.base_model_settings,
        )

        # Agent for ranking URLs by relevance (Stage 2)
        self.ranking_agent = Agent(
            model=model,
            output_type=URLRankingList,
            output_retries=MAX_RETRIES_DEFAULT,
            model_settings=self.base_model_settings,
        )

        # Query refinement agent with custom temperature for creativity (Stage 4)
        # Per Pydantic AI docs: Use ModelSettings to override temperature per agent
        # Note: output_type defined inline in refinement (dynamic model)
        refinement_settings = ModelSettings(
            temperature=0.5,  # More creative for query generation
            timeout=LLM_API_TIMEOUT_DEFAULT,
        )
        self.refinement_model_settings = refinement_settings
        self.openai_model = model  # Store for dynamic agent creation

        # Store configuration parameters from settings
        self.model_name = settings.model_choice
        self.temperature = settings.agentic_search_llm_temperature
        self.completeness_threshold = settings.agentic_search_completeness_threshold
        self.max_iterations = settings.agentic_search_max_iterations
        self.max_urls_per_iteration = settings.agentic_search_max_urls_per_iteration
        self.max_pages_per_iteration = settings.agentic_search_max_pages_per_iteration
        self.url_score_threshold = settings.agentic_search_url_score_threshold
        self.use_search_hints = settings.agentic_search_use_search_hints
        self.enable_url_filtering = settings.agentic_search_enable_url_filtering
        self.max_urls_to_rank = settings.agentic_search_max_urls_to_rank
        self.max_qdrant_results = settings.agentic_search_max_qdrant_results

        logger.info(
            f"Initialized agentic search configuration with model={self.model_name}, "
            f"threshold={self.completeness_threshold}, max_iterations={self.max_iterations}"
        )
