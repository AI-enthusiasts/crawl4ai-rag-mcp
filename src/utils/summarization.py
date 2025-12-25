"""Summarization utilities using Pydantic AI.

This module provides LLM-powered summarization following project architecture:
- Uses Pydantic AI (NOT OpenAI SDK directly) per AGENTS.md
- Singleton agent pattern for connection pooling
- Proper error handling with UnexpectedModelBehavior
"""

import logging
import os
import sys

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from src.config import get_settings
from src.core.constants import LLM_API_TIMEOUT_DEFAULT, MAX_RETRIES_DEFAULT

logger = logging.getLogger(__name__)

# Singleton agent instance
_summarization_agent: Agent[None, str] | None = None


def _get_agent() -> Agent[None, str]:
    """Get or create summarization agent (singleton pattern).

    Returns:
        Pydantic AI Agent configured for text summarization.
    """
    global _summarization_agent
    if _summarization_agent is None:
        settings = get_settings()

        # Create OpenAI model - API key read from OPENAI_API_KEY env var
        model = OpenAIModel(model_name=settings.model_choice)

        # Configure model settings per Pydantic AI docs
        model_settings = ModelSettings(
            temperature=0.3,
            timeout=LLM_API_TIMEOUT_DEFAULT,
        )

        # Create agent with string output (plain text summarization)
        _summarization_agent = Agent(
            model=model,
            output_type=str,
            output_retries=MAX_RETRIES_DEFAULT,
            model_settings=model_settings,
            system_prompt="You are a helpful assistant that provides concise library/tool/framework summaries.",
        )

        logger.debug(
            "Initialized summarization agent with model=%s", settings.model_choice
        )

    return _summarization_agent


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """Extract a summary for a source from its content using an LLM.

    This function uses Pydantic AI to generate a concise summary of the source content.

    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary

    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"

    if not content or len(content.strip()) == 0:
        return default_summary

    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content

    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""

    try:
        agent = _get_agent()
        result = agent.run_sync(prompt)
        summary = result.output.strip() if result.output else ""

        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary if summary else default_summary

    except UnexpectedModelBehavior as e:
        logger.exception(
            "LLM failed after retries for source %s: %s",
            source_id,
            e,
        )
        return default_summary
    except Exception as e:
        print(
            f"Error generating summary with LLM for {source_id}: {e}. Using default summary.",
            file=sys.stderr,
        )
        return default_summary
