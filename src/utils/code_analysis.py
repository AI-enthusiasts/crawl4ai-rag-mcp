"""Code analysis utilities for extracting and summarizing code examples.

This module provides:
- Code block extraction from markdown
- LLM-powered code example summarization using Pydantic AI

Uses Pydantic AI (NOT OpenAI SDK directly) per AGENTS.md architecture.
"""

import logging
import sys
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from src.config import get_settings
from src.core.constants import LLM_API_TIMEOUT_DEFAULT, MAX_RETRIES_DEFAULT

logger = logging.getLogger(__name__)

# Singleton agent instance
_code_summary_agent: Agent[None, str] | None = None


def _get_agent() -> Agent[None, str]:
    """Get or create code summary agent (singleton pattern).

    Returns:
        Pydantic AI Agent configured for code example summarization.
    """
    global _code_summary_agent
    if _code_summary_agent is None:
        settings = get_settings()

        # Create OpenAI model - API key read from OPENAI_API_KEY env var
        model = OpenAIModel(model_name=settings.model_choice)

        # Configure model settings per Pydantic AI docs
        model_settings = ModelSettings(
            temperature=0.3,
            timeout=LLM_API_TIMEOUT_DEFAULT,
        )

        # Create agent with string output (plain text summarization)
        _code_summary_agent = Agent(
            model=model,
            output_type=str,
            output_retries=MAX_RETRIES_DEFAULT,
            model_settings=model_settings,
            system_prompt="You are a helpful assistant that provides concise code example summaries.",
        )

        logger.debug(
            "Initialized code summary agent with model=%s", settings.model_choice
        )

    return _code_summary_agent


def extract_code_blocks(
    markdown_content: str,
    min_length: int = 1000,
) -> list[dict[str, Any]]:
    """Extract code blocks from markdown content along with context.

    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)

    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []

    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = 0
    while True:
        pos = markdown_content.find("```", pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3

    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]

        # Extract the content between backticks
        code_section = markdown_content[start_pos + 3 : end_pos]

        # Check if there's a language specifier on the first line
        lines = code_section.split("\n", 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and " " not in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1] if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section
        else:
            language = ""
            code_content = code_section

        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue

        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()

        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3 : context_end].strip()

        code_blocks.append(
            {
                "code": code_content,
                "language": language,
                "context_before": context_before,
                "context_after": context_after,
                "full_context": f"{context_before}\n\n{code_content}\n\n{context_after}",
            },
        )

        # Move to next pair (skip the closing backtick we just processed)
        i += 2

    return code_blocks


def generate_code_example_summary(
    code: str,
    context_before: str = "",
    context_after: str = "",
) -> str:
    """Generate a summary for a code example using its surrounding context.

    Uses Pydantic AI for LLM calls with proper error handling.

    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code

    Returns:
        A summary of what the code example demonstrates
    """
    default_summary = "Code example for demonstration purposes."

    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""

    try:
        agent = _get_agent()
        result = agent.run_sync(prompt)
        summary = result.output.strip() if result.output else ""
        return summary if summary else default_summary

    except UnexpectedModelBehavior as e:
        logger.exception("LLM failed after retries: %s", e)
        return default_summary
    except Exception as e:
        print(f"Error generating code example summary: {e}", file=sys.stderr)
        return default_summary


def process_code_example(args: tuple[str, str, str]) -> str:
    """Process a single code example to generate its summary.

    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (code, context_before, context_after)

    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)
