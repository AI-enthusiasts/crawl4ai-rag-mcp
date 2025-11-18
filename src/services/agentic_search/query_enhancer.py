"""Query enhancement for agentic search.

This module handles:
- Topic decomposition: Breaking query into essential topics
- Multi-query generation: Creating query variations for better recall

Per Anthropic research: "Search strategy should mirror expert human research:
explore the landscape before drilling into specifics."

Per RAG best practices: Generate 3-4 query variations BEFORE search
to improve recall through different phrasings and perspectives.
"""

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.core.constants import MAX_RETRIES_DEFAULT
from src.core.exceptions import LLMError
from src.services.agentic_models import (
    MultiQueryExpansion,
    QueryComplexity,
    TopicDecomposition,
)

from .config import AgenticSearchConfig

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Enhances queries with topic decomposition and multi-query expansion.

    This class implements the first phase of deep research:
    1. Decompose query into essential topics that MUST be covered
    2. Generate multiple query variations for each topic
    3. Enable topic-based completeness evaluation
    """

    def __init__(self, config: AgenticSearchConfig) -> None:
        """Initialize query enhancer with shared configuration.

        Args:
            config: Shared agentic search configuration with agents
        """
        self.config = config

        # Create specialized agents for query enhancement
        # Topic decomposition agent
        self.decomposition_agent = Agent(
            model=config.openai_model,
            output_type=TopicDecomposition,
            output_retries=MAX_RETRIES_DEFAULT,
            model_settings=config.base_model_settings,
        )

        # Multi-query expansion agent (slightly higher temperature for creativity)
        self.expansion_agent = Agent(
            model=config.openai_model,
            output_type=MultiQueryExpansion,
            output_retries=MAX_RETRIES_DEFAULT,
            model_settings=config.refinement_model_settings,  # 0.5 temperature
        )

    async def decompose_query(self, query: str) -> TopicDecomposition:
        """Decompose query into essential topics that must be covered.

        Per Anthropic: "Subagents plan, then use interleaved thinking after
        tool results to evaluate quality, identify gaps, and refine."

        Args:
            query: User's original query

        Returns:
            TopicDecomposition with topics and their specific queries

        Raises:
            LLMError: If decomposition fails after retries
        """
        logger.info("Decomposing query into topics: %s", query)

        prompt = f"""You are decomposing a user query into essential topics that MUST be covered for a complete answer.

User Query: "{query}"

Analyze the query and identify 3-6 essential topics. Consider:

For "What is X?" queries:
- Definition and core concept
- Key features/components
- Use cases and applications
- Code examples (if technical)

For "How to X?" queries:
- Prerequisites and setup
- Step-by-step process
- Code examples
- Common pitfalls and troubleshooting

For "Compare X vs Y" queries:
- X overview and strengths
- Y overview and strengths
- Key differences
- When to use each

For each topic, provide a specific search query that would find relevant information.

Also assess complexity:
- simple: Basic fact-finding (1 search, 3-10 results)
- moderate: Multi-aspect topic (2-4 searches, 10-15 results each)
- complex: Comprehensive research (5+ searches, many sources)

Return structured output with:
- topics: List of 3-6 essential topic names
- topic_queries: Dict mapping each topic to its search query
- complexity: Query complexity level"""

        try:
            result = await self.decomposition_agent.run(prompt)
            decomposition = result.output

            logger.info(
                "Decomposed into %d topics (complexity=%s): %s",
                len(decomposition.topics),
                decomposition.complexity.value,
                decomposition.topics,
            )

            return decomposition

        except UnexpectedModelBehavior as e:
            logger.error("Topic decomposition failed after retries: %s", e)
            raise LLMError("LLM topic decomposition failed after retries") from e

        except Exception as e:
            logger.exception("Unexpected error in topic decomposition: %s", e)
            raise LLMError(f"Topic decomposition failed: {e}") from e

    async def expand_query(
        self, query: str, topic: str | None = None
    ) -> MultiQueryExpansion:
        """Generate multiple query variations for better recall.

        Per RAG best practices: Multi-query generation improves recall
        by searching with different phrasings and perspectives.

        Args:
            query: Query to expand
            topic: Optional topic context for more focused expansion

        Returns:
            MultiQueryExpansion with query variations

        Raises:
            LLMError: If expansion fails after retries
        """
        context = f" (focusing on: {topic})" if topic else ""
        logger.info("Expanding query%s: %s", context, query)

        topic_context = f"\nTopic Focus: {topic}" if topic else ""

        prompt = f"""You are generating query variations to improve search recall.

Original Query: "{query}"{topic_context}

Generate multiple search query variations:

1. variations: 2-3 rephrased versions using:
   - Different terminology/synonyms
   - Different question structures
   - Different perspectives

2. broad_query: A more general "step-back" query that explores the broader context
   Example: "How to deploy Docker?" → "Docker deployment best practices"

3. specific_query: A more specific query with additional context
   Example: "How to deploy Docker?" → "Docker container deployment production environment"

The goal is to maximize recall by covering different ways the same information might be indexed.

Return structured output with:
- variations: List of 2-3 query variations
- broad_query: More general version
- specific_query: More specific version"""

        try:
            result = await self.expansion_agent.run(prompt)
            expansion = result.output

            logger.info(
                "Expanded into %d variations + broad + specific",
                len(expansion.variations),
            )

            return expansion

        except UnexpectedModelBehavior as e:
            logger.error("Query expansion failed after retries: %s", e)
            raise LLMError("LLM query expansion failed after retries") from e

        except Exception as e:
            logger.exception("Unexpected error in query expansion: %s", e)
            raise LLMError(f"Query expansion failed: {e}") from e

    async def enhance_query(
        self,
        query: str,
        decompose: bool = True,
        expand: bool = True,
    ) -> tuple[TopicDecomposition | None, list[str]]:
        """Full query enhancement: decomposition + multi-query expansion.

        This is the main entry point for query enhancement, combining
        topic decomposition and multi-query generation.

        Args:
            query: User's original query
            decompose: Whether to decompose into topics
            expand: Whether to generate query variations

        Returns:
            Tuple of (TopicDecomposition or None, list of all queries to execute)

        Raises:
            LLMError: If enhancement fails
        """
        decomposition = None
        all_queries: list[str] = []

        # Step 1: Topic decomposition
        if decompose:
            decomposition = await self.decompose_query(query)

            # Generate queries for each topic
            if expand:
                for topic, topic_query in decomposition.topic_queries.items():
                    # Expand each topic query
                    expansion = await self.expand_query(topic_query, topic)

                    # Collect all variations
                    all_queries.append(topic_query)  # Original topic query
                    all_queries.extend(expansion.variations)
                    all_queries.append(expansion.broad_query)
                    all_queries.append(expansion.specific_query)
            else:
                # Just use topic queries without expansion
                all_queries.extend(decomposition.topic_queries.values())
        elif expand:
            # No decomposition, just expand the original query
            expansion = await self.expand_query(query)
            all_queries.append(query)
            all_queries.extend(expansion.variations)
            all_queries.append(expansion.broad_query)
            all_queries.append(expansion.specific_query)
        else:
            # No enhancement, just use original query
            all_queries.append(query)

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in all_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        logger.info(
            "Query enhancement complete: %d unique queries from %d total",
            len(unique_queries),
            len(all_queries),
        )

        return decomposition, unique_queries

    async def generate_gap_queries(
        self,
        uncovered_topics: list[str],
        original_query: str,
    ) -> list[str]:
        """Generate specific queries to fill knowledge gaps.

        Per gap-driven iteration: After first search, search ONLY
        for uncovered topics to minimize redundant searches.

        Args:
            uncovered_topics: Topics that need more information
            original_query: Original user query for context

        Returns:
            List of queries targeting the gaps
        """
        if not uncovered_topics:
            return []

        logger.info(
            "Generating gap queries for %d uncovered topics", len(uncovered_topics)
        )

        gap_queries = []
        for topic in uncovered_topics:
            # Generate focused query for each gap
            expansion = await self.expand_query(
                f"{original_query} {topic}",
                topic=topic,
            )
            # Use the specific query as it's most targeted
            gap_queries.append(expansion.specific_query)
            # Also add one variation for diversity
            if expansion.variations:
                gap_queries.append(expansion.variations[0])

        # Remove duplicates
        seen = set()
        unique_gap_queries = []
        for q in gap_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_gap_queries.append(q)

        logger.info("Generated %d gap queries", len(unique_gap_queries))
        return unique_gap_queries
