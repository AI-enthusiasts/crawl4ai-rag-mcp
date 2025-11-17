"""Regression test for RAGResult validation bug.

Bug report:
When agentic_search returns empty results, it fails with:
  2 validation errors for RAGResult
  content
    String should have at least 1 character [type=string_too_short, input_value='', input_type=str]
  chunk_index
    Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]

Root cause:
1. In rag_queries.py:119, results are formatted with field "content"
2. In evaluator.py:86, code looks for field "chunk" (wrong name!)
3. result.get("chunk", "") always returns "" → validation error
4. chunk_index can be None → validation error

This test verifies the bug is fixed.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.agentic_models import RAGResult


class TestRAGResultValidationBug:
    """Test for RAGResult validation errors when processing empty/invalid results."""

    @pytest.mark.asyncio
    async def test_evaluator_handles_rag_response_with_content_field(self):
        """Test that evaluator correctly reads 'content' field from RAG response.

        Bug: evaluator.py:86 looks for 'chunk' but rag_queries.py:119 returns 'content'
        """
        from src.services.agentic_search.config import AgenticSearchConfig
        from src.services.agentic_search.evaluator import LocalKnowledgeEvaluator

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY required")

        # Create evaluator
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)

        # Mock FastMCP context
        mock_ctx = MagicMock()

        # Mock app context with database client
        mock_db_client = MagicMock()
        mock_app_ctx = MagicMock()
        mock_app_ctx.database_client = mock_db_client

        # Mock RAG response with CORRECT field name "content" (not "chunk")
        # This is what rag_queries.py actually returns
        rag_response = json.dumps({
            "success": True,
            "results": [
                {
                    "content": "Test content",  # ✓ Correct field name
                    "url": "https://example.com",
                    "similarity_score": 0.95,
                    "chunk_index": 0,
                }
            ]
        })

        with patch("src.services.agentic_search.evaluator.get_app_context", return_value=mock_app_ctx):
            with patch("src.services.agentic_search.evaluator.perform_rag_query", return_value=rag_response):
                with patch.object(evaluator, "_evaluate_completeness") as mock_eval:
                    # Mock completeness evaluation
                    from src.services.agentic_models import CompletenessEvaluation
                    mock_eval.return_value = CompletenessEvaluation(
                        score=0.5,
                        reasoning="Test",
                        gaps=[]
                    )

                    # This should NOT raise validation error
                    evaluation, rag_results = await evaluator.evaluate_local_knowledge(
                        ctx=mock_ctx,
                        query="test query",
                        iteration=1,
                        search_history=[],
                    )

                    # Should have successfully parsed 1 result
                    assert len(rag_results) == 1
                    assert rag_results[0].content == "Test content"
                    assert rag_results[0].url == "https://example.com"
                    assert rag_results[0].similarity_score == 0.95
                    assert rag_results[0].chunk_index == 0

    @pytest.mark.asyncio
    async def test_evaluator_skips_invalid_rag_results(self):
        """Test that evaluator skips results with empty content or missing chunk_index.

        This prevents validation errors when Qdrant returns malformed data.
        """
        from src.services.agentic_search.config import AgenticSearchConfig
        from src.services.agentic_search.evaluator import LocalKnowledgeEvaluator

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY required")

        # Create evaluator
        config = AgenticSearchConfig()
        evaluator = LocalKnowledgeEvaluator(config)

        # Mock FastMCP context
        mock_ctx = MagicMock()

        # Mock app context with database client
        mock_db_client = MagicMock()
        mock_app_ctx = MagicMock()
        mock_app_ctx.database_client = mock_db_client

        # Mock RAG response with INVALID results (empty content, missing chunk_index)
        rag_response = json.dumps({
            "success": True,
            "results": [
                {
                    "content": "",  # ❌ Empty content
                    "url": "https://example.com/empty",
                    "similarity_score": 0.5,
                    "chunk_index": None,  # ❌ Missing chunk_index
                },
                {
                    "content": None,  # ❌ None content
                    "url": "https://example.com/none",
                    "similarity_score": 0.4,
                    "chunk_index": 0,
                },
                {
                    "content": "Valid content",  # ✓ Valid
                    "url": "https://example.com/valid",
                    "similarity_score": 0.95,
                    "chunk_index": 0,
                },
            ]
        })

        with patch("src.services.agentic_search.evaluator.get_app_context", return_value=mock_app_ctx):
            with patch("src.services.agentic_search.evaluator.perform_rag_query", return_value=rag_response):
                with patch.object(evaluator, "_evaluate_completeness") as mock_eval:
                    # Mock completeness evaluation
                    from src.services.agentic_models import CompletenessEvaluation
                    mock_eval.return_value = CompletenessEvaluation(
                        score=0.3,
                        reasoning="Test",
                        gaps=["missing info"]
                    )

                    # This should NOT raise validation error
                    # Should skip invalid results and only return valid one
                    evaluation, rag_results = await evaluator.evaluate_local_knowledge(
                        ctx=mock_ctx,
                        query="test query",
                        iteration=1,
                        search_history=[],
                    )

                    # Should have skipped invalid results and kept only valid one
                    assert len(rag_results) == 1
                    assert rag_results[0].content == "Valid content"
                    assert rag_results[0].url == "https://example.com/valid"

    def test_rag_result_validation_strict(self):
        """Test that RAGResult strictly validates empty content and None chunk_index."""
        from pydantic import ValidationError

        # Empty content should fail
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            RAGResult(
                content="",  # ❌ Empty
                url="https://example.com",
                similarity_score=0.5,
                chunk_index=0,
            )

        # None chunk_index should fail
        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            RAGResult(
                content="Valid content",
                url="https://example.com",
                similarity_score=0.5,
                chunk_index=None,  # ❌ None
            )

        # Valid result should pass
        valid_result = RAGResult(
            content="Valid content",
            url="https://example.com",
            similarity_score=0.5,
            chunk_index=0,
        )
        assert valid_result.content == "Valid content"
        assert valid_result.chunk_index == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
