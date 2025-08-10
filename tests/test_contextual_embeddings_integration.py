"""
Integration tests for contextual embeddings feature.

This module tests the complete contextual embeddings pipeline including:
- Full document processing with contextual embeddings enabled
- Parallel processing with ThreadPoolExecutor
- Error handling and fallback mechanisms
- Performance comparisons
- Edge cases and large document handling
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor

import pytest

# Import the functions we're testing
import sys
sys.path.insert(0, 'src')

from utils.embeddings import (
    add_documents_to_database,
    generate_contextual_embedding,
    process_chunk_with_context,
)


# Test data fixtures
SMALL_DOCUMENT = "This is a small test document with minimal content."

MEDIUM_DOCUMENT = """
This is a medium-sized document that contains multiple paragraphs of content.
It includes various topics and sections that would benefit from contextual embeddings.

Section 1: Introduction
The introduction provides an overview of the document's purpose and scope.

Section 2: Main Content
The main content discusses the key topics in detail with examples and explanations.

Section 3: Conclusion
The conclusion summarizes the key points and provides recommendations.
"""

LARGE_DOCUMENT = "Large content section. " * 5000  # Creates a document > 25000 chars

CODE_DOCUMENT = """
def calculate_fibonacci(n):
    '''Calculate the nth Fibonacci number using dynamic programming.'''
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        self.data.append(item)
        return len(self.data)
"""


class TestContextualEmbeddingsIntegration:
    """Integration tests for contextual embeddings feature."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true", "CONTEXTUAL_EMBEDDING_MAX_WORKERS": "3"})
    @patch("utils.embeddings.openai.OpenAI")
    @patch("utils.embeddings.create_embedding")
    async def test_full_pipeline_with_contextual_embeddings(self, mock_create_embedding, mock_openai_class):
        """Test the complete pipeline with contextual embeddings enabled."""
        # Setup mocks
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This chunk discusses the main topic of the document."))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Mock embedding creation
        mock_create_embedding.return_value = [0.1] * 1536
        
        # Mock database
        mock_database = AsyncMock()
        mock_database.add_documents = AsyncMock()
        
        # Test data
        urls = ["http://example.com/doc1", "http://example.com/doc2"]
        chunk_numbers = [0, 1]
        contents = ["First chunk of content", "Second chunk of content"]
        metadatas = [{"url": urls[0]}, {"url": urls[1]}]
        url_to_full_document = {
            urls[0]: MEDIUM_DOCUMENT,
            urls[1]: MEDIUM_DOCUMENT
        }
        
        # Execute
        await add_documents_to_database(
            database=mock_database,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            url_to_full_document=url_to_full_document,
            batch_size=10
        )
        
        # Verify
        assert mock_openai.chat.completions.create.call_count == 2  # Two chunks
        mock_database.add_documents.assert_called_once()
        
        # Check that contextual_embedding flag was added to metadata
        call_args = mock_database.add_documents.call_args
        assert all('contextual_embedding' in m for m in call_args.kwargs['metadatas'])

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true", "CONTEXTUAL_EMBEDDING_MAX_WORKERS": "5"})
    @patch("utils.embeddings.ThreadPoolExecutor")
    @patch("utils.embeddings.openai.OpenAI")
    @patch("utils.embeddings.create_embedding")
    async def test_parallel_processing_validation(self, mock_create_embedding, mock_openai_class, mock_executor_class):
        """Test that chunks are processed in parallel using ThreadPoolExecutor."""
        # Setup mocks
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Context for chunk"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        mock_create_embedding.return_value = [0.1] * 1536
        
        # Mock ThreadPoolExecutor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Create futures that resolve immediately
        mock_futures = []
        for i in range(5):
            future = MagicMock()
            future.result.return_value = (f"Enhanced chunk {i}", [0.1] * 1536)
            mock_futures.append(future)
        
        mock_executor.submit.side_effect = mock_futures
        
        # Mock database
        mock_database = AsyncMock()
        mock_database.add_documents = AsyncMock()
        
        # Test data with 5 chunks
        urls = [f"http://example.com/doc{i}" for i in range(5)]
        chunk_numbers = list(range(5))
        contents = [f"Chunk {i} content" for i in range(5)]
        metadatas = [{"url": url} for url in urls]
        url_to_full_document = {url: MEDIUM_DOCUMENT for url in urls}
        
        # Mock as_completed to return futures in order
        with patch("concurrent.futures.as_completed", return_value=mock_futures):
            # Execute
            await add_documents_to_database(
                database=mock_database,
                urls=urls,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
                batch_size=10
            )
        
        # Verify parallel processing
        mock_executor_class.assert_called_once_with(max_workers=5)
        assert mock_executor.submit.call_count == 5

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true"})
    @patch("utils.embeddings.openai.OpenAI")
    @patch("utils.embeddings.create_embedding")
    @patch("utils.embeddings.create_embeddings_batch")
    async def test_error_handling_fallback(self, mock_create_batch, mock_create_embedding, mock_openai_class):
        """Test graceful fallback to standard embeddings when OpenAI API fails."""
        # Setup OpenAI to fail
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        mock_openai.chat.completions.create.side_effect = Exception("API Error")
        
        # Mock standard embedding creation
        mock_create_embedding.return_value = [0.2] * 1536
        mock_create_batch.return_value = [[0.2] * 1536, [0.2] * 1536]
        
        # Mock database
        mock_database = AsyncMock()
        mock_database.add_documents = AsyncMock()
        
        # Test data
        urls = ["http://example.com/doc1", "http://example.com/doc2"]
        chunk_numbers = [0, 1]
        contents = ["First chunk", "Second chunk"]
        metadatas = [{"url": urls[0]}, {"url": urls[1]}]
        url_to_full_document = {url: SMALL_DOCUMENT for url in urls}
        
        # Execute - should not raise exception
        await add_documents_to_database(
            database=mock_database,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            url_to_full_document=url_to_full_document,
            batch_size=10
        )
        
        # Verify fallback was used
        mock_database.add_documents.assert_called_once()
        # Should have called create_embedding for fallback
        assert mock_create_embedding.call_count == 2

    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        "USE_CONTEXTUAL_EMBEDDINGS": "true",
        "CONTEXTUAL_EMBEDDING_MAX_DOC_CHARS": "1000"
    })
    @patch("utils.embeddings.openai.OpenAI")
    @patch("utils.embeddings.create_embedding")
    async def test_large_document_truncation(self, mock_create_embedding, mock_openai_class):
        """Test that large documents are truncated to max character limit."""
        # Setup mocks
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        # Capture the prompt sent to OpenAI
        captured_prompt = None
        def capture_prompt(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs['messages'][1]['content']
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content="Truncated context"))
            ]
            return response
        
        mock_openai.chat.completions.create.side_effect = capture_prompt
        mock_create_embedding.return_value = [0.1] * 1536
        
        # Mock database
        mock_database = AsyncMock()
        mock_database.add_documents = AsyncMock()
        
        # Test with large document
        urls = ["http://example.com/large"]
        chunk_numbers = [0]
        contents = ["A chunk from a large document"]
        metadatas = [{"url": urls[0]}]
        url_to_full_document = {urls[0]: LARGE_DOCUMENT}
        
        # Execute
        await add_documents_to_database(
            database=mock_database,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            url_to_full_document=url_to_full_document,
            batch_size=10
        )
        
        # Verify document was truncated in the prompt
        assert captured_prompt is not None
        assert len(captured_prompt) < len(LARGE_DOCUMENT)
        assert "Large content section" in captured_prompt  # Still contains content

    @pytest.mark.asyncio
    @patch("utils.embeddings.openai.OpenAI")
    @patch("utils.embeddings.create_embedding")
    @patch("utils.embeddings.create_embeddings_batch")
    async def test_performance_comparison(self, mock_create_batch, mock_create_embedding, mock_openai_class):
        """Compare performance between standard and contextual embeddings."""
        # Setup mocks
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Context"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        mock_create_embedding.return_value = [0.1] * 1536
        mock_create_batch.return_value = [[0.1] * 1536] * 10
        
        # Mock database
        mock_database = AsyncMock()
        mock_database.add_documents = AsyncMock()
        
        # Test data - 10 chunks
        urls = [f"http://example.com/doc{i}" for i in range(10)]
        chunk_numbers = list(range(10))
        contents = [f"Chunk {i}" for i in range(10)]
        metadatas = [{"url": url} for url in urls]
        url_to_full_document = {url: MEDIUM_DOCUMENT for url in urls}
        
        # Test with standard embeddings
        with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "false"}):
            start_standard = time.time()
            await add_documents_to_database(
                database=mock_database,
                urls=urls,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
                batch_size=10
            )
            time_standard = time.time() - start_standard
        
        # Test with contextual embeddings
        with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true"}):
            start_contextual = time.time()
            await add_documents_to_database(
                database=mock_database,
                urls=urls,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
                batch_size=10
            )
            time_contextual = time.time() - start_contextual
        
        # Verify both methods were called
        assert mock_database.add_documents.call_count == 2
        
        # Log performance (for debugging)
        print(f"Standard embeddings: {time_standard:.3f}s")
        print(f"Contextual embeddings: {time_contextual:.3f}s")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true"})
    @patch("utils.embeddings.openai.OpenAI")
    @patch("utils.embeddings.create_embedding")
    async def test_metadata_validation(self, mock_create_embedding, mock_openai_class):
        """Test that metadata is correctly updated with contextual_embedding flag."""
        # Setup mocks
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        # First call succeeds, second fails
        def side_effect(**kwargs):
            if side_effect.call_count == 1:
                response = MagicMock()
                response.choices = [
                    MagicMock(message=MagicMock(content="Success context"))
                ]
                side_effect.call_count += 1
                return response
            else:
                raise Exception("API Error")
        side_effect.call_count = 0
        
        mock_openai.chat.completions.create.side_effect = side_effect
        mock_create_embedding.return_value = [0.1] * 1536
        
        # Mock database
        mock_database = AsyncMock()
        mock_database.add_documents = AsyncMock()
        
        # Test data
        urls = ["http://example.com/doc1", "http://example.com/doc2"]
        chunk_numbers = [0, 1]
        contents = ["First chunk", "Second chunk"]
        metadatas = [{"url": urls[0]}, {"url": urls[1]}]
        url_to_full_document = {url: SMALL_DOCUMENT for url in urls}
        
        # Execute
        await add_documents_to_database(
            database=mock_database,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            url_to_full_document=url_to_full_document,
            batch_size=10
        )
        
        # Verify metadata
        call_args = mock_database.add_documents.call_args
        metadatas_result = call_args.kwargs['metadatas']
        
        # First should have contextual_embedding=True (success)
        # Second should have contextual_embedding=False (failed)
        assert len(metadatas_result) == 2
        assert 'contextual_embedding' in metadatas_result[0]
        assert 'contextual_embedding' in metadatas_result[1]

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true"})
    @patch("utils.embeddings.openai.OpenAI")
    @patch("utils.embeddings.create_embedding")
    async def test_edge_cases(self, mock_create_embedding, mock_openai_class):
        """Test edge cases like empty documents, single chunks, special characters."""
        # Setup mocks
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Edge case context"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        mock_create_embedding.return_value = [0.1] * 1536
        
        # Mock database
        mock_database = AsyncMock()
        mock_database.add_documents = AsyncMock()
        
        # Test various edge cases
        test_cases = [
            # Empty document
            ([""], {"http://example.com/empty": ""}),
            # Single word
            (["Word"], {"http://example.com/word": "Word"}),
            # Special characters
            (["Test with special chars and emojis"], {"http://example.com/special": "Full doc with special characters"}),
            # Code snippets
            ([CODE_DOCUMENT[:100]], {"http://example.com/code": CODE_DOCUMENT}),
        ]
        
        for contents, url_to_full_document in test_cases:
            urls = list(url_to_full_document.keys())
            chunk_numbers = [0]
            metadatas = [{"url": urls[0]}]
            
            # Should not raise exceptions
            await add_documents_to_database(
                database=mock_database,
                urls=urls,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
                batch_size=10
            )
        
        # Verify all edge cases were processed
        assert mock_database.add_documents.call_count == len(test_cases)


class TestContextualEmbeddingFunctions:
    """Unit tests for individual contextual embedding functions."""
    
    @patch.dict(os.environ, {
        "CONTEXTUAL_EMBEDDING_MODEL": "gpt-4",
        "CONTEXTUAL_EMBEDDING_MAX_TOKENS": "150",
        "CONTEXTUAL_EMBEDDING_TEMPERATURE": "0.5"
    })
    @patch("utils.embeddings.openai.OpenAI")
    def test_generate_contextual_embedding_with_position(self, mock_openai_class):
        """Test contextual embedding generation with chunk position information."""
        # Setup mock
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is chunk 3 of 10 discussing the middle section."))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Test
        chunk = "Middle section content"
        full_doc = MEDIUM_DOCUMENT
        result = generate_contextual_embedding(chunk, full_doc, chunk_index=2, total_chunks=10)
        
        # Verify
        assert "This is chunk 3 of 10" in result
        assert chunk in result
        assert "---" in result  # Separator between context and chunk
        
        # Check API call parameters
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == 'gpt-4'
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 150

    def test_process_chunk_with_context(self):
        """Test the process_chunk_with_context function."""
        with patch("utils.embeddings.generate_contextual_embedding") as mock_gen:
            with patch("utils.embeddings.create_embedding") as mock_embed:
                # Setup mocks
                mock_gen.return_value = "Enhanced chunk with context"
                mock_embed.return_value = [0.5] * 1536
                
                # Test
                args = ("chunk text", "full document text", 1, 5)
                result_text, result_embedding = process_chunk_with_context(args)
                
                # Verify
                assert result_text == "Enhanced chunk with context"
                assert result_embedding == [0.5] * 1536
                mock_gen.assert_called_once_with("chunk text", "full document text", 1, 5)
                mock_embed.assert_called_once_with("Enhanced chunk with context")

    @patch.dict(os.environ, {
        "CONTEXTUAL_EMBEDDING_MAX_TOKENS": "invalid",
        "CONTEXTUAL_EMBEDDING_TEMPERATURE": "3.0",
        "CONTEXTUAL_EMBEDDING_MAX_DOC_CHARS": "-100"
    })
    @patch("utils.embeddings.openai.OpenAI")
    def test_configuration_validation(self, mock_openai_class):
        """Test that invalid configuration values are handled with defaults."""
        # Setup mock
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Context with defaults"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Test - should use defaults for invalid values
        result = generate_contextual_embedding("chunk", "document")
        
        # Verify defaults were used
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs['max_tokens'] == 200  # Default
        assert call_kwargs['temperature'] == 0.3  # Default
        assert "Context with defaults" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])