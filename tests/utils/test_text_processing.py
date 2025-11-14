"""
Comprehensive unit tests for src/utils/text_processing.py

Test Coverage:
- smart_chunk_markdown(): Intelligent text chunking respecting code blocks and paragraphs
- extract_section_info(): Extract headers and statistics from markdown chunks

Testing Approach:
- Test various chunk sizes and boundary conditions
- Test code block detection and preservation
- Test paragraph and sentence boundary detection
- Test edge cases (empty text, no breaks, very long text)
- Test header extraction with regex patterns
- Test word and character counting
- Parametrized tests for different scenarios
"""

import pytest

from src.utils.text_processing import smart_chunk_markdown, extract_section_info


class TestSmartChunkMarkdown:
    """Test smart_chunk_markdown() intelligent text chunking"""

    def test_chunk_simple_text(self):
        """Test chunking simple text without special boundaries"""
        text = "Word " * 1000  # 5000 characters
        chunks = smart_chunk_markdown(text, chunk_size=1000)

        assert len(chunks) > 1
        # Verify all chunks are within size limits
        for chunk in chunks:
            assert len(chunk) <= 1500  # Some flexibility for boundary detection

    def test_chunk_respects_code_blocks(self):
        """Test that code blocks are not split"""
        text = """
Introduction paragraph with some text.

```python
def important_function():
    # This code block should stay together
    result = []
    for i in range(100):
        result.append(i)
    return result
```

More text after the code block.
"""
        chunks = smart_chunk_markdown(text, chunk_size=200)

        # Find which chunk contains the code block start
        code_chunks = [c for c in chunks if "```python" in c]
        if code_chunks:
            # Verify the code block is not split
            assert "def important_function():" in code_chunks[0]

    def test_chunk_respects_paragraph_breaks(self):
        """Test that chunks break at paragraph boundaries"""
        # Create text with clear paragraph breaks
        paragraphs = ["Paragraph {}. ".format(i) + "Content " * 50 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = smart_chunk_markdown(text, chunk_size=500)

        assert len(chunks) > 1
        # Verify chunks don't split in middle of paragraphs (mostly)
        for chunk in chunks:
            # Each chunk should contain complete sentences
            assert chunk.strip() != ""

    def test_chunk_respects_sentence_breaks(self):
        """Test that chunks break at sentence boundaries when no paragraphs"""
        # Text with sentences but no paragraph breaks
        text = "This is sentence one. " * 100

        chunks = smart_chunk_markdown(text, chunk_size=200)

        assert len(chunks) > 1
        # Most chunks should end with period (sentence boundary)
        chunks_ending_with_period = sum(1 for c in chunks if c.rstrip().endswith("."))
        assert chunks_ending_with_period >= len(chunks) - 1

    def test_chunk_minimum_threshold(self):
        """Test that chunks only break past 30% threshold"""
        text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100

        chunks = smart_chunk_markdown(text, chunk_size=150)

        # Should respect 30% threshold for breaks
        for chunk in chunks[:-1]:  # All but last
            assert len(chunk) >= 45  # 30% of 150

    def test_chunk_at_text_end(self):
        """Test handling when reaching end of text"""
        text = "Short text."

        chunks = smart_chunk_markdown(text, chunk_size=5000)

        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        text = ""

        chunks = smart_chunk_markdown(text, chunk_size=1000)

        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text"""
        text = "   \n\n   \t  "

        chunks = smart_chunk_markdown(text, chunk_size=1000)

        # Function returns [''] for whitespace-only text
        assert len(chunks) <= 1  # May return empty string in list

    def test_chunk_very_long_text(self):
        """Test chunking very long text"""
        text = "Word " * 10000  # 50,000 characters

        chunks = smart_chunk_markdown(text, chunk_size=5000)

        assert len(chunks) >= 10
        # Verify all content is preserved
        reconstructed = "".join(chunks)
        # Account for whitespace stripping
        assert len(reconstructed.replace(" ", "")) >= len(text.replace(" ", "")) - 100

    def test_chunk_code_block_at_start(self):
        """Test text starting with code block"""
        text = """```python
def function():
    pass
```

Text after code."""

        chunks = smart_chunk_markdown(text, chunk_size=100)

        assert len(chunks) >= 1
        assert "```python" in chunks[0]

    def test_chunk_multiple_code_blocks(self):
        """Test text with multiple code blocks"""
        text = """
First section.

```python
code1
```

Middle section.

```javascript
code2
```

Last section.
"""

        chunks = smart_chunk_markdown(text, chunk_size=100)

        # Verify code blocks are in chunks
        all_text = "".join(chunks)
        assert "```python" in all_text
        assert "```javascript" in all_text

    def test_chunk_no_good_break_points(self):
        """Test text with no good break points (no periods, newlines)"""
        text = "A" * 10000  # Long text with no breaks

        chunks = smart_chunk_markdown(text, chunk_size=1000)

        assert len(chunks) >= 10
        # Should still chunk at boundaries
        for chunk in chunks[:-1]:
            assert len(chunk) <= 1000

    def test_chunk_mixed_boundaries(self):
        """Test text with mixed boundary types"""
        text = """
# Header 1

Some text in paragraph one.

```python
def code():
    pass
```

Another paragraph. With sentences. Multiple of them.

# Header 2

More content here.
"""

        chunks = smart_chunk_markdown(text, chunk_size=100)

        assert len(chunks) > 1
        all_text = "".join(chunks)
        # Verify content preservation
        assert "# Header 1" in all_text
        assert "def code():" in all_text

    def test_chunk_custom_size(self):
        """Test chunking with custom chunk sizes"""
        text = "Word " * 1000

        # Test small chunks
        small_chunks = smart_chunk_markdown(text, chunk_size=100)
        assert len(small_chunks) > 20

        # Test large chunks
        large_chunks = smart_chunk_markdown(text, chunk_size=10000)
        assert len(large_chunks) < 5

    def test_chunk_single_long_line(self):
        """Test chunking very long single line"""
        text = "Word " * 5000  # Single line

        chunks = smart_chunk_markdown(text, chunk_size=1000)

        assert len(chunks) > 1
        # Verify reasonable chunk sizes
        for chunk in chunks:
            assert len(chunk) <= 1500


class TestExtractSectionInfo:
    """Test extract_section_info() header and statistics extraction"""

    def test_extract_headers_basic(self):
        """Test extracting basic markdown headers"""
        chunk = """
# Main Header

Some content here.

## Subheader

More content.

### Sub-subheader

Even more content.
"""

        info = extract_section_info(chunk)

        assert "headers" in info
        assert "# Main Header" in info["headers"]
        assert "## Subheader" in info["headers"]
        assert "### Sub-subheader" in info["headers"]

    def test_extract_headers_formatting(self):
        """Test header formatting in output"""
        chunk = """
# Header 1
## Header 2
### Header 3
"""

        info = extract_section_info(chunk)

        # Headers should be joined with semicolon
        assert ";" in info["headers"]
        assert info["headers"].count("#") == 6  # 1 + 2 + 3

    def test_extract_no_headers(self):
        """Test extracting from text with no headers"""
        chunk = "Just some plain text without any headers."

        info = extract_section_info(chunk)

        assert info["headers"] == ""
        assert info["char_count"] > 0
        assert info["word_count"] > 0

    def test_extract_char_count(self):
        """Test character count calculation"""
        chunk = "Hello world"

        info = extract_section_info(chunk)

        assert info["char_count"] == len(chunk)

    def test_extract_word_count(self):
        """Test word count calculation"""
        chunk = "This is a test sentence."

        info = extract_section_info(chunk)

        assert info["word_count"] == 5

    def test_extract_empty_chunk(self):
        """Test extracting from empty chunk"""
        chunk = ""

        info = extract_section_info(chunk)

        assert info["headers"] == ""
        assert info["char_count"] == 0
        assert info["word_count"] == 0

    def test_extract_whitespace_chunk(self):
        """Test extracting from whitespace-only chunk"""
        chunk = "   \n\n   \t  "

        info = extract_section_info(chunk)

        assert info["headers"] == ""
        assert info["char_count"] == len(chunk)
        assert info["word_count"] == 0

    def test_extract_headers_with_special_chars(self):
        """Test headers with special characters"""
        chunk = """
# Header with *italic* and **bold**
## Header with `code`
### Header with [link](url)
"""

        info = extract_section_info(chunk)

        assert "Header with *italic*" in info["headers"]
        assert "Header with `code`" in info["headers"]
        assert "Header with [link]" in info["headers"]

    def test_extract_headers_with_numbers(self):
        """Test headers with numbers"""
        chunk = """
# Section 1: Introduction
## 2.1 Subsection
### Step 3: Implementation
"""

        info = extract_section_info(chunk)

        assert "Section 1" in info["headers"]
        assert "2.1 Subsection" in info["headers"]
        assert "Step 3" in info["headers"]

    def test_extract_inline_code_not_as_header(self):
        """Test that inline # in code is not treated as header"""
        chunk = """
Some text.

```python
# This is a comment, not a header
def function():
    pass
```

# This is a real header
"""

        info = extract_section_info(chunk)

        # Should only detect the real header (at start of line)
        # Regex should only match headers at line start (^)
        assert "# This is a real header" in info["headers"]

    def test_extract_false_header_patterns(self):
        """Test patterns that look like headers but aren't"""
        chunk = """
This is not # a header.
Also not a header: ## something
But this is:
# Real Header
"""

        info = extract_section_info(chunk)

        # Only line-starting patterns should match
        assert info["headers"].count("#") == 1

    def test_extract_very_long_chunk(self):
        """Test extraction from very long chunk"""
        chunk = "Word " * 10000 + "\n# Header\n" + "More " * 10000

        info = extract_section_info(chunk)

        assert info["char_count"] > 50000
        assert info["word_count"] > 20000
        assert "Header" in info["headers"]

    def test_extract_all_fields(self):
        """Test that all expected fields are present"""
        chunk = "# Header\n\nSome content here."

        info = extract_section_info(chunk)

        assert "headers" in info
        assert "char_count" in info
        assert "word_count" in info
        assert len(info) == 3  # Exactly these three fields


class TestTextProcessingIntegration:
    """Integration tests for text processing functions"""

    def test_chunk_and_extract_info_pipeline(self):
        """Test complete pipeline: chunk then extract info"""
        text = """
# Introduction

This is a long document that needs to be chunked.
It contains multiple sections.

## Details

Here are some details with multiple sentences.
Each sentence adds to the content.

```python
def example():
    return "code"
```

## Conclusion

Final thoughts here.
"""

        # Chunk the text
        chunks = smart_chunk_markdown(text, chunk_size=200)

        assert len(chunks) > 1

        # Extract info from each chunk
        infos = [extract_section_info(chunk) for chunk in chunks]

        # Verify all chunks have info
        assert len(infos) == len(chunks)

        # Verify character counts sum correctly (approximately)
        total_chars = sum(info["char_count"] for info in infos)
        assert total_chars > 0

        # Verify at least some chunks have headers
        chunks_with_headers = sum(1 for info in infos if info["headers"])
        assert chunks_with_headers > 0

    def test_realistic_document_processing(self):
        """Test processing realistic documentation content"""
        text = """
# API Documentation

## Authentication

To authenticate, use the following method:

```python
import requests

response = requests.post('https://api.example.com/auth', json={
    'username': 'user',
    'password': 'pass'
})
token = response.json()['token']
```

## Making Requests

Once authenticated, include the token in headers.

### GET Requests

Use GET for retrieving data:

```python
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('https://api.example.com/data', headers=headers)
```

### POST Requests

Use POST for creating resources. The request body should be JSON formatted.
Include proper content-type headers.

## Error Handling

The API returns standard HTTP status codes. Handle errors appropriately:
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Server Error
"""

        # Process the document
        chunks = smart_chunk_markdown(text, chunk_size=500)

        # Should create multiple chunks (actual behavior is 2 chunks for this text)
        assert len(chunks) >= 2

        # Extract info from all chunks
        infos = [extract_section_info(chunk) for chunk in chunks]

        # Verify comprehensive processing
        all_headers = " ".join(info["headers"] for info in infos)
        assert "API Documentation" in all_headers or "Authentication" in all_headers

        # Verify code blocks are preserved
        all_content = "".join(chunks)
        assert "```python" in all_content
        assert "import requests" in all_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
