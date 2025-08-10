#!/usr/bin/env python3
"""
Final debug script to fix the remaining test issues in tests/test_utils.py

Root Cause Analysis:
1. The tests are making actual API calls instead of using mocks
2. This happens because the openai client is created inside the functions and not mocked at the client level
3. Need to mock openai.OpenAI() constructor or patch at the right level

Final fixes needed:
- Mock openai.OpenAI for contextual and code summary functions
- Update test expectations to be more realistic
"""

import os
import sys
import subprocess


def apply_final_fixes():
    """Apply final fixes to ensure all tests use mocks instead of real API calls"""
    
    test_file = "/home/krashnicov/crawl4aimcp/tests/test_utils.py"
    
    # Read the file
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Final fixes to prevent real API calls
    final_fixes = [
        # Fix contextual embedding tests to mock OpenAI client
        (
            '''    @patch.dict(os.environ, {"CONTEXTUAL_EMBEDDING_MODEL": "gpt-4"})
    @patch("utils.embeddings.openai.chat.completions.create")
    def test_generate_contextual_embedding_success(self, mock_create):''',
            '''    @patch.dict(os.environ, {"CONTEXTUAL_EMBEDDING_MODEL": "gpt-4"})
    @patch("utils.embeddings.openai.OpenAI")
    def test_generate_contextual_embedding_success(self, mock_client):'''
        ),
        
        (
            '''        mock_create.return_value = mock_response

        # Test
        full_doc = "This is a full document about testing in Python"
        chunk = "Testing is important"
        result = generate_contextual_embedding(chunk, full_doc, 0, 1)

        # Verify
        assert "This chunk discusses testing" in result
        assert chunk in result
        mock_create.assert_called_once()''',
            '''        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        full_doc = "This is a full document about testing in Python"
        chunk = "Testing is important"
        result = generate_contextual_embedding(chunk, full_doc, 0, 1)

        # Verify
        assert "This chunk discusses testing" in result
        assert chunk in result
        mock_client_instance.chat.completions.create.assert_called_once()'''
        ),
        
        (
            '''    @patch("utils.embeddings.openai.chat.completions.create")
    def test_generate_contextual_embedding_error(self, mock_create):''',
            '''    @patch("utils.embeddings.openai.OpenAI")
    def test_generate_contextual_embedding_error(self, mock_client):'''
        ),
        
        (
            '''        # Mock error
        mock_create.side_effect = Exception("API Error")

        # Test
        chunk = "Test chunk"
        result = generate_contextual_embedding(chunk, "Full doc", 0, 1)

        # Verify
        assert result == chunk  # Returns original chunk on error  # Returns original chunk on error''',
            '''        # Mock error
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

        # Test
        chunk = "Test chunk"
        result = generate_contextual_embedding(chunk, "Full doc", 0, 1)

        # Verify
        assert result == chunk  # Returns original chunk on error  # Returns original chunk on error'''
        ),
        
        # Fix code summary tests to mock OpenAI client
        (
            '''    @patch("utils.code_analysis.openai.chat.completions.create")
    def test_generate_code_example_summary_success(self, mock_create):''',
            '''    @patch("utils.code_analysis.openai.OpenAI")
    def test_generate_code_example_summary_success(self, mock_client):'''
        ),
        
        (
            '''        mock_create.return_value = mock_response

        # Test
        summary = generate_code_example_summary(
            code="def test(): pass",
            context_before="Before",
            context_after="After",
        )

        assert summary == "A test function example"''',
            '''        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        summary = generate_code_example_summary(
            code="def test(): pass",
            context_before="Before",
            context_after="After",
        )

        assert summary == "A test function example"'''
        ),
        
        (
            '''    @patch("utils.code_analysis.openai.chat.completions.create")
    def test_generate_code_example_summary_error(self, mock_create):''',
            '''    @patch("utils.code_analysis.openai.OpenAI")
    def test_generate_code_example_summary_error(self, mock_client):'''
        ),
        
        (
            '''        # Mock error
        mock_create.side_effect = Exception("API Error")

        # Test
        summary = generate_code_example_summary("code", "before", "after")

        assert summary == "Code example for demonstration purposes."''',
            '''        # Mock error
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

        # Test
        summary = generate_code_example_summary("code", "before", "after")

        assert summary == "Code example for demonstration purposes."'''
        ),
        
        # Fix source summary tests
        (
            '''    @patch("utils.summarization.openai.chat.completions.create")
    def test_extract_source_summary_success(self, mock_create):''',
            '''    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_success(self, mock_client):'''
        ),
        
        (
            '''        mock_create.return_value = mock_response

        # Test
        summary = extract_source_summary(
            "pytest.org",
            "Content about pytest testing framework",
        )

        assert summary == "This is a testing library"''',
            '''        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        summary = extract_source_summary(
            "pytest.org",
            "Content about pytest testing framework",
        )

        assert summary == "This is a testing library"'''
        ),
        
        # Fix remaining tests similarly
        (
            '''    @patch("utils.summarization.openai.chat.completions.create")
    def test_extract_source_summary_error(self, mock_create):''',
            '''    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_error(self, mock_client):'''
        ),
        
        (
            '''        # Mock error
        mock_create.side_effect = Exception("API Error")

        # Test
        summary = extract_source_summary("example.com", "Some content")

        assert summary == "Content from example.com"''',
            '''        # Mock error
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

        # Test
        summary = extract_source_summary("example.com", "Some content")

        assert summary == "Content from example.com"'''
        ),
        
        (
            '''    @patch("utils.summarization.openai.chat.completions.create")
    def test_extract_source_summary_max_length(self, mock_create):''',
            '''    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_max_length(self, mock_client):'''
        ),
        
        (
            '''        mock_create.return_value = mock_response

        # Test
        summary = extract_source_summary("example.com", "content", max_length=500)

        assert len(summary) == 503  # 500 + "..."
        assert summary.endswith("...")''',
            '''        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        summary = extract_source_summary("example.com", "content", max_length=500)

        assert len(summary) == 503  # 500 + "..."
        assert summary.endswith("...")'''
        ),
        
        # Fix remaining contextual embedding tests
        (
            '''    @patch("utils.embeddings.openai.chat.completions.create")
    def test_generate_contextual_embedding_token_limit(self, mock_create):''',
            '''    @patch("utils.embeddings.openai.OpenAI")
    def test_generate_contextual_embedding_token_limit(self, mock_client):'''
        ),
        
        (
            '''        mock_create.return_value = mock_response

        # Test with very long document (should be truncated to 25000 chars)
        very_long_document = "A" * 30000
        chunk = "This is a test chunk"

        result = generate_contextual_embedding(chunk, very_long_document)

        # Verify the document was truncated in the prompt
        assert "Context for chunk" in result
        call_args = mock_create.call_args[1]  # Use keyword arguments
        prompt_content = call_args["messages"][1]["content"]
        # Check that document section is within the 25000 character limit
        document_section = (
            prompt_content.split("</document>")[0].split("<document>")[1].strip()
        )
        assert len(document_section) == 25000''',
            '''        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test with very long document (should be truncated to 25000 chars)
        very_long_document = "A" * 30000
        chunk = "This is a test chunk"

        result = generate_contextual_embedding(chunk, very_long_document)

        # Verify the document was truncated in the prompt
        assert "Context for chunk" in result
        call_args = mock_client_instance.chat.completions.create.call_args[1]  # Use keyword arguments
        prompt_content = call_args["messages"][1]["content"]
        # Check that document section is within the 25000 character limit
        document_section = (
            prompt_content.split("</document>")[0].split("<document>")[1].strip()
        )
        assert len(document_section) == 25000'''
        ),
        
        (
            '''    @patch("utils.code_analysis.openai.chat.completions.create")
    def test_generate_code_example_summary_truncation(self, mock_create):''',
            '''    @patch("utils.code_analysis.openai.OpenAI")
    def test_generate_code_example_summary_truncation(self, mock_client):'''
        ),
        
        (
            '''        mock_create.return_value = mock_response

        # Test with very long inputs that should be truncated
        long_code = "def function():\\n" + "    # comment\\n" * 1000  # Very long code
        long_context_before = "B" * 1000  # Long context
        long_context_after = "A" * 1000  # Long context

        summary = generate_code_example_summary(
            long_code,
            long_context_before,
            long_context_after,
        )

        assert summary == "Summary of code"

        # Verify truncation was applied in the prompt
        call_args = mock_create.call_args[1]  # Use keyword arguments
        prompt_content = call_args["messages"][1]["content"]

        # Check that code was truncated to 1500 chars
        code_section = (
            prompt_content.split("</code_example>")[0]
            .split("<code_example>")[1]
            .strip()
        )
        assert len(code_section) <= 1500

        # Check that context sections were truncated to 500 chars
        context_before_section = (
            prompt_content.split("</context_before>")[0]
            .split("<context_before>")[1]
            .strip()
        )
        assert len(context_before_section) <= 500

        context_after_section = (
            prompt_content.split("</context_after>")[0]
            .split("<context_after>")[1]
            .strip()
        )
        assert len(context_after_section) <= 500''',
            '''        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test with very long inputs that should be truncated
        long_code = "def function():\\n" + "    # comment\\n" * 1000  # Very long code
        long_context_before = "B" * 1000  # Long context
        long_context_after = "A" * 1000  # Long context

        summary = generate_code_example_summary(
            long_code,
            long_context_before,
            long_context_after,
        )

        assert summary == "Summary of code"

        # Verify truncation was applied in the prompt
        call_args = mock_client_instance.chat.completions.create.call_args[1]  # Use keyword arguments
        prompt_content = call_args["messages"][1]["content"]

        # Check that code was truncated to 1500 chars
        code_section = (
            prompt_content.split("</code_example>")[0]
            .split("<code_example>")[1]
            .strip()
        )
        assert len(code_section) <= 1500

        # Check that context sections were truncated to 500 chars
        context_before_section = (
            prompt_content.split("</context_before>")[0]
            .split("<context_before>")[1]
            .strip()
        )
        assert len(context_before_section) <= 500

        context_after_section = (
            prompt_content.split("</context_after>")[0]
            .split("<context_after>")[1]
            .strip()
        )
        assert len(context_after_section) <= 500'''
        ),
    ]
    
    # Apply all final fixes
    print(f"Applying {len(final_fixes)} final fixes to {test_file}...")
    
    modified_content = content
    for old_pattern, new_pattern in final_fixes:
        if old_pattern in modified_content:
            modified_content = modified_content.replace(old_pattern, new_pattern)
            print(f"✓ Applied fix for pattern starting with: {old_pattern[:30]}...")
        else:
            print(f"⚠ Pattern not found for: {old_pattern[:30]}...")
    
    # Write the fixed content back
    with open(test_file, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Applied all final fixes to {test_file}")
    return True


def run_final_test():
    """Run all tests to verify final fixes"""
    print("\n🧪 Running full test suite to verify all fixes...")
    
    # Change to the project directory
    os.chdir("/home/krashnicov/crawl4aimcp")
    
    try:
        # Run all tests in the test_utils.py file
        result = subprocess.run([
            "uv", "run", "pytest", "tests/test_utils.py", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=180)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print(f"❌ Some tests still failing with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 180 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main debug script"""
    print("🔧 Final Debug Script: Fix OpenAI client mocking in test_utils.py")
    print("="*80)
    
    print("📋 Final issues identified:")
    print("1. Tests are making real API calls instead of using mocks")
    print("2. Need to mock openai.OpenAI() client constructor instead of specific methods") 
    print("3. Update test assertions to use client instance mocks")
    print()
    
    # Apply final fixes
    if apply_final_fixes():
        # Run all tests to verify
        if run_final_test():
            print("\n🎉 SUCCESS: All tests are now passing!")
        else:
            print("\n⚠️  Some tests may still be failing. Check output above for details.")
    else:
        print("\n❌ FAILED: Could not apply fixes")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())