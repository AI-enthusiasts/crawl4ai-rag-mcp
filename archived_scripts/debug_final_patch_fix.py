#!/usr/bin/env python3
"""
Final targeted fix for the last 5 failing tests in tests/test_utils.py

Issues remaining:
1. One more concurrent.futures patch path in line 330
2. Source summary tests are hitting real API instead of mocks

Final fixes needed:
- Fix remaining concurrent.futures patch
- The source summary tests are still making real API calls despite patching
"""

import os
import sys
import subprocess


def apply_final_targeted_fixes():
    """Apply targeted fixes for the last 5 failing tests"""
    
    test_file = "/home/krashnicov/crawl4aimcp/tests/test_utils.py"
    
    # Read the file
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Final targeted fixes
    targeted_fixes = [
        # Fix the remaining concurrent.futures patch
        (
            '"utils.concurrent.futures.ThreadPoolExecutor"',
            '"utils.embeddings.concurrent.futures.ThreadPoolExecutor"'
        ),
        
        # The source summary tests are still hitting real API, need proper environment setup
        (
            '''    @patch.dict(os.environ, {"MODEL_CHOICE": "gpt-4"})
    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_success(self, mock_client):
        """Test successful source summary extraction"""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is a testing library")),
        ]
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        summary = extract_source_summary(
            "pytest.org",
            "Content about pytest testing framework",
        )

        assert summary == "This is a testing library"''',
            '''    @patch.dict(os.environ, {"MODEL_CHOICE": "gpt-4", "OPENAI_API_KEY": "test-key"})
    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_success(self, mock_client):
        """Test successful source summary extraction"""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is a testing library")),
        ]
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        summary = extract_source_summary(
            "pytest.org",
            "Content about pytest testing framework",
        )

        assert summary == "This is a testing library"'''
        ),
        
        (
            '''    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_error(self, mock_client):
        """Test source summary with API error"""
        # Mock error
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

        # Test
        summary = extract_source_summary("example.com", "Some content")

        assert summary == "Content from example.com"''',
            '''    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_error(self, mock_client):
        """Test source summary with API error"""
        # Mock error
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

        # Test
        summary = extract_source_summary("example.com", "Some content")

        assert summary == "Content from example.com"'''
        ),
        
        (
            '''    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_max_length(self, mock_client):
        """Test source summary respects max length"""
        # Mock very long response
        long_summary = "A" * 600
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=long_summary))]
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        summary = extract_source_summary("example.com", "content", max_length=500)

        assert len(summary) == 503  # 500 + "..."
        assert summary.endswith("...")''',
            '''    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("utils.summarization.openai.OpenAI")
    def test_extract_source_summary_max_length(self, mock_client):
        """Test source summary respects max length"""
        # Mock very long response
        long_summary = "A" * 600
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=long_summary))]
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test
        summary = extract_source_summary("example.com", "content", max_length=500)

        assert len(summary) == 503  # 500 + "..."
        assert summary.endswith("...")'''
        ),
    ]
    
    # Apply targeted fixes
    print(f"Applying {len(targeted_fixes)} targeted fixes...")
    
    modified_content = content
    for old_pattern, new_pattern in targeted_fixes:
        if old_pattern in modified_content:
            modified_content = modified_content.replace(old_pattern, new_pattern)
            print(f"✓ Applied fix for: {old_pattern[:50]}...")
        else:
            print(f"⚠ Pattern not found for: {old_pattern[:50]}...")
    
    # Write the fixed content back
    with open(test_file, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Applied all targeted fixes")
    return True


def run_final_test_check():
    """Run tests to check the final result"""
    print("\n🧪 Running final test check...")
    
    # Change to the project directory
    os.chdir("/home/krashnicov/crawl4aimcp")
    
    try:
        # Run the specific failing tests first
        failing_tests = [
            "tests/test_utils.py::TestDocumentOperations::test_add_documents_contextual_processing_error",
            "tests/test_utils.py::TestSourceSummary::test_extract_source_summary_success"
        ]
        
        for test in failing_tests:
            print(f"\n📊 Testing: {test.split('::')[-1]}")
            result = subprocess.run([
                "uv", "run", "pytest", test, "-v"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                print(f"Error: {result.stdout.split('FAILED')[-1][:200]}...")
        
        print("\n🔬 Running full test suite...")
        # Run all tests
        result = subprocess.run([
            "uv", "run", "pytest", "tests/test_utils.py", "-q"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("🎉 SUCCESS: All tests are now passing!")
            return True
        else:
            # Parse the summary line
            lines = result.stdout.split('\n')
            summary = [line for line in lines if 'failed' in line and 'passed' in line and '=' in line]
            if summary:
                print(f"📊 Result: {summary[-1]}")
            else:
                print(f"❌ Tests still failing (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main script"""
    print("🎯 Final Targeted Fix: Last 5 failing tests")
    print("="*60)
    
    # Apply targeted fixes
    if apply_final_targeted_fixes():
        # Test the results
        if run_final_test_check():
            print("\n🏆 MISSION ACCOMPLISHED! All tests are now passing.")
            
            # Clean up debug scripts
            print("\n🧹 Moving debug scripts to archived_scripts...")
            archive_dir = "/home/krashnicov/crawl4aimcp/archived_scripts"
            os.makedirs(archive_dir, exist_ok=True)
            
            # Move all debug scripts
            for script in os.listdir("/home/krashnicov/crawl4aimcp"):
                if script.startswith("debug_") and script.endswith(".py"):
                    src = f"/home/krashnicov/crawl4aimcp/{script}"
                    dst = f"{archive_dir}/{script}"
                    os.rename(src, dst)
                    print(f"   ✓ Moved {script}")
            
            print("\n✨ All test fixes complete and debug scripts archived!")
        else:
            print("\n⚠️ Some tests may still need attention.")
    
    return 0


if __name__ == "__main__":
    exit(main())