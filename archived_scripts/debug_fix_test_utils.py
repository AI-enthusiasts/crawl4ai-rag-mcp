#!/usr/bin/env python3
"""
Debug script to fix all failing tests in tests/test_utils.py

Root Cause Analysis:
1. Tests are patching incorrect import paths - trying to patch 'utils.openai.chat.completions' 
   but should patch the correct module paths based on where openai is actually imported
2. test_process_chunk_with_context has wrong function signature and expected return values

Issues to Fix:
- Lines with @patch("utils.openai.chat.completions.create") need to be updated to patch the correct module
- test_process_chunk_with_context needs signature and return value fixes
"""

import os
import sys
import subprocess


def fix_test_utils():
    """Fix all the failing import path and function signature issues in test_utils.py"""
    
    test_file = "/home/krashnicov/crawl4aimcp/tests/test_utils.py"
    
    # Read the file
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Define all the fixes needed
    fixes = [
        # Fix 1: Line 166 - contextual embedding test
        ('    @patch("utils.openai.chat.completions.create")\n    def test_generate_contextual_embedding_success(self, mock_create):',
         '    @patch("utils.embeddings.openai.chat.completions.create")\n    def test_generate_contextual_embedding_success(self, mock_create):'),
        
        # Fix 2: Line 186 - contextual embedding error test  
        ('    @patch("utils.openai.chat.completions.create")\n    def test_generate_contextual_embedding_error(self, mock_create):',
         '    @patch("utils.embeddings.openai.chat.completions.create")\n    def test_generate_contextual_embedding_error(self, mock_create):'),
        
        # Fix 3: Lines 201-208 - Fix process_chunk_with_context test
        ('    def test_process_chunk_with_context(self):\n        """Test chunk processing helper function"""\n        with patch("utils.generate_contextual_embedding") as mock_gen:\n            mock_gen.return_value = ("Enhanced chunk", True)\n\n            args = ("http://example.com", "chunk content", "full document")\n            result = process_chunk_with_context(args)\n\n            assert result == ("Enhanced chunk", True)\n            mock_gen.assert_called_once_with("full document", "chunk content")',
         '    def test_process_chunk_with_context(self):\n        """Test chunk processing helper function"""\n        with patch("utils.generate_contextual_embedding") as mock_gen, \\\n             patch("utils.create_embedding") as mock_embed:\n            mock_gen.return_value = "Enhanced chunk"\n            mock_embed.return_value = [0.1] * 1536\n\n            args = ("chunk content", "full document", 0, 1)\n            result = process_chunk_with_context(args)\n\n            assert result == ("Enhanced chunk", [0.1] * 1536)\n            mock_gen.assert_called_once_with("chunk content", "full document", 0, 1)'),
        
        # Fix 4: Line 578 - code example summary test
        ('    @patch("utils.openai.chat.completions.create")\n    def test_generate_code_example_summary_success(self, mock_create):',
         '    @patch("utils.code_analysis.openai.chat.completions.create")\n    def test_generate_code_example_summary_success(self, mock_create):'),
        
        # Fix 5: Line 597 - code example summary error test
        ('    @patch("utils.openai.chat.completions.create")\n    def test_generate_code_example_summary_error(self, mock_create):',
         '    @patch("utils.code_analysis.openai.chat.completions.create")\n    def test_generate_code_example_summary_error(self, mock_create):'),
        
        # Fix 6: Line 689 - source summary success test
        ('    @patch("utils.openai.chat.completions.create")\n    def test_extract_source_summary_success(self, mock_create):',
         '    @patch("utils.summarization.openai.chat.completions.create")\n    def test_extract_source_summary_success(self, mock_create):'),
        
        # Fix 7: Line 712 - source summary error test  
        ('    @patch("utils.openai.chat.completions.create")\n    def test_extract_source_summary_error(self, mock_create):',
         '    @patch("utils.summarization.openai.chat.completions.create")\n    def test_extract_source_summary_error(self, mock_create):'),
        
        # Fix 8: Line 723 - source summary max length test
        ('    @patch("utils.openai.chat.completions.create")\n    def test_extract_source_summary_max_length(self, mock_create):',
         '    @patch("utils.summarization.openai.chat.completions.create")\n    def test_extract_source_summary_max_length(self, mock_create):'),
        
        # Fix 9: Line 836 - contextual embedding token limit test
        ('    @patch("utils.openai.chat.completions.create")\n    def test_generate_contextual_embedding_token_limit(self, mock_create):',
         '    @patch("utils.embeddings.openai.chat.completions.create")\n    def test_generate_contextual_embedding_token_limit(self, mock_create):'),
        
        # Fix 10: Line 862 - code example summary truncation test
        ('    @patch("utils.openai.chat.completions.create")\n    def test_generate_code_example_summary_truncation(self, mock_create):',
         '    @patch("utils.code_analysis.openai.chat.completions.create")\n    def test_generate_code_example_summary_truncation(self, mock_create):'),
    ]
    
    # Apply all fixes
    print(f"Applying {len(fixes)} fixes to {test_file}...")
    
    modified_content = content
    for old_pattern, new_pattern in fixes:
        if old_pattern in modified_content:
            modified_content = modified_content.replace(old_pattern, new_pattern)
            print(f"✓ Applied fix: {old_pattern[:50]}...")
        else:
            print(f"⚠ Pattern not found: {old_pattern[:50]}...")
    
    # Write the fixed content back
    with open(test_file, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Applied all fixes to {test_file}")
    return True


def run_tests():
    """Run the tests to verify fixes"""
    print("\n🧪 Running tests to verify fixes...")
    
    # Change to the project directory
    os.chdir("/home/krashnicov/crawl4aimcp")
    
    try:
        # Run only the specific test file
        result = subprocess.run([
            "uv", "run", "pytest", "tests/test_utils.py", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print(f"❌ Tests failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main debug script"""
    print("🔧 Debug Script: Fixing test_utils.py import paths and function signatures")
    print("="*80)
    
    print("📋 Issues identified:")
    print("1. @patch decorators using incorrect import paths (utils.openai.* instead of utils.module.openai.*)")
    print("2. test_process_chunk_with_context has wrong function signature and expected return values")
    print()
    
    # Apply fixes
    if fix_test_utils():
        # Run tests to verify
        if run_tests():
            print("\n🎉 SUCCESS: All fixes applied and tests are now passing!")
        else:
            print("\n⚠️  Fixes applied but some tests may still be failing. Check output above.")
    else:
        print("\n❌ FAILED: Could not apply fixes")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())