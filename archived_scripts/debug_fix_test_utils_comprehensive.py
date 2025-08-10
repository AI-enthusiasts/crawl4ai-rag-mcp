#!/usr/bin/env python3
"""
Debug script to comprehensively fix all failing tests in tests/test_utils.py

Root Cause Analysis:
1. Tests still have incorrect patch paths for openai embeddings 
2. Test still expecting API to be mocked but actual API calls are being made
3. Need to patch the correct locations where openai modules are imported

Additional fixes needed:
- Patch "utils.openai.embeddings.create" paths
- Patch "utils.time.sleep" paths  
- The contextual embedding tests are actually calling the API instead of being mocked
"""

import os
import sys
import subprocess


def fix_test_utils_comprehensive():
    """Fix all the remaining failing import path and function signature issues in test_utils.py"""
    
    test_file = "/home/krashnicov/crawl4aimcp/tests/test_utils.py"
    
    # Read the file
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Define all the additional fixes needed
    additional_fixes = [
        # Fix remaining openai.embeddings.create patches
        ('@patch("utils.openai.embeddings.create")', '@patch("utils.embeddings.openai.embeddings.create")'),
        
        # Fix time.sleep patches 
        ('@patch("utils.time.sleep")', '@patch("utils.embeddings.time.sleep")'),
        
        # Fix process_chunk_with_context patches that are still wrong
        ('with patch("utils.generate_contextual_embedding") as mock_gen', 'with patch("utils.embeddings.generate_contextual_embedding") as mock_gen'),
        ('patch("utils.create_embedding") as mock_embed', 'patch("utils.embeddings.create_embedding") as mock_embed'),
        
        # Fix create_embeddings_batch patches
        ('@patch("utils.create_embeddings_batch")', '@patch("utils.embeddings.create_embeddings_batch")'),
        
        # Fix create_embedding patches  
        ('@patch("utils.create_embedding")', '@patch("utils.embeddings.create_embedding")'),
        
    ]
    
    # Apply all fixes
    print(f"Applying {len(additional_fixes)} additional fixes to {test_file}...")
    
    modified_content = content
    for old_pattern, new_pattern in additional_fixes:
        if old_pattern in modified_content:
            modified_content = modified_content.replace(old_pattern, new_pattern)
            print(f"✓ Applied fix: {old_pattern[:50]}...")
        else:
            print(f"⚠ Pattern not found: {old_pattern[:50]}...")
    
    # Write the fixed content back
    with open(test_file, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Applied all additional fixes to {test_file}")
    return True


def run_specific_tests():
    """Run specific failing tests to verify fixes"""
    print("\n🧪 Running specific tests to verify fixes...")
    
    # Change to the project directory
    os.chdir("/home/krashnicov/crawl4aimcp")
    
    failing_tests = [
        "tests/test_utils.py::TestEmbeddingFunctions::test_create_embedding_success",
        "tests/test_utils.py::TestContextualEmbedding::test_process_chunk_with_context", 
        "tests/test_utils.py::TestCodeBlockExtraction::test_generate_code_example_summary_success"
    ]
    
    for test in failing_tests:
        print(f"\n📊 Running: {test}")
        try:
            result = subprocess.run([
                "uv", "run", "pytest", test, "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                if result.stderr:
                    print("STDERR:", result.stderr[-500:])
                    
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT")
        except Exception as e:
            print(f"❌ ERROR: {e}")


def main():
    """Main debug script"""
    print("🔧 Debug Script: Comprehensive fix for test_utils.py import paths")
    print("="*80)
    
    print("📋 Additional issues identified:")
    print("1. @patch decorators for openai.embeddings.create still using incorrect paths")
    print("2. time.sleep patches need correct module paths")
    print("3. Function patches in process_chunk_with_context test need full module paths")
    print()
    
    # Apply fixes
    if fix_test_utils_comprehensive():
        # Run specific tests to verify
        run_specific_tests()
    else:
        print("\n❌ FAILED: Could not apply fixes")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())