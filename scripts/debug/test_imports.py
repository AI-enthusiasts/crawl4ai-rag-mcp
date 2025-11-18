#!/usr/bin/env python3
"""Test that the imports are working correctly."""

import sys

sys.path.insert(0, "src")

try:
    # Test the problematic imports
    print("Testing imports...")

    # Test 1: Import from services.crawling
    from services.crawling import crawl_batch, process_urls_for_mcp  # noqa: F401
    print("✓ Successfully imported from services.crawling")

    # Test 2: Import from services.validated_search
    from services.validated_search import ValidatedCodeSearchService  # noqa: F401
    print("✓ Successfully imported from services.validated_search")

    # Test 3: Import from utils
    from utils import add_documents_to_database, create_embeddings_batch  # noqa: F401
    print("✓ Successfully imported functions from utils package")

    # Test 4: Import tools module
    from tools import register_tools  # noqa: F401
    print("✓ Successfully imported tools module")

    print("\n✅ All imports successful! The import issues have been resolved.")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
