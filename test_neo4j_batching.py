#\!/usr/bin/env python3
"""
Test script to verify Neo4j transaction batching functionality.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def test_batch_configuration():
    """Test that batch configuration is properly loaded"""
    # Set custom batch size via environment
    os.environ["NEO4J_BATCH_SIZE"] = "25"
    os.environ["NEO4J_BATCH_TIMEOUT"] = "60"

    # Initialize extractor
    extractor = DirectNeo4jExtractor(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    # Verify configuration
    assert extractor.batch_size == 25, f"Expected batch_size=25, got {extractor.batch_size}"
    assert extractor.batch_timeout_seconds == 60, f"Expected timeout=60, got {extractor.batch_timeout_seconds}"

    logger.info("✓ Batch configuration test passed")
    logger.info(f"  - Batch size: {extractor.batch_size}")
    logger.info(f"  - Batch timeout: {extractor.batch_timeout_seconds} seconds")

    return extractor

async def test_batch_processing_methods():
    """Test that batch processing methods exist and are callable"""
    extractor = await test_batch_configuration()

    # Check if new methods exist
    assert hasattr(extractor, "_process_modules_in_batches"), "Missing _process_modules_in_batches method"
    assert hasattr(extractor, "_process_batch_transaction"), "Missing _process_batch_transaction method"
    assert callable(extractor._process_modules_in_batches), "_process_modules_in_batches is not callable"
    assert callable(extractor._process_batch_transaction), "_process_batch_transaction is not callable"

    logger.info("✓ Batch processing methods test passed")
    logger.info("  - _process_modules_in_batches: Found")
    logger.info("  - _process_batch_transaction: Found")

async def test_create_graph_uses_batching():
    """Test that _create_graph method uses batching"""
    extractor = await test_batch_configuration()

    # Read the source to verify batching is used in _create_graph
    import inspect
    source = inspect.getsource(extractor._create_graph)

    # Check if batching method is called
    assert "_process_modules_in_batches" in source, "_create_graph doesn't use batching"

    logger.info("✓ _create_graph uses batching test passed")
    logger.info("  - _create_graph calls _process_modules_in_batches")

async def simulate_batch_processing():
    """Simulate batch processing with mock data"""
    logger.info("\n--- Simulating batch processing ---")

    # Create mock modules data
    mock_modules = []
    for i in range(120):  # Create 120 mock modules
        mock_modules.append({
            "file_path": f"src/module_{i}.py",
            "module_name": f"module_{i}",
            "language": "Python",
            "classes": [],
            "functions": [],
            "imports": [],
            "line_count": 100 + i,
        })

    logger.info(f"Created {len(mock_modules)} mock modules")

    # Calculate batches with default size
    batch_size = int(os.getenv("NEO4J_BATCH_SIZE", "50"))
    num_batches = (len(mock_modules) + batch_size - 1) // batch_size

    logger.info(f"Will process in {num_batches} batches of {batch_size} modules each:")

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(mock_modules))
        logger.info(f"  - Batch {batch_num + 1}: modules {start + 1}-{end}")

async def main():
    """Run all tests"""
    logger.info("=== Neo4j Transaction Batching Tests ===\n")

    try:
        # Test 1: Configuration
        await test_batch_configuration()

        # Test 2: Methods exist
        await test_batch_processing_methods()

        # Test 3: _create_graph uses batching
        await test_create_graph_uses_batching()

        # Test 4: Simulate batch processing
        await simulate_batch_processing()

        logger.info("\n=== All tests passed\\! ===")
        logger.info("\nBatching implementation summary:")
        logger.info("  1. Configurable batch size via NEO4J_BATCH_SIZE (default: 50)")
        logger.info("  2. Configurable timeout via NEO4J_BATCH_TIMEOUT (default: 120 seconds)")
        logger.info("  3. Each batch processed in separate transaction")
        logger.info("  4. Error handling per batch - continues on failure")
        logger.info("  5. Progress logging for large repositories")

    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
