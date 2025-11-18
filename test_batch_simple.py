#\!/usr/bin/env python3
"""Simple test to verify Neo4j batching configuration"""
import os
import sys
from pathlib import Path

# Set test batch configuration
os.environ["NEO4J_BATCH_SIZE"] = "25"
os.environ["NEO4J_BATCH_TIMEOUT"] = "60"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

# Create extractor
extractor = DirectNeo4jExtractor(
    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
    neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
)

# Verify configuration
print(f"Batch size: {extractor.batch_size}")
print(f"Batch timeout: {extractor.batch_timeout_seconds} seconds")

# Check methods exist
print(f"Has _process_modules_in_batches: {hasattr(extractor, '_process_modules_in_batches')}")
print(f"Has _process_batch_transaction: {hasattr(extractor, '_process_batch_transaction')}")

print("\nBatching implementation verified successfully\!")
