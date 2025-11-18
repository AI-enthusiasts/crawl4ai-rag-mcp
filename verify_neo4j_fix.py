#!/usr/bin/env python3
"""Verify Neo4j dependencies can be imported correctly."""

import sys
import os

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nAttempting to import knowledge_graph modules...")

# Try the fix we implemented
try:
    src_dir = str(Path(__file__).parent.parent.absolute())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        print(f"Added {src_dir} to sys.path")
    
    from knowledge_graph.knowledge_graph_validator import KnowledgeGraphValidator
    from knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor
    
    print("✓ Successfully imported knowledge_graph modules!")
    print(f"  KnowledgeGraphValidator: {KnowledgeGraphValidator}")
    print(f"  DirectNeo4jExtractor: {DirectNeo4jExtractor}")
    
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("\nChecking if knowledge_graph directory exists:")

    src_dir = Path(__file__).parent.parent.absolute()
    kg_path = src_dir / "src" / "knowledge_graph"

    if kg_path.exists():
        print(f"  {kg_path} exists")
        print(f"  Contents: {os.listdir(kg_path)}")
    else:
        print(f"  {kg_path} does NOT exist")

        # Check alternative path
        alt_path = src_dir / "knowledge_graph"
        if alt_path.exists():
            print(f"  Alternative path {alt_path} exists")
            print(f"  Contents: {os.listdir(alt_path)}")