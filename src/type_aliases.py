"""Common type aliases for the Crawl4AI MCP Server.

This module defines type aliases used throughout the codebase to improve
readability and maintain consistency.
"""

from typing import Any

# Document and content types
DocumentChunk = dict[str, Any]
"""A document chunk with metadata.

Expected keys:
- content: str - The text content
- metadata: dict[str, Any] - Metadata including source, chunk_id, etc.
"""

SourceMetadata = dict[str, str | int | None]
"""Metadata about a data source.

Common keys:
- source: str - Source URL or identifier
- domain: str - Domain name
- timestamp: int - Unix timestamp
- chunk_index: int - Chunk number
"""

# Vector and embedding types
EmbeddingVector = list[float]
"""A vector embedding representation (typically 1536 or 3072 dimensions)."""

SimilarityScore = float
"""Similarity score between 0.0 and 1.0."""

# Search result types
SearchResult = dict[str, Any]
"""A search result with content and metadata.

Expected keys:
- content: str - The content
- score: float - Similarity/relevance score
- source: str - Source URL
- metadata: dict[str, Any] - Additional metadata
"""

SearchResults = list[SearchResult]
"""List of search results."""

# Database types
PointID = str | int
"""Identifier for a vector database point (Qdrant uses str or int)."""

CollectionName = str
"""Name of a vector database collection."""

# URL and network types
URLString = str
"""A valid URL string."""

URLList = list[URLString]
"""List of URLs."""

# Neo4j and knowledge graph types
CypherQuery = str
"""A Cypher query string for Neo4j."""

NodeProperties = dict[str, Any]
"""Properties of a Neo4j node."""

# Validation types
ValidationStatus = str
"""Validation status: 'extracted', 'validated', 'verified', 'failed'."""

ConfidenceScore = float
"""Confidence score between 0.0 and 1.0."""

# MCP and tool types
ToolName = str
"""Name of an MCP tool."""

ToolArguments = dict[str, Any]
"""Arguments passed to an MCP tool."""

# Code analysis types
CodeType = str
"""Type of code element: 'class', 'method', 'function'."""

ParameterList = list[str]
"""List of parameter names or signatures."""

# Error and response types
ErrorMessage = str
"""Error message string."""

JSONResponse = dict[str, Any]
"""JSON response object."""

# Batch operation types
BatchSize = int
"""Size of a batch for batch operations."""

MaxConcurrent = int
"""Maximum number of concurrent operations."""
