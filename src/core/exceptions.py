"""Custom exceptions for the Crawl4AI MCP server."""


class MCPToolError(Exception):
    """Custom exception for MCP tool errors that should be returned as JSON-RPC errors."""

    def __init__(self, message: str, code: int = -32000):
        self.message = message
        self.code = code
        super().__init__(message)


# ========================================
# Base Exceptions
# ========================================


class Crawl4AIError(Exception):
    """Base exception for all Crawl4AI MCP errors."""


# ========================================
# Database Exceptions
# ========================================


class DatabaseError(Crawl4AIError):
    """Base exception for database-related errors."""


class ConnectionError(DatabaseError):
    """Database connection failed."""


class QueryError(DatabaseError):
    """Database query execution failed."""


class VectorStoreError(DatabaseError):
    """Vector store operation failed."""


class EmbeddingError(DatabaseError):
    """Embedding generation failed."""


# ========================================
# Network Exceptions
# ========================================


class NetworkError(Crawl4AIError):
    """Base exception for network-related errors."""


class FetchError(NetworkError):
    """HTTP fetch operation failed."""


class CrawlError(NetworkError):
    """Web crawling operation failed."""


class SearchError(NetworkError):
    """Search operation failed."""


# ========================================
# Validation Exceptions
# ========================================


class ValidationError(Crawl4AIError):
    """Base exception for validation errors."""


class ConfigurationError(ValidationError):
    """Configuration validation failed."""


class InputValidationError(ValidationError):
    """Input parameter validation failed."""


class SchemaValidationError(ValidationError):
    """Schema validation failed."""


# ========================================
# Knowledge Graph Exceptions
# ========================================


class KnowledgeGraphError(Crawl4AIError):
    """Base exception for knowledge graph operations."""


class RepositoryError(KnowledgeGraphError):
    """Repository operation failed."""


class GitError(KnowledgeGraphError):
    """Git operation failed."""


class ParsingError(KnowledgeGraphError):
    """Code parsing failed."""


class AnalysisError(KnowledgeGraphError):
    """Code analysis failed."""


# ========================================
# File I/O Exceptions
# ========================================


class FileOperationError(Crawl4AIError):
    """Base exception for file operations."""


class FileReadError(FileOperationError):
    """File read operation failed."""


class FileWriteError(FileOperationError):
    """File write operation failed."""


# ========================================
# External Service Exceptions
# ========================================


class ExternalServiceError(Crawl4AIError):
    """Base exception for external service errors."""


class LLMError(ExternalServiceError):
    """LLM API call failed."""


class EmbeddingServiceError(ExternalServiceError):
    """Embedding service call failed."""
