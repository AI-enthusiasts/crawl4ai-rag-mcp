"""Validated code search service.

This package combines Qdrant semantic search with Neo4j structural validation
to provide high-confidence code search results with:
- Semantic similarity matching via Qdrant
- Structural validation via Neo4j knowledge graph
- Confidence scoring and filtering
- Intelligent suggestion generation
- Performance caching and optimization
"""

# Import from modular structure
from .models import ValidationResult
from .neo4j_client import Neo4jValidationClient
from .service import ValidatedCodeSearchService
from .validator import CodeValidator

__all__ = [
    # Main service
    "ValidatedCodeSearchService",
    # Data models
    "ValidationResult",
    # Sub-services
    "Neo4jValidationClient",
    "CodeValidator",
]
