"""Utility functions for the Crawl4AI MCP server."""

# Import code analysis functions
from .code_analysis import (
    extract_code_blocks,
    generate_code_example_summary,
    process_code_example,
)

# Import embedding functions from local embeddings module
from .embeddings import (
    add_code_examples_to_database,
    add_documents_to_database,
    create_embedding,
    create_embeddings_batch,
    generate_contextual_embedding,
    process_chunk_with_context,
    search_code_examples,
    search_documents,
)
from .reranking import rerank_results

# Import summarization functions
from .summarization import extract_source_summary
from .text_processing import (
    extract_section_info,
    smart_chunk_markdown,
)
from .url_helpers import (
    extract_domain_from_url,
    is_sitemap,
    is_txt,
    normalize_url,
    parse_sitemap,
    parse_sitemap_content,
)
from .validation import (
    validate_github_url,
    validate_neo4j_connection,
    validate_script_path,
)

__all__ = [
    "add_code_examples_to_database",
    # Database and embedding functions
    "add_documents_to_database",
    "create_embedding",
    "create_embeddings_batch",
    # Code analysis functions
    "extract_code_blocks",
    # URL helpers
    "extract_domain_from_url",
    # Text processing functions
    "extract_section_info",
    # Summarization functions
    "extract_source_summary",
    "generate_code_example_summary",
    "generate_contextual_embedding",
    "is_sitemap",
    "is_txt",
    "normalize_url",
    "parse_sitemap",
    "parse_sitemap_content",
    "process_chunk_with_context",
    "process_code_example",
    # Reranking
    "rerank_results",
    "search_code_examples",
    "search_documents",
    "smart_chunk_markdown",
    # Validation
    "validate_github_url",
    "validate_neo4j_connection",
    "validate_script_path",
]
