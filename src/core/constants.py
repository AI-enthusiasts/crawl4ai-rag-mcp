"""Application-wide constants for Crawl4AI MCP Server.

This module contains all magic values extracted from the codebase
for better maintainability and discoverability.
"""

# ========================================
# Agentic Search Constants
# ========================================

# Completeness thresholds
COMPLETENESS_THRESHOLD_DEFAULT = 0.95  # Default threshold for answer completeness
COMPLETENESS_THRESHOLD_STRICT = 0.95  # Strict threshold requiring comprehensive answers
COMPLETENESS_THRESHOLD_MODERATE = 0.8  # Moderate threshold for good answers
COMPLETENESS_THRESHOLD_LENIENT = 0.5  # Lenient threshold for basic answers

# URL scoring
URL_SCORE_THRESHOLD_DEFAULT = 0.7  # Default minimum relevance score for URLs
URL_SCORE_THRESHOLD_HIGH = 0.8  # High relevance threshold
URL_SCORE_THRESHOLD_LOW = 0.5  # Low relevance threshold

# Iteration limits
MAX_ITERATIONS_DEFAULT = 3  # Default maximum search iterations
MAX_ITERATIONS_EXTENSIVE = 5  # Extensive search iterations
MAX_ITERATIONS_QUICK = 1  # Quick single-pass search

MAX_URLS_PER_ITERATION_DEFAULT = 5  # Default URLs to crawl per iteration
MAX_URLS_PER_ITERATION_EXTENSIVE = 10  # More URLs for comprehensive search
MAX_URLS_PER_ITERATION_QUICK = 1  # Single URL for quick search

# Recursive crawling limits
MAX_PAGES_PER_ITERATION_DEFAULT = 50  # Maximum pages to crawl across all URLs in iteration
MAX_PAGES_PER_ITERATION_EXTENSIVE = 100  # More pages for comprehensive search
MAX_PAGES_PER_ITERATION_QUICK = 10  # Fewer pages for quick search

MAX_CRAWL_DEPTH_DEFAULT = 3  # Default depth for recursive crawling
MAX_CRAWL_DEPTH_SHALLOW = 1  # Single level (no recursion)
MAX_CRAWL_DEPTH_DEEP = 5  # Deep crawling for comprehensive coverage

# Memory protection limits
MAX_VISITED_URLS_LIMIT = 10000  # Maximum visited URLs to track (prevents memory exhaustion)

# LLM parameters
LLM_TEMPERATURE_DETERMINISTIC = 0.3  # Deterministic for completeness/ranking
LLM_TEMPERATURE_BALANCED = 0.5  # Balanced for query refinement
LLM_TEMPERATURE_CREATIVE = 0.7  # Creative for ideation

# LLM call optimization thresholds
SCORE_IMPROVEMENT_THRESHOLD = 0.1  # Skip refinement if score improved by this much
MAX_URLS_TO_RANK_DEFAULT = 20  # Default number of URLs to rank with LLM

# Qdrant limits
MAX_QDRANT_RESULTS_DEFAULT = 10  # Default results from Qdrant
MAX_QDRANT_RESULTS_COMPREHENSIVE = 20  # More results for thorough analysis
MAX_QDRANT_RESULTS_QUICK = 5  # Fewer results for speed

# URL filtering patterns (avoid these in recursive crawling)
URL_FILTER_PATTERNS = [
    # GitHub patterns to avoid infinite crawling
    r"/commit/",  # Individual commits
    r"/commits/",  # Commit history pages
    r"/blame/",  # Blame pages
    r"/compare/",  # Compare pages
    r"/pull/\d+/commits",  # PR commit pages
    r"/pull/\d+/files",  # PR files pages
    r"/issues/\d+/events",  # Issue events
    r"/actions/runs/",  # GitHub Actions runs
    r"/network/dependencies",  # Dependency graph
    r"/pulse",  # Pulse/activity pages
    r"/graphs/",  # Graph pages
    r"/security/",  # Security advisories
    # GitLab patterns
    r"/-/commit/",
    r"/-/commits/",
    r"/-/merge_requests/\d+/diffs",
    # Documentation patterns to filter
    r"/search\?",  # Search result pages
    r"/tag/",  # Tag pages
    r"/tags\?",  # Tag listing pages
    r"/releases\?",  # Release listing pages
    # General patterns
    r"\?page=\d+$",  # Pagination pages (often duplicates)
    r"/archive/",  # Archive pages
    r"\.git$",  # Git repositories
    r"\.zip$",  # Downloads
    r"\.tar\.gz$",  # Downloads
    r"/rss$",  # RSS feeds
    r"/atom$",  # Atom feeds
]

# ========================================
# Vector Database Constants
# ========================================

# OpenAI embedding dimensions
OPENAI_EMBEDDING_DIMENSION = 1536  # text-embedding-3-small
OPENAI_EMBEDDING_LARGE_DIMENSION = 3072  # text-embedding-3-large

# Batch sizes
QDRANT_BATCH_SIZE = 100  # Qdrant can handle larger batches
SUPABASE_BATCH_SIZE = 50  # Supabase batch size
NEO4J_BATCH_SIZE_DEFAULT = 50  # Default Neo4j batch size

# ========================================
# Repository Limits
# ========================================

# Size limits (defined in Settings, referenced here for code usage)
REPO_MAX_SIZE_MB_DEFAULT = 500  # Default maximum repository size
REPO_MAX_FILE_COUNT_DEFAULT = 10000  # Default maximum file count

# ========================================
# Network & Timeout Constants
# ========================================

# Timeouts (seconds)
SEARXNG_TIMEOUT_DEFAULT = 30
NEO4J_BATCH_TIMEOUT_DEFAULT = 120
HTTP_REQUEST_TIMEOUT_DEFAULT = 30
CRAWL_TIMEOUT_DEFAULT = 60
LLM_API_TIMEOUT_DEFAULT = 60  # OpenAI API timeout (default is 10 minutes - too long)
LLM_API_CONNECT_TIMEOUT = 5  # Connection timeout for LLM API
LLM_API_READ_TIMEOUT = 60  # Read timeout for LLM responses

# Retry limits
MAX_RETRIES_DEFAULT = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff base (2s, 4s, 8s...)

# ========================================
# Security Constants
# ========================================

# Input size limits (bytes)
MAX_INPUT_SIZE = 50000  # 50KB max input for safety
MAX_URL_LENGTH = 2048  # Maximum URL length
MAX_QUERY_LENGTH = 1000  # Maximum search query length

# ========================================
# Test Constants
# ========================================

# Test models
TEST_MODEL_CHEAP = "gpt-4.1-nano"  # Cheap model for integration tests
TEST_MODEL_FALLBACK = "gpt-4o-mini"  # Fallback if nano not available

# Test cost estimates (USD)
TEST_COST_PER_1K_TOKENS_NANO = 0.00015  # gpt-4.1-nano cost
TEST_COST_PER_RUN_BASIC = 0.001  # Basic test run cost
TEST_COST_PER_RUN_COMPREHENSIVE = 0.005  # Comprehensive test run cost

# ========================================
# HTTP Status Codes
# ========================================

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_ERROR = 500
