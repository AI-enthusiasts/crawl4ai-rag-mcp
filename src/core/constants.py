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

MAX_URLS_PER_ITERATION_DEFAULT = 3  # Default URLs to crawl per iteration
MAX_URLS_PER_ITERATION_EXTENSIVE = 5  # More URLs for comprehensive search
MAX_URLS_PER_ITERATION_QUICK = 1  # Single URL for quick search

# LLM parameters
LLM_TEMPERATURE_DETERMINISTIC = 0.3  # Deterministic for completeness/ranking
LLM_TEMPERATURE_BALANCED = 0.5  # Balanced for query refinement
LLM_TEMPERATURE_CREATIVE = 0.7  # Creative for ideation

# Qdrant limits
MAX_QDRANT_RESULTS_DEFAULT = 10  # Default results from Qdrant
MAX_QDRANT_RESULTS_COMPREHENSIVE = 20  # More results for thorough analysis
MAX_QDRANT_RESULTS_QUICK = 5  # Fewer results for speed

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
