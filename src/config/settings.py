"""Configuration settings for Crawl4AI MCP Server using Pydantic Settings.

This module provides type-safe configuration management with automatic validation,
environment variable loading, and documentation generation.
"""

import logging
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Central configuration management with Pydantic validation.

    All settings are loaded from environment variables with automatic type conversion
    and validation. Default values are provided for non-critical settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars
        validate_default=True,
    )

    # ========================================
    # Debug Settings
    # ========================================
    debug: bool = Field(
        default=False,
        alias="MCP_DEBUG",
        description="Enable debug mode with verbose logging",
    )

    # ========================================
    # API Keys
    # ========================================
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for LLM operations",
    )

    # ========================================
    # Server Settings
    # ========================================
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )

    port: int = Field(
        default=8051,
        ge=1024,
        le=65535,
        description="Server port number",
    )

    # ========================================
    # Database Settings
    # ========================================
    vector_database: str = Field(
        default="qdrant",
        description="Vector database type (qdrant or supabase)",
    )

    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL",
    )

    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key for authentication",
    )

    # ========================================
    # Neo4j Settings
    # ========================================
    neo4j_uri: str | None = Field(
        default=None,
        description="Neo4j database URI",
    )

    neo4j_username: str | None = Field(
        default=None,
        description="Neo4j username",
    )

    neo4j_password: str | None = Field(
        default=None,
        description="Neo4j password",
    )

    use_knowledge_graph: bool = Field(
        default=False,
        description="Enable knowledge graph features",
    )

    neo4j_batch_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Batch size for Neo4j transaction processing",
    )

    neo4j_batch_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout in seconds for Neo4j batch operations",
    )

    # ========================================
    # SearXNG Settings
    # ========================================
    searxng_url: str | None = Field(
        default="http://localhost:8080",
        description="SearXNG instance URL for web search",
    )

    searxng_user_agent: str = Field(
        default="MCP-Crawl4AI-RAG-Server/1.0",
        description="User agent for SearXNG requests",
    )

    searxng_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout in seconds for SearXNG requests",
    )

    searxng_default_engines: str = Field(
        default="",
        description="Comma-separated list of default search engines",
    )

    # ========================================
    # Feature Flags
    # ========================================
    use_reranking: bool = Field(
        default=False,
        description="Enable result reranking for improved relevance",
    )

    use_test_env: bool = Field(
        default=False,
        alias="USE_TEST_ENV",
        description="Use test environment configuration",
    )

    use_agentic_rag: bool = Field(
        default=False,
        description="Enable agentic RAG features",
    )

    # ========================================
    # Agentic Search Settings
    # ========================================
    agentic_search_enabled: bool = Field(
        default=False,
        description="Enable agentic search with iterative refinement",
    )

    agentic_search_completeness_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Completeness threshold (0.0-1.0) for determining when answer is sufficient",
    )

    agentic_search_max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of search iterations",
    )

    agentic_search_max_urls_per_iteration: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum starting URLs to crawl per iteration",
    )

    agentic_search_max_pages_per_iteration: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum total pages to crawl recursively across all URLs in iteration",
    )

    agentic_search_url_score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (0.0-1.0) for URLs to be crawled",
    )

    agentic_search_use_search_hints: bool = Field(
        default=False,
        description="Generate search hints from crawled content",
    )

    agentic_search_enable_url_filtering: bool = Field(
        default=True,
        description="Enable smart URL filtering to avoid GitHub commits, pagination, etc.",
    )

    agentic_search_max_urls_to_rank: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Maximum number of search results to rank with LLM (reduce for lower costs)",
    )

    agentic_search_llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM temperature for agentic search evaluations",
    )

    agentic_search_max_qdrant_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to retrieve from Qdrant per query",
    )

    model_choice: str = Field(
        default="gpt-4o-mini",
        description="LLM model for evaluations and completeness checks",
    )

    # ========================================
    # Crawler Settings
    # ========================================
    max_concurrent_sessions: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum concurrent browser sessions (global limit)",
    )

    # ========================================
    # Transport Settings
    # ========================================
    transport: str = Field(
        default="http",
        description="Transport mode (http or stdio)",
    )

    mcp_api_key: str | None = Field(
        default=None,
        description="MCP API key for authentication",
    )

    # ========================================
    # OAuth2 Settings
    # ========================================
    use_oauth2: bool = Field(
        default=False,
        description="Enable OAuth2 authentication",
    )

    oauth2_issuer: str | None = Field(
        default=None,
        description="OAuth2 issuer URL",
    )

    oauth2_secret_key: str = Field(
        default="change-me-in-production",
        description="OAuth2 JWT secret key",
    )

    oauth2_scopes: str = Field(
        default="read:data,write:data",
        description="Comma-separated list of valid OAuth2 scopes",
    )

    oauth2_required_scopes: str = Field(
        default="read:data",
        description="Comma-separated list of required OAuth2 scopes",
    )

    # ========================================
    # Repository Size Limits
    # ========================================
    repo_max_size_mb: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum repository size in MB",
    )

    repo_max_file_count: int = Field(
        default=10000,
        ge=1,
        le=1000000,
        description="Maximum file count for repository",
    )

    repo_min_free_space_gb: float = Field(
        default=1.0,
        ge=0.1,
        le=1000.0,
        description="Minimum free disk space required in GB",
    )

    repo_allow_size_override: bool = Field(
        default=False,
        description="Allow overriding size limits",
    )

    # ========================================
    # Test Settings
    # ========================================
    test_openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for tests (falls back to openai_api_key)",
    )

    test_model_choice: str = Field(
        default="gpt-4.1-nano",
        description="LLM model for integration tests (cheap and fast)",
    )

    # ========================================
    # Validators
    # ========================================
    @field_validator("oauth2_issuer", mode="before")
    @classmethod
    def set_oauth2_issuer(cls, v: str | None, info: Any) -> str:
        """Set OAuth2 issuer default from host and port if not provided."""
        if v:
            return v
        # Access other field values during validation
        host = info.data.get("host", "0.0.0.0")
        port = info.data.get("port", 8051)
        return f"https://{host}:{port}"

    # ========================================
    # Helper Methods
    # ========================================
    def has_neo4j_config(self) -> bool:
        """Check if Neo4j environment variables are configured."""
        return all([self.neo4j_uri, self.neo4j_username, self.neo4j_password])

    def get_neo4j_config(self) -> dict[str, Any]:
        """Get Neo4j configuration as a dictionary."""
        return {
            "uri": self.neo4j_uri or "",
            "auth": (self.neo4j_username or "", self.neo4j_password or ""),
        }

    def get_oauth2_scopes_list(self) -> list[str]:
        """Get OAuth2 scopes as a list."""
        return [s.strip() for s in self.oauth2_scopes.split(",") if s.strip()]

    def get_oauth2_required_scopes_list(self) -> list[str]:
        """Get required OAuth2 scopes as a list."""
        return [
            s.strip() for s in self.oauth2_required_scopes.split(",") if s.strip()
        ]

    def to_dict(self) -> dict[str, Any]:
        """Export settings as a dictionary (safe version without secrets)."""
        return {
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "vector_database": self.vector_database,
            "use_knowledge_graph": self.use_knowledge_graph,
            "use_reranking": self.use_reranking,
            "use_test_env": self.use_test_env,
            "has_neo4j": self.has_neo4j_config(),
            "has_searxng": bool(self.searxng_url),
            "has_openai": bool(self.openai_api_key),
            "neo4j_batch_size": self.neo4j_batch_size,
            "neo4j_batch_timeout": self.neo4j_batch_timeout,
            "repo_max_size_mb": self.repo_max_size_mb,
            "repo_max_file_count": self.repo_max_file_count,
            "repo_min_free_space_gb": self.repo_min_free_space_gb,
            "repo_allow_size_override": self.repo_allow_size_override,
            "agentic_search_enabled": self.agentic_search_enabled,
            "agentic_search_completeness_threshold": self.agentic_search_completeness_threshold,
            "model_choice": self.model_choice,
            "test_model_choice": self.test_model_choice,
        }


# Singleton pattern with proper typing
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        logger.info("Settings initialized from environment")
        logger.debug("Vector database: %s", _settings_instance.vector_database)
        if not _settings_instance.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is missing. OpenAI features will be unavailable.",
            )
    return _settings_instance


def reset_settings() -> None:
    """Reset settings instance (useful for testing)."""
    global _settings_instance
    _settings_instance = None
