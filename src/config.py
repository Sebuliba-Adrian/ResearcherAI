"""
Centralized configuration management using Pydantic Settings.
"""
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "ResearcherAI"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "production"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False

    # Google Gemini API
    google_api_key: str

    # Neo4j (Production Graph Database)
    use_neo4j: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Qdrant (Production Vector Database)
    use_qdrant: bool = True
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "research_papers"
    qdrant_vector_size: int = 384

    # Kafka (Event Streaming)
    use_kafka: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topics: list[str] = [
        "paper.collected",
        "paper.processed",
        "graph.updated",
        "vector.indexed",
        "query.received",
        "query.completed",
    ]

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_device: str = "cpu"

    # Cache Configuration
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    cache_l2_enabled: bool = True
    cache_l2_ttl_seconds: int = 86400

    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_expected_exception: type = Exception

    # Token Budget
    token_budget_enabled: bool = True
    token_budget_max_input: int = 30000
    token_budget_max_output: int = 8000
    token_budget_warning_threshold: float = 0.8

    # Data Collection
    data_sources_enabled: list[str] = [
        "arxiv",
        "semantic_scholar",
        "zenodo",
        "pubmed",
        "web",
        "huggingface",
        "kaggle",
    ]
    max_papers_per_source: int = 10
    collection_timeout_seconds: int = 300

    # Reasoning
    reasoning_max_turns: int = 5
    reasoning_temperature: float = 0.7
    reasoning_max_tokens: int = 2000

    # Summarization
    summarization_modes: list[str] = ["simple", "detailed", "comparative", "hierarchical"]
    summarization_default_mode: str = "simple"

    # Session Management
    session_persistence_enabled: bool = True
    session_storage_path: str = "./sessions"
    session_max_age_hours: int = 24

    # Scheduler
    scheduler_enabled: bool = False
    scheduler_interval_hours: int = 6
    scheduler_auto_start: bool = False

    # Database (SQLAlchemy)
    database_url: Optional[str] = None
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None

    # CORS
    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ["development", "dev"]

    @property
    def graph_backend(self) -> str:
        """Determine which graph backend to use."""
        return "neo4j" if self.use_neo4j else "networkx"

    @property
    def vector_backend(self) -> str:
        """Determine which vector backend to use."""
        return "qdrant" if self.use_qdrant else "faiss"

    @property
    def effective_database_url(self) -> str:
        """Get effective database URL with fallback to SQLite."""
        if self.database_url:
            return self.database_url
        # Default to SQLite for development
        return f"sqlite:///{self.session_storage_path}/researcherai.db"

    def get_kafka_config(self) -> dict:
        """Get Kafka configuration."""
        if not self.use_kafka:
            return {}
        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "group.id": "researcherai-group",
            "auto.offset.reset": "earliest",
        }


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Cached settings object
    """
    return Settings()


# Export convenience function
settings = get_settings()
