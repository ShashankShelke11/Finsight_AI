"""
Configuration Management for FinSight AI
Supports environment variables, .env file loading, and sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Base repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in ("true", "1", "yes", "on")


def _load_env_file(filepath: Path) -> None:
    """Lightweight .env loader without external dependencies."""
    if not filepath.exists():
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k not in os.environ:
                os.environ[k] = v


# Load .env if present
_load_env_file(REPO_ROOT / ".env")


@dataclass
class Settings:
    # Environment & Logging
    ENVIRONMENT: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    DEBUG: bool = field(
        default_factory=lambda: _parse_bool(os.getenv("DEBUG", "true"))
    )

    # Server Settings
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    API_V1_PREFIX: str = field(
        default_factory=lambda: os.getenv("API_V1_PREFIX", "/api/v1")
    )

    # Database
    DATABASE_URL: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{REPO_ROOT / 'finsight.db'}"
        )
    )

    # Vector & Retrieval
    USE_MOCK_EMBEDDINGS: bool = field(
        default_factory=lambda: _parse_bool(os.getenv("USE_MOCK_EMBEDDINGS", "true"))
    )
    EMBEDDING_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    EMBEDDING_DIMENSION: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "384"))
    )
    HYBRID_SEARCH_RRF_K: int = field(
        default_factory=lambda: int(os.getenv("HYBRID_SEARCH_RRF_K", "60"))
    )
    RERANKER_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    TOP_K_RETRIEVAL: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "10"))
    )
    TOP_K_RERANK: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RERANK", "5"))
    )

    # LLM & Agent Orchestration
    LLM_PROVIDER: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "local_mock")
    )
    LLM_MODEL: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini-1.5-pro")
    )
    GEMINI_API_KEY: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    OPENAI_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    ANTHROPIC_API_KEY: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Security & Guardrails
    MAX_UPLOAD_SIZE_MB: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    )
    MAX_QUERY_LENGTH: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    )
    MAX_RETRIES_PER_QUERY: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES_PER_QUERY", "2"))
    )
    AUDIT_LOG_ENABLED: bool = field(
        default_factory=lambda: _parse_bool(os.getenv("AUDIT_LOG_ENABLED", "true"))
    )

    # Paths
    BASE_DIR: Path = REPO_ROOT
    DATA_DIR: Path = REPO_ROOT / "data"
    SAMPLE_DATA_DIR: Path = REPO_ROOT / "data" / "sample"

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")


# Global singleton instance
settings = Settings()
