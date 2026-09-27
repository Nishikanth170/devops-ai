"""Application settings.

All runtime configuration is environment driven (12-factor).  A ``.env`` file is
read when present (see ``.env.example``).

The defaults are *development* safe values so that the platform can boot and be
tested offline.  :meth:`Settings.assert_production_ready` fails fast when the
platform is started in production with placeholder secrets.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Placeholder values that must never reach production.
DEV_JWT_SECRET = "dev-only-insecure-jwt-secret"
DEV_GITHUB_WEBHOOK_SECRET = "dev-github-webhook-secret"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------ App
    APP_NAME: str = "DevOps Agent Platform"
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    DEBUG: bool = True

    # ------------------------------------------------------------- Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/devops_agents",
        description="Async SQLAlchemy connection string.",
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    #: Set to False when the schema is owned by Alembic (`alembic upgrade head`).
    DATABASE_CREATE_SCHEMA: bool = True

    # -------------------------------------------------------- Vector store
    #: ``pgvector`` uses the vector_memory table with an HNSW index.
    #: ``memory`` keeps embeddings in-process (no pgvector needed) - handy for
    #: tests and local runs.  Both satisfy the same VectorStore interface.
    VECTOR_BACKEND: Literal["pgvector", "memory"] = "pgvector"
    VECTOR_DIMENSIONS: int = 1536  # OpenAI text-embedding-3-small
    VECTOR_SIMILARITY_THRESHOLD: float = 0.65

    # ---------------------------------------------------------------- Queue
    QUEUE_BACKEND: Literal["redis", "memory"] = "redis"
    REDIS_URL: str = "redis://localhost:6379/0"
    EVENT_QUEUE_NAME: str = "events"
    PRIORITY_QUEUE_NAME: str = "events-priority"
    EVENT_CONSUMER_BLOCK_SECONDS: int = 5
    #: The API process also consumes events in-process (single-node deployments).
    #: Set false when dedicated worker pods run `python -m app.events.consumer`.
    EVENT_CONSUMER_IN_PROCESS: bool = True

    # ------------------------------------------------------------------ LLM
    #: ``fake`` is a deterministic offline gateway (no API key, no network).
    LLM_PROVIDER: Literal["openai", "fake"] = "openai"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_EMBEDDING_CHARS: int = 32_000

    # ----------------------------------------------------------------- Auth
    OAUTH2_CLIENT_ID: str = "dev-client-id"
    OAUTH2_CLIENT_SECRET: str = "dev-client-secret"
    OAUTH2_AUTHORIZATION_URL: str = "https://sso.example.com/oauth2/authorize"
    OAUTH2_TOKEN_URL: str = "https://sso.example.com/oauth2/token"
    OAUTH2_USERINFO_URL: str = "https://sso.example.com/oauth2/userinfo"
    OAUTH2_REDIRECT_URI: str = "http://localhost:8000/auth/callback"
    OAUTH2_SCOPES: str = "openid profile email"
    OAUTH2_ADMIN_EMAILS: list[str] = Field(default_factory=list)
    JWT_SECRET_KEY: str = DEV_JWT_SECRET
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DEFAULT_USER_ROLE: str = "on-call-engineer"

    # ----------------------------------------------------------- Team routing
    DEFAULT_TEAM_NAME: str = "platform"
    #: Maps a repo full name (``acme/payments``) or service name to a team name.
    TEAM_ROUTING: dict[str, str] = Field(default_factory=dict)

    # ----------------------------------------------- Integrations: git / CI
    GITHUB_WEBHOOK_SECRET: str = DEV_GITHUB_WEBHOOK_SECRET
    GITHUB_API_BASE_URL: str = "https://api.github.com"
    GITHUB_TOKEN: Optional[str] = None
    GITHUB_APP_ID: Optional[str] = None
    GITHUB_APP_PRIVATE_KEY: Optional[str] = None

    GITLAB_WEBHOOK_SECRET: str = "dev-gitlab-webhook-secret"
    GITLAB_API_BASE_URL: str = "https://gitlab.com/api/v4"
    GITLAB_TOKEN: Optional[str] = None

    JENKINS_BASE_URL: Optional[str] = None
    JENKINS_USER: Optional[str] = None
    JENKINS_API_TOKEN: Optional[str] = None
    JENKINS_WEBHOOK_SECRET: str = "dev-jenkins-webhook-secret"

    # ------------------------------------------ Integrations: monitoring/ops
    DATADOG_API_KEY: Optional[str] = None
    DATADOG_APP_KEY: Optional[str] = None
    DATADOG_WEBHOOK_SECRET: str = ""
    PAGERDUTY_INTEGRATION_KEY: Optional[str] = None

    JIRA_BASE_URL: Optional[str] = None
    JIRA_EMAIL: Optional[str] = None
    JIRA_API_TOKEN: Optional[str] = None
    JIRA_WEBHOOK_SECRET: str = ""

    # --------------------------------------------------------------- helpers
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    def assert_production_ready(self) -> None:
        """Raise ``RuntimeError`` when production runs with dev placeholders."""
        if not self.is_production:
            return

        problems: list[str] = []
        if self.JWT_SECRET_KEY == DEV_JWT_SECRET:
            problems.append("JWT_SECRET_KEY still uses the development default")
        if self.GITHUB_WEBHOOK_SECRET == DEV_GITHUB_WEBHOOK_SECRET:
            problems.append("GITHUB_WEBHOOK_SECRET still uses the development default")
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            problems.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.DEBUG:
            problems.append("DEBUG must be false in production")
        if self.QUEUE_BACKEND == "memory":
            problems.append("QUEUE_BACKEND=memory cannot be used in production (no durability)")

        if problems:
            raise RuntimeError(
                "Refusing to start in production with insecure configuration: "
                + "; ".join(problems)
            )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (safe to call at import time)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests)."""
    get_settings.cache_clear()
