"""Database engine, session factory and schema bootstrap.

Sessions are *request scoped*: FastAPI dependencies open one session per request
(``get_session``) and long running code opens one per unit of work
(``session_scope``).  Nothing in the platform keeps a session as a global.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        settings = get_settings()
        kwargs: dict[str, object] = {"echo": settings.DEBUG, "pool_pre_ping": True}
        if not settings.DATABASE_URL.startswith("sqlite"):
            kwargs["pool_size"] = settings.DATABASE_POOL_SIZE
            kwargs["max_overflow"] = settings.DATABASE_MAX_OVERFLOW
        _engine = create_async_engine(settings.DATABASE_URL, **kwargs)
        logger.debug("Created database engine for %s", settings.DATABASE_URL.split("@")[-1])
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory bound to the engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False, autoflush=False
        )
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional unit of work: commit on success, rollback on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session."""
    async with session_scope() as session:
        yield session


async def init_db(create_schema: Optional[bool] = None) -> None:
    """Create the schema when configured to do so and prepare the vector index."""
    settings = get_settings()
    should_create = settings.DATABASE_CREATE_SCHEMA if create_schema is None else create_schema

    if should_create:
        await create_schema_tables()
    await ensure_vector_index()


async def create_schema_tables() -> None:
    """``create_all`` for every model (Alembic owns this in production)."""
    # Import models so they are registered on ``Base.metadata``.
    from app.memory import models as _models  # noqa: F401
    from app.memory import vector_store as _vector_store  # noqa: F401

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(_create_all_but_skipped)


def _create_all_but_skipped(sync_conn) -> None:
    """Create tables, skipping ``vector_memory`` when the in-memory backend is on."""
    settings = get_settings()
    tables: Sequence = Base.metadata.sorted_tables
    if settings.VECTOR_BACKEND == "memory":
        tables = [t for t in tables if t.name != "vector_memory"]
    Base.metadata.create_all(sync_conn, tables=list(tables))


async def ensure_vector_index() -> None:
    """Enable pgvector and create the HNSW index (PostgreSQL only)."""
    settings = get_settings()
    if settings.VECTOR_BACKEND != "pgvector":
        return
    if not settings.DATABASE_URL.startswith("postgresql"):
        return

    engine = get_engine()
    dims = settings.VECTOR_DIMENSIONS
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vector_memory_embedding_hnsw "
                    "ON vector_memory USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                )
            )
        logger.info("pgvector extension ready (embedding dims=%s, HNSW index ensured)", dims)
    except Exception as exc:  # pragma: no cover - requires a live PostgreSQL
        logger.warning("Could not prepare pgvector index: %s", exc)


async def close_db() -> None:
    """Dispose the engine (called on application shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.debug("Database engine disposed")
    _engine = None
    _session_factory = None
