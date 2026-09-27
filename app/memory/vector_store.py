"""Vector (semantic) memory.

Two backends implement the same interface:

``pgvector``  (default, production)
    Stores embeddings in the ``vector_memory`` table and uses the cosine
    distance operator with an HNSW index for approximate nearest neighbour
    search.

``memory``    (development / tests)
    Keeps embeddings in-process and computes exact cosine similarity in Python.
    No PostgreSQL extension required, which makes the whole platform testable
    offline.

Every lookup returns :class:`VectorRecord` (a plain dataclass) so callers never
depend on the ORM model or on which backend is active.

Note: the ORM attribute is called ``meta`` because ``metadata`` is reserved by
SQLAlchemy's declarative API; the physical column is still named ``metadata``.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import JSON, DateTime, Index, String, Text, Uuid, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import Settings, get_settings
from app.core.database import Base
from app.core.exceptions import ConfigurationError
from app.core.utils import utcnow

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly
    from pgvector.sqlalchemy import Vector as PgVector
except ImportError:  # pragma: no cover
    PgVector = None  # type: ignore[assignment]


def _embedding_column_type():
    """Return the column type for embeddings, honouring the active backend."""
    settings = get_settings()
    if PgVector is None:
        if settings.VECTOR_BACKEND == "pgvector":
            raise ConfigurationError(
                "VECTOR_BACKEND=pgvector requires the 'pgvector' package: pip install pgvector"
            )
        return JSON
    return PgVector(settings.VECTOR_DIMENSIONS)


class VectorMemory(Base):
    """Vector embeddings linked back to their source relational record."""

    __tablename__ = "vector_memory"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source_table: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(_embedding_column_type(), nullable=False)
    #: Physical column name is ``metadata``; the attribute avoids clashing with
    #: ``Base.metadata``.
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_vector_memory_source", "source_table", "source_id"),
    )


@dataclass
class VectorRecord:
    """Backend-agnostic view of a stored embedding."""

    id: uuid.UUID
    source_table: str
    source_id: uuid.UUID
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``; ``0.0`` for degenerate vectors."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = na = nb = 0.0
    for i in range(length):
        x, y = float(a[i]), float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class _InMemoryBackend:
    """Process-local embedding store used when ``VECTOR_BACKEND=memory``."""

    records: dict[str, list[VectorRecord]] = {}

    @classmethod
    def add(cls, record: VectorRecord) -> None:
        cls.records.setdefault(record.source_table, []).append(record)

    @classmethod
    def clear(cls) -> None:
        cls.records.clear()


class VectorStore:
    """Semantic retrieval over the memory layer."""

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.backend = self.settings.VECTOR_BACKEND
        if self.backend == "pgvector" and session is None:
            raise ConfigurationError("The pgvector backend requires an AsyncSession")

    # ------------------------------------------------------------- writes
    async def add_memory(
        self,
        source_table: str,
        source_id: uuid.UUID,
        content: str,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> VectorRecord:
        """Persist one embedding for a relational record."""
        meta = metadata or {}
        if self.backend == "memory":
            record = VectorRecord(
                id=uuid.uuid4(),
                source_table=source_table,
                source_id=source_id,
                content=content,
                embedding=list(embedding),
                metadata=meta,
                created_at=utcnow(),
            )
            _InMemoryBackend.add(record)
            return record

        row = VectorMemory(
            source_table=source_table,
            source_id=source_id,
            content=content,
            embedding=list(embedding),
            meta=meta,
        )
        self.session.add(row)  # type: ignore[union-attr]
        await self.session.flush()  # type: ignore[union-attr]
        return self._to_record(row)

    async def get_by_source(self, source_table: str, source_id: uuid.UUID) -> list[VectorRecord]:
        """All embeddings belonging to one relational record."""
        if self.backend == "memory":
            return [
                r
                for r in _InMemoryBackend.records.get(source_table, [])
                if r.source_id == source_id
            ]

        stmt = select(VectorMemory).where(
            VectorMemory.source_table == source_table,
            VectorMemory.source_id == source_id,
        )
        result = await self.session.execute(stmt)  # type: ignore[union-attr]
        return [self._to_record(row) for row in result.scalars().all()]

    async def delete_by_source(self, source_table: str, source_id: uuid.UUID) -> int:
        """Delete every embedding for a relational record. Returns rows removed."""
        if self.backend == "memory":
            existing = _InMemoryBackend.records.get(source_table, [])
            kept = [r for r in existing if r.source_id != source_id]
            removed = len(existing) - len(kept)
            _InMemoryBackend.records[source_table] = kept
            return removed

        stmt = delete(VectorMemory).where(
            VectorMemory.source_table == source_table,
            VectorMemory.source_id == source_id,
        )
        result = await self.session.execute(stmt)  # type: ignore[union-attr]
        return int(result.rowcount or 0)

    async def count(self, source_table: Optional[str] = None) -> int:
        """Number of stored embeddings (optionally scoped to one table)."""
        if self.backend == "memory":
            if source_table:
                return len(_InMemoryBackend.records.get(source_table, []))
            return sum(len(v) for v in _InMemoryBackend.records.values())

        stmt = select(func.count(VectorMemory.id))
        if source_table:
            stmt = stmt.where(VectorMemory.source_table == source_table)
        return int(await self.session.scalar(stmt) or 0)  # type: ignore[union-attr]

    # -------------------------------------------------------------- search
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_table: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> list[tuple[VectorRecord, float]]:
        """Cosine similarity search.

        Returns ``(record, similarity)`` pairs ordered by descending similarity.
        ``threshold`` defaults to ``settings.VECTOR_SIMILARITY_THRESHOLD``.
        """
        min_score = (
            self.settings.VECTOR_SIMILARITY_THRESHOLD if threshold is None else threshold
        )

        if self.backend == "memory":
            return self._search_memory(
                query_embedding, top_k, source_table, metadata_filter, min_score
            )

        # pgvector: 1 - cosine_distance == cosine similarity.
        similarity = (1 - VectorMemory.embedding.cosine_distance(query_embedding)).label(
            "similarity"
        )
        stmt = select(VectorMemory, similarity).where(similarity >= min_score)
        if source_table:
            stmt = stmt.where(VectorMemory.source_table == source_table)
        for key, value in (metadata_filter or {}).items():
            stmt = stmt.where(VectorMemory.meta[key].astext == str(value))
        stmt = stmt.order_by(similarity.desc()).limit(top_k)

        result = await self.session.execute(stmt)  # type: ignore[union-attr]
        return [(self._to_record(row[0]), float(row[1])) for row in result.all()]

    def _search_memory(
        self,
        query_embedding: list[float],
        top_k: int,
        source_table: Optional[str],
        metadata_filter: Optional[dict[str, Any]],
        threshold: float,
    ) -> list[tuple[VectorRecord, float]]:
        candidates: Iterable[VectorRecord]
        if source_table:
            candidates = _InMemoryBackend.records.get(source_table, [])
        else:
            candidates = [r for bucket in _InMemoryBackend.records.values() for r in bucket]

        scored: list[tuple[VectorRecord, float]] = []
        for record in candidates:
            if metadata_filter and any(
                str(record.metadata.get(k)) != str(v) for k, v in metadata_filter.items()
            ):
                continue
            score = cosine_similarity(query_embedding, record.embedding)
            if score >= threshold:
                scored.append((record, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _to_record(row: VectorMemory) -> VectorRecord:
        return VectorRecord(
            id=row.id,
            source_table=row.source_table,
            source_id=row.source_id,
            content=row.content,
            embedding=list(row.embedding or []),
            metadata=dict(row.meta or {}),
            created_at=row.created_at,
        )

    @staticmethod
    def clear_memory_backend() -> None:
        """Drop the in-process store (tests / dev restarts)."""
        _InMemoryBackend.clear()
