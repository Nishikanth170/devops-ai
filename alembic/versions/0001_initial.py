"""initial schema

Revision ID: 0001
Revises:
Create Date: bootstrap migration

Creates every table from the SQLAlchemy metadata, enables the pgvector
extension and adds the HNSW similarity index (PostgreSQL only).

Using ``metadata.create_all`` for the *bootstrap* revision is deliberate: the
schema is fully described by the models, and subsequent revisions should use
explicit ``op.*`` operations so that upgrades stay reviewable.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from app.core.config import get_settings
from app.core.database import Base
from app.memory import models as _models  # noqa: F401  (registers tables)
from app.memory import vector_store as _vector_store  # noqa: F401

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_HNSW_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_vector_memory_embedding_hnsw "
    "ON vector_memory USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
)


def _tables():
    settings = get_settings()
    tables = list(Base.metadata.sorted_tables)
    if settings.VECTOR_BACKEND == "memory":
        tables = [table for table in tables if table.name != "vector_memory"]
    return tables


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    Base.metadata.create_all(bind=bind, tables=_tables())

    if is_postgres and get_settings().VECTOR_BACKEND == "pgvector":
        op.execute(_HNSW_INDEX)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind, tables=list(reversed(_tables())))
