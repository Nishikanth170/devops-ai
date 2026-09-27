"""Test configuration.

The suite runs fully offline:

* ``ENVIRONMENT=test``, ``LLM_PROVIDER=fake`` (deterministic hashed embeddings
  and structural chat answers - no API key, no network).
* ``VECTOR_BACKEND=memory`` (in-process embeddings - no pgvector needed).
* ``QUEUE_BACKEND=memory`` (asyncio priority queue - no Redis needed).
* A throwaway SQLite file database (deleted after the session) so the full
  relational + FastAPI stack is exercised for real.

Env vars are set *before* any ``app.*`` module is imported, because the vector
store reads the embedding column type at class-definition time.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TEST_DB_DIR = Path(__file__).resolve().parent / "_tmp"
_TEST_DB_PATH = _TEST_DB_DIR / f"test-{uuid.uuid4().hex}.db"

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("VECTOR_BACKEND", "memory")
os.environ.setdefault("QUEUE_BACKEND", "memory")
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_DB_PATH.as_posix()}")
os.environ.setdefault("DATABASE_CREATE_SCHEMA", "true")
os.environ.setdefault("VECTOR_DIMENSIONS", "64")
os.environ.setdefault("VECTOR_SIMILARITY_THRESHOLD", "0.3")
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret")
os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-github-secret")
os.environ.setdefault("GITLAB_WEBHOOK_SECRET", "test-gitlab-secret")
os.environ.setdefault("JENKINS_WEBHOOK_SECRET", "test-jenkins-secret")
os.environ.setdefault("DEFAULT_TEAM_NAME", "platform-test")

_TEST_DB_DIR.mkdir(exist_ok=True)

from app.auth.rbac import UserContext  # noqa: E402
from app.core.config import get_settings, reset_settings_cache  # noqa: E402
from app.core.database import Base  # noqa: E402
from app.llm.fake import FakeLLMGateway  # noqa: E402
from app.main import create_app  # noqa: E402
from app.memory.models import Team, User  # noqa: E402
from app.memory.service import MemoryService  # noqa: E402
from app.memory.vector_store import VectorStore  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db_file():
    yield
    try:
        _TEST_DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass


@pytest_asyncio.fixture(autouse=True)
async def _clear_vector_memory():
    yield
    VectorStore.clear_memory_backend()


@pytest.fixture()
def settings():
    reset_settings_cache()
    return get_settings()


@pytest_asyncio.fixture()
async def db_engine(settings):
    """Per-test database with the relational schema (minus vector_memory)."""
    from app.memory import models as _models  # noqa: F401
    from app.memory import vector_store as _vector_store  # noqa: F401

    engine = create_async_engine(settings.DATABASE_URL, future=True)
    tables = [t for t in Base.metadata.sorted_tables if t.name != "vector_memory"]
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all, tables=list(tables))
        await conn.run_sync(Base.metadata.create_all, tables=list(tables))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session(db_engine) -> AsyncIterator[AsyncSession]:
    """Request/processor-scoped session on the per-test database.

    Both HTTP requests (via lifespan ``init_db``) and worker-style processors
    must share *these* semantics: the per-test engine and its schema. The
    lifespan in ``api_client`` below rebinds the global factories onto
    ``db_engine`` so ``session_scope()`` sees the same rows the tests seed.
    """
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture()
async def api_client(db_engine) -> AsyncIterator[AsyncClient]:
    """Full FastAPI app over the per-test SQLite file database.

    The lifespan runs ``init_db`` and starts an in-memory event consumer, so
    webhook -> queue -> processor flows run end to end.
    """
    import app.core.database as database

    previous_engine = database._engine
    previous_factory = database._session_factory
    database._engine = db_engine
    database._session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    app = create_app()
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                yield client
    finally:
        database._engine = previous_engine
        database._session_factory = previous_factory


@pytest.fixture()
def fake_llm(settings) -> FakeLLMGateway:
    return FakeLLMGateway(settings)


@pytest_asyncio.fixture()
async def memory_service(db_session, fake_llm, settings) -> MemoryService:
    return MemoryService(db_session, fake_llm, settings)


async def seed_team(session: AsyncSession, name: str = "platform-test") -> Team:
    """Seed and commit: HTTP requests and processors use their own sessions."""
    team = Team(name=name)
    session.add(team)
    await session.flush()
    await session.commit()
    return team


async def seed_user(
    session: AsyncSession,
    team: Team,
    email: str = "oncall@example.com",
    role: str = "admin",
    name: str = "Oncall Test",
) -> User:
    user = User(
        email=email,
        name=name,
        role=role,
        team_id=team.id,
        external_id=f"sso-{email}",
    )
    session.add(user)
    await session.flush()
    await session.commit()
    return user


@pytest_asyncio.fixture()
async def team(db_session) -> Team:
    return await seed_team(db_session)


@pytest_asyncio.fixture()
async def admin_user(db_session, team) -> User:
    return await seed_user(db_session, team, role="admin")


@pytest_asyncio.fixture()
async def user_ctx(admin_user) -> UserContext:
    return UserContext(
        user_id=admin_user.id,
        email=admin_user.email,
        name=admin_user.name,
        role=admin_user.role,
        team_id=admin_user.team_id,
    )


@pytest.fixture()
def auth_headers(settings):
    """Bearer-token factory for API tests (bypasses SSO for fixture users)."""
    from jose import jwt

    def _headers(user: User) -> dict[str, str]:
        token = jwt.encode(
            {"sub": user.external_id, "email": user.email, "type": "access"},
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        return {"Authorization": f"Bearer {token}"}

    return _headers


@pytest.fixture()
def make_incident():
    from app.memory.models import Incident, IncidentStatus, SeverityLevel

    def _make(
        team_id,
        title="Database pool exhausted",
        service="payments",
        severity=SeverityLevel.SEV2,
        status=IncidentStatus.RESOLVED,
        root_cause="connection leak in retry logic",
        resolution="bounded pool size, fixed retry backoff",
    ) -> Incident:
        return Incident(
            title=title,
            description="connections exhausted during traffic spike",
            service=service,
            severity=severity,
            status=status,
            team_id=team_id,
            root_cause=root_cause,
            resolution_steps=resolution,
        )

    return _make
