"""FastAPI application factory.

``create_app()`` wires configuration, middleware, routers and the lifespan that
builds/tears down the :class:`~app.container.AppServices` container.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app import __version__
from app.api.routes import ROUTERS
from app.api.schemas import MessageResponse
from app.auth.oauth2 import router as auth_router
from app.auth.rbac import ROLE_ADMIN, ROLE_TEAM_LEAD, require_role
from app.container import AppServices
from app.core.config import Settings, get_settings
from app.core.database import close_db, init_db
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
AI-powered engineering and DevOps agent platform: incident response, code
review and pipeline risk agents backed by a hybrid (vector + relational)
memory layer, driven by inbound webhooks and an internal event queue.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build services, prepare the database, start the queue consumer."""
    settings: Settings = get_settings()
    settings.assert_production_ready()
    configure_logging()
    logger.info(
        "starting %s (%s, llm=%s, vector=%s, queue=%s)",
        settings.APP_NAME,
        settings.ENVIRONMENT,
        settings.LLM_PROVIDER,
        settings.VECTOR_BACKEND,
        settings.QUEUE_BACKEND,
    )

    await init_db()
    services = AppServices.build(settings)
    if settings.EVENT_CONSUMER_IN_PROCESS:
        await services.queue.start_consumer()
    app.state.services = services

    logger.info("ready: %d queue handlers registered", services.queue.handler_count())
    yield

    logger.info("shutting down: %s", services.queue.stats())
    await services.shutdown()
    await close_db()
    app.state.services = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (also the module-level ``app`` for uvicorn)."""
    settings = settings or get_settings()

    application = FastAPI(
        title=settings.APP_NAME,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        debug=settings.DEBUG,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(application)

    for router in ROUTERS:
        application.include_router(router)
    application.include_router(auth_router)

    # ---------------------------------------------------------- health/docs
    @application.get("/health", tags=["system"], response_model=MessageResponse)
    async def health_check() -> MessageResponse:
        return MessageResponse(status="healthy", detail=settings.APP_NAME)

    @application.get("/health/ready", tags=["system"])
    async def readiness(request: Request) -> dict:
        """Database + queue readiness (used by container orchestrators)."""
        from app.core.database import session_scope

        checks: dict[str, str] = {"database": "ok", "queue": "ok"}
        try:
            async with session_scope() as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:  # pragma: no cover - requires a live database
            checks["database"] = f"error: {exc}"

        services = getattr(request.app.state, "services", None)
        if services is None:
            checks["queue"] = "error: services not initialised"

        ready = all(not value.startswith("error") for value in checks.values())
        detail: dict[str, object] = {"checks": checks}
        if services is not None:
            detail["queue"] = services.queue.stats()
            detail["handlers"] = services.queue.handler_count()
        return {"status": "ready" if ready else "degraded", **detail}

    @application.get(
        "/admin/llm-stats",
        tags=["admin"],
        dependencies=[Depends(require_role(ROLE_ADMIN, ROLE_TEAM_LEAD))],
    )
    async def llm_stats(request: Request) -> dict:
        """Token/cost, memory effectiveness and queue counters."""
        services: AppServices = request.app.state.services
        return {
            "llm": services.llm.get_usage_stats(),
            "memory": await _memory_stats(services),
            "queue": services.queue.stats(),
        }

    return application


async def _memory_stats(services: AppServices) -> dict:
    """Memory effectiveness stats using a short-lived session of its own."""
    from app.core.database import session_scope
    from app.memory.service import MemoryService

    async with session_scope() as session:
        memory = MemoryService(session, services.llm, services.settings)
        return await memory.get_retrieval_stats()


#: Uvicorn entry point: ``uvicorn app.main:app``
app = create_app()


def main() -> None:  # pragma: no cover - manual runner
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
