"""FastAPI dependency wiring.

Everything the routes need is derived from two roots: the request-scoped
database session and the process-wide :class:`AppServices` container.  That
keeps route handlers free of construction logic and makes them trivial to test
by overriding ``get_services``/``get_session``.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.rbac import UserContext, get_current_user, require_role
from app.container import AppServices
from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.core.exceptions import PlatformError
from app.events.ingestion import WebhookIngestion
from app.events.queue import BaseEventQueue
from app.integrations.base import IntegrationAdapter
from app.llm.gateway import BaseLLMGateway
from app.memory.service import MemoryService
from app.orchestrator import AgentOrchestrator


def get_services(request: Request) -> AppServices:
    """Return the service container created by the application lifespan."""
    services = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - only when lifespan is bypassed
        raise PlatformError(
            "Application services are not initialised (lifespan did not run)"
        )
    return services


def get_settings_dep() -> Settings:
    """Injectable settings accessor (overridable in tests)."""
    return get_settings()


def get_llm_gateway(services: AppServices = Depends(get_services)) -> BaseLLMGateway:
    return services.llm


def get_event_queue(services: AppServices = Depends(get_services)) -> BaseEventQueue:
    return services.queue


def get_webhook_ingestion(
    services: AppServices = Depends(get_services),
) -> WebhookIngestion:
    return services.ingestion


def get_adapters(
    services: AppServices = Depends(get_services),
) -> dict[str, IntegrationAdapter]:
    return services.adapters


def get_memory_service(
    db: AsyncSession = Depends(get_session),
    llm: BaseLLMGateway = Depends(get_llm_gateway),
    settings: Settings = Depends(get_settings_dep),
) -> MemoryService:
    """Memory service bound to this request's session."""
    return MemoryService(db, llm, settings)


def get_orchestrator(
    memory: MemoryService = Depends(get_memory_service),
    llm: BaseLLMGateway = Depends(get_llm_gateway),
    settings: Settings = Depends(get_settings_dep),
) -> AgentOrchestrator:
    """Orchestrator bound to this request's session (no shared state)."""
    return AgentOrchestrator(memory, llm, settings)


__all__: list[str] = [
    "UserContext",
    "get_adapters",
    "get_current_user",
    "get_event_queue",
    "get_llm_gateway",
    "get_memory_service",
    "get_orchestrator",
    "get_services",
    "get_session",
    "get_settings_dep",
    "get_webhook_ingestion",
    "require_role",
]

# Re-exported for annotating handler return values in tests.
JsonDict = dict[str, Any]
