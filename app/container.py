"""Application service container.

One instance is created in the FastAPI lifespan and stored on ``app.state``;
dependencies (``app/api/deps.py``) read from it.  This replaces module-level
globals *and* avoids the "one session for the whole process" bug: the
orchestrator is built per request/event from a request-scoped ``MemoryService``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings, get_settings
from app.events.ingestion import WebhookIngestion
from app.events.processors import EventProcessors
from app.events.queue import BaseEventQueue, build_event_queue
from app.integrations import build_adapters
from app.integrations.base import IntegrationAdapter
from app.llm.gateway import BaseLLMGateway, build_llm_gateway


@dataclass
class AppServices:
    """Long-lived collaborators shared by the API and the queue consumers."""

    settings: Settings
    llm: BaseLLMGateway
    queue: BaseEventQueue
    ingestion: WebhookIngestion
    processors: EventProcessors
    adapters: dict[str, IntegrationAdapter]

    @classmethod
    def build(cls, settings: Optional[Settings] = None) -> "AppServices":
        settings = settings or get_settings()
        llm = build_llm_gateway(settings)
        queue = build_event_queue(settings)
        adapters = build_adapters(settings)
        processors = EventProcessors(queue, llm, settings, adapters=adapters)
        processors.register(queue)
        ingestion = WebhookIngestion(queue, settings, adapters=adapters)
        return cls(
            settings=settings,
            llm=llm,
            queue=queue,
            ingestion=ingestion,
            processors=processors,
            adapters=adapters,
        )

    async def shutdown(self) -> None:
        """Stop consumers and release external resources."""
        await self.queue.stop()
        for adapter in self.adapters.values():
            closer = getattr(adapter, "aclose", None)
            if closer is not None:
                await closer()
        await self.llm.aclose()
