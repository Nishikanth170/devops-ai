"""Webhook ingestion: verify -> parse -> normalise -> enqueue.

The ingestion layer is deliberately thin: it never talks to the LLM and never
blocks on agent work, so vendor webhooks are acknowledged in milliseconds and
processed asynchronously by the queue consumers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Optional
from uuid import UUID

from app.core.config import Settings, get_settings
from app.core.database import session_scope
from app.core.exceptions import NotFoundError, WebhookVerificationError
from app.events.queue import BaseEventQueue
from app.events.routing import resolve_team_id
from app.events.schemas import InternalEvent, determine_priority
from app.integrations import build_adapters
from app.integrations.base import IntegrationAdapter, IntegrationEvent

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


class WebhookIngestion:
    """Entry point for ``POST /webhooks/{source}``."""

    def __init__(
        self,
        event_queue: BaseEventQueue,
        settings: Optional[Settings] = None,
        adapters: Optional[dict[str, IntegrationAdapter]] = None,
        session_factory: SessionFactory = session_scope,
    ) -> None:
        self.queue = event_queue
        self.settings = settings or get_settings()
        self.adapters = adapters if adapters is not None else build_adapters(self.settings)
        self._session_factory = session_factory

    @property
    def sources(self) -> list[str]:
        return sorted(self.adapters)

    def get_adapter(self, source: str) -> IntegrationAdapter:
        adapter = self.adapters.get(source)
        if adapter is None:
            raise NotFoundError(
                f"Unknown webhook source {source!r}. Known sources: {', '.join(self.sources)}"
            )
        return adapter

    async def handle_webhook(
        self,
        source: str,
        headers: Optional[Mapping[str, Any]],
        body: bytes,
    ) -> dict[str, Any]:
        """Verify, parse and enqueue every event contained in a webhook."""
        adapter = self.get_adapter(source)

        if not await adapter.verify_webhook(headers or {}, body):
            raise WebhookVerificationError(
                f"Signature verification failed for {source} webhook", source=source
            )

        events = adapter.parse_webhook(headers or {}, body)
        published: list[str] = []

        for event in events:
            internal = await self._to_internal_event(event)
            await self.queue.publish(internal)
            published.append(internal.event_type)
            logger.info(
                "ingested %s from %s (priority=%s, team=%s)",
                internal.event_type,
                internal.source,
                internal.priority,
                internal.team_id,
            )

        return {
            "status": "accepted",
            "source": source,
            "events_published": len(published),
            "event_types": published,
        }

    async def _to_internal_event(self, event: IntegrationEvent) -> InternalEvent:
        """Normalise a vendor event and resolve the owning team."""
        team_id = await self._resolve_team(event)
        priority = determine_priority(event.event_type)
        # Incidents and alerts are always critical, regardless of mapping drift.
        if event.event_type.startswith("incident.") and event.event_type.endswith("triggered"):
            priority = "critical"

        return InternalEvent(
            event_type=event.event_type,
            source=event.source,
            team_id=team_id,
            payload=event.payload,
            priority=priority,
            delivery_id=event.delivery_id,
        )

    async def _resolve_team(self, event: IntegrationEvent) -> Optional[UUID]:
        """Look the team up from configuration, creating it on first use."""
        payload = event.payload or {}
        repo = payload.get("repo") or payload.get("project")
        service = payload.get("service")

        try:
            async with self._session_factory() as session:
                return await resolve_team_id(
                    session, repo=repo, service=service, settings=self.settings
                )
        except Exception as exc:  # pragma: no cover - depends on database availability
            logger.warning("could not resolve team for %s: %s", event.event_type, exc)
            return None
