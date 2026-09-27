"""Event layer tests: queue semantics, ingestion and full event flows."""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import func, select

from app.events.ingestion import WebhookIngestion
from app.events.processors import EventProcessors
from app.events.queue import (
    InMemoryEventQueue,
    RedisEventQueue,
    build_event_queue,
    priority_for,
)
from app.events.routing import get_or_create_team, resolve_team_id, team_name_for
from app.events.schemas import InternalEvent, determine_priority
from app.memory.models import (
    CodeReview,
    Deployment,
    DeploymentStatus,
    Incident,
    IncidentStatus,
    ReviewComment,
)


class TestPriorities:
    def test_determine_priority(self):
        assert determine_priority("incident.triggered") == "critical"
        assert determine_priority("pull_request.opened") == "high"
        assert determine_priority("ticket.closed") == "low"
        assert determine_priority("something.unknown") == "normal"

    def test_queue_backend_selection(self, settings):
        assert isinstance(build_event_queue(settings), InMemoryEventQueue)
        settings.QUEUE_BACKEND = "redis"
        assert isinstance(build_event_queue(settings), RedisEventQueue)


class TestInMemoryQueue:
    @pytest.mark.asyncio
    async def test_publish_and_consume(self, settings):
        queue = InMemoryEventQueue(settings)
        seen: list[str] = []

        async def record(event):
            seen.append(event.event_type)

        queue.register_handler("ping", record)
        event = InternalEvent(event_type="ping", source="test", priority="normal")
        await queue.publish(event)
        assert await queue.drain() == 1
        assert seen == ["ping"]

    @pytest.mark.asyncio
    async def test_critical_events_jump_the_queue(self, settings):
        queue = InMemoryEventQueue(settings)
        order: list[str] = []

        async def record_low(event):
            order.append(event.event_type)

        async def record_critical(event):
            order.append(event.event_type)

        queue.register_handler("low", record_low)
        queue.register_handler("critical-job", record_critical)
        await queue.publish(InternalEvent(event_type="low", source="t", priority="low"))
        await queue.publish(
            InternalEvent(event_type="critical-job", source="t", priority="critical")
        )
        await queue.drain()
        assert order == ["critical-job", "low"]

    @pytest.mark.asyncio
    async def test_handler_failures_land_in_dead_letters(self, settings):
        queue = InMemoryEventQueue(settings)

        async def boom(event):
            raise RuntimeError("kaput")

        queue.register_handler("bad", boom)
        await queue.publish(InternalEvent(event_type="bad", source="t"))
        await queue.drain()
        assert queue.failed == 1
        assert len(queue.dead_letters) == 1
        stats = queue.stats()
        assert stats["published"] == 1 and stats["processed"] == 0

    def test_priority_for(self):
        assert priority_for("alert.fired") == "critical"


class TestTeamRouting:
    @pytest.mark.asyncio
    async def test_get_or_create_is_stable(self, db_session):
        first = await get_or_create_team(db_session, "payments")
        second = await get_or_create_team(db_session, "payments")
        assert first.id == second.id

    def test_team_name_for_prefers_repo_leaf(self, settings):
        settings.TEAM_ROUTING = {"acme/payments": "payments"}
        assert team_name_for(repo="acme/payments", settings=settings) == "payments"
        assert team_name_for(service="checkout", settings=settings) == settings.DEFAULT_TEAM_NAME

    @pytest.mark.asyncio
    async def test_resolve_team_id_creates_team(self, db_session, settings):
        team_id = await resolve_team_id(db_session, service="checkout", settings=settings)
        assert isinstance(team_id, UUID)


def _session_factory_for(db_session):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def factory():
        yield db_session

    return factory


class TestProcessorFlows:
    @pytest.mark.asyncio
    async def test_alert_creates_incident_and_suggestion(
        self, settings, fake_llm, db_session, team
    ):
        from app.events.queue import InMemoryEventQueue

        queue: InMemoryEventQueue = InMemoryEventQueue(settings)
        processors = EventProcessors(
            queue, fake_llm, settings, session_factory=_session_factory_for(db_session)
        )
        processors.adapters = {}  # no outbound integrations in tests

        event = InternalEvent(
            event_type="alert.fired",
            source="datadog",
            team_id=team.id,
            priority="critical",
            payload={
                "id": "alert-1",
                "title": "Pool saturation",
                "service": "payments",
                "severity": "SEV2",
                "description": "connection pool 95% full during traffic spike",
            },
        )
        await processors.handle_alert_event(event)

        incident = (
            await db_session.execute(select(Incident).where(Incident.team_id == team.id))
        ).scalars().one()
        assert incident.title == "Pool saturation"
        assert incident.status == IncidentStatus.OPEN
        assert incident.severity.value == "SEV2"

    @pytest.mark.asyncio
    async def test_pr_event_persists_findings(
        self, settings, fake_llm, db_session, team
    ):
        from app.events.queue import InMemoryEventQueue

        queue: InMemoryEventQueue = InMemoryEventQueue(settings)
        processors = EventProcessors(
            queue, fake_llm, settings, session_factory=_session_factory_for(db_session)
        )
        processors.adapters = {}

        event = InternalEvent(
            event_type="pull_request.opened",
            source="github",
            team_id=team.id,
            priority="high",
            payload={
                "number": 42,
                "repo": "acme/payments",
                "title": "Fix connection leak",
                "description": "Bounds the pool.",
                "author": "dev",
                "diff": "diff --git a/app/db.py b/app/db.py\n+++ b/app/db.py\n+pool = get_pool()",
            },
        )
        review_id = await processors.handle_pr_event(event)
        assert isinstance(review_id, UUID)

        review = await db_session.get(CodeReview, review_id)
        assert review is not None
        assert review.status == "reviewed"

        comment_count = await db_session.scalar(
            select(func.count())
            .select_from(ReviewComment)
            .where(ReviewComment.review_id == review.id)
        )
        assert (comment_count or 0) >= 1

    @pytest.mark.asyncio
    async def test_deployment_lifecycle(
        self, settings, fake_llm, db_session, team
    ):
        from app.events.queue import InMemoryEventQueue

        queue: InMemoryEventQueue = InMemoryEventQueue(settings)
        processors = EventProcessors(
            queue, fake_llm, settings, session_factory=_session_factory_for(db_session)
        )
        processors.adapters = {}

        started = InternalEvent(
            event_type="deployment.started",
            source="github",
            team_id=team.id,
            priority="high",
            payload={
                "service": "payments",
                "commit_sha": "abc123",
                "environment": "production",
                "changes": [
                    {
                        "change_type": "database_migration",
                        "description": "add index",
                        "file_paths": ["migrations/001.sql"],
                    }
                ],
            },
        )
        deployment_id = await processors.handle_deployment_started(started)
        deployment = await db_session.get(Deployment, deployment_id)
        assert deployment is not None
        assert deployment.status == DeploymentStatus.IN_PROGRESS
        # intrinsic migration risk (0.8 * 0.4 = 0.32) plus baseline history
        assert 0.0 < deployment.risk_score <= 0.95

        failed = InternalEvent(
            event_type="deployment.failed",
            source="github",
            team_id=team.id,
            priority="critical",
            payload={
                "service": "payments",
                "commit_sha": "abc123",
                "failure_reason": "migration deadlock",
            },
        )
        assert await processors.handle_deployment_failed(failed) == deployment_id
        await db_session.refresh(deployment)
        assert deployment.status == DeploymentStatus.FAILED
        assert deployment.failure_reason == "migration deadlock"

    @pytest.mark.asyncio
    async def test_ingestion_rejects_bad_signature(self, settings, db_session):
        from app.core.exceptions import WebhookVerificationError
        from app.events.queue import InMemoryEventQueue
        from app.integrations import build_adapters

        queue: InMemoryEventQueue = InMemoryEventQueue(settings)
        adapters = build_adapters(settings)
        ingestion = WebhookIngestion(
            queue, settings, adapters=adapters,
            session_factory=_session_factory_for(db_session),
        )
        with pytest.raises(WebhookVerificationError):
            await ingestion.handle_webhook("github", {}, b'{"action":"opened"}')
