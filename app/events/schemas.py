"""Internal event schemas.

``IntegrationEvent`` (vendor-shaped, produced by adapters) is normalised into
``InternalEvent`` (platform-shaped) exactly once, at the ingestion boundary.
Everything downstream - the queue, the processors, the agents - speaks only
``InternalEvent``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.utils import utcnow

EventPriority = Literal["low", "normal", "high", "critical"]


class EventTypes:
    """Canonical event type strings."""

    # Git events
    PR_OPENED = "pull_request.opened"
    PR_UPDATED = "pull_request.updated"
    PR_MERGED = "pull_request.merged"
    PR_CLOSED = "pull_request.closed"
    REVIEW_SUBMITTED = "pull_request.review_submitted"

    # CI/CD events
    BUILD_STARTED = "build.started"
    BUILD_COMPLETED = "build.completed"
    BUILD_FAILED = "build.failed"
    DEPLOYMENT_STARTED = "deployment.started"
    DEPLOYMENT_COMPLETED = "deployment.completed"
    DEPLOYMENT_FAILED = "deployment.failed"
    DEPLOYMENT_ROLLED_BACK = "deployment.rolled_back"

    # Monitoring events
    ALERT_FIRED = "alert.fired"
    ALERT_RESOLVED = "alert.resolved"
    INCIDENT_TRIGGERED = "incident.triggered"
    INCIDENT_ACKNOWLEDGED = "incident.acknowledged"
    INCIDENT_RESOLVED = "incident.resolved"

    # Ticketing events
    TICKET_CREATED = "ticket.created"
    TICKET_UPDATED = "ticket.updated"
    TICKET_CLOSED = "ticket.closed"

    #: Event types that always go to the priority queue.
    CRITICAL = frozenset(
        {
            INCIDENT_TRIGGERED,
            ALERT_FIRED,
            DEPLOYMENT_FAILED,
            DEPLOYMENT_ROLLED_BACK,
        }
    )
    HIGH = frozenset(
        {
            PR_OPENED,
            REVIEW_SUBMITTED,
            DEPLOYMENT_STARTED,
            BUILD_STARTED,
            BUILD_FAILED,
            INCIDENT_ACKNOWLEDGED,
        }
    )


def determine_priority(event_type: str) -> EventPriority:
    """Map an event type onto a queue priority lane."""
    if event_type in EventTypes.CRITICAL:
        return "critical"
    if event_type in EventTypes.HIGH:
        return "high"
    if event_type in {EventTypes.ALERT_RESOLVED, EventTypes.TICKET_CLOSED}:
        return "low"
    return "normal"


class InternalEvent(BaseModel):
    """Normalised, queueable platform event."""

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    event_type: str
    source: str
    team_id: Optional[UUID] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: EventPriority = "normal"
    created_at: datetime = Field(default_factory=utcnow)
    #: Vendor delivery id (``X-GitHub-Delivery``, ...) for traceability.
    delivery_id: Optional[str] = None
    #: Monotonic attempt counter maintained by the consumer on retry.
    attempts: int = 0

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> "InternalEvent":
        return cls.model_validate_json(raw)
