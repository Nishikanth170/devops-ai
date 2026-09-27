"""Incident routes: browse, create, resolve and give feedback."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    UserContext,
    get_current_user,
    get_memory_service,
    get_orchestrator,
    get_session,
)
from app.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    IncidentCreateRequest,
    IncidentListResponse,
    IncidentResolveRequest,
    IncidentResponse,
    SuggestionResponse,
)
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.utils import utcnow
from app.events.routing import resolve_team_id
from app.memory.models import AgentSuggestion, Incident, IncidentStatus, Runbook, SeverityLevel
from app.memory.service import MemoryService
from app.orchestrator import AgentOrchestrator

router = APIRouter(prefix="/incidents", tags=["incidents"])

ACTIVE_STATUSES = (IncidentStatus.OPEN, IncidentStatus.INVESTIGATING, IncidentStatus.MITIGATED)


async def _suggestions_for(
    session: AsyncSession, incident_id: UUID, limit: int = 5
) -> list[AgentSuggestion]:
    result = await session.execute(
        select(AgentSuggestion)
        .where(AgentSuggestion.incident_id == incident_id)
        .order_by(desc(AgentSuggestion.created_at))
        .limit(limit)
    )
    return list(result.scalars().all())


def _ensure_team_access(user: UserContext, team_id: UUID) -> None:
    if not user.is_admin and user.team_id and team_id != user.team_id:
        raise PermissionDeniedError("Access to another team's incident is not permitted")


@router.get("", response_model=IncidentListResponse)
async def list_incidents(
    service: Optional[str] = Query(None),
    severity: Optional[SeverityLevel] = Query(None),
    incident_status: Optional[IncidentStatus] = Query(None, alias="status"),
    team_id: Optional[UUID] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> IncidentListResponse:
    """List incidents with filters, scoped to the caller's team unless admin."""
    query = select(Incident)
    scoped_team = user.scope_team_id(team_id)
    if scoped_team:
        query = query.where(Incident.team_id == scoped_team)
    if service:
        query = query.where(Incident.service == service)
    if severity:
        query = query.where(Incident.severity == severity)
    if incident_status:
        query = query.where(Incident.status == incident_status)

    total = int(await db.scalar(select(func.count()).select_from(query.subquery())) or 0)
    result = await db.execute(
        query.order_by(desc(Incident.started_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    incidents = list(result.scalars().all())

    return IncidentListResponse(
        incidents=[IncidentResponse.model_validate(incident) for incident in incidents],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/active", response_model=IncidentListResponse)
async def list_active_incidents(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> IncidentListResponse:
    """Open/investigating/mitigated incidents - the on-call view."""
    query = select(Incident).where(Incident.status.in_(ACTIVE_STATUSES))
    scoped_team = user.scope_team_id()
    if scoped_team:
        query = query.where(Incident.team_id == scoped_team)

    result = await db.execute(query.order_by(desc(Incident.started_at)).limit(100))
    incidents = list(result.scalars().all())
    return IncidentListResponse(
        incidents=[IncidentResponse.model_validate(incident) for incident in incidents],
        total=len(incidents),
        page=1,
        page_size=len(incidents) or 1,
    )


@router.post("", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
async def create_incident(
    payload: IncidentCreateRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> IncidentResponse:
    """Open an incident manually and (optionally) get an agent assessment."""
    team_id = payload.team_id or user.team_id or await resolve_team_id(
        db, service=payload.service
    )
    if not user.is_admin and user.team_id and team_id != user.team_id:
        raise PermissionDeniedError("Cannot open an incident for another team")

    incident = Incident(
        title=payload.title,
        description=payload.description,
        severity=SeverityLevel(payload.severity),
        status=IncidentStatus.OPEN,
        service=payload.service,
        team_id=team_id,
        runbook_id=payload.runbook_id,
        external_alert_id=payload.external_alert_id,
    )
    db.add(incident)
    await db.flush()

    if payload.run_agent:
        await orchestrator.process_incident_alert(
            {
                "title": incident.title,
                "service": incident.service,
                "severity": incident.severity.value,
                "description": incident.description or "",
                "incident_id": incident.id,
            },
            team_id,
        )

    await orchestrator.memory.record_incident(incident)
    await db.flush()
    return IncidentResponse.model_validate(incident)


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(
    incident_id: UUID,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    memory: MemoryService = Depends(get_memory_service),
) -> IncidentResponse:
    """Incident detail with agent suggestions and semantically similar history."""
    context = await memory.get_incident_with_context(incident_id)
    if context is None:
        raise NotFoundError("Incident not found")

    incident = context.incident
    _ensure_team_access(user, incident.team_id)

    response = IncidentResponse.model_validate(incident)
    response.agent_suggestions = [
        SuggestionResponse.model_validate(suggestion)
        for suggestion in await _suggestions_for(db, incident_id)
    ]
    response.similar_incidents = [
        {
            "incident_id": str(item.incident.id) if item.incident else None,
            "title": item.incident.title if item.incident else item.matched_text[:120],
            "similarity": round(item.similarity, 3),
            "root_cause": item.incident.root_cause if item.incident else None,
            "resolution_steps": item.incident.resolution_steps if item.incident else None,
        }
        for item in context.similar_incidents
    ]
    return response


@router.get("/{incident_id}/similar", response_model=list[dict])
async def similar_incidents(
    incident_id: UUID,
    top_k: int = Query(5, ge=1, le=20),
    user: UserContext = Depends(get_current_user),
    memory: MemoryService = Depends(get_memory_service),
) -> list[dict]:
    """Semantic retrieval only - handy for inspecting the memory layer."""
    context = await memory.get_incident_with_context(incident_id, similar_top_k=top_k)
    if context is None:
        raise NotFoundError("Incident not found")
    _ensure_team_access(user, context.incident.team_id)

    return [
        {
            "incident_id": str(item.incident.id) if item.incident else None,
            "title": item.incident.title if item.incident else None,
            "similarity": round(item.similarity, 3),
            "matched_text": item.matched_text,
            "runbook": item.runbook.name if item.runbook else None,
        }
        for item in context.similar_incidents
    ]


@router.post("/{incident_id}/resolve", response_model=IncidentResponse)
async def resolve_incident(
    incident_id: UUID,
    payload: IncidentResolveRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    memory: MemoryService = Depends(get_memory_service),
) -> IncidentResponse:
    """Close the loop: store root cause + resolution and re-embed the incident."""
    incident = await db.get(Incident, incident_id)
    if incident is None:
        raise NotFoundError("Incident not found")
    _ensure_team_access(user, incident.team_id)

    incident.root_cause = payload.root_cause
    incident.resolution_steps = payload.resolution_steps
    incident.status = IncidentStatus.RESOLVED
    incident.resolved_at = utcnow()
    if payload.runbook_id:
        incident.runbook_id = payload.runbook_id

    if incident.runbook_id and payload.runbook_success is not None:
        runbook = await db.get(Runbook, incident.runbook_id)
        if runbook is not None:
            runbook.last_used_at = utcnow()
            if payload.runbook_success:
                runbook.success_count += 1
            else:
                runbook.failure_count += 1

    # Replace the incident embedding with the resolved version so future
    # retrievals match on the actual fix, not only on the symptoms.
    await memory.vector_store.delete_by_source("incidents", incident.id)
    await memory.record_incident(incident)
    await db.flush()
    return IncidentResponse.model_validate(incident)


@router.post("/{incident_id}/feedback", response_model=FeedbackResponse)
async def incident_feedback(
    incident_id: UUID,
    payload: FeedbackRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    memory: MemoryService = Depends(get_memory_service),
) -> FeedbackResponse:
    """Rate the most recent agent suggestion for this incident."""
    incident = await db.get(Incident, incident_id)
    if incident is None:
        raise NotFoundError("Incident not found")
    _ensure_team_access(user, incident.team_id)

    suggestions = await _suggestions_for(db, incident_id, limit=1)
    if not suggestions:
        raise NotFoundError("No agent suggestion recorded for this incident")

    suggestion_id = suggestions[0].id
    await memory.record_feedback(suggestion_id, payload.feedback)
    return FeedbackResponse(status="recorded", suggestion_id=suggestion_id)
