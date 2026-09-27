"""Ad-hoc agent query endpoint (chat interface to any specialist agent)."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentType
from app.api.deps import (
    UserContext,
    get_current_user,
    get_orchestrator,
    get_session,
)
from app.api.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    FeedbackRequest,
    FeedbackResponse,
    SuggestionResponse,
)
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.utils import utcnow
from app.memory.models import AgentSuggestion, CodeReview, Deployment, Incident
from app.orchestrator import AgentOrchestrator

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/types", response_model=list[str])
async def agent_types() -> list[str]:
    """List the available agent types (useful for clients and discovery)."""
    return [agent.value for agent in AgentType]


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
    payload: AgentQueryRequest,
    user: UserContext = Depends(get_current_user),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> AgentQueryResponse:
    """Ask any agent a question; routing is automatic unless ``agent_type`` is set."""
    context = dict(payload.context or {})
    if payload.agent_type:
        context["agent_type"] = payload.agent_type

    response = await orchestrator.process(
        query=payload.query,
        payload=context,
        user_id=user.user_id,
        team_id=user.team_id,
        conversation_history=payload.conversation_history,
        agent_type=AgentType(payload.agent_type) if payload.agent_type else None,
    )

    data = response.as_dict()
    data["agent_type"] = str(
        data.get("metadata", {}).get("agent_type") or payload.agent_type or "unknown"
    )
    return AgentQueryResponse(**data)


@router.get("/suggestions", response_model=list[SuggestionResponse])
async def list_suggestions(
    agent_type: Optional[str] = Query(None),
    unrated_only: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> list[SuggestionResponse]:
    """Recent agent suggestions (optionally only those still unrated)."""
    if not user.is_admin and user.team_id is None:
        # No team scope -> nothing is visible (admins see everything).
        return []

    query = select(AgentSuggestion)
    if agent_type:
        query = query.where(AgentSuggestion.agent_type == agent_type)
    if unrated_only:
        query = query.where(AgentSuggestion.user_feedback.is_(None))
    if not user.is_admin:
        incident_ids = select(Incident.id).where(Incident.team_id == user.team_id)
        review_ids = select(CodeReview.id).where(CodeReview.team_id == user.team_id)
        deployment_ids = select(Deployment.id).where(Deployment.team_id == user.team_id)
        query = query.where(
            or_(
                AgentSuggestion.incident_id.in_(incident_ids),
                AgentSuggestion.review_id.in_(review_ids),
                AgentSuggestion.deployment_id.in_(deployment_ids),
            )
        )

    result = await db.execute(query.order_by(desc(AgentSuggestion.created_at)).limit(limit))
    return [SuggestionResponse.model_validate(row) for row in result.scalars().all()]


@router.post("/suggestions/{suggestion_id}/feedback", response_model=FeedbackResponse)
async def suggestion_feedback(
    suggestion_id: UUID,
    payload: FeedbackRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> FeedbackResponse:
    """Rate any agent suggestion - feeds the retrieval quality loop."""
    suggestion = await db.get(AgentSuggestion, suggestion_id)
    if suggestion is None:
        raise NotFoundError("Suggestion not found")

    if not user.is_admin and suggestion.incident_id and user.team_id:
        incident = await db.get(Incident, suggestion.incident_id)
        if incident is not None and incident.team_id != user.team_id:
            raise PermissionDeniedError(
                "Access to another team's suggestion is not permitted"
            )

    suggestion.user_feedback = payload.feedback
    suggestion.feedback_at = utcnow()
    return FeedbackResponse(status="recorded", suggestion_id=suggestion.id)
