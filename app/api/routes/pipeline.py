"""Pipeline routes: deployment history, detail and on-demand risk assessment."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    UserContext,
    get_current_user,
    get_orchestrator,
    get_session,
)
from app.api.schemas import (
    DeploymentResponse,
    FeedbackRequest,
    FeedbackResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
)
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.utils import utcnow
from app.memory.models import AgentSuggestion, Deployment, DeploymentStatus, PipelineChange
from app.orchestrator import AgentOrchestrator

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

ACTIVE_STATUSES = (DeploymentStatus.IN_PROGRESS, DeploymentStatus.SUCCESS)


def _ensure_team_access(user: UserContext, team_id: UUID) -> None:
    if not user.is_admin and user.team_id and team_id != user.team_id:
        raise PermissionDeniedError("Access to another team's deployment is not permitted")


async def _with_changes(
    session: AsyncSession, deployments: list[Deployment]
) -> list[DeploymentResponse]:
    responses: list[DeploymentResponse] = []
    for deployment in deployments:
        changes = list(
            (
                await session.execute(
                    select(PipelineChange).where(
                        PipelineChange.deployment_id == deployment.id
                    )
                )
            ).scalars().all()
        )
        response = DeploymentResponse.model_validate(deployment)
        response.changes = changes  # type: ignore[assignment]
        responses.append(response)
    return responses


@router.get("/history", response_model=list[DeploymentResponse])
async def deployment_history(
    service: Optional[str] = Query(None),
    environment: Optional[str] = Query("production"),
    deploy_status: Optional[DeploymentStatus] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> list[DeploymentResponse]:
    """Deployment history with risk scores and change sets."""
    query = select(Deployment)
    scoped_team = user.scope_team_id()
    if scoped_team:
        query = query.where(Deployment.team_id == scoped_team)
    if service:
        query = query.where(Deployment.service == service)
    if environment:
        query = query.where(Deployment.environment == environment)
    if deploy_status:
        query = query.where(Deployment.status == deploy_status)

    result = await db.execute(
        query.order_by(desc(Deployment.started_at)).limit(limit)
    )
    return await _with_changes(db, list(result.scalars().all()))


@router.get("/stats", response_model=dict)
async def failure_stats(
    days: int = Query(30, ge=1, le=365),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict:
    """Failure rate and top risky services over the last ``days`` days."""
    from datetime import timedelta

    from sqlalchemy import case

    cutoff = utcnow() - timedelta(days=days)

    query = select(
        Deployment.service,
        func.count(Deployment.id).label("total"),
        func.sum(
            case(
                (Deployment.status.in_([DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]), 1),
                else_=0,
            )
        ).label("failures"),
        func.avg(Deployment.risk_score).label("avg_risk"),
    ).group_by(Deployment.service)
    scoped_team = user.scope_team_id()
    if scoped_team:
        query = query.where(Deployment.team_id == scoped_team)
    query = query.where(Deployment.started_at >= cutoff).order_by(desc("failures"))

    rows = (await db.execute(query)).all()
    return {
        "window_days": days,
        "by_service": [
            {
                "service": row.service,
                "total": int(row.total),
                "failures": int(row.failures or 0),
                "failure_rate": round((row.failures or 0) / row.total, 3)
                if row.total
                else 0.0,
                "avg_risk_score": round(float(row.avg_risk or 0.0), 3),
            }
            for row in rows
        ],
    }


@router.get("/active", response_model=list[DeploymentResponse])
async def active_deployments(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> list[DeploymentResponse]:
    """In-flight and recently completed deployments (the pipeline watch view)."""
    query = select(Deployment).where(Deployment.status.in_(ACTIVE_STATUSES))
    scoped_team = user.scope_team_id()
    if scoped_team:
        query = query.where(Deployment.team_id == scoped_team)
    result = await db.execute(query.order_by(desc(Deployment.started_at)).limit(50))
    return await _with_changes(db, list(result.scalars().all()))


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: UUID,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> DeploymentResponse:
    """Deployment detail with its changes and the agent's risk notes."""
    deployment = await db.get(Deployment, deployment_id)
    if deployment is None:
        raise NotFoundError("Deployment not found")
    _ensure_team_access(user, deployment.team_id)
    hydrated = await _with_changes(db, [deployment])
    return hydrated[0]


@router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_risk(
    payload: RiskAssessmentRequest,
    user: UserContext = Depends(get_current_user),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> RiskAssessmentResponse:
    """Score a hypothetical deployment before it is started (pre-flight gate)."""
    changes = payload.changes or []
    response = await orchestrator.process_deployment_event(
        {
            "service": payload.service,
            "commit_sha": payload.commit_sha,
            "environment": payload.environment,
            "status": "in_progress",
            "changes": changes,
            "deployment_id": str(payload.deployment_id) if payload.deployment_id else None,
            "event_type": "deployment",
        },
        user.team_id,
    )
    metadata = response.metadata
    return RiskAssessmentResponse(
        risk_score=float(metadata.get("risk_score", 0.0)),
        risk_band=str(metadata.get("risk_band", "medium")),
        confidence=response.confidence,
        response=response.response,
        risk_factors=response.suggestions,
        recommendations=response.follow_up_actions,
        similar_deployments=[str(match) for match in response.historical_matches],
    )


@router.post("/deployments/{deployment_id}/feedback", response_model=FeedbackResponse)
async def deployment_feedback(
    deployment_id: UUID,
    payload: FeedbackRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> FeedbackResponse:
    """Rate the agent's risk assessment for a deployment."""
    deployment = await db.get(Deployment, deployment_id)
    if deployment is None:
        raise NotFoundError("Deployment not found")
    _ensure_team_access(user, deployment.team_id)

    suggestion = (
        await db.execute(
            select(AgentSuggestion)
            .where(AgentSuggestion.deployment_id == deployment.id)
            .order_by(desc(AgentSuggestion.created_at))
        )
    ).scalars().first()
    if suggestion is None:
        raise NotFoundError("No agent suggestion recorded for this deployment")

    suggestion.user_feedback = payload.feedback
    suggestion.feedback_at = utcnow()
    return FeedbackResponse(status="recorded", suggestion_id=suggestion.id)
