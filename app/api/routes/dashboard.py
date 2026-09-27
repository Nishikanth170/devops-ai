"""Dashboard routes: aggregate KPIs + agent/queue/memory observability."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    UserContext,
    get_current_user,
    get_event_queue,
    get_llm_gateway,
    get_memory_service,
    get_session,
)
from app.api.schemas import AgentStatsResponse, DashboardSummaryResponse
from app.auth.rbac import ROLE_ADMIN, ROLE_TEAM_LEAD, require_role
from app.core.utils import utcnow
from app.events.queue import BaseEventQueue
from app.llm.gateway import BaseLLMGateway
from app.memory.models import (
    CodeReview,
    Deployment,
    DeploymentStatus,
    Incident,
    IncidentStatus,
)
from app.memory.service import MemoryService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

OPEN_STATUSES = [
    IncidentStatus.OPEN,
    IncidentStatus.INVESTIGATING,
    IncidentStatus.MITIGATED,
]
FAILED_STATUSES = [DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]
# Query with enum members: SQLAlchemy binds them to whatever the column stores
# (member names on PostgreSQL enums, values elsewhere), so this stays portable.


@router.get("/summary", response_model=DashboardSummaryResponse)
async def dashboard_summary(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> DashboardSummaryResponse:
    """Overview metrics for the engineering dashboard."""
    now = utcnow()
    since_24h = now - timedelta(hours=24)
    since_30d = now - timedelta(days=30)

    team_filter: Optional[Any] = user.scope_team_id()

    # --- open incidents -----------------------------------------------------
    incident_stmt = select(func.count(Incident.id)).where(Incident.status.in_(OPEN_STATUSES))
    if team_filter:
        incident_stmt = incident_stmt.where(Incident.team_id == team_filter)
    open_incidents = int(await db.scalar(incident_stmt) or 0)

    # --- PRs awaiting review ------------------------------------------------
    review_stmt = select(func.count(CodeReview.id)).where(
        CodeReview.status.in_(["open", "reviewed"])
    )
    if team_filter:
        review_stmt = review_stmt.where(CodeReview.team_id == team_filter)
    prs_awaiting = int(await db.scalar(review_stmt) or 0)

    # --- recent deployments -------------------------------------------------
    deploy_stmt = select(func.count(Deployment.id)).where(
        Deployment.started_at >= since_24h
    )
    if team_filter:
        deploy_stmt = deploy_stmt.where(Deployment.team_id == team_filter)
    recent_deployments = int(await db.scalar(deploy_stmt) or 0)

    # --- deployment failure rate (30d) --------------------------------------
    failure_stmt = select(
        func.count(Deployment.id),
        func.sum(
            case((Deployment.status.in_(FAILED_STATUSES), 1), else_=0)
        ),
    ).where(Deployment.started_at >= since_30d)
    if team_filter:
        failure_stmt = failure_stmt.where(Deployment.team_id == team_filter)
    total_deploys, failed_deploys = (await db.execute(failure_stmt)).one()
    failure_rate = (failed_deploys or 0) / total_deploys if total_deploys else 0.0

    # --- incident trend (daily) ---------------------------------------------
    trend_stmt = (
        select(func.date(Incident.started_at).label("day"), func.count(Incident.id))
        .where(Incident.started_at >= since_30d)
        .group_by("day")
        .order_by("day")
    )
    if team_filter:
        trend_stmt = trend_stmt.where(Incident.team_id == team_filter)
    incident_trend = [
        {"date": str(row.day), "count": int(row[1])} for row in (await db.execute(trend_stmt)).all()
    ]

    # --- failures by service ------------------------------------------------
    by_service_stmt = (
        select(
            Deployment.service,
            func.count(Deployment.id).label("total"),
            func.sum(
                case((Deployment.status.in_(FAILED_STATUSES), 1), else_=0)
            ).label("failures"),
        )
        .where(Deployment.started_at >= since_30d)
        .group_by(Deployment.service)
        .order_by(desc("failures"))
    )
    if team_filter:
        by_service_stmt = by_service_stmt.where(Deployment.team_id == team_filter)
    failure_by_service = [
        {
            "service": row.service,
            "total": int(row.total),
            "failures": int(row.failures or 0),
            "failure_rate": round((row.failures or 0) / row.total, 3) if row.total else 0.0,
        }
        for row in (await db.execute(by_service_stmt)).all()
    ]

    # --- review turnaround (computed in Python for dialect portability) -----
    review_stmt = select(
        CodeReview.created_at, CodeReview.merged_at
    ).where(
        CodeReview.merged_at.is_not(None), CodeReview.created_at >= since_30d
    )
    if team_filter:
        review_stmt = review_stmt.where(CodeReview.team_id == team_filter)
    turnaround_buckets: dict[str, list[float]] = defaultdict(list)
    for created_at, merged_at in (await db.execute(review_stmt)).all():
        hours = (merged_at - created_at).total_seconds() / 3600
        turnaround_buckets[str(created_at.date())].append(hours)
    review_turnaround_time = [
        {"date": day, "avg_hours": round(sum(values) / len(values), 1)}
        for day, values in sorted(turnaround_buckets.items())
    ]

    return DashboardSummaryResponse(
        open_incidents=open_incidents,
        prs_awaiting_review=prs_awaiting,
        recent_deployments=recent_deployments,
        deployment_failure_rate=round(failure_rate, 3),
        incident_trend=incident_trend,
        deployment_failure_by_service=failure_by_service,
        review_turnaround_time=review_turnaround_time,
    )


@router.get(
    "/agents",
    response_model=AgentStatsResponse,
    dependencies=[Depends(require_role(ROLE_ADMIN, ROLE_TEAM_LEAD))],
)
async def agent_stats(
    user: UserContext = Depends(get_current_user),
    llm: BaseLLMGateway = Depends(get_llm_gateway),
    queue: BaseEventQueue = Depends(get_event_queue),
    memory: MemoryService = Depends(get_memory_service),
) -> AgentStatsResponse:
    """LLM cost/usage, memory effectiveness and queue health in one call.

    Restricted to team leads and admins because LLM spend and feedback rates
    are organisational metrics.
    """
    return AgentStatsResponse(
        llm=llm.get_usage_stats(),
        memory=await memory.get_retrieval_stats(),
        queue=queue.stats(),
    )
