"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

FeedbackValue = Literal["helpful", "not_helpful", "incorrect"]


class ORMModel(BaseModel):
    """Base model that can be built directly from SQLAlchemy rows."""

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------- agent
class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    agent_type: Optional[Literal["incident_response", "code_review", "devops_pipeline"]] = None
    context: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class AgentQueryResponse(BaseModel):
    response: str
    confidence: float
    agent_type: str
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    historical_matches: list[str] = Field(default_factory=list)
    follow_up_actions: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuggestionResponse(ORMModel):
    id: UUID
    agent_type: str
    suggestion: str
    confidence: float
    historical_matches: list[str] = Field(default_factory=list)
    user_feedback: Optional[str] = None
    created_at: datetime


# ----------------------------------------------------------------- incidents
class IncidentCreateRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=500)
    description: Optional[str] = None
    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"] = "SEV3"
    service: str = Field(..., min_length=1, max_length=100)
    team_id: Optional[UUID] = None
    external_alert_id: Optional[str] = None
    runbook_id: Optional[UUID] = None
    #: Ask the Incident Response Agent for an initial assessment.
    run_agent: bool = True


class IncidentResolveRequest(BaseModel):
    root_cause: str = Field(..., min_length=3)
    resolution_steps: str = Field(..., min_length=3)
    runbook_id: Optional[UUID] = None
    runbook_success: Optional[bool] = None


class IncidentResponse(ORMModel):
    id: UUID
    title: str
    description: Optional[str] = None
    severity: str
    status: str
    service: str
    team_id: UUID
    started_at: datetime
    resolved_at: Optional[datetime] = None
    root_cause: Optional[str] = None
    resolution_steps: Optional[str] = None
    runbook_id: Optional[UUID] = None
    external_alert_id: Optional[str] = None
    agent_suggestions: list[SuggestionResponse] = Field(default_factory=list)
    similar_incidents: list[dict[str, Any]] = Field(default_factory=list)


class IncidentListResponse(BaseModel):
    incidents: list[IncidentResponse]
    total: int
    page: int
    page_size: int


# --------------------------------------------------------------- code review
class ReviewCommentResponse(ORMModel):
    id: UUID
    file_path: str
    line_number: int
    comment: str
    severity: str
    category: str
    author_type: str
    confidence: float
    historical_reference_id: Optional[UUID] = None


class CodeReviewResponse(ORMModel):
    id: UUID
    pr_id: str
    repo: str
    author: str
    title: str
    description: Optional[str] = None
    status: str
    diff_hash: str
    reviewer_agent_notes: Optional[str] = None
    human_feedback: Optional[str] = None
    merged_at: Optional[datetime] = None
    created_at: datetime
    comments: list[ReviewCommentResponse] = Field(default_factory=list)
    patterns: list[dict[str, Any]] = Field(default_factory=list)


class ReviewListResponse(BaseModel):
    reviews: list[CodeReviewResponse]
    total: int


# ---------------------------------------------------------------- deployment
class PipelineChangeResponse(ORMModel):
    id: UUID
    change_type: str
    description: str
    risk_score: float
    file_paths: list[str] = Field(default_factory=list)


class DeploymentResponse(ORMModel):
    id: UUID
    service: str
    team_id: UUID
    commit_sha: str
    branch: Optional[str] = None
    environment: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    rollback: bool
    failure_reason: Optional[str] = None
    risk_score: float
    agent_notes: Optional[str] = None
    changes: list[PipelineChangeResponse] = Field(default_factory=list)


class RiskAssessmentRequest(BaseModel):
    service: str
    commit_sha: Optional[str] = None
    environment: str = "production"
    changes: list[dict[str, Any]] = Field(default_factory=list)
    deployment_id: Optional[UUID] = None


class RiskAssessmentResponse(BaseModel):
    risk_score: float
    risk_band: str
    confidence: float
    response: str
    risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    similar_deployments: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ dashboard
class DashboardSummaryResponse(BaseModel):
    open_incidents: int
    prs_awaiting_review: int
    recent_deployments: int
    deployment_failure_rate: float
    incident_trend: list[dict[str, Any]] = Field(default_factory=list)
    deployment_failure_by_service: list[dict[str, Any]] = Field(default_factory=list)
    review_turnaround_time: list[dict[str, Any]] = Field(default_factory=list)


class AgentStatsResponse(BaseModel):
    llm: dict[str, Any]
    memory: dict[str, Any]
    queue: dict[str, Any]


# ------------------------------------------------------------------- feedback
class FeedbackRequest(BaseModel):
    feedback: FeedbackValue


class FeedbackResponse(BaseModel):
    status: str
    suggestion_id: UUID


class MessageResponse(BaseModel):
    status: str
    detail: Optional[str] = None


class WebhookAcceptedResponse(BaseModel):
    status: str
    source: str
    events_published: int
    event_types: list[str] = Field(default_factory=list)
