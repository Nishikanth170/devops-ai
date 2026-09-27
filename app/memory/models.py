"""Relational (structured) memory: the system of record for the platform.

Design notes
------------
* ``sqlalchemy.Uuid`` (the 2.0 generic type) is used instead of the
  PostgreSQL-specific ``UUID`` so the schema also runs on SQLite for tests.
* Enum columns are stored as native PostgreSQL enums with ``name=`` set
  explicitly, which keeps Alembic migrations deterministic.
* Timestamps are naive UTC via :func:`app.core.utils.utcnow`.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.core.utils import utcnow


class SeverityLevel(str, enum.Enum):
    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"


class IncidentStatus(str, enum.Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class DeploymentStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    IN_PROGRESS = "in_progress"


class ChangeType(str, enum.Enum):
    CONFIG_CHANGE = "config_change"
    DEPENDENCY_BUMP = "dependency_bump"
    INFRA_CHANGE = "infra_change"
    CODE_CHANGE = "code_change"
    DATABASE_MIGRATION = "database_migration"
    FEATURE_FLAG = "feature_flag"


class UserRole(str, enum.Enum):
    ON_CALL_ENGINEER = "on-call-engineer"
    TEAM_LEAD = "team-lead"
    ADMIN = "admin"


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    users: Mapped[List["User"]] = relationship(back_populates="team")
    incidents: Mapped[List["Incident"]] = relationship(back_populates="team")
    deployments: Mapped[List["Deployment"]] = relationship(back_populates="team")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50), nullable=False, default=UserRole.ON_CALL_ENGINEER.value
    )
    team_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("teams.id", ondelete="SET NULL"), index=True
    )
    #: Stable subject claim issued by the identity provider (SSO).
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    team: Mapped[Optional["Team"]] = relationship(back_populates="users")


class Runbook(Base):
    __tablename__ = "runbooks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    steps: Mapped[str] = mapped_column(Text, nullable=False)  # Markdown or structured JSON
    service: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    severity: Mapped[SeverityLevel] = mapped_column(
        Enum(SeverityLevel, name="severity_level"), nullable=False
    )
    status: Mapped[IncidentStatus] = mapped_column(
        Enum(IncidentStatus, name="incident_status"),
        default=IncidentStatus.OPEN,
        nullable=False,
    )
    service: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    team_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, index=True, nullable=False
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    root_cause: Mapped[Optional[str]] = mapped_column(Text)
    resolution_steps: Mapped[Optional[str]] = mapped_column(Text)
    runbook_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("runbooks.id", ondelete="SET NULL")
    )
    external_alert_id: Mapped[Optional[str]] = mapped_column(String(200), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    team: Mapped["Team"] = relationship(back_populates="incidents")
    runbook: Mapped[Optional["Runbook"]] = relationship()
    suggestions: Mapped[List["AgentSuggestion"]] = relationship(back_populates="incident")

    __table_args__ = (
        Index("ix_incidents_service_started", "service", "started_at"),
        Index("ix_incidents_team_status", "team_id", "status"),
    )


class CodeReview(Base):
    __tablename__ = "code_reviews"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    pr_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # GitHub PR number
    repo: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    author: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    #: SHA-256 of the diff - makes agent re-runs on the same revision idempotent.
    diff_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)
    reviewer_agent_notes: Mapped[Optional[str]] = mapped_column(Text)
    human_feedback: Mapped[Optional[str]] = mapped_column(Text)
    team_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("teams.id", ondelete="SET NULL"), index=True
    )
    merged_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    comments: Mapped[List["ReviewComment"]] = relationship(
        back_populates="review", cascade="all, delete-orphan"
    )
    patterns: Mapped[List["ReviewPattern"]] = relationship(
        back_populates="review", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_reviews_repo_status", "repo", "status"),
        Index("ix_reviews_author_created", "author", "created_at"),
        Index("ix_reviews_repo_pr_diff", "repo", "pr_id", "diff_hash"),
    )


class ReviewComment(Base):
    __tablename__ = "review_comments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    review_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("code_reviews.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="info", nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="style")
    author_type: Mapped[str] = mapped_column(String(20), default="agent", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    historical_reference_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    review: Mapped["CodeReview"] = relationship(back_populates="comments")


class ReviewPattern(Base):
    __tablename__ = "review_patterns"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    review_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("code_reviews.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pattern_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    review: Mapped["CodeReview"] = relationship(back_populates="patterns")


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    service: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    team_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    branch: Mapped[Optional[str]] = mapped_column(String(100))
    environment: Mapped[str] = mapped_column(String(50), default="production", nullable=False)
    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus, name="deployment_status"),
        default=DeploymentStatus.IN_PROGRESS,
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, index=True, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    rollback: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text)
    #: 0-1 risk score computed by the DevOps Pipeline Agent.
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    agent_notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    team: Mapped["Team"] = relationship(back_populates="deployments")
    changes: Mapped[List["PipelineChange"]] = relationship(
        back_populates="deployment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_deployments_service_started", "service", "started_at"),
        Index("ix_deployments_status_env", "status", "environment"),
    )


class PipelineChange(Base):
    __tablename__ = "pipeline_changes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    deployment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    change_type: Mapped[ChangeType] = mapped_column(
        Enum(ChangeType, name="change_type"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    file_paths: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    deployment: Mapped["Deployment"] = relationship(back_populates="changes")


class AgentSuggestion(Base):
    """Every agent suggestion is recorded - this is the learning/feedback loop."""

    __tablename__ = "agent_suggestions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    incident_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("incidents.id", ondelete="CASCADE"), index=True
    )
    review_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("code_reviews.id", ondelete="CASCADE"), index=True
    )
    deployment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("deployments.id", ondelete="CASCADE"), index=True
    )
    query_context: Mapped[str] = mapped_column(Text, nullable=False)
    suggestion: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    #: Stringified UUIDs of the historical records that informed the suggestion.
    historical_matches: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    #: helpful | not_helpful | incorrect
    user_feedback: Mapped[Optional[str]] = mapped_column(String(50))
    feedback_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    incident: Mapped[Optional["Incident"]] = relationship(back_populates="suggestions")

    __table_args__ = (
        Index("ix_suggestions_agent_created", "agent_type", "created_at"),
        Index("ix_suggestions_feedback", "user_feedback"),
    )
