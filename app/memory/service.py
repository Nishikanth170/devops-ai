"""Memory Service - the retrieval + storage facade of the memory layer.

Combines *semantic* retrieval (vector similarity over historical resolutions,
review findings, deployment outcomes) with *structured* lookup of the full
relational record, and records every agent suggestion so that human feedback
becomes retrievable signal for later runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.utils import normalize_whitespace, utcnow
from app.llm.gateway import BaseLLMGateway
from app.memory.models import (
    AgentSuggestion,
    CodeReview,
    Deployment,
    Incident,
    PipelineChange,
    ReviewPattern,
)
from app.memory.models import Runbook as RunbookModel
from app.memory.vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)

FEEDBACK_VALUES = {"helpful", "not_helpful", "incorrect"}


def _enum_value(value: object) -> str:
    """Return ``enum.value`` when applicable, else ``str(value)``."""
    return getattr(value, "value", value)  # type: ignore[return-value]


@dataclass
class RetrievedContext:
    """A semantic hit plus the full relational record it points at."""

    similarity: float
    vector_match: Optional[VectorRecord] = None
    incident: Optional[Incident] = None
    runbook: Optional[RunbookModel] = None
    code_review: Optional[CodeReview] = None
    review_pattern: Optional[ReviewPattern] = None
    deployment: Optional[Deployment] = None
    changes: list[PipelineChange] = field(default_factory=list)

    @property
    def source_id(self) -> Optional[UUID]:
        return self.vector_match.source_id if self.vector_match else None

    @property
    def matched_text(self) -> str:
        return self.vector_match.content if self.vector_match else ""


@dataclass
class IncidentContext:
    """An incident together with its runbook and semantically similar history."""

    incident: Incident
    runbook: Optional[RunbookModel] = None
    similar_incidents: list[RetrievedContext] = field(default_factory=list)


class MemoryService:
    """Hybrid memory: vector search + relational lookup + feedback capture."""

    def __init__(
        self,
        session: AsyncSession,
        llm_gateway: BaseLLMGateway,
        settings: Optional[Settings] = None,
    ) -> None:
        self.session = session
        self.llm = llm_gateway
        self.settings = settings or get_settings()
        self.vector_store = VectorStore(session, self.settings)

    # ---------------------------------------------------------- embeddings
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string through the LLM gateway."""
        return await self.llm.generate_embedding(text or "")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.llm.generate_embeddings_batch(texts)

    # ---------------------------------------------------- incident memory
    @staticmethod
    def build_incident_text(incident: Incident) -> str:
        """Text representation used for semantic search over past incidents."""
        parts = [
            f"Service: {incident.service}",
            f"Severity: {_enum_value(incident.severity)}",
            f"Title: {incident.title}",
        ]
        if incident.description:
            parts.append(f"Symptoms: {incident.description}")
        if incident.root_cause:
            parts.append(f"Root cause: {incident.root_cause}")
        if incident.resolution_steps:
            parts.append(f"Resolution: {incident.resolution_steps}")
        return normalize_whitespace(" | ".join(parts))

    async def record_incident(
        self, incident: Incident, resolution_summary: Optional[str] = None
    ) -> None:
        """Embed an incident so future alerts can be matched against it."""
        content = resolution_summary or self.build_incident_text(incident)
        embedding = await self.embed_text(content)
        await self.vector_store.add_memory(
            source_table="incidents",
            source_id=incident.id,
            content=content,
            embedding=embedding,
            metadata={
                "service": incident.service,
                "severity": _enum_value(incident.severity),
                "status": _enum_value(incident.status),
                "team_id": str(incident.team_id),
            },
        )

    async def record_runbook(self, runbook: RunbookModel) -> None:
        content = normalize_whitespace(
            f"Runbook {runbook.name} for {runbook.service}: "
            f"{runbook.description or ''} {runbook.steps or ''}"
        )
        embedding = await self.embed_text(content)
        await self.vector_store.add_memory(
            source_table="runbooks",
            source_id=runbook.id,
            content=content,
            embedding=embedding,
            metadata={"service": runbook.service, "name": runbook.name},
        )

    async def find_similar_incidents(
        self,
        query: str,
        service: Optional[str] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
        exclude_incident_id: Optional[UUID] = None,
    ) -> list[RetrievedContext]:
        """Find semantically similar past incidents and hydrate them."""
        query_embedding = await self.embed_text(query)
        metadata_filter = {"service": service} if service else None

        hits = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k + 1 if exclude_incident_id else top_k,
            source_table="incidents",
            metadata_filter=metadata_filter,
            threshold=threshold if threshold is not None else 0.65,
        )

        contexts: list[RetrievedContext] = []
        for record, similarity in hits:
            if exclude_incident_id and record.source_id == exclude_incident_id:
                continue
            incident = await self.session.get(Incident, record.source_id)
            if incident is None:
                logger.debug("Vector hit without relational row: %s", record.source_id)
                continue
            runbook = (
                await self.session.get(RunbookModel, incident.runbook_id)
                if incident.runbook_id
                else None
            )
            contexts.append(
                RetrievedContext(
                    similarity=similarity,
                    vector_match=record,
                    incident=incident,
                    runbook=runbook,
                )
            )
            if len(contexts) >= top_k:
                break
        return contexts

    async def get_incident_with_context(
        self, incident_id: UUID, similar_top_k: int = 3
    ) -> Optional[IncidentContext]:
        """Hydrate an incident with its runbook and similar historical incidents."""
        incident = await self.session.get(Incident, incident_id)
        if incident is None:
            return None

        runbook = (
            await self.session.get(RunbookModel, incident.runbook_id)
            if incident.runbook_id
            else None
        )
        similar = await self.find_similar_incidents(
            query=self.build_incident_text(incident),
            service=incident.service,
            top_k=similar_top_k,
            exclude_incident_id=incident.id,
        )
        return IncidentContext(incident=incident, runbook=runbook, similar_incidents=similar)

    # ------------------------------------------------- code review memory
    @staticmethod
    def build_review_text(review: CodeReview) -> str:
        parts = [f"Repo: {review.repo}", f"PR #{review.pr_id}", f"Title: {review.title}"]
        if review.description:
            parts.append(f"Description: {review.description}")
        if review.reviewer_agent_notes:
            parts.append(f"Review findings: {review.reviewer_agent_notes}")
        if review.human_feedback:
            parts.append(f"Human review: {review.human_feedback}")
        return normalize_whitespace(" | ".join(parts))

    async def record_code_review(
        self,
        review: CodeReview,
        analysis_summary: Optional[str] = None,
        patterns: Optional[list[ReviewPattern]] = None,
    ) -> None:
        """Embed a completed review and every pattern it surfaced.

        ``patterns`` is passed explicitly by callers that already hold the rows:
        iterating a lazy relationship would perform IO outside a greenlet
        context under asyncio.
        """
        content = analysis_summary or self.build_review_text(review)
        embedding = await self.embed_text(content)
        await self.vector_store.add_memory(
            source_table="code_reviews",
            source_id=review.id,
            content=content,
            embedding=embedding,
            metadata={
                "repo": review.repo,
                "author": review.author,
                "status": review.status,
                "team_id": str(review.team_id) if review.team_id else "",
            },
        )
        for pattern in patterns if patterns is not None else list(review.patterns):
            await self.record_review_pattern(pattern, repo=review.repo)

    async def record_review_pattern(
        self, pattern: ReviewPattern, repo: Optional[str] = None
    ) -> None:
        content = f"{pattern.pattern_type}: {pattern.description}"
        embedding = await self.embed_text(content)
        await self.vector_store.add_memory(
            source_table="review_patterns",
            source_id=pattern.id,
            content=content,
            embedding=embedding,
            metadata={"pattern_type": pattern.pattern_type, "repo": repo or ""},
        )

    async def find_similar_review_patterns(
        self,
        diff_summary: str,
        repo: Optional[str] = None,
        top_k: int = 8,
        threshold: Optional[float] = None,
    ) -> list[RetrievedContext]:
        """Find historical review patterns that resemble the current diff."""
        query_embedding = await self.embed_text(diff_summary)
        hits = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_table="review_patterns",
            metadata_filter={"repo": repo} if repo else None,
            threshold=threshold if threshold is not None else 0.65,
        )

        contexts: list[RetrievedContext] = []
        for record, similarity in hits:
            pattern = await self.session.get(ReviewPattern, record.source_id)
            if pattern is None:
                continue
            review = await self.session.get(CodeReview, pattern.review_id)
            contexts.append(
                RetrievedContext(
                    similarity=similarity,
                    vector_match=record,
                    review_pattern=pattern,
                    code_review=review,
                )
            )
        return contexts

    # -------------------------------------------------- deployment memory
    @staticmethod
    def build_deployment_text(deployment: Deployment, changes: Optional[list] = None) -> str:
        parts = [
            f"Service: {deployment.service}",
            f"Environment: {deployment.environment}",
            f"Commit: {deployment.commit_sha[:8]}",
            f"Status: {_enum_value(deployment.status)}",
        ]
        if deployment.failure_reason:
            parts.append(f"Failure: {deployment.failure_reason}")
        change_descriptions = [
            f"{_enum_value(getattr(c, 'change_type', c))}: {getattr(c, 'description', c)}"
            for c in (changes or [])
        ]
        if change_descriptions:
            parts.append("Changes: " + "; ".join(change_descriptions))
        return normalize_whitespace(" | ".join(parts))

    async def record_deployment(
        self,
        deployment: Deployment,
        change_summary: Optional[str] = None,
        changes: Optional[list[PipelineChange]] = None,
    ) -> None:
        """Embed a deployment outcome for future risk assessment."""
        content = change_summary or self.build_deployment_text(deployment, changes)
        embedding = await self.embed_text(content)
        await self.vector_store.add_memory(
            source_table="deployments",
            source_id=deployment.id,
            content=content,
            embedding=embedding,
            metadata={
                "service": deployment.service,
                "status": _enum_value(deployment.status),
                "environment": deployment.environment,
                "risk_score": deployment.risk_score,
                "team_id": str(deployment.team_id),
            },
        )

    async def find_similar_deployments(
        self,
        query: str,
        service: Optional[str] = None,
        top_k: int = 6,
        threshold: Optional[float] = None,
    ) -> list[RetrievedContext]:
        """Find similar historical deployments for risk assessment."""
        query_embedding = await self.embed_text(query)
        hits = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_table="deployments",
            metadata_filter={"service": service} if service else None,
            threshold=threshold if threshold is not None else 0.65,
        )

        contexts: list[RetrievedContext] = []
        for record, similarity in hits:
            deployment = await self.session.get(Deployment, record.source_id)
            if deployment is None:
                continue
            result = await self.session.execute(
                select(PipelineChange).where(PipelineChange.deployment_id == deployment.id)
            )
            contexts.append(
                RetrievedContext(
                    similarity=similarity,
                    vector_match=record,
                    deployment=deployment,
                    changes=list(result.scalars().all()),
                )
            )
        return contexts

    # --------------------------------------------- agent feedback loop
    async def record_suggestion(
        self,
        agent_type: str,
        query_context: str,
        suggestion: str,
        confidence: float,
        historical_matches: Optional[list[UUID]] = None,
        incident_id: Optional[UUID] = None,
        review_id: Optional[UUID] = None,
        deployment_id: Optional[UUID] = None,
    ) -> AgentSuggestion:
        """Persist an agent suggestion so humans can rate it later."""
        record = AgentSuggestion(
            agent_type=agent_type,
            incident_id=incident_id,
            review_id=review_id,
            deployment_id=deployment_id,
            query_context=query_context,
            suggestion=suggestion,
            confidence=confidence,
            historical_matches=[str(m) for m in (historical_matches or [])],
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_suggestion(self, suggestion_id: UUID) -> Optional[AgentSuggestion]:
        return await self.session.get(AgentSuggestion, suggestion_id)

    async def record_feedback(self, suggestion_id: UUID, feedback: str) -> bool:
        """Record human feedback (``helpful`` | ``not_helpful`` | ``incorrect``)."""
        if feedback not in FEEDBACK_VALUES:
            raise ValueError(
                f"feedback must be one of {sorted(FEEDBACK_VALUES)}, got {feedback!r}"
            )
        suggestion = await self.session.get(AgentSuggestion, suggestion_id)
        if suggestion is None:
            return False
        suggestion.user_feedback = feedback
        suggestion.feedback_at = utcnow()
        await self.session.flush()
        return True

    async def get_retrieval_stats(self) -> dict:
        """Memory effectiveness metrics for the admin/dashboard views."""
        feedback_result = await self.session.execute(
            select(AgentSuggestion.user_feedback, func.count(AgentSuggestion.id)).group_by(
                AgentSuggestion.user_feedback
            )
        )
        feedback_counts = {key or "unrated": count for key, count in feedback_result.all()}
        rated = sum(count for key, count in feedback_counts.items() if key != "unrated")
        helpful = feedback_counts.get("helpful", 0)

        vector_count = await self.vector_store.count()
        suggestion_count = int(
            await self.session.scalar(select(func.count(AgentSuggestion.id))) or 0
        )

        return {
            "total_vector_entries": vector_count,
            "total_suggestions": suggestion_count,
            "feedback_distribution": feedback_counts,
            "helpful_rate": round(helpful / rated, 3) if rated else 0.0,
            "vector_backend": self.settings.VECTOR_BACKEND,
        }
