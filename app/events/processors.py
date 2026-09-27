"""Event processors: the glue between inbound events, memory and the agents.

Each handler opens its own unit of work (``session_scope``), so a failing event
can never poison another handler's session, and long agent calls never hold a
transaction open across the queue's blocking reads.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentResponse
from app.core.config import Settings, get_settings
from app.core.database import session_scope
from app.core.exceptions import IntegrationError
from app.core.utils import sha256_hex, truncate, utcnow
from app.events.queue import BaseEventQueue
from app.events.routing import resolve_team_id
from app.events.schemas import EventTypes, InternalEvent
from app.integrations import build_adapters
from app.integrations.base import IntegrationAdapter
from app.llm.gateway import BaseLLMGateway
from app.memory.models import (
    AgentSuggestion,
    ChangeType,
    CodeReview,
    Deployment,
    DeploymentStatus,
    Incident,
    IncidentStatus,
    PipelineChange,
    ReviewComment,
    ReviewPattern,
    SeverityLevel,
)
from app.memory.service import MemoryService
from app.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]

SEVERITY_ALIASES = {
    "P1": SeverityLevel.SEV1,
    "P2": SeverityLevel.SEV2,
    "P3": SeverityLevel.SEV3,
    "P4": SeverityLevel.SEV4,
    "CRITICAL": SeverityLevel.SEV1,
    "HIGH": SeverityLevel.SEV2,
    "MEDIUM": SeverityLevel.SEV3,
    "LOW": SeverityLevel.SEV4,
}
FAILED_EVENT_TYPES = {EventTypes.DEPLOYMENT_FAILED, EventTypes.DEPLOYMENT_ROLLED_BACK}


class EventProcessors:
    """Handlers that turn internal events into memory writes + agent runs."""

    def __init__(
        self,
        event_queue: BaseEventQueue,
        llm_gateway: BaseLLMGateway,
        settings: Optional[Settings] = None,
        session_factory: SessionFactory = session_scope,
        adapters: Optional[dict[str, IntegrationAdapter]] = None,
    ) -> None:
        self.queue = event_queue
        self.llm = llm_gateway
        self.settings = settings or get_settings()
        self.adapters = adapters if adapters is not None else build_adapters(self.settings)
        self._session_factory = session_factory

    # --------------------------------------------------------------- wiring
    def register(self, queue: Optional[BaseEventQueue] = None) -> BaseEventQueue:
        """Attach every handler to the queue."""
        target = queue or self.queue
        target.register_handlers(
            {
                EventTypes.PR_OPENED: self.handle_pr_event,
                EventTypes.PR_UPDATED: self.handle_pr_event,
                EventTypes.REVIEW_SUBMITTED: self.handle_review_submitted,
                EventTypes.DEPLOYMENT_STARTED: self.handle_deployment_started,
                EventTypes.DEPLOYMENT_COMPLETED: self.handle_deployment_completed,
                EventTypes.DEPLOYMENT_FAILED: self.handle_deployment_failed,
                EventTypes.DEPLOYMENT_ROLLED_BACK: self.handle_deployment_failed,
                EventTypes.ALERT_FIRED: self.handle_alert_event,
                EventTypes.INCIDENT_TRIGGERED: self.handle_alert_event,
                EventTypes.BUILD_FAILED: self.handle_build_event,
            }
        )
        return target

    def orchestrator(self, session: AsyncSession) -> AgentOrchestrator:
        """Build an orchestrator bound to one session/unit of work."""
        memory = MemoryService(session, self.llm, self.settings)
        return AgentOrchestrator(memory, self.llm, self.settings)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def map_severity(value: Any) -> SeverityLevel:
        """Normalise vendor severities onto the internal ``SEV1..SEV4`` scale."""
        if isinstance(value, SeverityLevel):
            return value
        key = str(value or "").strip().upper()
        if key in SeverityLevel.__members__:
            return SeverityLevel[key]
        return SEVERITY_ALIASES.get(key, SeverityLevel.SEV3)

    @staticmethod
    def map_change_type(value: Any) -> ChangeType:
        key = str(getattr(value, "value", value) or "").strip().lower()
        try:
            return ChangeType(key)
        except ValueError:
            return ChangeType.CODE_CHANGE

    async def _team_id(self, session: AsyncSession, event: InternalEvent) -> UUID:
        if event.team_id:
            return event.team_id
        return await resolve_team_id(
            session,
            repo=event.payload.get("repo"),
            service=event.payload.get("service"),
            settings=self.settings,
        )

    async def _notify_on_call(self, incident: Incident, response: AgentResponse) -> Optional[str]:
        """Raise an external incident when the agent recommends escalation."""
        adapter = self.adapters.get("datadog")
        if adapter is None or not adapter.enabled:
            return None
        escalations = [
            action
            for action in response.follow_up_actions
            if action.get("action") in {"escalate", "execute_runbook"}
        ]
        if not escalations:
            return None
        try:
            return await adapter.create_ticket(
                title=f"[{self.map_severity(incident.severity).value}] {incident.title}",
                description=truncate(response.response, 4000),
                service=incident.service,
                severity=self.map_severity(incident.severity).value,
            )
        except IntegrationError as exc:
            logger.warning("could not raise external incident: %s", exc)
            return None

    # ------------------------------------------------------- incident/alert
    async def handle_alert_event(self, event: InternalEvent) -> Optional[UUID]:
        """Create an incident record, run the Incident Agent, embed the result."""
        payload = event.payload
        async with self._session_factory() as session:
            team_id = await self._team_id(session, event)
            severity = self.map_severity(payload.get("severity"))

            incident = Incident(
                title=truncate(payload.get("title") or f"Alert from {event.source}", 500),
                description=truncate(payload.get("description") or "", 20000),
                severity=severity,
                status=IncidentStatus.OPEN,
                service=payload.get("service") or "unknown",
                team_id=team_id,
                external_alert_id=str(payload.get("id") or "") or None,
            )
            session.add(incident)
            await session.flush()

            orchestrator = self.orchestrator(session)
            response = await orchestrator.process_incident_alert(
                {**payload, "incident_id": incident.id}, team_id
            )

            # Embed the incident so future alerts can match against it.
            await orchestrator.memory.record_incident(incident)
            incident_id = incident.id
            external_ref = await self._notify_on_call(incident, response)

            logger.info(
                "incident %s handled by incident agent (confidence=%.2f, external=%s)",
                incident_id,
                response.confidence,
                external_ref or "n/a",
            )
            return incident_id

    async def handle_review_submitted(self, event: InternalEvent) -> Optional[UUID]:
        """Turn a human review into feedback on the agent's suggestion."""
        payload = event.payload
        state = str(payload.get("state") or "").upper()
        review_id: Optional[UUID] = None

        async with self._session_factory() as session:
            review = (
                await session.execute(
                    select(CodeReview)
                    .where(
                        CodeReview.repo == payload.get("repo"),
                        CodeReview.pr_id == str(payload.get("pr_number") or ""),
                    )
                    .order_by(CodeReview.created_at.desc())
                )
            ).scalars().first()
            if review is None:
                logger.debug(
                    "review not found for %s#%s",
                    payload.get("repo"),
                    payload.get("pr_number"),
                )
                return None

            review.human_feedback = (
                f"{payload.get('reviewer') or 'human'}: {state} "
                f"{truncate(payload.get('body') or '', 2000)}"
            ).strip()
            review_id = review.id

            feedback = {"APPROVED": "helpful", "CHANGES_REQUESTED": "not_helpful"}.get(state)
            if feedback:
                suggestion = (
                    await session.execute(
                        select(AgentSuggestion)
                        .where(AgentSuggestion.review_id == review.id)
                        .order_by(AgentSuggestion.created_at.desc())
                    )
                ).scalars().first()
                if suggestion is not None:
                    suggestion.user_feedback = feedback
                    suggestion.feedback_at = utcnow()
                    logger.info(
                        "recorded %s feedback on suggestion %s", feedback, suggestion.id
                    )
        return review_id

    # ----------------------------------------------------------- code review
    async def handle_pr_event(self, event: InternalEvent) -> Optional[UUID]:
        """Run the Code Review Agent over a PR and store its findings."""
        payload = dict(event.payload)
        repo = payload.get("repo")
        pr_id = str(payload.get("number") or payload.get("pr_number") or "")

        async with self._session_factory() as session:
            team_id = await self._team_id(session, event)
            diff = payload.get("diff") or await self._fetch_diff(repo, pr_id)
            diff_hash = payload.get("diff_hash") or sha256_hex(
                diff or f"{repo}#{pr_id}:{payload.get('title', '')}"
            )

            review = await self._get_or_create_review(
                session, repo, pr_id, payload, diff_hash, team_id
            )

            # Same revision already analysed -> skip the LLM call (idempotency).
            if review.reviewer_agent_notes and review.diff_hash == diff_hash:
                logger.info("PR %s#%s already reviewed for this revision", repo, pr_id)
                return review.id

            orchestrator = self.orchestrator(session)
            response = await orchestrator.process_pr_event(
                {**payload, "diff": diff, "diff_hash": diff_hash, "review_id": review.id},
                team_id,
            )

            review.reviewer_agent_notes = response.response
            review.diff_hash = diff_hash
            review.status = "reviewed"
            patterns = await self._persist_findings(session, review, response)

            await orchestrator.memory.record_code_review(review, patterns=patterns)
            posted = await self._post_review_comments(repo, pr_id, response)
            review_id = review.id

            logger.info(
                "PR %s#%s reviewed: %s finding(s), confidence=%.2f, posted=%s",
                repo,
                pr_id,
                len(response.suggestions),
                response.confidence,
                posted,
            )
            return review_id

    async def _fetch_diff(self, repo: Optional[str], pr_id: str) -> str:
        """Try to fetch the raw diff from the git provider (best effort)."""
        if not repo or not pr_id:
            return ""
        for source in ("github", "gitlab"):
            adapter = self.adapters.get(source)
            if adapter is None or not adapter.enabled:
                continue
            try:
                if source == "github":
                    return await adapter.fetch_pr_diff(repo, pr_id)
                return await adapter.fetch_mr_changes(repo, pr_id)
            except IntegrationError as exc:
                logger.warning("could not fetch diff from %s: %s", source, exc)
            except AttributeError:  # pragma: no cover - provider without diff support
                continue
        return ""

    async def _get_or_create_review(
        self,
        session: AsyncSession,
        repo: Optional[str],
        pr_id: str,
        payload: dict[str, Any],
        diff_hash: str,
        team_id: UUID,
    ) -> CodeReview:
        existing = (
            await session.execute(
                select(CodeReview)
                .where(CodeReview.repo == repo, CodeReview.pr_id == pr_id)
                .order_by(CodeReview.created_at.desc())
            )
        ).scalars().first()
        if existing is not None:
            if payload.get("title"):
                existing.title = truncate(str(payload["title"]), 500)
            return existing

        review = CodeReview(
            pr_id=pr_id,
            repo=str(repo or "unknown"),
            author=str(payload.get("author") or "unknown"),
            title=truncate(str(payload.get("title") or f"PR #{pr_id}"), 500),
            description=truncate(str(payload.get("description") or ""), 20000),
            diff_hash=diff_hash,
            status="open",
            team_id=team_id,
        )
        session.add(review)
        await session.flush()
        return review

    @staticmethod
    async def _persist_findings(
        session: AsyncSession, review: CodeReview, response: AgentResponse
    ) -> list[ReviewPattern]:
        """Store inline comments and derive reusable review patterns.

        Rows are added through the session (not through the relationship
        collections) so that no lazy IO happens outside a greenlet context.
        """
        comments: list[ReviewComment] = []
        patterns_by_type: dict[str, ReviewPattern] = {}

        for finding in response.suggestions:
            reference = finding.get("historical_reference_id")
            historical_id: Optional[UUID] = None
            if reference:
                try:
                    historical_id = UUID(str(reference))
                except (ValueError, TypeError):
                    historical_id = None

            comments.append(
                ReviewComment(
                    review_id=review.id,
                    file_path=truncate(str(finding.get("file_path") or "unknown"), 500),
                    line_number=int(finding.get("line_number") or 1),
                    comment=truncate(str(finding.get("message") or ""), 8000),
                    severity=str(finding.get("severity") or "info"),
                    category=str(finding.get("category") or "style"),
                    author_type="agent",
                    confidence=float(finding.get("confidence") or 0.0),
                    historical_reference_id=historical_id,
                )
            )

            pattern_type = f"{finding.get('category') or 'style'}_finding"
            pattern = patterns_by_type.get(pattern_type)
            if pattern is None:
                patterns_by_type[pattern_type] = ReviewPattern(
                    review_id=review.id,
                    pattern_type=pattern_type,
                    description=truncate(str(finding.get("message") or pattern_type), 2000),
                )
            else:
                pattern.frequency += 1

        patterns = list(patterns_by_type.values())
        session.add_all([*comments, *patterns])
        await session.flush()
        return patterns

    async def _post_review_comments(
        self, repo: Optional[str], pr_id: str, response: AgentResponse
    ) -> int:
        """Publish agent findings back onto the PR (best effort)."""
        adapter = self.adapters.get("github")
        if adapter is None or not adapter.enabled or not repo or not pr_id:
            return 0
        try:
            return await adapter.post_review_comments(repo, pr_id, response.suggestions)
        except IntegrationError as exc:
            logger.warning("could not publish review comments: %s", exc)
            return 0

    # ----------------------------------------------------------- deployments
    async def handle_deployment_started(self, event: InternalEvent) -> Optional[UUID]:
        """Record a deployment, score its risk and embed the outcome."""
        payload = event.payload
        changes = payload.get("changes") or []

        async with self._session_factory() as session:
            team_id = await self._team_id(session, event)
            deployment = await self._get_or_create_deployment(session, payload, team_id)
            change_rows = await self._ensure_changes(session, deployment, changes)

            orchestrator = self.orchestrator(session)
            response = await orchestrator.process_deployment_event(
                {**payload, "deployment_id": deployment.id, "changes": changes}, team_id
            )
            # The agent wrote risk_score/agent_notes back onto the row itself.
            await orchestrator.memory.record_deployment(deployment, changes=change_rows)
            deployment_id = deployment.id

            logger.info(
                "deployment %s (%s) risk=%s band=%s",
                deployment_id,
                deployment.service,
                response.metadata.get("risk_score"),
                response.metadata.get("risk_band"),
            )
            return deployment_id

    @staticmethod
    async def _ensure_changes(
        session: AsyncSession, deployment: Deployment, changes: list[dict[str, Any]]
    ) -> list[PipelineChange]:
        """Create the change rows once, returning them for embedding.

        Changes are queried/inserted explicitly: touching the lazy
        ``Deployment.changes`` relationship would run IO outside a greenlet.
        """
        result = await session.execute(
            select(PipelineChange).where(PipelineChange.deployment_id == deployment.id)
        )
        existing = list(result.scalars().all())
        if existing or not changes:
            return existing

        rows = [
            PipelineChange(
                deployment_id=deployment.id,
                change_type=EventProcessors.map_change_type(
                    change.get("change_type") or change.get("type")
                ),
                description=truncate(str(change.get("description") or ""), 4000),
                risk_score=float(change.get("risk_score") or 0.0),
                file_paths=list(change.get("file_paths") or []),
            )
            for change in changes
        ]
        session.add_all(rows)
        await session.flush()
        return rows

    async def handle_deployment_completed(self, event: InternalEvent) -> Optional[UUID]:
        return await self._finalize_deployment(event, DeploymentStatus.SUCCESS)

    async def handle_deployment_failed(self, event: InternalEvent) -> Optional[UUID]:
        status = (
            DeploymentStatus.ROLLED_BACK
            if event.event_type == EventTypes.DEPLOYMENT_ROLLED_BACK
            else DeploymentStatus.FAILED
        )
        return await self._finalize_deployment(event, status)

    async def _finalize_deployment(
        self, event: InternalEvent, status: DeploymentStatus
    ) -> Optional[UUID]:
        """Close out a deployment and embed the outcome for future risk scoring."""
        payload = event.payload
        async with self._session_factory() as session:
            deployment = await self._find_deployment(session, payload)
            if deployment is None:
                logger.warning(
                    "no deployment record for %s @ %s",
                    payload.get("service"),
                    str(payload.get("commit_sha"))[:8],
                )
                return None

            deployment.status = status
            deployment.completed_at = utcnow()
            deployment.rollback = status == DeploymentStatus.ROLLED_BACK
            if payload.get("failure_reason"):
                deployment.failure_reason = truncate(str(payload["failure_reason"]), 4000)

            memory = MemoryService(session, self.llm, self.settings)
            change_result = await session.execute(
                select(PipelineChange).where(PipelineChange.deployment_id == deployment.id)
            )
            await memory.record_deployment(
                deployment, changes=list(change_result.scalars().all())
            )

            logger.info(
                "deployment %s marked %s (%s)",
                deployment.id,
                status.value,
                deployment.failure_reason or "no failure reason",
            )
            return deployment.id

    async def _get_or_create_deployment(
        self, session: AsyncSession, payload: dict[str, Any], team_id: UUID
    ) -> Deployment:
        existing = await self._find_deployment(session, payload)
        if existing is not None:
            return existing

        deployment = Deployment(
            service=str(payload.get("service") or "unknown"),
            team_id=team_id,
            commit_sha=str(payload.get("commit_sha") or "unknown"),
            branch=payload.get("branch"),
            environment=str(payload.get("environment") or "production"),
            status=DeploymentStatus.IN_PROGRESS,
        )
        session.add(deployment)
        await session.flush()
        return deployment

    @staticmethod
    async def _find_deployment(
        session: AsyncSession, payload: dict[str, Any]
    ) -> Optional[Deployment]:
        commit_sha = payload.get("commit_sha")
        if not commit_sha:
            return None
        stmt = select(Deployment).where(Deployment.commit_sha == str(commit_sha))
        if payload.get("service"):
            stmt = stmt.where(Deployment.service == payload["service"])
        return (
            await session.execute(stmt.order_by(Deployment.started_at.desc()))
        ).scalars().first()

    # ----------------------------------------------------------------- build
    async def handle_build_event(self, event: InternalEvent) -> Optional[UUID]:
        """Annotate the deployment affected by a failing CI build."""
        payload = event.payload
        commit_sha = payload.get("head_sha") or payload.get("commit_sha")
        if not commit_sha:
            logger.debug("build event without commit sha: %s", payload.get("run_id"))
            return None

        async with self._session_factory() as session:
            deployment = await self._find_deployment(session, {"commit_sha": commit_sha})
            if deployment is None:
                logger.info("build failure for unknown commit %s", str(commit_sha)[:8])
                return None

            job = payload.get("name") or payload.get("job_name") or payload.get("run_id")
            reason = payload.get("conclusion") or payload.get("failure_reason") or "unknown"
            note = f"CI build {job} failed ({reason})"
            deployment.agent_notes = f"{deployment.agent_notes or ''}\n{note}".strip()
            logger.info("annotated deployment %s: %s", deployment.id, note)
            return deployment.id
