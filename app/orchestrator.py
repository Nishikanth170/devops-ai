"""Agent Orchestrator - routing, context assembly and agent lifecycle.

The orchestrator is cheap to construct and is therefore built per unit of work
(one per HTTP request / one per consumed event) around a request-scoped
``MemoryService``.  It owns no global state and no database session.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from app.agents.base import AgentContext, AgentResponse, AgentType, BaseAgent
from app.agents.code_review import CodeReviewAgent
from app.agents.devops_pipeline import DevOpsPipelineAgent
from app.agents.incident_response import IncidentResponseAgent
from app.core.config import Settings, get_settings
from app.core.utils import sha256_hex, truncate
from app.llm.gateway import BaseLLMGateway
from app.memory.service import MemoryService

logger = logging.getLogger(__name__)

INCIDENT_KEYWORDS = (
    "incident",
    "alert",
    "error",
    "down",
    "outage",
    "sev",
    "pagerduty",
    "datadog",
    "on-call",
    "saturation",
    "latency",
)
CODE_REVIEW_KEYWORDS = (
    "pr",
    "pull request",
    "review",
    "diff",
    "commit",
    "merge",
    "github",
    "gitlab",
    "lint",
)
PIPELINE_KEYWORDS = (
    "deploy",
    "deployment",
    "pipeline",
    "build",
    "release",
    "rollout",
    "canary",
    "jenkins",
    "github actions",
    "rollback",
)

#: Payload ``event_type`` values mapped directly onto an agent.
EVENT_TYPE_ROUTING: dict[str, AgentType] = {
    "incident": AgentType.INCIDENT_RESPONSE,
    "incident.triggered": AgentType.INCIDENT_RESPONSE,
    "incident.acknowledged": AgentType.INCIDENT_RESPONSE,
    "incident.resolved": AgentType.INCIDENT_RESPONSE,
    "alert": AgentType.INCIDENT_RESPONSE,
    "alert.fired": AgentType.INCIDENT_RESPONSE,
    "alert.resolved": AgentType.INCIDENT_RESPONSE,
    "pull_request": AgentType.CODE_REVIEW,
    "pull_request.opened": AgentType.CODE_REVIEW,
    "pull_request.updated": AgentType.CODE_REVIEW,
    "pull_request.review_submitted": AgentType.CODE_REVIEW,
    "pr": AgentType.CODE_REVIEW,
    "deployment": AgentType.DEVOPS_PIPELINE,
    "deployment.started": AgentType.DEVOPS_PIPELINE,
    "deployment.completed": AgentType.DEVOPS_PIPELINE,
    "deployment.failed": AgentType.DEVOPS_PIPELINE,
    "deployment.rolled_back": AgentType.DEVOPS_PIPELINE,
    "build": AgentType.DEVOPS_PIPELINE,
    "build.completed": AgentType.DEVOPS_PIPELINE,
    "build.started": AgentType.DEVOPS_PIPELINE,
    "ci.job": AgentType.DEVOPS_PIPELINE,
}


class AgentOrchestrator:
    """Routes requests to the right agent and runs the agent lifecycle."""

    def __init__(
        self,
        memory_service: MemoryService,
        llm_gateway: BaseLLMGateway,
        settings: Optional[Settings] = None,
    ) -> None:
        self.memory = memory_service
        self.llm = llm_gateway
        self.settings = settings or get_settings()
        self.agents: dict[AgentType, BaseAgent] = {
            AgentType.INCIDENT_RESPONSE: IncidentResponseAgent(
                memory_service, llm_gateway, self.settings
            ),
            AgentType.CODE_REVIEW: CodeReviewAgent(memory_service, llm_gateway, self.settings),
            AgentType.DEVOPS_PIPELINE: DevOpsPipelineAgent(
                memory_service, llm_gateway, self.settings
            ),
        }

    # --------------------------------------------------------------- routing
    def get_agent(self, agent_type: AgentType) -> BaseAgent:
        try:
            return self.agents[AgentType(agent_type)]
        except ValueError as exc:
            raise ValueError(f"Unknown agent type: {agent_type}") from exc

    def determine_agent_type(self, query: str, payload: dict[str, Any]) -> AgentType:
        """Pick an agent from explicit hints, event type, then keyword scoring."""
        explicit = payload.get("agent_type")
        if explicit:
            try:
                return AgentType(explicit)
            except ValueError:
                logger.warning("Ignoring unknown agent_type hint: %s", explicit)

        event_type = str(payload.get("event_type", "")).lower()
        if event_type:
            if event_type in EVENT_TYPE_ROUTING:
                return EVENT_TYPE_ROUTING[event_type]
            # tolerate prefixes like "incident.triggered.v2" / "github.pull_request"
            for known, agent_type in EVENT_TYPE_ROUTING.items():
                if known and known in event_type:
                    return agent_type

        query_lower = (query or "").lower()
        scores = {
            AgentType.INCIDENT_RESPONSE: sum(1 for k in INCIDENT_KEYWORDS if k in query_lower),
            AgentType.CODE_REVIEW: sum(1 for k in CODE_REVIEW_KEYWORDS if k in query_lower),
            AgentType.DEVOPS_PIPELINE: sum(1 for k in PIPELINE_KEYWORDS if k in query_lower),
        }
        best = max(scores, key=lambda agent: scores[agent])
        if scores[best] == 0:
            return AgentType.INCIDENT_RESPONSE  # safe default: diagnostic reasoning
        return best

    # ----------------------------------------------------------------- entry
    async def process(
        self,
        query: str,
        payload: Optional[dict[str, Any]] = None,
        user_id: Optional[UUID] = None,
        team_id: Optional[UUID] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
        agent_type: Optional[AgentType] = None,
    ) -> AgentResponse:
        """Route to an agent and run retrieve -> generate -> record."""
        payload = dict(payload or {})
        selected = agent_type or self.determine_agent_type(query, payload)
        agent = self.get_agent(selected)

        context = AgentContext(
            query=query,
            payload=payload,
            user_id=user_id,
            team_id=team_id,
            conversation_history=list(conversation_history or []),
        )
        return await agent.process(context)

    # ------------------------------------------------------ typed entrypoints
    async def process_incident_alert(
        self, alert_payload: dict[str, Any], team_id: Optional[UUID] = None
    ) -> AgentResponse:
        """Handle an incoming alert/incident (high priority path)."""
        query = "\n".join(
            [
                f"Alert: {alert_payload.get('title', 'Unknown alert')}",
                f"Service: {alert_payload.get('service', 'unknown')}",
                f"Severity: {alert_payload.get('severity', 'unknown')}",
                f"Description: {truncate(alert_payload.get('description') or '', 4000)}",
            ]
        )
        payload = {
            "event_type": "incident.triggered",
            "service": alert_payload.get("service"),
            "severity": alert_payload.get("severity"),
            "external_alert_id": alert_payload.get("id"),
            "incident_id": alert_payload.get("incident_id"),
        }
        return await self.process(
            query=query,
            payload=payload,
            team_id=team_id,
            agent_type=AgentType.INCIDENT_RESPONSE,
        )

    async def process_pr_event(
        self, pr_payload: dict[str, Any], team_id: Optional[UUID] = None
    ) -> AgentResponse:
        """Handle a pull request event (opened/synchronize/reopened)."""
        diff = pr_payload.get("diff") or ""
        diff_summary = pr_payload.get("diff_summary") or self._summarize_diff(diff)

        query = "\n".join(
            [
                f"PR #{pr_payload.get('number', pr_payload.get('pr_number', '?'))} in "
                f"{pr_payload.get('repo', 'unknown')}: {pr_payload.get('title', '')}",
                f"Author: {pr_payload.get('author', 'unknown')}",
                f"Description: {truncate(pr_payload.get('description') or '', 2000)}",
                f"Diff summary: {diff_summary}",
            ]
        )
        payload = {
            "event_type": "pull_request.opened",
            "repo": pr_payload.get("repo"),
            "pr_number": pr_payload.get("number") or pr_payload.get("pr_number"),
            "title": pr_payload.get("title"),
            "description": pr_payload.get("description"),
            "author": pr_payload.get("author"),
            "diff": diff,
            "diff_summary": diff_summary,
            "diff_hash": pr_payload.get("diff_hash") or sha256_hex(diff or query),
            "review_id": pr_payload.get("review_id"),
        }
        return await self.process(
            query=query,
            payload=payload,
            team_id=team_id,
            agent_type=AgentType.CODE_REVIEW,
        )

    async def process_deployment_event(
        self, deploy_payload: dict[str, Any], team_id: Optional[UUID] = None
    ) -> AgentResponse:
        """Handle a deployment event and score its risk."""
        changes = deploy_payload.get("changes") or []
        query = "\n".join(
            [
                f"Deployment: {deploy_payload.get('service', 'unknown')} @ "
                f"{(deploy_payload.get('commit_sha') or '')[:8]}",
                f"Environment: {deploy_payload.get('environment', 'production')}",
                f"Status: {deploy_payload.get('status', 'in_progress')}",
                f"Changes: {changes}",
            ]
        )
        payload = {
            "event_type": "deployment.started",
            "service": deploy_payload.get("service"),
            "commit_sha": deploy_payload.get("commit_sha"),
            "environment": deploy_payload.get("environment"),
            "status": deploy_payload.get("status"),
            "changes": changes,
            "deployment_id": deploy_payload.get("deployment_id"),
        }
        return await self.process(
            query=query,
            payload=payload,
            team_id=team_id,
            agent_type=AgentType.DEVOPS_PIPELINE,
        )

    @staticmethod
    def _summarize_diff(diff: str, max_files: int = 8) -> str:
        """Cheap, deterministic diff summary used when none is supplied."""
        if not diff:
            return "no diff provided"
        files: list[str] = []
        added = removed = 0
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                path = line[6:].strip()
                if path not in files:
                    files.append(path)
            elif line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        listed = ", ".join(files[:max_files]) or "no files detected"
        more = f" (+{len(files) - max_files} more)" if len(files) > max_files else ""
        return f"{len(files)} file(s) changed [{listed}{more}], +{added}/-{removed} lines"
