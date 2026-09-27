"""Common agent interface.

Every agent implements the same lifecycle:

1. ``retrieve_context`` - pull historically relevant records out of memory
2. ``generate_response`` - call the LLM with that context and structure the answer
3. ``record_outcome``   - persist the suggestion so humans can rate it

``BaseAgent.process`` orchestrates those three steps, which is why the
orchestrator and the event processors can treat all agents identically.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from app.core.utils import truncate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.llm.gateway import BaseLLMGateway
    from app.memory.service import MemoryService, RetrievedContext


class AgentType(str, enum.Enum):
    INCIDENT_RESPONSE = "incident_response"
    CODE_REVIEW = "code_review"
    DEVOPS_PIPELINE = "devops_pipeline"


@dataclass
class AgentContext:
    """Everything an agent needs to answer one request."""

    query: str
    payload: dict[str, Any] = field(default_factory=dict)
    #: Filled in by ``BaseAgent.process`` after retrieval.
    historical_context: list["RetrievedContext"] = field(default_factory=list)
    user_id: Optional[UUID] = None
    team_id: Optional[UUID] = None
    #: Prior turns, oldest first, for chat-style queries.
    conversation_history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Structured agent answer consumed by the API and the event processors."""

    response: str
    confidence: float = 0.0
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    historical_matches: list[UUID] = field(default_factory=list)
    follow_up_actions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (UUIDs become strings)."""
        return {
            "response": self.response,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "historical_matches": [str(match) for match in self.historical_matches],
            "follow_up_actions": self.follow_up_actions,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """Abstract base class shared by all specialist agents."""

    def __init__(
        self,
        agent_type: AgentType,
        memory_service: "MemoryService",
        llm_gateway: "BaseLLMGateway",
        system_prompt: str,
    ) -> None:
        self.agent_type = agent_type
        self.memory = memory_service
        self.llm = llm_gateway
        self.system_prompt = system_prompt

    # ------------------------------------------------------------ interface
    @abstractmethod
    async def retrieve_context(self, context: AgentContext) -> list["RetrievedContext"]:
        """Return historically relevant records for this request."""

    @abstractmethod
    async def generate_response(self, context: AgentContext) -> AgentResponse:
        """Produce the answer using the retrieved context."""

    @abstractmethod
    async def record_outcome(
        self,
        context: AgentContext,
        response: AgentResponse,
        feedback: Optional[str] = None,
    ) -> None:
        """Persist the suggestion for the human feedback loop."""

    # -------------------------------------------------------------- template
    async def process(self, context: AgentContext) -> AgentResponse:
        """Retrieve -> generate -> record. Used by orchestrator and workers."""
        context.historical_context = await self.retrieve_context(context)
        response = await self.generate_response(context)
        await self.record_outcome(context, response)
        return response

    # ------------------------------------------------------------- prompting
    def _build_prompt(
        self, context: AgentContext, historical_context: list["RetrievedContext"]
    ) -> list[dict[str, str]]:
        """Assemble the message list: system, history, context, current query."""
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

        for message in context.conversation_history[-6:]:
            if message.get("content"):
                messages.append(
                    {"role": message.get("role", "user"), "content": message["content"]}
                )

        if historical_context:
            context_block = self._format_historical_context(historical_context)
        else:
            context_block = "none found for this request."

        messages.append(
            {"role": "system", "content": f"Relevant historical context:\n{context_block}"}
        )
        messages.append({"role": "user", "content": context.query})
        return messages

    def format_context_item(self, index: int, item: "RetrievedContext") -> str:
        """Format one retrieved record. Override for agent-specific rendering."""
        prefix = f"[{index}] (similarity {item.similarity:.2f})"

        if item.incident is not None:
            incident = item.incident
            runbook_note = f" | Runbook: {item.runbook.name}" if item.runbook else ""
            return (
                f"{prefix} Incident: {incident.title} | Service: {incident.service} | "
                f"Root cause: {incident.root_cause or 'unknown'} | "
                f"Resolution: {truncate(incident.resolution_steps or 'none', 300)}"
                f"{runbook_note}"
            )

        if item.review_pattern is not None:
            pattern = item.review_pattern
            return (
                f"{prefix} Review pattern: {pattern.pattern_type}: {pattern.description} "
                f"(seen {pattern.frequency}x)"
            )

        if item.code_review is not None:
            review = item.code_review
            return (
                f"{prefix} Code review: PR #{review.pr_id} in {review.repo} by {review.author} | "
                f"Notes: {truncate(review.reviewer_agent_notes or 'none', 300)}"
            )

        if item.deployment is not None:
            deployment = item.deployment
            status = getattr(deployment.status, "value", deployment.status)
            return (
                f"{prefix} Deployment: {deployment.service} @ {deployment.commit_sha[:8]} | "
                f"Status: {status} | Risk: {deployment.risk_score:.2f} | "
                f"Failure: {truncate(deployment.failure_reason or 'none', 200)}"
            )

        if item.matched_text:
            return f"{prefix} {truncate(item.matched_text, 300)}"
        return ""

    def _format_historical_context(self, historical_context: list["RetrievedContext"]) -> str:
        """Render retrieved records as compact, citation-numbered context."""
        rendered = [
            line
            for line in (
                self.format_context_item(index, item)
                for index, item in enumerate(historical_context, start=1)
            )
            if line
        ]
        return "\n".join(rendered) if rendered else "No relevant historical context found."

    # ---------------------------------------------------------------- maths
    @staticmethod
    def _average_similarity(historical_context: list["RetrievedContext"]) -> float:
        if not historical_context:
            return 0.0
        return sum(item.similarity for item in historical_context) / len(historical_context)

    @staticmethod
    def _historical_match_ids(historical_context: list["RetrievedContext"]) -> list[UUID]:
        return [
            item.vector_match.source_id
            for item in historical_context
            if item.vector_match is not None
        ]
