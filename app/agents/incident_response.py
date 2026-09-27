"""Incident Response Agent.

Given an alert/incident (or an ad-hoc question), it retrieves semantically
similar resolved incidents + their runbooks, asks the LLM to reason over that
history, and returns actionable mitigations with an explicit confidence score.
"""

from __future__ import annotations

from typing import Any, Optional

from app.agents.base import AgentContext, AgentResponse, AgentType, BaseAgent
from app.core.config import Settings
from app.llm.gateway import BaseLLMGateway
from app.llm.prompts import INCIDENT_RESPONSE_PROMPT
from app.memory.service import MemoryService, RetrievedContext

#: Similarity below which we recommend a human escalation instead of acting.
LOW_CONFIDENCE_THRESHOLD = 0.7


class IncidentResponseAgent(BaseAgent):
    def __init__(
        self,
        memory_service: MemoryService,
        llm_gateway: BaseLLMGateway,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(
            agent_type=AgentType.INCIDENT_RESPONSE,
            memory_service=memory_service,
            llm_gateway=llm_gateway,
            system_prompt=INCIDENT_RESPONSE_PROMPT,
        )
        self.settings = settings or memory_service.settings

    async def retrieve_context(self, context: AgentContext) -> list[RetrievedContext]:
        """Find similar past incidents from the alert text / error message."""
        return await self.memory.find_similar_incidents(
            query=context.query,
            service=context.payload.get("service"),
            top_k=5,
            threshold=self.settings.VECTOR_SIMILARITY_THRESHOLD,
        )

    async def generate_response(self, context: AgentContext) -> AgentResponse:
        historical = context.historical_context
        messages = self._build_prompt(context, historical)

        response_text = await self.llm.chat_completion(
            messages=messages,
            temperature=0.2,  # low temperature for consistent diagnostics
            max_tokens=1500,
        )

        avg_similarity = self._average_similarity(historical)
        confidence = min(avg_similarity + 0.1, 0.95) if historical else 0.3
        suggestions = self._build_suggestions(historical)

        follow_up: list[dict[str, Any]] = []
        if avg_similarity < LOW_CONFIDENCE_THRESHOLD:
            follow_up.append(
                {
                    "action": "escalate",
                    "reason": (
                        "Low similarity to historical incidents - page the subject "
                        "matter expert before making changes"
                    ),
                    "priority": "high" if not historical else "medium",
                }
            )
        if any(item.runbook is not None for item in historical):
            follow_up.append(
                {
                    "action": "execute_runbook",
                    "reason": "A matching runbook was found in similar incidents",
                    "priority": "medium",
                }
            )

        return AgentResponse(
            response=response_text,
            confidence=round(confidence, 3),
            suggestions=suggestions,
            historical_matches=self._historical_match_ids(historical),
            follow_up_actions=follow_up,
            metadata={
                "agent_type": self.agent_type.value,
                "similar_incidents_count": len(historical),
                "avg_similarity": round(avg_similarity, 3),
                "runbook_matched": any(item.runbook is not None for item in historical),
            },
        )

    def _build_suggestions(self, historical: list[RetrievedContext]) -> list[dict[str, Any]]:
        """Turn retrieved history into concrete, deduplicated suggestions."""
        suggestions: list[dict[str, Any]] = []
        seen_runbooks: set[str] = set()

        for item in historical:
            if item.runbook is not None and str(item.runbook.id) not in seen_runbooks:
                seen_runbooks.add(str(item.runbook.id))
                suggestions.append(
                    {
                        "type": "runbook",
                        "title": item.runbook.name,
                        "description": item.runbook.description
                        or "Runbook used for a similar incident",
                        "runbook_id": str(item.runbook.id),
                        "confidence": round(item.similarity, 3),
                    }
                )
            if item.incident is not None and item.incident.root_cause:
                suggestions.append(
                    {
                        "type": "root_cause_hypothesis",
                        "title": f"Possible root cause: {item.incident.root_cause[:120]}",
                        "description": f"Based on incident: {item.incident.title}",
                        "incident_id": str(item.incident.id),
                        "confidence": round(item.similarity, 3),
                    }
                )
        return suggestions

    async def record_outcome(
        self,
        context: AgentContext,
        response: AgentResponse,
        feedback: Optional[str] = None,
    ) -> None:
        suggestion = await self.memory.record_suggestion(
            agent_type=self.agent_type.value,
            query_context=context.query,
            suggestion=response.response,
            confidence=response.confidence,
            historical_matches=response.historical_matches,
            incident_id=context.payload.get("incident_id"),
        )
        if feedback:
            await self.memory.record_feedback(suggestion.id, feedback)
