"""DevOps Pipeline Agent.

Scores deployment risk from historical outcomes (weighted by semantic
similarity) combined with the intrinsic risk of each change type, recommends
pipeline gates, and writes the risk score back onto the deployment record.
"""

from __future__ import annotations

from typing import Any, Optional

from app.agents.base import AgentContext, AgentResponse, AgentType, BaseAgent
from app.core.config import Settings
from app.llm.gateway import BaseLLMGateway
from app.llm.prompts import DEVOPS_PIPELINE_PROMPT
from app.memory.models import Deployment, DeploymentStatus
from app.memory.service import MemoryService, RetrievedContext

#: Intrinsic risk per change type (0-1). Tunable from historical outcomes.
CHANGE_TYPE_RISK: dict[str, float] = {
    "database_migration": 0.8,
    "infra_change": 0.6,
    "dependency_bump": 0.5,
    "config_change": 0.4,
    "code_change": 0.3,
    "feature_flag": 0.2,
}
DEFAULT_CHANGE_RISK = 0.3

FAILED_STATUSES = {DeploymentStatus.FAILED.value, DeploymentStatus.ROLLED_BACK.value}

#: Weights blending learned history with intrinsic change risk.
HISTORY_WEIGHT = 0.6
CHANGE_WEIGHT = 0.4


class DevOpsPipelineAgent(BaseAgent):
    def __init__(
        self,
        memory_service: MemoryService,
        llm_gateway: BaseLLMGateway,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(
            agent_type=AgentType.DEVOPS_PIPELINE,
            memory_service=memory_service,
            llm_gateway=llm_gateway,
            system_prompt=DEVOPS_PIPELINE_PROMPT,
        )
        self.settings = settings or memory_service.settings

    async def retrieve_context(self, context: AgentContext) -> list[RetrievedContext]:
        """Find similar past deployments (by service and change description)."""
        payload = context.payload
        query = self._build_query(context)
        return await self.memory.find_similar_deployments(
            query=query,
            service=payload.get("service"),
            top_k=6,
            threshold=self.settings.VECTOR_SIMILARITY_THRESHOLD,
        )

    @staticmethod
    def _build_query(context: AgentContext) -> str:
        query = context.query
        changes = context.payload.get("changes") or []
        if changes:
            described = ", ".join(
                f"{DevOpsPipelineAgent._change_type(change)}: "
                f"{DevOpsPipelineAgent._change_description(change)}"
                for change in changes
            )
            query = f"{query} Changes: {described}"
        return query

    @staticmethod
    def _change_type(change: Any) -> str:
        if isinstance(change, dict):
            return str(
                getattr(change.get("change_type"), "value", None)
                or change.get("change_type")
                or change.get("type")
                or "code_change"
            )
        return str(getattr(getattr(change, "change_type", None), "value", "code_change"))

    @staticmethod
    def _change_description(change: Any) -> str:
        if isinstance(change, dict):
            return str(change.get("description", ""))
        return str(getattr(change, "description", ""))

    async def generate_response(self, context: AgentContext) -> AgentResponse:
        historical = context.historical_context
        messages = self._build_prompt(context, historical)

        response_text = await self.llm.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )

        risk_score = self.compute_risk_score(historical, context.payload)
        confidence = self._confidence(risk_score, historical)

        return AgentResponse(
            response=response_text,
            confidence=confidence,
            suggestions=self._extract_risk_factors(historical),
            historical_matches=self._historical_match_ids(historical),
            follow_up_actions=self.get_pipeline_recommendations(risk_score, context.payload),
            metadata={
                "agent_type": self.agent_type.value,
                "risk_score": risk_score,
                "risk_band": self._risk_band(risk_score),
                "similar_deployments": len(historical),
                "failed_similar_deployments": sum(
                    1
                    for item in historical
                    if item.deployment is not None
                    and getattr(item.deployment.status, "value", item.deployment.status)
                    in FAILED_STATUSES
                ),
            },
        )

    # --------------------------------------------------------------- scoring
    def compute_risk_score(
        self,
        historical: list[RetrievedContext],
        payload: dict[str, Any],
    ) -> float:
        """Blend learned failure rate with intrinsic change-type risk.

        ``risk = HISTORY_WEIGHT * similarity_weighted_failure_rate
                  + CHANGE_WEIGHT * max(change_type_risk)``
        capped at 0.95. Without comparable history we fall back to a neutral
        0.3 baseline failure rate.
        """
        total_weight = 0.0
        weighted_failure = 0.0
        for item in historical:
            if item.deployment is None:
                continue
            total_weight += item.similarity
            status = getattr(item.deployment.status, "value", item.deployment.status)
            if status in FAILED_STATUSES:
                weighted_failure += item.similarity

        historical_failure_rate = weighted_failure / total_weight if total_weight else 0.3

        change_risk = DEFAULT_CHANGE_RISK
        for change in payload.get("changes") or []:
            change_risk = max(
                change_risk, CHANGE_TYPE_RISK.get(self._change_type(change), DEFAULT_CHANGE_RISK)
            )

        score = historical_failure_rate * HISTORY_WEIGHT + change_risk * CHANGE_WEIGHT
        return round(min(0.95, max(0.05, score)), 3)

    @staticmethod
    def _risk_band(risk_score: float) -> str:
        if risk_score >= 0.7:
            return "high"
        if risk_score >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _confidence(risk_score: float, historical: list[RetrievedContext]) -> float:
        """Confidence grows with comparable history and with score decisiveness."""
        if not historical:
            return 0.4
        decisiveness = abs(risk_score - 0.5) * 2  # 0 when ambiguous, 1 when extreme
        return round(min(0.95, 0.55 + 0.25 * min(len(historical), 4) / 4 + 0.15 * decisiveness), 3)

    def _extract_risk_factors(self, historical: list[RetrievedContext]) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []
        for item in historical:
            deployment = item.deployment
            if deployment is None:
                continue
            status = getattr(deployment.status, "value", deployment.status)
            if status in FAILED_STATUSES:
                factors.append(
                    {
                        "factor": (
                            f"Similar deployment {status}: "
                            f"{deployment.failure_reason or 'reason unknown'}"
                        ),
                        "impact": "high",
                        "evidence": f"{deployment.service} @ {deployment.commit_sha[:8]}",
                        "similarity": round(item.similarity, 3),
                    }
                )
        return factors

    def get_pipeline_recommendations(
        self, risk_score: float, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Gates/actions implied by the risk score and the change types present."""
        recommendations: list[dict[str, Any]] = []

        if risk_score > 0.7:
            recommendations.extend(
                [
                    {
                        "action": "require_manual_approval",
                        "reason": "High risk deployment",
                        "priority": "high",
                    },
                    {
                        "action": "enable_canary",
                        "reason": "Gradual rollout limits blast radius",
                        "priority": "high",
                    },
                    {
                        "action": "extend_monitoring_window",
                        "reason": "Watch the service for 30 minutes post-deploy",
                        "priority": "medium",
                    },
                ]
            )
        elif risk_score > 0.4:
            recommendations.extend(
                [
                    {
                        "action": "run_integration_tests",
                        "reason": "Medium risk - verify in staging first",
                        "priority": "medium",
                    },
                    {
                        "action": "notify_on_call",
                        "reason": "Awareness for a medium risk deploy",
                        "priority": "low",
                    },
                ]
            )

        for change in payload.get("changes") or []:
            change_type = self._change_type(change)
            if change_type == "database_migration":
                recommendations.append(
                    {
                        "action": "verify_rollback_plan",
                        "reason": "Database migrations require a tested rollback path",
                        "priority": "high",
                    }
                )
            elif change_type == "dependency_bump":
                recommendations.append(
                    {
                        "action": "run_dependency_audit",
                        "reason": "Check for breaking changes in the bumped dependencies",
                        "priority": "medium",
                    }
                )
            elif change_type == "config_change":
                recommendations.append(
                    {
                        "action": "diff_runtime_config",
                        "reason": "Config drift is a common failure cause",
                        "priority": "medium",
                    }
                )
        return recommendations

    async def record_outcome(
        self,
        context: AgentContext,
        response: AgentResponse,
        feedback: Optional[str] = None,
    ) -> None:
        suggestion = await self.memory.record_suggestion(
            agent_type=self.agent_type.value,
            query_context=context.query[:2000],
            suggestion=response.response,
            confidence=response.confidence,
            historical_matches=response.historical_matches,
            deployment_id=context.payload.get("deployment_id"),
        )
        if feedback:
            await self.memory.record_feedback(suggestion.id, feedback)

        # Persist the computed risk onto the deployment record itself.
        deployment_id = context.payload.get("deployment_id")
        risk_score = response.metadata.get("risk_score")
        if deployment_id and risk_score is not None:
            deployment = await self.memory.session.get(Deployment, deployment_id)
            if deployment is not None:
                deployment.risk_score = float(risk_score)
                deployment.agent_notes = response.response
                await self.memory.session.flush()
