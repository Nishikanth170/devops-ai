"""Unit tests for the three specialist agents.

These tests use small mocks for memory/LLM plus the offline fake gateway, so
no network or database server is required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.agents.base import AgentContext, AgentType
from app.agents.code_review import CodeReviewAgent
from app.agents.devops_pipeline import DevOpsPipelineAgent
from app.agents.incident_response import IncidentResponseAgent
from app.llm.gateway import BaseLLMGateway
from app.memory.service import MemoryService, RetrievedContext
from app.memory.vector_store import VectorRecord


@pytest.fixture()
def test_settings():
    from app.core.config import Settings

    return Settings(
        ENVIRONMENT="test",
        LLM_PROVIDER="fake",
        VECTOR_BACKEND="memory",
        QUEUE_BACKEND="memory",
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        VECTOR_DIMENSIONS=64,
        VECTOR_SIMILARITY_THRESHOLD=0.3,
        DATABASE_CREATE_SCHEMA=False,
    )


@pytest.fixture()
def mock_memory(test_settings):
    memory = AsyncMock(spec=MemoryService)
    memory.settings = test_settings
    memory.find_similar_incidents = AsyncMock(return_value=[])
    memory.find_similar_review_patterns = AsyncMock(return_value=[])
    memory.find_similar_deployments = AsyncMock(return_value=[])
    memory.record_suggestion = AsyncMock()
    return memory


@pytest.fixture()
def mock_llm():
    llm = AsyncMock(spec=BaseLLMGateway)
    llm.chat_completion = AsyncMock(return_value="Test response")
    llm.generate_embedding = AsyncMock(return_value=[0.1] * 64)
    return llm


@pytest.fixture()
def incident_context():
    return AgentContext(
        query="Database connection pool exhausted in payments-service",
        payload={
            "service": "payments-service",
            "severity": "SEV2",
            "event_type": "incident.triggered",
        },
        team_id=uuid4(),
    )


def make_vector_record(content: str = "similar incident") -> VectorRecord:
    return VectorRecord(
        id=uuid4(),
        source_table="incidents",
        source_id=uuid4(),
        content=content,
        embedding=[0.1] * 64,
        metadata={},
    )


def make_incident_history(vector_match=None, similarity=0.9):
    incident = MagicMock()
    incident.title = "Past DB pool exhaustion"
    incident.service = "payments-service"
    incident.severity = MagicMock(value="SEV2")
    incident.root_cause = "Connection leak in retry logic"
    incident.resolution_steps = "Fixed retry logic, increased pool"
    incident.runbook_id = None
    return RetrievedContext(
        similarity=similarity,
        vector_match=vector_match or make_vector_record(),
        incident=incident,
    )


class TestIncidentResponseAgent:
    @pytest.mark.asyncio
    async def test_retrieve_context_calls_memory(
        self, mock_memory, mock_llm, incident_context
    ):
        agent = IncidentResponseAgent(mock_memory, mock_llm)
        mock_memory.find_similar_incidents.return_value = [make_incident_history()]

        result = await agent.retrieve_context(incident_context)

        assert len(result) == 1
        assert result[0].similarity == pytest.approx(0.9)
        mock_memory.find_similar_incidents.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_response_embeds_runbook_suggestions(
        self, mock_memory, mock_llm, incident_context
    ):
        agent = IncidentResponseAgent(mock_memory, mock_llm)
        runbook = MagicMock()
        runbook.id = uuid4()
        runbook.name = "db-pool-exhaustion"
        runbook.description = "Drain and restart the pool"

        history = make_incident_history()
        history.runbook = runbook
        incident_context.historical_context = [history]

        response = await agent.generate_response(incident_context)

        assert response.confidence > 0
        assert response.response == "Test response"
        runbook_suggestions = [
            s for s in response.suggestions if s["type"] == "runbook"
        ]
        assert runbook_suggestions and runbook_suggestions[0]["title"] == "db-pool-exhaustion"
        assert response.metadata["runbook_matched"] is True

    @pytest.mark.asyncio
    async def test_low_similarity_triggers_escalation(
        self, mock_memory, mock_llm, incident_context
    ):
        agent = IncidentResponseAgent(mock_memory, mock_llm)
        incident_context.historical_context = [make_incident_history(similarity=0.4)]

        response = await agent.generate_response(incident_context)

        escalations = [
            action for action in response.follow_up_actions if action["action"] == "escalate"
        ]
        assert escalations, "expected an escalation follow-up on low confidence"

    @pytest.mark.asyncio
    async def test_no_history_stays_honest(self, mock_memory, mock_llm, incident_context):
        agent = IncidentResponseAgent(mock_memory, mock_llm)
        incident_context.historical_context = []

        response = await agent.generate_response(incident_context)

        assert response.confidence <= 0.3
        assert any(
            action["action"] == "escalate" for action in response.follow_up_actions
        )


class TestCodeReviewAgent:
    @pytest.mark.asyncio
    async def test_retrieve_context_searches_patterns(self, mock_memory, mock_llm):
        agent = CodeReviewAgent(mock_memory, mock_llm)
        context = AgentContext(
            query="PR with SQL query in loop",
            payload={
                "repo": "backend/api",
                "diff_summary": "Added user lookup in for loop",
                "review_id": uuid4(),
            },
            team_id=uuid4(),
        )
        mock_memory.find_similar_review_patterns.return_value = [
            RetrievedContext(
                similarity=0.88,
                vector_match=make_vector_record(),
                review_pattern=MagicMock(
                    pattern_type="n_plus_one_finding",
                    description="N+1 query in loop",
                    frequency=3,
                ),
                code_review=MagicMock(
                    pr_id="123",
                    repo="backend/api",
                    reviewer_agent_notes="N+1 query detected",
                ),
            )
        ]

        result = await agent.retrieve_context(context)

        assert len(result) == 1
        mock_memory.find_similar_review_patterns.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_json_response_is_parsed_into_comments(self, mock_memory, mock_llm):
        structured = {
            "overall_assessment": "medium risk",
            "inline_comments": [
                {
                    "file_path": "app/db.py",
                    "line_number": 12,
                    "severity": "error",
                    "category": "performance",
                    "message": "N+1 query in loop",
                    "confidence": 0.9,
                }
            ],
            "patterns_matched": ["n_plus_one_finding"],
            "positive_notes": ["good test coverage"],
        }
        mock_llm.chat_completion = AsyncMock(
            return_value=f"**Overall Assessment**: medium\n\n```json\n{json.dumps(structured)}\n```"
        )
        agent = CodeReviewAgent(mock_memory, mock_llm)
        context = AgentContext(
            query="Review PR",
            payload={"repo": "backend/api", "diff": "diff...", "review_id": uuid4()},
            team_id=uuid4(),
        )

        response = await agent.generate_response(context)

        assert len(response.suggestions) == 1
        comment = response.suggestions[0]
        assert comment["file_path"] == "app/db.py"
        assert comment["severity"] == "error"
        assert comment["line_number"] == 12
        assert any(
            action["action"] == "human_review_required"
            for action in response.follow_up_actions
        )

    def test_extract_json_handles_fenced_and_bare(self):
        payload = {"inline_comments": []}
        fenced = f"text ```json\n{json.dumps(payload)}\n``` tail"
        assert CodeReviewAgent.extract_json(fenced) == payload
        assert CodeReviewAgent.extract_json(json.dumps(payload)) == payload
        assert CodeReviewAgent.extract_json("no json here") is None


class TestDevOpsPipelineAgent:
    @pytest.mark.asyncio
    async def test_compute_risk_score_high_for_db_migration(self, mock_memory, mock_llm):
        agent = DevOpsPipelineAgent(mock_memory, mock_llm)
        deployment = MagicMock()
        deployment.status = MagicMock(value="failed")
        deployment.failure_reason = "Migration deadlock"
        deployment.service = "orders-service"
        history = RetrievedContext(
            similarity=0.9,
            vector_match=make_vector_record(),
            deployment=deployment,
        )
        payload = {
            "service": "orders-service",
            "changes": [
                {"type": "database_migration", "description": "Add index to orders table"}
            ],
        }

        risk = agent.compute_risk_score([history], payload)

        assert risk > 0.7

    @pytest.mark.asyncio
    async def test_recommendations_cover_high_risk(self, mock_memory, mock_llm):
        agent = DevOpsPipelineAgent(mock_memory, mock_llm)
        recommendations = agent.get_pipeline_recommendations(
            risk_score=0.85,
            payload={"changes": [{"type": "database_migration"}]},
        )
        actions = [item["action"] for item in recommendations]
        assert "require_manual_approval" in actions
        assert "enable_canary" in actions
        assert "verify_rollback_plan" in actions

    @pytest.mark.asyncio
    async def test_risk_band_boundaries(self, mock_memory, mock_llm):
        agent = DevOpsPipelineAgent(mock_memory, mock_llm)
        assert agent._risk_band(0.85) == "high"
        assert agent._risk_band(0.5) == "medium"
        assert agent._risk_band(0.1) == "low"


class TestAgentTypes:
    def test_agent_type_values_match_registry(self):
        assert {agent.value for agent in AgentType} == {
            "incident_response",
            "code_review",
            "devops_pipeline",
        }
