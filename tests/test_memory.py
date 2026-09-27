"""Memory tests: embeddings, hybrid retrieval and the feedback loop.

Uses the offline fake gateway and the in-memory vector backend against a real
SQLite database, so the hybrid retrieval path is exercised for real.
"""

from __future__ import annotations

import uuid

import pytest

from app.memory.service import FEEDBACK_VALUES
from app.memory.vector_store import VectorStore, cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_degenerate_vectors_score_zero(self):
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_round_trip_and_filters(self, settings):
        store = VectorStore(session=None, settings=settings)
        first = await store.add_memory(
            "incidents", uuid.uuid4(), "db pool exhausted",
            [1.0, 0.0], {"service": "payments"},
        )
        second = await store.add_memory(
            "incidents", uuid.uuid4(), "frontend 500s",
            [0.0, 1.0], {"service": "web"},
        )

        hits = await store.similarity_search([1.0, 0.0], top_k=5, threshold=0.99)
        assert [(record.id, score) for record, score in hits] == [(first.id, 1.0)]

        filtered = await store.similarity_search(
            [1.0, 0.0],
            top_k=5,
            source_table="incidents",
            metadata_filter={"service": "web"},
            threshold=0.0,
        )
        assert [record.id for record, _ in filtered] == [second.id]

        assert await store.count("incidents") == 2
        assert len(await store.get_by_source("incidents", first.source_id)) == 1
        assert await store.delete_by_source("incidents", first.source_id) == 1
        assert await store.count("incidents") == 1

    @pytest.mark.asyncio
    async def test_requires_session_for_pgvector(self, settings):
        from app.core.exceptions import ConfigurationError

        settings.VECTOR_BACKEND = "pgvector"
        with pytest.raises(ConfigurationError):
            VectorStore(session=None, settings=settings)


class TestMemoryServiceEmbeddings:
    @pytest.mark.asyncio
    async def test_embed_text_calls_llm(self, memory_service, fake_llm, settings):
        embedding = await memory_service.embed_text("test incident")
        assert len(embedding) == settings.VECTOR_DIMENSIONS
        assert any(value != 0.0 for value in embedding)

    @pytest.mark.asyncio
    async def test_similar_texts_embed_similarly(self, memory_service):
        first = await memory_service.embed_text("database connection pool exhausted")
        second = await memory_service.embed_text("database connection pool is exhausted")
        other = await memory_service.embed_text("frontend css alignment glitch")
        assert cosine_similarity(first, second) > cosine_similarity(first, other)


class TestIncidentMemory:
    @pytest.mark.asyncio
    async def test_record_and_retrieve_similar(
        self, memory_service, db_session, team, make_incident
    ):
        past = make_incident(team.id)
        db_session.add(past)
        await db_session.flush()
        await memory_service.record_incident(past)

        unrelated = make_incident(
            team.id,
            title="CSS button alignment off by two pixels",
            service="web-frontend",
            root_cause="flexbox gap",
            resolution="fixed margin",
        )
        unrelated.description = "button misaligned on the settings page"
        db_session.add(unrelated)
        await db_session.flush()
        await memory_service.record_incident(unrelated)

        query = (
            "Database connection pool exhausted in payments-service during traffic spike"
        )
        contexts = await memory_service.find_similar_incidents(
            query, service="payments", top_k=3, threshold=0.0
        )

        assert contexts, "expected at least one similar incident"
        top = contexts[0]
        assert top.incident is not None
        assert top.incident.title == "Database pool exhausted"
        assert top.similarity > contexts[-1].similarity or len(contexts) == 1

    @pytest.mark.asyncio
    async def test_get_incident_with_context_excludes_self(
        self, memory_service, db_session, team, make_incident
    ):
        incident = make_incident(team.id)
        db_session.add(incident)
        await db_session.flush()
        await memory_service.record_incident(incident)

        context = await memory_service.get_incident_with_context(incident.id)

        assert context is not None
        assert context.incident.id == incident.id
        assert context.similar_incidents == []


class TestFeedbackLoop:
    @pytest.mark.asyncio
    async def test_record_and_rate_suggestion(self, memory_service, db_session):
        record = await memory_service.record_suggestion(
            agent_type="incident_response",
            query_context="pool exhausted",
            suggestion="increase pool size",
            confidence=0.8,
            historical_matches=[uuid.uuid4()],
        )

        assert await memory_service.record_feedback(record.id, "helpful") is True
        fetched = await memory_service.get_suggestion(record.id)
        assert fetched is not None
        assert fetched.user_feedback == "helpful"
        assert fetched.feedback_at is not None

    @pytest.mark.asyncio
    async def test_feedback_rejects_unknown_values(self, memory_service):
        with pytest.raises(ValueError):
            await memory_service.record_feedback(uuid.uuid4(), "meh")

    @pytest.mark.asyncio
    async def test_feedback_missing_suggestion_returns_false(self, memory_service):
        assert await memory_service.record_feedback(uuid.uuid4(), "helpful") is False

    def test_feedback_values_documented(self):
        assert FEEDBACK_VALUES == {"helpful", "not_helpful", "incorrect"}

    @pytest.mark.asyncio
    async def test_retrieval_stats(self, memory_service, db_session, team, make_incident):
        incident = make_incident(team.id)
        db_session.add(incident)
        await db_session.flush()
        await memory_service.record_incident(incident)

        record = await memory_service.record_suggestion(
            agent_type="code_review",
            query_context="diff",
            suggestion="fix n+1",
            confidence=0.7,
        )
        await memory_service.record_feedback(record.id, "helpful")

        stats = await memory_service.get_retrieval_stats()

        assert stats["total_vector_entries"] == 1
        assert stats["total_suggestions"] == 1
        assert stats["feedback_distribution"] == {"helpful": 1}
        assert stats["helpful_rate"] == 1.0
        assert stats["vector_backend"] == "memory"

    @pytest.mark.asyncio
    async def test_gateway_batch_embeddings(self, fake_llm):
        embeddings = await fake_llm.generate_embeddings_batch(["a", "b", "c"])
        assert len(embeddings) == 3
        assert all(isinstance(vector, list) for vector in embeddings)
