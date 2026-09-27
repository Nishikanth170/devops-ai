"""API smoke tests: the full stack over a real (SQLite) database."""

from __future__ import annotations

import pytest

from app.memory.models import CodeReview, Incident, IncidentStatus


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_and_readiness(self, api_client):
        health = await api_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        ready = await api_client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"

    @pytest.mark.asyncio
    async def test_protected_endpoints_need_auth(self, api_client):
        response = await api_client.get("/incidents")
        assert response.status_code in {401, 403}

    @pytest.mark.asyncio
    async def test_unknown_webhook_source_is_404(self, api_client, admin_user, auth_headers):
        response = await api_client.post(
            "/webhooks/nope", content=b"{}", headers=auth_headers(admin_user)
        )
        assert response.status_code == 404


class TestIncidentsApi:
    @pytest.mark.asyncio
    async def test_create_list_resolve_feedback(
        self, api_client, admin_user, auth_headers, db_session, team
    ):
        headers = auth_headers(admin_user)

        created = await api_client.post(
            "/incidents",
            json={
                "title": "Pool saturation",
                "description": "connection pool 95% full during traffic spike",
                "severity": "SEV2",
                "service": "payments",
                "team_id": str(team.id),
                "run_agent": True,
            },
            headers=headers,
        )
        assert created.status_code == 201, created.text
        incident_id = created.json()["id"]

        listed = await api_client.get("/incidents", headers=headers)
        assert listed.status_code == 200
        assert listed.json()["total"] == 1

        detail = await api_client.get(f"/incidents/{incident_id}", headers=headers)
        assert detail.status_code == 200
        body = detail.json()
        assert body["title"] == "Pool saturation"
        assert len(body["agent_suggestions"]) >= 1

        resolved = await api_client.post(
            f"/incidents/{incident_id}/resolve",
            json={
                "root_cause": "connection leak in retry logic",
                "resolution_steps": "bounded pool size, fixed retry backoff",
            },
            headers=headers,
        )
        assert resolved.status_code == 200
        assert resolved.json()["status"] == "resolved"

        rated = await api_client.post(
            f"/incidents/{incident_id}/feedback",
            json={"feedback": "helpful"},
            headers=headers,
        )
        assert rated.status_code == 200
        assert rated.json()["status"] == "recorded"


class TestPipelineApi:
    @pytest.mark.asyncio
    async def test_assess_and_stats(self, api_client, admin_user, auth_headers):
        headers = auth_headers(admin_user)
        response = await api_client.post(
            "/pipeline/assess",
            json={
                "service": "payments",
                "commit_sha": "abc123",
                "environment": "production",
                "changes": [
                    {
                        "change_type": "database_migration",
                        "description": "add index to orders",
                        "file_paths": ["migrations/001.sql"],
                    }
                ],
            },
            headers=headers,
        )
        assert response.status_code == 200
        assessed = response.json()
        assert assessed["risk_score"] > 0.0
        assert assessed["risk_band"] in {"low", "medium", "high"}
        assert "verify_rollback_plan" in [
            item["action"] for item in assessed["recommendations"]
        ]

        stats = await api_client.get("/pipeline/stats", headers=headers)
        assert stats.status_code == 200
        assert stats.json()["by_service"] == []

    @pytest.mark.asyncio
    async def test_dashboard_summary(
        self, api_client, admin_user, auth_headers, db_session, team
    ):
        db_session.add(
            Incident(
                title="open incident",
                service="payments",
                severity="SEV3",
                status=IncidentStatus.OPEN,
                team_id=team.id,
            )
        )
        db_session.add(
            CodeReview(
                pr_id="1",
                repo="acme/payments",
                author="dev",
                title="Fix leak",
                diff_hash="x",
                status="open",
                team_id=team.id,
            )
        )
        await db_session.flush()
        await db_session.commit()

        headers = auth_headers(admin_user)
        summary = await api_client.get("/dashboard/summary", headers=headers)
        assert summary.status_code == 200
        body = summary.json()
        assert body["open_incidents"] >= 1
        assert body["prs_awaiting_review"] >= 1
        assert isinstance(body["deployment_failure_rate"], float)


class TestWebhookToAgentFlow:
    @pytest.mark.asyncio
    async def test_github_webhook_is_accepted(self, api_client, admin_user, auth_headers):
        import hashlib
        import hmac
        import json

        headers = auth_headers(admin_user)
        body = json.dumps(
            {
                "action": "opened",
                "repository": {"full_name": "acme/payments"},
                "pull_request": {
                    "number": 42,
                    "title": "Fix leak",
                    "body": "",
                    "user": {"login": "dev"},
                    "diff_url": "",
                    "head": {"sha": "abc"},
                    "base": {"sha": "def"},
                    "updated_at": "2024-01-01T00:00:00Z",
                },
            }
        ).encode()
        digest = hmac.new(b"test-github-secret", body, hashlib.sha256).hexdigest()
        response = await api_client.post(
            "/webhooks/github",
            content=body,
            headers={
                **headers,
                "X-Hub-Signature-256": f"sha256={digest}",
                "X-GitHub-Event": "pull_request",
            },
        )
        assert response.status_code == 200
        accepted = response.json()
        assert accepted["events_published"] == 1
        assert accepted["event_types"] == ["pull_request.opened"]

    @pytest.mark.asyncio
    async def test_rejected_signature_is_401(self, api_client, admin_user, auth_headers):
        response = await api_client.post(
            "/webhooks/github",
            content=b"{}",
            headers={**auth_headers(admin_user), "X-Hub-Signature-256": "sha256=bad"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_agent_query_routes_to_incidents(
        self, api_client, admin_user, auth_headers
    ):
        response = await api_client.post(
            "/agent/query",
            json={"query": "the database pool is exhausted, what do I do?"},
            headers=auth_headers(admin_user),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_type"] == "incident_response"
        assert body["confidence"] >= 0.0
