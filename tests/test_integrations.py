"""Integration adapter tests: signature verification and webhook parsing."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from app.integrations.datadog import DatadogAdapter
from app.integrations.github import GitHubAdapter
from app.integrations.gitlab import GitLabAdapter
from app.integrations.jenkins import JenkinsAdapter
from app.integrations.jira import JiraAdapter


@pytest.fixture()
def github() -> GitHubAdapter:
    return GitHubAdapter({"webhook_secret": "s3cret", "token": "tok"})


@pytest.fixture()
def github_pr_payload() -> bytes:
    return json.dumps(
        {
            "action": "opened",
            "repository": {"full_name": "acme/payments"},
            "pull_request": {
                "number": 42,
                "title": "Fix connection leak",
                "body": "Bounds the pool.",
                "user": {"login": "dev"},
                "diff_url": "https://api.github.com/diff",
                "head": {"sha": "abc123"},
                "base": {"sha": "def456"},
                "updated_at": "2024-01-01T00:00:00Z",
            },
        }
    ).encode()


def sign_github(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class TestGitHubAdapter:
    @pytest.mark.asyncio
    async def test_verify_ok_and_rejects_tampering(self, github, github_pr_payload):
        headers = {"X-Hub-Signature-256": sign_github("s3cret", github_pr_payload)}
        assert await github.verify_webhook(headers, github_pr_payload) is True
        assert await github.verify_webhook(headers, github_pr_payload + b" ") is False

    @pytest.mark.asyncio
    async def test_verify_rejects_missing_signature(self, github, github_pr_payload):
        assert await github.verify_webhook({}, github_pr_payload) is False

    def test_parse_pull_request(self, github, github_pr_payload):
        events = github.parse_webhook(
            {"X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "d-1"},
            github_pr_payload,
        )
        assert len(events) == 1
        event = events[0]
        assert event.event_type == "pull_request.opened"
        assert event.source == "github"
        assert event.delivery_id == "d-1"
        assert event.payload["repo"] == "acme/payments"
        assert event.payload["number"] == 42
        assert event.payload["action"] == "opened"

    def test_parse_ignores_closed_prs(self, github):
        body = json.dumps(
            {"action": "closed", "repository": {"full_name": "acme/payments"}}
        ).encode()
        assert github.parse_webhook({"X-GitHub-Event": "pull_request"}, body) == []

    def test_parse_deployment_failure(self, github):
        body = json.dumps(
            {
                "deployment": {"id": 7, "environment": "production", "sha": "abc123"},
                "deployment_status": {"state": "failure", "description": "timeout"},
            }
        ).encode()
        events = github.parse_webhook({"X-GitHub-Event": "deployment_status"}, body)
        assert events[0].event_type == "deployment.failed"
        assert events[0].payload["failure_reason"] == "timeout"

    def test_parse_pr_reference(self, github):
        assert github.parse_pr_reference("acme/payments#42") == ("acme/payments", "42")


class TestGitLabAdapter:
    @pytest.fixture()
    def adapter(self) -> GitLabAdapter:
        return GitLabAdapter({"webhook_secret": "gitlab-secret"})

    @pytest.mark.asyncio
    async def test_verify_token(self, adapter):
        assert await adapter.verify_webhook({"X-Gitlab-Token": "gitlab-secret"}, b"{}")
        assert not await adapter.verify_webhook({"X-Gitlab-Token": "nope"}, b"{}")

    def test_parse_merge_request(self, adapter):
        body = json.dumps(
            {
                "object_kind": "merge_request",
                "object_attributes": {
                    "action": "open",
                    "iid": 3,
                    "title": "Add index",
                    "description": "db",
                    "updated_at": "2024-01-01",
                },
                "project": {"path_with_namespace": "acme/api"},
                "user": {"username": "dev"},
            }
        ).encode()
        events = adapter.parse_webhook({}, body)
        assert len(events) == 1
        assert events[0].event_type == "pull_request.opened"
        assert events[0].payload["repo"] == "acme/api"

    def test_parse_reference_splits(self, adapter):
        assert adapter.parse_reference("acme/api!3") == ("acme/api", "3", "merge_request")
        assert adapter.parse_reference("acme/api#9") == ("acme/api", "9", "issue")


class TestJenkinsAdapter:
    @pytest.fixture()
    def adapter(self) -> JenkinsAdapter:
        return JenkinsAdapter({"webhook_secret": "jenkins-secret"})

    @pytest.mark.asyncio
    async def test_verify_secret(self, adapter):
        assert await adapter.verify_webhook({"X-Jenkins-Secret": "jenkins-secret"}, b"{}")
        assert not await adapter.verify_webhook({"X-Jenkins-Secret": "x"}, b"{}")

    def test_parse_failed_build(self, adapter):
        body = json.dumps(
            {
                "name": "payments-deploy",
                "build": {
                    "phase": "FINALIZED",
                    "status": "FAILURE",
                    "number": 10,
                    "url": "http://ci/job/10",
                },
            }
        ).encode()
        events = adapter.parse_webhook({}, body)
        assert events[0].event_type == "build.failed"
        assert events[0].payload["build_number"] == 10
        assert "FAILURE" in events[0].payload["failure_reason"]


class TestDatadogAdapter:
    @pytest.fixture()
    def adapter(self) -> DatadogAdapter:
        return DatadogAdapter({})

    @pytest.mark.asyncio
    async def test_unsigned_webhooks_accepted_with_warning(self, adapter):
        assert await adapter.verify_webhook({}, b"{}") is True

    def test_parse_alert(self, adapter):
        body = json.dumps(
            {
                "alert": {
                    "id": "9",
                    "title": "Pool saturation",
                    "priority": "P1",
                    "message": "pool 95% full",
                    "tags": ["service:payments"],
                }
            }
        ).encode()
        events = adapter.parse_webhook({}, body)
        assert len(events) == 1
        event = events[0]
        assert event.event_type == "alert.fired"
        assert event.payload["service"] == "payments"
        assert event.payload["severity"] == "SEV1"

    def test_parse_pagerduty_acknowledged(self, adapter):
        body = json.dumps(
            {
                "event": "incident.acknowledged",
                "incident": {
                    "id": "PD1",
                    "title": "Outage",
                    "urgency": "high",
                    "service": {"summary": "payments"},
                },
            }
        ).encode()
        events = adapter.parse_webhook({}, body)
        assert events[0].event_type == "incident.acknowledged"
        assert events[0].source == "pagerduty"


class TestJiraAdapter:
    @pytest.fixture()
    def adapter(self) -> JiraAdapter:
        return JiraAdapter({})

    def test_parse_issue_created(self, adapter):
        body = json.dumps(
            {
                "webhookEvent": "jira:issue_created",
                "issue": {
                    "id": "1001",
                    "key": "ENG-1",
                    "fields": {
                        "summary": "Pool exhaustion",
                        "status": {"name": "To Do"},
                        "assignee": {"emailAddress": "dev@example.com"},
                        "updated": "2024-01-01T00:00:00Z",
                    },
                },
            }
        ).encode()
        events = adapter.parse_webhook({}, body)
        assert events[0].event_type == "ticket.created"
        assert events[0].payload["key"] == "ENG-1"
        assert events[0].payload["assignee"] == "dev@example.com"

    def test_adf_wraps_text(self, adapter):
        doc = JiraAdapter._adf("line one\nline two")
        assert doc["type"] == "doc"
        assert len(doc["content"]) == 2
