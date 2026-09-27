"""GitHub integration adapter (webhooks in, REST API out).

Webhook signatures use ``X-Hub-Signature-256`` (HMAC-SHA256 of the raw body).
Outbound calls authenticate with a personal/installation token in
``Authorization: Bearer ...``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, ClassVar, Optional

import httpx

from app.core.exceptions import IntegrationError
from app.core.utils import truncate
from app.integrations.base import IntegrationAdapter, IntegrationEvent, normalize_headers

logger = logging.getLogger(__name__)


class GitHubAdapter(IntegrationAdapter):
    """GitHub (and GitHub Enterprise) integration."""

    source: ClassVar[str] = "github"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.webhook_secret = self.config.get("webhook_secret") or ""
        self.base_url = (self.config.get("base_url") or "https://api.github.com").rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ---------------------------------------------------------------- inbound
    async def verify_webhook(self, headers: Any, body: bytes) -> bool:
        if not self.webhook_secret:
            logger.warning("github webhook secret not configured - rejecting request")
            return False
        headers = normalize_headers(headers)
        signature = headers.get("X-Hub-Signature-256") or headers.get("X-Hub-Signature") or ""
        if not signature.startswith("sha256="):
            return False
        expected = "sha256=" + hmac.new(
            self.webhook_secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def parse_webhook(self, headers: Any, body: bytes) -> list[IntegrationEvent]:
        headers = normalize_headers(headers)
        event_name = headers.get("X-GitHub-Event", "")
        delivery_id = headers.get("X-GitHub-Delivery")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise IntegrationError(
                f"Invalid GitHub webhook body: {exc}", source=self.source
            ) from exc

        repo_name = (payload.get("repository") or {}).get("full_name", "")
        events: list[IntegrationEvent] = []

        if event_name == "pull_request":
            action = payload.get("action")
            if action in {"opened", "reopened", "synchronize", "ready_for_review"}:
                pr = payload.get("pull_request") or {}
                events.append(
                    IntegrationEvent(
                        event_type="pull_request.opened",
                        source=self.source,
                        timestamp=pr.get("updated_at") or pr.get("created_at") or "",
                        delivery_id=delivery_id,
                        payload={
                            "number": pr.get("number"),
                            "repo": repo_name,
                            "title": pr.get("title"),
                            "description": pr.get("body"),
                            "author": (pr.get("user") or {}).get("login"),
                            "diff_url": pr.get("diff_url"),
                            "head_sha": (pr.get("head") or {}).get("sha"),
                            "base_sha": (pr.get("base") or {}).get("sha"),
                            "action": action,
                        },
                    )
                )

        elif event_name == "pull_request_review":
            review = payload.get("review") or {}
            pr = payload.get("pull_request") or {}
            events.append(
                IntegrationEvent(
                    event_type="pull_request.review_submitted",
                    source=self.source,
                    timestamp=review.get("submitted_at") or "",
                    delivery_id=delivery_id,
                    payload={
                        "pr_number": pr.get("number"),
                        "repo": repo_name,
                        "reviewer": (review.get("user") or {}).get("login"),
                        "state": review.get("state"),  # APPROVED|CHANGES_REQUESTED|COMMENTED
                        "body": truncate(review.get("body") or "", 4000),
                    },
                )
            )

        elif event_name == "deployment_status":
            events.extend(self._parse_deployment_status(payload, repo_name, delivery_id))

        elif event_name == "workflow_job":
            events.extend(self._parse_workflow_job(payload, repo_name, delivery_id))

        elif event_name == "ping":
            logger.info("GitHub ping received for %s", repo_name or "unknown repository")

        return events

    def _parse_deployment_status(
        self, payload: dict[str, Any], repo_name: str, delivery_id: Optional[str]
    ) -> list[IntegrationEvent]:
        deployment = payload.get("deployment") or {}
        status = payload.get("deployment_status") or {}
        state = (status.get("state") or "").lower()
        event_type = {
            "success": "deployment.completed",
            "failure": "deployment.failed",
            "error": "deployment.failed",
            "inactive": "deployment.rolled_back",
            "in_progress": "deployment.started",
        }.get(state, "deployment.started")
        return [
            IntegrationEvent(
                event_type=event_type,
                source=self.source,
                timestamp=status.get("updated_at") or status.get("created_at") or "",
                delivery_id=delivery_id,
                payload={
                    "deployment_id": str(deployment.get("id") or ""),
                    "status": state,
                    "environment": deployment.get("environment"),
                    "repo": repo_name,
                    "commit_sha": deployment.get("sha"),
                    "branch": deployment.get("ref"),
                    "failure_reason": (
                        status.get("description") if state in {"failure", "error"} else None
                    ),
                },
            )
        ]

    def _parse_workflow_job(
        self, payload: dict[str, Any], repo_name: str, delivery_id: Optional[str]
    ) -> list[IntegrationEvent]:
        job = payload.get("workflow_job") or {}
        conclusion = job.get("conclusion")
        event_type = {
            "completed": "build.completed",
            "in_progress": "build.started",
            "queued": "build.started",
        }.get(job.get("status") or "", "build.started")
        return [
            IntegrationEvent(
                event_type=event_type,
                source=self.source,
                timestamp=job.get("started_at") or job.get("created_at") or "",
                delivery_id=delivery_id,
                payload={
                    "run_id": job.get("run_id"),
                    "job_id": job.get("id"),
                    "name": job.get("name"),
                    "repo": repo_name,
                    "status": job.get("status"),
                    "conclusion": conclusion,
                    "head_sha": job.get("head_sha"),
                    "failure_reason": (
                        conclusion if conclusion in {"failure", "timed_out"} else None
                    ),
                },
            )
        ]

    # -------------------------------------------------------------- outbound
    def _http(self) -> httpx.AsyncClient:
        """Lazily build the authenticated API client."""
        token = self.config.get("token")
        if not token:
            raise IntegrationError(
                "GitHub token is not configured (set GITHUB_TOKEN)", source=self.source
            )
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=30.0,
            )
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http().request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise IntegrationError(
                f"GitHub {method} {path} failed: {exc.response.status_code} "
                f"{exc.response.text[:200]}",
                source=self.source,
            ) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError(
                f"GitHub {method} {path} failed: {exc}", source=self.source
            ) from exc

    @staticmethod
    def parse_pr_reference(resource_id: str) -> tuple[str, str]:
        """Parse ``owner/repo#123`` into ``("owner/repo", "123")``."""
        if "#" not in resource_id:
            raise IntegrationError(
                f"Expected a pull request reference like 'owner/repo#123', got {resource_id!r}",
                source="github",
            )
        repo, number = resource_id.split("#", 1)
        return repo.strip(), number.strip()

    async def post_comment(self, resource_id: str, comment: str, **kwargs: Any) -> bool:
        """Post a review on a pull request identified by ``owner/repo#number``."""
        repo, number = self.parse_pr_reference(resource_id)
        await self._request(
            "POST",
            f"/repos/{repo}/pulls/{number}/reviews",
            json={"body": comment, "event": kwargs.get("event", "COMMENT")},
        )
        return True

    async def post_review_comments(
        self, repo: str, pr_number: str | int, comments: list[dict[str, Any]]
    ) -> int:
        """Publish inline comments for agent findings; returns how many posted."""
        posted = 0
        for comment in comments:
            path = comment.get("file_path") or comment.get("path")
            line = comment.get("line_number") or comment.get("line") or 1
            body = comment.get("message") or comment.get("comment") or ""
            if not path or not body:
                continue
            try:
                await self._request(
                    "POST",
                    f"/repos/{repo}/pulls/{pr_number}/comments",
                    json={
                        "body": f"[{comment.get('severity', 'info')}] {body}",
                        "path": path,
                        "line": int(line),
                    },
                )
                posted += 1
            except IntegrationError as exc:
                logger.warning("Could not post review comment on %s:%s - %s", path, line, exc)
        return posted

    async def fetch_details(self, resource_id: str) -> dict[str, Any]:
        """Fetch pull request metadata (``owner/repo#number``)."""
        repo, number = self.parse_pr_reference(resource_id)
        response = await self._request("GET", f"/repos/{repo}/pulls/{number}")
        data = response.json()
        return {
            "number": data.get("number"),
            "repo": repo,
            "title": data.get("title"),
            "description": data.get("body"),
            "author": (data.get("user") or {}).get("login"),
            "state": data.get("state"),
            "merged": data.get("merged"),
            "head_sha": (data.get("head") or {}).get("sha"),
            "diff_url": data.get("diff_url"),
            "changed_files": data.get("changed_files"),
            "additions": data.get("additions"),
            "deletions": data.get("deletions"),
        }

    async def fetch_pr_diff(self, repo: str, pr_number: str | int) -> str:
        """Fetch the raw unified diff of a pull request (for review analysis)."""
        try:
            response = await self._http().get(
                f"/repos/{repo}/pulls/{pr_number}",
                headers={"Accept": "application/vnd.github.v3.diff"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise IntegrationError(
                f"Could not fetch diff for {repo}#{pr_number}: {exc.response.status_code}",
                source=self.source,
            ) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError(
                f"Could not fetch diff for {repo}#{pr_number}: {exc}", source=self.source
            ) from exc
        return response.text

    async def create_ticket(self, title: str, description: str, **kwargs: Any) -> str:
        """Create a GitHub issue (e.g. follow-up work generated from an incident)."""
        repo = kwargs.get("repo")
        if not repo:
            raise IntegrationError(
                "GitHub issue creation requires a 'repo' argument", source=self.source
            )
        payload: dict[str, Any] = {"title": title, "body": description}
        for key in ("labels", "assignees", "milestone"):
            if kwargs.get(key):
                payload[key] = kwargs[key]
        response = await self._request("POST", f"/repos/{repo}/issues", json=payload)
        data = response.json()
        return f"{repo}#{data.get('number')}"

    async def update_ticket(self, ticket_id: str, **kwargs: Any) -> bool:
        """Update an existing issue (``owner/repo#number``)."""
        repo, number = self.parse_pr_reference(ticket_id)
        payload = {
            key: value
            for key, value in kwargs.items()
            if key in {"title", "body", "state", "labels", "assignees"} and value is not None
        }
        if not payload:
            return False
        await self._request("PATCH", f"/repos/{repo}/issues/{number}", json=payload)
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
