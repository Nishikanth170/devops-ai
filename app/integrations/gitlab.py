"""GitLab integration adapter.

GitLab authenticates webhooks with a static secret in ``X-Gitlab-Token``
(compared in constant time) rather than an HMAC signature.
"""

from __future__ import annotations

import hmac
import json
import logging
from typing import Any, ClassVar, Optional
from urllib.parse import quote

import httpx

from app.core.exceptions import IntegrationError
from app.integrations.base import IntegrationAdapter, IntegrationEvent, normalize_headers

logger = logging.getLogger(__name__)


class GitLabAdapter(IntegrationAdapter):
    """GitLab (SaaS or self-managed) integration."""

    source: ClassVar[str] = "gitlab"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.webhook_secret = self.config.get("webhook_secret") or ""
        self.base_url = (self.config.get("base_url") or "https://gitlab.com/api/v4").rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # --------------------------------------------------------------- inbound
    async def verify_webhook(self, headers: Any, body: bytes) -> bool:
        if not self.webhook_secret:
            logger.warning("gitlab webhook secret not configured - rejecting request")
            return False
        token = normalize_headers(headers).get("X-Gitlab-Token") or ""
        return hmac.compare_digest(token, self.webhook_secret)

    def parse_webhook(self, headers: Any, body: bytes) -> list[IntegrationEvent]:
        headers = normalize_headers(headers)
        delivery_id = headers.get("X-Gitlab-Event-UUID") or headers.get("X-Gitlab-Event")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise IntegrationError(
                f"Invalid GitLab webhook body: {exc}", source=self.source
            ) from exc

        kind = payload.get("object_kind", "")
        project = (payload.get("project") or {}).get("path_with_namespace", "")

        if kind == "merge_request":
            return self._parse_merge_request(payload, project, delivery_id)
        if kind == "pipeline":
            return self._parse_pipeline(payload, project, delivery_id)
        return []

    def _parse_merge_request(
        self, payload: dict[str, Any], project: str, delivery_id: Optional[str]
    ) -> list[IntegrationEvent]:
        attributes = payload.get("object_attributes") or {}
        if attributes.get("action") not in {"open", "reopen", "update"}:
            return []
        return [
            IntegrationEvent(
                event_type="pull_request.opened",
                source=self.source,
                timestamp=attributes.get("updated_at") or attributes.get("created_at") or "",
                delivery_id=delivery_id,
                payload={
                    "number": attributes.get("iid"),
                    "repo": project,
                    "title": attributes.get("title"),
                    "description": attributes.get("description"),
                    "author": (payload.get("user") or {}).get("username"),
                    "head_sha": (attributes.get("last_commit") or {}).get("id"),
                    "source_branch": attributes.get("source_branch"),
                    "target_branch": attributes.get("target_branch"),
                    "action": attributes.get("action"),
                },
            )
        ]

    def _parse_pipeline(
        self, payload: dict[str, Any], project: str, delivery_id: Optional[str]
    ) -> list[IntegrationEvent]:
        attributes = payload.get("object_attributes") or {}
        status = attributes.get("status")
        event_type = {
            "running": "build.started",
            "pending": "build.started",
            "success": "build.completed",
            "failed": "build.failed",
            "canceled": "build.failed",
        }.get(status, "build.started")
        builds = payload.get("builds") or [{}]
        return [
            IntegrationEvent(
                event_type=event_type,
                source=self.source,
                timestamp=attributes.get("created_at") or "",
                delivery_id=delivery_id,
                payload={
                    "run_id": attributes.get("id"),
                    "repo": project,
                    "status": status,
                    "ref": attributes.get("ref"),
                    "head_sha": attributes.get("sha"),
                    "job_name": builds[0].get("name") if builds else None,
                    "failure_reason": attributes.get("detailed_status")
                    if status in {"failed", "canceled"}
                    else None,
                },
            )
        ]

    # -------------------------------------------------------------- outbound
    def _http(self) -> httpx.AsyncClient:
        token = self.config.get("token")
        if not token:
            raise IntegrationError(
                "GitLab token is not configured (set GITLAB_TOKEN)", source=self.source
            )
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"PRIVATE-TOKEN": token},
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
                f"GitLab {method} {path} failed: {exc.response.status_code} "
                f"{exc.response.text[:200]}",
                source=self.source,
            ) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError(
                f"GitLab {method} {path} failed: {exc}", source=self.source
            ) from exc

    @staticmethod
    def parse_reference(resource_id: str) -> tuple[str, str, str]:
        """Parse ``group/project!12`` or ``group/project#34`` into its parts."""
        if "!" in resource_id:
            project, iid = resource_id.rsplit("!", 1)
            return project, iid, "merge_request"
        if "#" in resource_id:
            project, iid = resource_id.rsplit("#", 1)
            return project, iid, "issue"
        raise IntegrationError(
            f"Expected 'group/project!12' or 'group/project#34', got {resource_id!r}",
            source="gitlab",
        )

    async def post_comment(self, resource_id: str, comment: str, **kwargs: Any) -> bool:
        """Comment on a merge request or issue."""
        project, iid, kind = self.parse_reference(resource_id)
        project_id = quote(project, safe="")
        if kind == "merge_request":
            path = f"/projects/{project_id}/merge_requests/{iid}/notes"
        else:
            path = f"/projects/{project_id}/issues/{iid}/notes"
        await self._request("POST", path, json={"body": comment})
        return True

    async def fetch_mr_changes(self, project: str, mr_iid: str | int) -> str:
        """Return the unified diff (``changes``) of a merge request."""
        project_id = quote(project, safe="")
        response = await self._request(
            "GET", f"/projects/{project_id}/merge_requests/{mr_iid}/changes"
        )
        changes = response.json().get("changes") or []
        return "\n".join(
            f"--- {c.get('old_path')}\n+++ {c.get('new_path')}\n{c.get('diff', '')}"
            for c in changes
        )

    async def fetch_details(self, resource_id: str) -> dict[str, Any]:
        project, iid, kind = self.parse_reference(resource_id)
        project_id = quote(project, safe="")
        if kind == "merge_request":
            data = (
                await self._request("GET", f"/projects/{project_id}/merge_requests/{iid}")
            ).json()
            return {
                "number": data.get("iid"),
                "repo": project,
                "title": data.get("title"),
                "description": data.get("description"),
                "author": (data.get("author") or {}).get("username"),
                "state": data.get("state"),
                "head_sha": (data.get("diff_refs") or {}).get("head_sha"),
                "source_branch": data.get("source_branch"),
                "target_branch": data.get("target_branch"),
            }
        data = (await self._request("GET", f"/projects/{project_id}/issues/{iid}")).json()
        return {
            "key": f"{project}#{data.get('iid')}",
            "title": data.get("title"),
            "description": data.get("description"),
            "status": data.get("state"),
        }

    async def create_ticket(self, title: str, description: str, **kwargs: Any) -> str:
        project = kwargs.get("repo") or kwargs.get("project")
        if not project:
            raise IntegrationError(
                "GitLab issue creation requires a 'repo' argument", source=self.source
            )
        project_id = quote(project, safe="")
        payload: dict[str, Any] = {"title": title, "description": description}
        if kwargs.get("labels"):
            payload["labels"] = ",".join(kwargs["labels"])
        if kwargs.get("assignee_ids"):
            payload["assignee_ids"] = kwargs["assignee_ids"]
        data = (
            await self._request("POST", f"/projects/{project_id}/issues", json=payload)
        ).json()
        return f"{project}#{data.get('iid')}"

    async def update_ticket(self, ticket_id: str, **kwargs: Any) -> bool:
        project, iid, kind = self.parse_reference(ticket_id)
        project_id = quote(project, safe="")
        segment = "merge_requests" if kind == "merge_request" else "issues"
        payload = {
            key: value
            for key, value in {
                "title": kwargs.get("title"),
                "description": kwargs.get("description") or kwargs.get("body"),
                "state_event": kwargs.get("state_event"),
                "add_labels": ",".join(kwargs["labels"]) if kwargs.get("labels") else None,
            }.items()
            if value is not None
        }
        if not payload:
            return False
        await self._request("PUT", f"/projects/{project_id}/{segment}/{iid}", json=payload)
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
