"""Jira integration adapter (Cloud REST API v3).

Inbound: Jira issue webhooks (shared-secret header).  Outbound: create/update
issues and comment on them - used to turn incident follow-ups into tracked work.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
from typing import Any, ClassVar, Optional

import httpx

from app.core.exceptions import IntegrationError
from app.core.utils import truncate
from app.integrations.base import IntegrationAdapter, IntegrationEvent, normalize_headers

logger = logging.getLogger(__name__)


class JiraAdapter(IntegrationAdapter):
    source: ClassVar[str] = "jira"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.base_url = (self.config.get("base_url") or "").rstrip("/")
        self.webhook_secret = self.config.get("webhook_secret") or ""
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.config.get("email") and self.config.get("api_token"))

    # --------------------------------------------------------------- inbound
    async def verify_webhook(self, headers: Any, body: bytes) -> bool:
        if not self.webhook_secret:
            logger.warning("jira webhook secret not configured - accepting unsigned webhook")
            return True
        provided = normalize_headers(headers).get("X-Webhook-Secret") or ""
        return hmac.compare_digest(provided, self.webhook_secret)

    def parse_webhook(self, headers: Any, body: bytes) -> list[IntegrationEvent]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise IntegrationError(
                f"Invalid Jira webhook body: {exc}", source=self.source
            ) from exc

        issue = payload.get("issue")
        if not issue:
            logger.info("Jira webhook without issue payload: %s", payload.get("webhookEvent"))
            return []

        fields = issue.get("fields") or {}
        webhook_event = str(payload.get("webhookEvent") or "jira:issue_updated")
        event_type = "ticket.updated"
        if "created" in webhook_event:
            event_type = "ticket.created"
        elif "deleted" in webhook_event or fields.get("status", {}).get("name") in {
            "Done",
            "Closed",
        }:
            event_type = "ticket.closed"

        assignee = fields.get("assignee") or {}
        return [
            IntegrationEvent(
                event_type=event_type,
                source=self.source,
                timestamp=str(fields.get("updated") or fields.get("created") or ""),
                delivery_id=payload.get("webhookEvent") and issue.get("id"),
                payload={
                    "id": issue.get("id"),
                    "key": issue.get("key"),
                    "summary": fields.get("summary"),
                    "description": truncate(str(fields.get("description") or ""), 4000),
                    "status": (fields.get("status") or {}).get("name"),
                    "assignee": assignee.get("emailAddress") or assignee.get("displayName"),
                    "project": (fields.get("project") or {}).get("key"),
                    "priority": (fields.get("priority") or {}).get("name"),
                    "webhook_event": webhook_event,
                },
            )
        ]

    # -------------------------------------------------------------- outbound
    def _http(self) -> httpx.AsyncClient:
        if not self.enabled:
            raise IntegrationError(
                "Jira is not configured (JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN)",
                source=self.source,
            )
        if self._client is None:
            credentials = base64.b64encode(
                f"{self.config['email']}:{self.config['api_token']}".encode()
            ).decode()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Accept": "application/json",
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
                f"Jira {method} {path} failed: {exc.response.status_code} "
                f"{exc.response.text[:200]}",
                source=self.source,
            ) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError(
                f"Jira {method} {path} failed: {exc}", source=self.source
            ) from exc

    @staticmethod
    def _adf(text: str) -> dict[str, Any]:
        """Wrap plain text in Atlassian Document Format (Jira v3 requires ADF)."""
        paragraphs = [line for line in (text or "").splitlines() if line.strip()] or [""]
        return {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": truncate(line, 30000)}],
                }
                for line in paragraphs
            ],
        }

    async def create_ticket(self, title: str, description: str, **kwargs: Any) -> str:
        """Create an issue; returns the issue key (e.g. ``ENG-123``)."""
        project_key = kwargs.get("project_key")
        if not project_key:
            raise IntegrationError(
                "Jira issue creation requires a 'project_key' argument", source=self.source
            )
        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": truncate(title, 250),
            "description": self._adf(description),
            "issuetype": {"name": kwargs.get("issue_type", "Task")},
        }
        if kwargs.get("labels"):
            fields["labels"] = kwargs["labels"]
        if kwargs.get("priority"):
            fields["priority"] = {"name": kwargs["priority"]}

        payload: dict[str, Any] = {"fields": fields}
        if kwargs.get("assignee_account_id"):
            payload["update"] = {
                "assignee": [{"set": {"accountId": kwargs["assignee_account_id"]}}]
            }

        response = await self._request("POST", "/rest/api/3/issue", json=payload)
        data = response.json()
        return str(data.get("key") or "")

    async def update_ticket(self, ticket_id: str, **kwargs: Any) -> bool:
        """Update summary/description/labels/priority of an issue."""
        fields: dict[str, Any] = {}
        if kwargs.get("title"):
            fields["summary"] = truncate(str(kwargs["title"]), 250)
        if kwargs.get("description"):
            fields["description"] = self._adf(str(kwargs["description"]))
        if kwargs.get("labels"):
            fields["labels"] = kwargs["labels"]
        if kwargs.get("priority"):
            fields["priority"] = {"name": kwargs["priority"]}
        if not fields:
            return False
        await self._request("PUT", f"/rest/api/3/issue/{ticket_id}", json={"fields": fields})
        return True

    async def post_comment(self, resource_id: str, comment: str, **kwargs: Any) -> bool:
        """Add a comment to an issue; use ``POST /transitions`` separately to move it."""
        await self._request(
            "POST",
            f"/rest/api/3/issue/{resource_id}/comment",
            json={"body": self._adf(comment)},
        )
        return True

    async def transition_ticket(self, ticket_id: str, transition_name: str) -> bool:
        """Move an issue through its workflow by transition name."""
        response = await self._request(
            "GET", f"/rest/api/3/issue/{ticket_id}/transitions"
        )
        for transition in response.json().get("transitions", []):
            if transition.get("name", "").lower() == transition_name.lower():
                await self._request(
                    "POST",
                    f"/rest/api/3/issue/{ticket_id}/transitions",
                    json={"transition": {"id": transition["id"]}},
                )
                return True
        logger.warning("Transition %r not available for %s", transition_name, ticket_id)
        return False

    async def fetch_details(self, resource_id: str) -> dict[str, Any]:
        """Fetch issue fields (summary, status, assignee, priority)."""
        response = await self._request("GET", f"/rest/api/3/issue/{resource_id}")
        data = response.json()
        fields = data.get("fields") or {}
        return {
            "key": data.get("key"),
            "summary": fields.get("summary"),
            "description": str(fields.get("description") or ""),
            "status": (fields.get("status") or {}).get("name"),
            "assignee": ((fields.get("assignee") or {}) or {}).get("emailAddress"),
            "priority": (fields.get("priority") or {}).get("name"),
            "project": (fields.get("project") or {}).get("key"),
            "url": f"{self.base_url}/browse/{data.get('key')}" if self.base_url else None,
        }

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
