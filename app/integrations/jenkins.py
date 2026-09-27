"""Jenkins integration adapter.

Inbound: Jenkins "notification" plugin webhooks (``X-Jenkins-Secret`` shared
secret).  Outbound: trigger builds, poll build status/console output - used by
the DevOps Pipeline Agent to gate or roll back a release.
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

_PHASE_TO_EVENT = {
    "STARTED": "build.started",
    "FINALIZED": "build.completed",
    "COMPLETED": "build.completed",
}

#: Terminal statuses that imply failure once a build leaves the "running"
#: phase. The Jenkins notification plugin reports both `status` ("SUCCESS" /
#: "FAILURE" / "ABORTED") and `phase` ("STARTED" / "COMPLETED" / "FINALIZED").
_FAILED_STATUSES = {"FAILURE", "FAILED", "ABORTED", "UNSTABLE", "TIMED_OUT"}


class JenkinsAdapter(IntegrationAdapter):
    source: ClassVar[str] = "jenkins"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.base_url = (self.config.get("base_url") or "").rstrip("/")
        self.webhook_secret = self.config.get("webhook_secret") or ""
        self._client: Optional[httpx.AsyncClient] = None

    # --------------------------------------------------------------- inbound
    async def verify_webhook(self, headers: Any, body: bytes) -> bool:
        if not self.webhook_secret:
            logger.warning("jenkins webhook secret not configured - rejecting request")
            return False
        provided = normalize_headers(headers).get("X-Jenkins-Secret") or ""
        return hmac.compare_digest(provided, self.webhook_secret)

    def parse_webhook(self, headers: Any, body: bytes) -> list[IntegrationEvent]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise IntegrationError(
                f"Invalid Jenkins webhook body: {exc}", source=self.source
            ) from exc

        build = payload.get("build") or payload
        job_name = build.get("full_url") or build.get("job_name") or "unknown-job"
        parameters = build.get("parameters") if isinstance(build.get("parameters"), dict) else {}
        phase = str(build.get("phase") or "").upper()
        status = str(build.get("status") or "").upper()
        result = str(build.get("result") or "").upper()
        # Depending on the plugin version the terminal outcome arrives in
        # either `status` (legacy) or `result`.
        outcome = result or status

        if phase == "STARTED" or not outcome:
            event_type = _PHASE_TO_EVENT.get(phase, "build.started")
        elif outcome in _FAILED_STATUSES:
            event_type = "build.failed"
        else:
            event_type = "build.completed"

        return [
            IntegrationEvent(
                event_type=event_type,
                source=self.source,
                timestamp=str(build.get("timestamp") or ""),
                delivery_id=normalize_headers(headers).get("X-Jenkins-Delivery"),
                payload={
                    "run_id": f"{job_name}#{build.get('number')}",
                    "job_name": job_name,
                    "build_number": build.get("number"),
                    "status": (status or phase).lower(),
                    "result": outcome.lower() or None,
                    "repo": job_name,
                    "environment": parameters.get("ENVIRONMENT"),
                    "commit_sha": parameters.get("GIT_COMMIT"),
                    "failure_reason": (
                        f"Jenkins job {job_name} #{build.get('number')} finished with {outcome}"
                        if outcome in _FAILED_STATUSES
                        else None
                    ),
                },
            )
        ]

    # -------------------------------------------------------------- outbound
    def _http(self) -> httpx.AsyncClient:
        if not self.base_url:
            raise IntegrationError(
                "Jenkins base URL is not configured (set JENKINS_BASE_URL)", source=self.source
            )
        if self._client is None:
            auth = None
            if self.config.get("user") and self.config.get("api_token"):
                auth = (self.config["user"], self.config["api_token"])
            self._client = httpx.AsyncClient(base_url=self.base_url, auth=auth, timeout=30.0)
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http().request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise IntegrationError(
                f"Jenkins {method} {path} failed: {exc.response.status_code}",
                source=self.source,
            ) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError(
                f"Jenkins {method} {path} failed: {exc}", source=self.source
            ) from exc

    @staticmethod
    def _job_path(job_name: str) -> str:
        """Build the nested job path Jenkins uses for folder-based jobs."""
        return "/job/" + "/job/".join(
            quote(part, safe="") for part in job_name.strip("/").split("/")
        )

    async def trigger_build(
        self, job_name: str, parameters: Optional[dict[str, Any]] = None
    ) -> bool:
        """Trigger a job (with parameters when supplied)."""
        job_path = self._job_path(job_name)
        if parameters:
            await self._request("POST", f"{job_path}/buildWithParameters", data=parameters)
        else:
            await self._request("POST", f"{job_path}/build")
        return True

    async def fetch_details(self, resource_id: str) -> dict[str, Any]:
        """Fetch build metadata for ``job/name#buildNumber`` (or last build)."""
        if "#" in resource_id:
            job_name, build_number = resource_id.rsplit("#", 1)
        else:
            job_name, build_number = resource_id, "lastBuild"
        response = await self._request(
            "GET", f"{self._job_path(job_name)}/{build_number}/api/json", params={"tree": "*"}
        )
        data = response.json()
        actions = data.get("actions") or [{}]
        return {
            "job_name": job_name,
            "build_number": data.get("number"),
            "status": data.get("result") or "running",
            "url": data.get("url"),
            "duration_ms": data.get("duration"),
            "commit_sha": (actions[0].get("lastBuiltRevision") or {}).get("SHA1"),
        }

    async def fetch_console_output(self, job_name: str, build_number: str | int) -> str:
        """Read the tail of a build's console log (useful for failure triage)."""
        response = await self._request(
            "GET", f"{self._job_path(job_name)}/{build_number}/consoleText"
        )
        return response.text[-20000:]

    async def post_comment(self, resource_id: str, comment: str, **kwargs: Any) -> bool:
        """Jenkins has no comment API; the job description is updated instead."""
        job_name = resource_id.split("#", 1)[0]
        await self._request(
            "POST", f"{self._job_path(job_name)}/description", data={"description": comment}
        )
        return True

    async def create_ticket(self, title: str, description: str, **kwargs: Any) -> str:
        raise IntegrationError(
            "Jenkins has no ticketing system - route ticket creation to Jira or GitHub",
            source=self.source,
        )

    async def update_ticket(self, ticket_id: str, **kwargs: Any) -> bool:
        raise IntegrationError(
            "Jenkins has no ticketing system - route ticket updates to Jira or GitHub",
            source=self.source,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
