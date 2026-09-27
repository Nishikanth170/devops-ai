"""Monitoring integration adapter: Datadog alerts + PagerDuty incidents.

Both vendors deliver JSON webhooks.  Severity is normalised to the platform's
``SEV1..SEV4`` scale so incidents arriving from different tools can be compared.

Outbound, ``create_ticket`` raises a PagerDuty incident (Events API v2) when an
integration key is configured, otherwise it publishes a Datadog event.
"""

from __future__ import annotations

import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx

from app.core.exceptions import IntegrationError
from app.core.utils import truncate
from app.integrations.base import IntegrationAdapter, IntegrationEvent, normalize_headers

logger = logging.getLogger(__name__)

PRIORITY_TO_SEVERITY = {"P1": "SEV1", "P2": "SEV2", "P3": "SEV3", "P4": "SEV4"}
URGENCY_TO_SEVERITY = {"high": "SEV2", "low": "SEV4"}
DATADOG_EVENTS_URL = "https://api.datadoghq.com/api/v1/events"
PAGERDUTY_ENQUEUE_URL = "https://events.pagerduty.com/v2/enqueue"


class DatadogAdapter(IntegrationAdapter):
    source: ClassVar[str] = "datadog"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.app_key = self.config.get("app_key")
        self.pagerduty_key = self.config.get("pagerduty_integration_key")
        self.webhook_secret = self.config.get("webhook_secret") or ""
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key or self.pagerduty_key)

    # --------------------------------------------------------------- inbound
    async def verify_webhook(self, headers: Any, body: bytes) -> bool:
        """Verify the shared-secret header when one is configured.

        Datadog/PagerDuty webhooks have no HMAC scheme; the recommended approach
        is a custom header (``X-Webhook-Secret``) configured with the webhook.
        When no secret is configured we accept the request but log a warning so
        operators notice the gap.
        """
        if not self.webhook_secret:
            logger.warning("monitoring webhook secret not configured - accepting unsigned webhook")
            return True
        provided = normalize_headers(headers).get("X-Webhook-Secret") or ""
        return hmac.compare_digest(provided, self.webhook_secret)

    def parse_webhook(self, headers: Any, body: bytes) -> list[IntegrationEvent]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise IntegrationError(
                f"Invalid monitoring webhook body: {exc}", source=self.source
            ) from exc

        if "alert" in payload:
            return self._parse_datadog_alert(payload["alert"])
        if "incident" in payload:
            return self._parse_pagerduty_incident(payload["incident"], payload)
        if payload.get("event_type"):
            return self._parse_pagerduty_webhook(payload)
        logger.info("Unrecognised monitoring webhook keys: %s", sorted(payload.keys()))
        return []

    def _parse_datadog_alert(self, alert: dict[str, Any]) -> list[IntegrationEvent]:
        status = str(alert.get("status") or alert.get("event_type") or "triggered").lower()
        event_type = "alert.resolved" if status in {"resolved", "recovered"} else "alert.fired"
        tags = alert.get("tags") or []
        return [
            IntegrationEvent(
                event_type=event_type,
                source="datadog",
                timestamp=str(
                    alert.get("last_triggered_ts")
                    or alert.get("date")
                    or datetime.now(timezone.utc).isoformat()
                ),
                delivery_id=str(alert.get("id") or "") or None,
                payload={
                    "id": str(alert.get("id") or alert.get("monitor_id") or ""),
                    "title": alert.get("title") or "Datadog monitor alert",
                    "service": self._extract_service(tags, alert),
                    "severity": self._map_severity(alert.get("priority", "P3")),
                    "description": truncate(alert.get("message") or "", 4000),
                    "monitor_id": alert.get("monitor_id"),
                    "tags": tags,
                    "environment": self._extract_tag(tags, "env"),
                },
            )
        ]

    @staticmethod
    def _extract_tag(tags: list[str], prefix: str) -> Optional[str]:
        for tag in tags:
            if str(tag).startswith(f"{prefix}:"):
                return str(tag).split(":", 1)[1]
        return None

    def _extract_service(self, tags: list[str], alert: dict[str, Any]) -> str:
        return (
            alert.get("service")
            or self._extract_tag(tags, "service")
            or self._extract_tag(tags, "kube_namespace")
            or "unknown"
        )

    @staticmethod
    def _map_severity(priority: str) -> str:
        key = str(priority or "").strip().upper()
        if key in PRIORITY_TO_SEVERITY:
            return PRIORITY_TO_SEVERITY[key]
        return URGENCY_TO_SEVERITY.get(str(priority or "").lower(), "SEV3")

    def _parse_pagerduty_incident(
        self, incident: dict[str, Any], payload: dict[str, Any]
    ) -> list[IntegrationEvent]:
        raw_event = str(payload.get("event") or payload.get("event_type") or "triggered").lower()
        mapping = (
            ("incident.resolved", ("resolved",)),
            ("incident.acknowledged", ("acknowledged",)),
            ("incident.triggered", ("triggered", "incident.triggered")),
        )
        mapped = "incident.triggered"
        for candidate, values in mapping:
            if any(value in raw_event for value in values):
                mapped = candidate
                break

        service = incident.get("service") or {}
        priority = incident.get("priority")
        severity_source = (
            priority.get("summary")
            if isinstance(priority, dict)
            else incident.get("urgency", "high")
        )
        return [
            IntegrationEvent(
                event_type=mapped,
                source="pagerduty",
                timestamp=str(
                    incident.get("created_at") or datetime.now(timezone.utc).isoformat()
                ),
                delivery_id=str(incident.get("id") or "") or None,
                payload={
                    "id": str(incident.get("id") or ""),
                    "dedup_key": incident.get("dedup_key") or payload.get("dedup_key"),
                    "title": incident.get("title") or "PagerDuty incident",
                    "service": service.get("summary")
                    if isinstance(service, dict)
                    else str(service),
                    "severity": self._map_severity(severity_source),
                    "description": truncate(incident.get("description") or "", 4000),
                    "incident_number": incident.get("incident_number"),
                    "url": incident.get("html_url"),
                },
            )
        ]

    def _parse_pagerduty_webhook(self, payload: dict[str, Any]) -> list[IntegrationEvent]:
        event = payload.get("event") or {}
        data = event.get("data") or {}
        return self._parse_pagerduty_incident(data, {"event": event.get("event_type")})

    # -------------------------------------------------------------- outbound
    async def _post(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise IntegrationError(
                    f"Monitoring API call failed: {exc.response.status_code} "
                    f"{exc.response.text[:200]}",
                    source=self.source,
                ) from exc
            except httpx.HTTPError as exc:
                raise IntegrationError(
                    f"Monitoring API call failed: {exc}", source=self.source
                ) from exc
            return response.json() if response.content else {}

    async def create_ticket(self, title: str, description: str, **kwargs: Any) -> str:
        """Raise a PagerDuty incident (preferred) or publish a Datadog event.

        Returns the external identifier: a PagerDuty ``dedup_key`` or a Datadog
        event id.
        """
        service = kwargs.get("service", "unknown")
        severity = kwargs.get("severity", "SEV3")

        if self.pagerduty_key:
            data = await self._post(
                PAGERDUTY_ENQUEUE_URL,
                {
                    "routing_key": self.pagerduty_key,
                    "event_action": "trigger",
                    "dedup_key": kwargs.get("dedup_key") or f"{service}:{truncate(title, 60)}",
                    "payload": {
                        "summary": truncate(title, 1024),
                        "source": service,
                        "severity": "critical" if severity in {"SEV1", "SEV2"} else "warning",
                        "custom_details": {"description": description, "severity": severity},
                    },
                },
                {"Content-Type": "application/json"},
            )
            return str(data.get("dedup_key") or "")

        if self.api_key:
            data = await self._post(
                DATADOG_EVENTS_URL,
                {
                    "title": truncate(title, 100),
                    "text": description,
                    "alert_type": "error" if severity in {"SEV1", "SEV2"} else "warning",
                    "source_type_name": "devops-agent-platform",
                    "tags": [f"service:{service}", f"severity:{severity}"],
                },
                {"DD-API-KEY": self.api_key, "Content-Type": "application/json"},
            )
            return str((data.get("event") or {}).get("id") or "")

        raise IntegrationError(
            "No monitoring credentials configured (set PAGERDUTY_INTEGRATION_KEY or "
            "DATADOG_API_KEY) to raise incidents",
            source=self.source,
        )

    async def update_ticket(self, ticket_id: str, **kwargs: Any) -> bool:
        """Acknowledge or resolve a PagerDuty incident by ``dedup_key``."""
        action = kwargs.get("action")
        if not self.pagerduty_key or action not in {"acknowledge", "resolve"}:
            return False
        await self._post(
            PAGERDUTY_ENQUEUE_URL,
            {"routing_key": self.pagerduty_key, "event_action": action, "dedup_key": ticket_id},
            {"Content-Type": "application/json"},
        )
        return True

    async def post_comment(self, resource_id: str, comment: str, **kwargs: Any) -> bool:
        """Publish a Datadog event, used as a timeline annotation."""
        if not self.api_key:
            raise IntegrationError(
                "DATADOG_API_KEY is required to publish events", source=self.source
            )
        await self._post(
            DATADOG_EVENTS_URL,
            {
                "title": truncate(f"DevOps agent note for {resource_id}", 100),
                "text": comment,
                "alert_type": "info",
                "source_type_name": "devops-agent-platform",
                "tags": [f"resource:{resource_id}"],
            },
            {"DD-API-KEY": self.api_key, "Content-Type": "application/json"},
        )
        return True

    async def fetch_details(self, resource_id: str) -> dict[str, Any]:
        """Fetch a Datadog monitor definition (requires API + application key)."""
        if not (self.api_key and self.app_key):
            raise IntegrationError(
                "DATADOG_API_KEY and DATADOG_APP_KEY are required to read monitors",
                source=self.source,
            )
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"https://api.datadoghq.com/api/v1/monitor/{resource_id}",
                    headers={"DD-API-KEY": self.api_key, "DD-APPLICATION-KEY": self.app_key},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise IntegrationError(
                    f"Could not fetch monitor {resource_id}: {exc}", source=self.source
                ) from exc
            data = response.json()
        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "type": data.get("type"),
            "query": data.get("query"),
            "message": data.get("message"),
            "tags": data.get("tags"),
            "overall_state": data.get("overall_state"),
        }

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
