"""Integration adapter interface.

Adapters are the only place that knows about third-party payload shapes.  They
translate vendor webhooks into :class:`IntegrationEvent` objects (the platform's
normalised inbound event) and expose the outbound operations the agents need
(comment on a PR, create a ticket, fetch details).

Headers are wrapped in a case-insensitive mapping because vendors disagree:
GitHub sends ``X-Hub-Signature-256``, GitLab sends ``X-Gitlab-Token``, Jenkins
sends ``X-Jenkins-Secret`` and proxies may lowercase all of them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from app.core.exceptions import IntegrationError

logger = logging.getLogger(__name__)


class CaseInsensitiveHeaders(Mapping[str, str]):
    """Read-only, case-insensitive view over an HTTP header mapping."""

    def __init__(self, headers: Optional[Mapping[str, Any]] = None) -> None:
        self._store: dict[str, str] = {
            str(key).lower(): str(value) for key, value in dict(headers or {}).items()
        }

    def __getitem__(self, key: str) -> str:
        return self._store[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self._store.get(str(key).lower(), default)


def normalize_headers(headers: Optional[Mapping[str, Any]]) -> CaseInsensitiveHeaders:
    if isinstance(headers, CaseInsensitiveHeaders):
        return headers
    return CaseInsensitiveHeaders(headers)


@dataclass
class IntegrationEvent:
    """Normalised internal event produced by an adapter."""

    event_type: str  # e.g. "pull_request.opened", "alert.fired"
    source: str  # e.g. "github", "datadog", "jenkins"
    timestamp: str  # ISO 8601
    payload: dict[str, Any] = field(default_factory=dict)
    #: External id of the delivery, used for idempotency/dedup.
    delivery_id: Optional[str] = None
    #: Resolved later by the ingestion layer (repo/service -> team).
    team_id: Optional[str] = None


class IntegrationAdapter(ABC):
    """Base class for every inbound/outbound integration."""

    #: Short source key used in webhook URLs: ``/webhooks/{source}``.
    source: ClassVar[str] = "unknown"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}

    @property
    def enabled(self) -> bool:
        """Whether outbound credentials are configured."""
        return bool(self.config.get("token") or self.config.get("api_token"))

    # ----------------------------------------------------------------- inbound
    @abstractmethod
    async def verify_webhook(self, headers: Mapping[str, Any], body: bytes) -> bool:
        """Return True when the request really comes from the vendor."""

    @abstractmethod
    def parse_webhook(self, headers: Mapping[str, Any], body: bytes) -> list[IntegrationEvent]:
        """Translate a vendor webhook into normalised events (may be empty)."""

    # ---------------------------------------------------------------- outbound
    @abstractmethod
    async def post_comment(self, resource_id: str, comment: str, **kwargs: Any) -> bool:
        """Publish a comment/review onto the external resource."""

    @abstractmethod
    async def fetch_details(self, resource_id: str) -> dict[str, Any]:
        """Fetch the full details of an external resource."""

    @abstractmethod
    async def create_ticket(self, title: str, description: str, **kwargs: Any) -> str:
        """Create a ticket/issue; returns its external identifier."""

    @abstractmethod
    async def update_ticket(self, ticket_id: str, **kwargs: Any) -> bool:
        """Update an existing ticket/issue."""

    # ----------------------------------------------------------------- helpers
    def _require(self, *keys: str) -> None:
        """Raise a clear error when outbound credentials are missing."""
        missing = [key for key in keys if not self.config.get(key)]
        if missing:
            raise IntegrationError(
                f"{self.source} adapter is missing configuration: {', '.join(missing)}",
                source=self.source,
            )
