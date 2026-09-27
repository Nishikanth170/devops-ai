"""Domain exceptions and their HTTP mapping.

Raising a typed exception deep in the domain layer lets the API layer translate
it once (see :func:`register_exception_handlers`) instead of raising raw
``HTTPException`` from services/agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


class PlatformError(Exception):
    """Base class for all platform errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR


class ConfigurationError(PlatformError):
    """Invalid or missing configuration."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class NotFoundError(PlatformError):
    """Requested resource does not exist."""

    status_code = status.HTTP_404_NOT_FOUND


class PermissionDeniedError(PlatformError):
    """Authenticated principal is not allowed to perform the action."""

    status_code = status.HTTP_403_FORBIDDEN


class IntegrationError(PlatformError):
    """Upstream integration (GitHub, Jira, Jenkins, ...) failed."""

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self, message: str, *, source: Optional[str] = None) -> None:
        super().__init__(message)
        self.source = source


class WebhookVerificationError(IntegrationError):
    """Webhook signature / shared secret did not validate."""

    status_code = status.HTTP_401_UNAUTHORIZED


class AgentError(PlatformError):
    """Agent failed to produce a response."""

    def __init__(self, message: str, *, agent_type: Optional[str] = None) -> None:
        super().__init__(message)
        self.agent_type = agent_type


class QueueError(PlatformError):
    """Event queue unavailable or event could not be published."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class LLMGatewayError(PlatformError):
    """LLM provider call failed after retries."""

    status_code = status.HTTP_502_BAD_GATEWAY


def register_exception_handlers(app: FastAPI) -> None:
    """Map :class:`PlatformError` subclasses onto JSON API responses."""

    async def _handle_platform_error(request: Request, exc: PlatformError) -> JSONResponse:
        logger.warning("%s on %s: %s", type(exc).__name__, request.url.path, exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc), "error": type(exc).__name__},
        )

    app.add_exception_handler(PlatformError, _handle_platform_error)  # type: ignore[arg-type]
