"""Webhook ingestion endpoints (GitHub, GitLab, Jenkins, Datadog, Jira)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_webhook_ingestion
from app.api.schemas import WebhookAcceptedResponse
from app.events.ingestion import WebhookIngestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.get("/sources", response_model=list[str])
async def known_sources(
    ingestion: WebhookIngestion = Depends(get_webhook_ingestion),
) -> list[str]:
    """Sources accepted by ``POST /webhooks/{source}``."""
    return ingestion.sources


@router.post("/{source}", response_model=WebhookAcceptedResponse)
async def ingest_webhook(
    source: str,
    request: Request,
    ingestion: WebhookIngestion = Depends(get_webhook_ingestion),
) -> WebhookAcceptedResponse:
    """Verify signature, normalise the payload and enqueue it for processing.

    The endpoint acknowledges immediately: agents run asynchronously through the
    event queue so vendor timeouts never cause retries that duplicate work.
    """
    body = await request.body()
    result = await ingestion.handle_webhook(source, request.headers, body)
    logger.debug("webhook %s accepted: %s", source, result)
    return WebhookAcceptedResponse(**result)
