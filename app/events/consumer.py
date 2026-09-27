"""Standalone queue worker.

Run with ``python -m app.events.consumer`` (the ``worker`` service in
docker-compose).  The API process also consumes events in-process for
single-node deployments; scaling out beyond one node should run API-only pods
plus N worker processes so agent latency never blocks webhook acknowledgements.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from app.container import AppServices
from app.core.config import get_settings
from app.core.database import close_db, init_db
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    settings = get_settings()
    settings.assert_production_ready()
    configure_logging()

    await init_db()
    services = AppServices.build(settings)
    await services.queue.start_consumer()

    logger.info(
        "worker consuming %s (%s handlers registered)",
        settings.QUEUE_BACKEND,
        services.queue.handler_count(),
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, AttributeError):  # pragma: no cover - Windows
            logger.debug("signal handler registration unsupported for %s", sig)

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover - manual stop
        logger.info("worker interrupted")
    finally:
        logger.info("worker shutting down: %s", services.queue.stats())
        await services.shutdown()
        await close_db()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
