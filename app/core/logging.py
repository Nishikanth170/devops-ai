"""Logging setup.

Uses ``structlog`` for structured (JSON-friendly) logs when it is installed and
transparently falls back to the standard library otherwise, so the platform has
no hard dependency on structlog.
"""

from __future__ import annotations

import logging
import sys

from app.core.config import get_settings


def configure_logging(level: int | None = None) -> None:
    """Configure application logging once, at process startup."""
    settings = get_settings()
    log_level = level if level is not None else (logging.DEBUG if settings.DEBUG else logging.INFO)

    try:  # pragma: no cover - depends on optional dependency
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer()
                if settings.ENVIRONMENT in {"development", "test"}
                else structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)
    except ImportError:
        logging.basicConfig(
            level=log_level,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
        )

    # Third-party noise control.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.DEBUG else logging.WARNING
    )
