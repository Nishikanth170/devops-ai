"""Small shared helpers (time, hashing, text)."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

_WS_RE = re.compile(r"\s+")


def utcnow() -> datetime:
    """Naive UTC timestamp.

    ``datetime.utcnow()`` is deprecated in Python 3.12+ and the schema uses
    naive ``TIMESTAMP`` columns, so we centralise the conversion here.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def sha256_hex(value: str | bytes) -> str:
    """Hex SHA-256 digest, used for diff de-duplication and idempotency keys."""
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (keeps embedded text compact for prompts)."""
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str | None, limit: int, suffix: str = "...") -> str:
    """Truncate ``text`` to ``limit`` characters, appending ``suffix`` if cut."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def slugify(value: str, fallback: str = "unknown") -> str:
    """Lowercase dash-separated slug (team / service names)."""
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or fallback
