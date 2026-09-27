"""JWT access tokens (issued after SSO login)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from jose import JWTError, jwt

from app.core.config import Settings, get_settings
from app.core.exceptions import PermissionDeniedError


def create_access_token(
    subject: str,
    settings: Optional[Settings] = None,
    *,
    role: Optional[str] = None,
    team_id: Optional[UUID | str] = None,
    email: Optional[str] = None,
    expires_minutes: Optional[int] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """Sign an access token whose ``sub`` identifies the SSO subject."""
    settings = settings or get_settings()
    expires = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires)).timestamp()),
    }
    if role:
        payload["role"] = role
    if team_id:
        payload["team_id"] = str(team_id)
    if email:
        payload["email"] = email
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str, settings: Optional[Settings] = None) -> dict[str, Any]:
    """Decode/verify a JWT, raising :class:`PermissionDeniedError` when invalid."""
    settings = settings or get_settings()
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as exc:
        raise PermissionDeniedError("Could not validate credentials") from exc
