"""Role-based access control.

Roles are intentionally coarse (three levels) because the platform scopes most
access by *team* rather than by fine-grained permission: an on-call engineer sees
their team's incidents/deployments, a team lead additionally manages runbooks and
sees agents' feedback metrics, and an admin sees everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.tokens import decode_access_token
from app.core.database import get_session
from app.core.exceptions import PermissionDeniedError
from app.memory.models import User

ROLE_ON_CALL_ENGINEER = "on-call-engineer"
ROLE_TEAM_LEAD = "team-lead"
ROLE_ADMIN = "admin"
ALL_ROLES = (ROLE_ON_CALL_ENGINEER, ROLE_TEAM_LEAD, ROLE_ADMIN)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


@dataclass
class UserContext:
    """Authenticated principal attached to each request."""

    user_id: UUID
    email: str
    name: str
    role: str
    team_id: Optional[UUID] = None

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    @property
    def can_manage(self) -> bool:
        return self.role in (ROLE_ADMIN, ROLE_TEAM_LEAD)

    def scope_team_id(self, requested_team_id: Optional[UUID] = None) -> Optional[UUID]:
        """Team filter to apply: admins may query any team, others only their own."""
        if self.is_admin:
            return requested_team_id
        return self.team_id


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_session),
) -> UserContext:
    """Resolve the bearer token into a :class:`UserContext`."""
    if not token:
        raise PermissionDeniedError("Missing bearer token")

    payload = decode_access_token(token)
    subject = payload.get("sub")
    if not subject:
        raise PermissionDeniedError("Token is missing the subject claim")

    user = (
        await db.execute(select(User).where(User.external_id == str(subject)))
    ).scalar_one_or_none()
    if user is None:
        raise PermissionDeniedError("User is not provisioned on this platform")

    return UserContext(
        user_id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        team_id=user.team_id,
    )


def require_role(*allowed_roles: str):
    """Dependency factory enforcing that the caller has one of ``allowed_roles``."""

    async def _checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if user.role not in allowed_roles:
            raise PermissionDeniedError(
                f"Requires one of the following roles: {', '.join(allowed_roles)}"
            )
        return user

    return _checker


def ensure_team_access(user: UserContext, team_id: Optional[UUID]) -> None:
    """Raise when a non-admin tries to touch another team's data."""
    if user.is_admin or team_id is None or user.team_id is None:
        return
    if team_id != user.team_id:
        raise PermissionDeniedError("Access to another team's data is not permitted")
