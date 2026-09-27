"""Team routing: map a repo/service to a team (and create teams on demand)."""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.utils import slugify
from app.memory.models import Team

logger = logging.getLogger(__name__)


def team_name_for(
    repo: Optional[str] = None,
    service: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> str:
    """Resolve the configured team name for a repo/service pair."""
    settings = settings or get_settings()
    routing = settings.TEAM_ROUTING or {}
    lowered = {str(key).lower(): value for key, value in routing.items()}

    for candidate in (repo, service):
        if not candidate:
            continue
        if candidate in routing:
            return routing[candidate]
        if str(candidate).lower() in lowered:
            return lowered[str(candidate).lower()]
        # Also match by the trailing path segment (e.g. "acme/payments" -> "payments").
        leaf = str(candidate).split("/")[-1].lower()
        if leaf in lowered:
            return lowered[leaf]

    return settings.DEFAULT_TEAM_NAME


async def get_or_create_team(session: AsyncSession, name: str) -> Team:
    """Fetch a team by name, creating it when it does not exist yet."""
    team = (await session.execute(select(Team).where(Team.name == name))).scalar_one_or_none()
    if team is None:
        team = Team(name=slugify(name, fallback="platform"))
        session.add(team)
        await session.flush()
        logger.info("created team %s", team.name)
    return team


async def resolve_team_id(
    session: AsyncSession,
    repo: Optional[str] = None,
    service: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> UUID:
    """Return the team UUID responsible for a repo/service."""
    name = team_name_for(repo=repo, service=service, settings=settings)
    team = await get_or_create_team(session, name)
    return team.id
