"""SSO / OAuth2 (OIDC) integration.

Flow implemented:

1. ``GET /auth/login``            -> redirect to the identity provider
2. ``GET /auth/callback?code=...`` -> exchange the code, read userinfo, upsert
                                      the local ``users`` row and issue our JWT
3. ``POST /auth/token``           -> the same exchange for API clients
                                      (``OAuth2PasswordBearer`` compatible)
4. ``GET /auth/me``               -> caller identity from the JWT

``POST /auth/dev-login`` issues a token directly for a seeded user and is
disabled outside development/staging.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlencode
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.rbac import ROLE_ADMIN, UserContext, get_current_user
from app.auth.tokens import create_access_token
from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.core.exceptions import IntegrationError
from app.core.utils import utcnow
from app.memory.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    team_id: Optional[UUID] = None


class OAuth2SSOClient:
    """Thin OIDC client: authorization redirect, code exchange, userinfo."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def authorization_url(self, state: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.settings.OAUTH2_CLIENT_ID,
            "redirect_uri": self.settings.OAUTH2_REDIRECT_URI,
            "scope": self.settings.OAUTH2_SCOPES,
            "state": state,
        }
        return f"{self.settings.OAUTH2_AUTHORIZATION_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: Optional[str] = None) -> dict[str, Any]:
        """Exchange an authorization code for provider tokens."""
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri or self.settings.OAUTH2_REDIRECT_URI,
            "client_id": self.settings.OAUTH2_CLIENT_ID,
            "client_secret": self.settings.OAUTH2_CLIENT_SECRET,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self.settings.OAUTH2_TOKEN_URL,
                    data=payload,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise IntegrationError(
                    f"SSO token exchange failed: {exc}", source="oauth2"
                ) from exc
            return response.json()

    async def fetch_userinfo(self, access_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    self.settings.OAUTH2_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise IntegrationError(
                    f"SSO userinfo lookup failed: {exc}", source="oauth2"
                ) from exc
            return response.json()


async def upsert_user(
    session: AsyncSession, userinfo: dict[str, Any], settings: Optional[Settings] = None
) -> User:
    """Create or refresh the local user from the SSO userinfo payload."""
    settings = settings or get_settings()
    subject = str(userinfo.get("sub") or userinfo.get("email") or "")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SSO userinfo did not contain a subject",
        )

    email = str(userinfo.get("email") or f"{subject}@unknown")
    name = str(userinfo.get("name") or userinfo.get("preferred_username") or email)
    is_admin_email = email.lower() in {e.lower() for e in settings.OAUTH2_ADMIN_EMAILS}

    user = (
        await session.execute(select(User).where(User.external_id == subject))
    ).scalar_one_or_none()

    if user is None:
        user = User(
            email=email,
            name=name,
            role=ROLE_ADMIN if is_admin_email else settings.DEFAULT_USER_ROLE,
            external_id=subject,
        )
        session.add(user)
        await session.flush()
        logger.info("provisioned user %s (%s)", email, user.role)
    else:
        user.email = email
        user.name = name
        if is_admin_email:
            user.role = ROLE_ADMIN

    user.last_login_at = utcnow()
    return user


def _token_response(user: User, settings: Settings) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(
            user.external_id,
            settings,
            role=user.role,
            team_id=user.team_id,
            email=user.email,
        ),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=user.role,
        team_id=user.team_id,
    )


@router.get("/login", summary="Redirect to the identity provider")
async def login(request: Request, settings: Settings = Depends(get_settings)) -> RedirectResponse:
    client = OAuth2SSOClient(settings)
    # Short-lived signed state protects the callback against CSRF/replay.
    state = create_access_token(
        subject="oauth-state", settings=settings, expires_minutes=10, extra_claims={"use": "sso"}
    )
    return RedirectResponse(client.authorization_url(state))


@router.get("/callback", response_model=TokenResponse, summary="SSO redirect target")
async def callback(
    code: str,
    state: Optional[str] = None,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_session),
) -> TokenResponse:
    client = OAuth2SSOClient(settings)
    tokens = await client.exchange_code(code)
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="SSO did not return an access token")
    userinfo = await client.fetch_userinfo(access_token)
    user = await upsert_user(db, userinfo, settings)
    return _token_response(user, settings)


@router.post("/token", response_model=TokenResponse, summary="Exchange an SSO code for a JWT")
async def token(
    code: str = Form(..., description="Authorization code from the identity provider"),
    redirect_uri: Optional[str] = Form(default=None),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_session),
) -> TokenResponse:
    client = OAuth2SSOClient(settings)
    tokens = await client.exchange_code(code, redirect_uri=redirect_uri)
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="SSO did not return an access token")
    userinfo = await client.fetch_userinfo(access_token)
    user = await upsert_user(db, userinfo, settings)
    return _token_response(user, settings)


@router.get("/me", summary="Current authenticated user")
async def me(user: UserContext = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user_id": str(user.user_id),
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "team_id": str(user.team_id) if user.team_id else None,
    }


@router.post(
    "/dev-login",
    response_model=TokenResponse,
    summary="Issue a token for an existing user (non-production only)",
)
async def dev_login(
    email: str,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_session),
) -> TokenResponse:
    if settings.is_production:
        raise HTTPException(status_code=403, detail="dev-login is disabled in production")
    user = (
        await db.execute(select(User).where(User.email == email))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail=f"No user with email {email}")
    return _token_response(user, settings)
