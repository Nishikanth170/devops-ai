"""Auth tests: JWT issuance/decoding, RBAC gating and SSO provisioning."""

from __future__ import annotations

import pytest

from app.auth.oauth2 import TokenResponse, upsert_user
from app.auth.rbac import (
    ROLE_ADMIN,
    ensure_team_access,
    get_current_user,
    require_role,
)
from app.auth.tokens import create_access_token, decode_access_token
from app.core.exceptions import PermissionDeniedError


class TestTokens:
    def test_round_trip(self, settings):
        token = create_access_token(
            "sso-user", settings, role="admin", email="admin@example.com"
        )
        payload = decode_access_token(token, settings)
        assert payload["sub"] == "sso-user"
        assert payload["role"] == "admin"

    def test_invalid_token_raises(self, settings):
        with pytest.raises(PermissionDeniedError):
            decode_access_token("not-a-token", settings)

    def test_wrong_secret_raises(self, settings):
        token = create_access_token("sso-user", settings)
        settings.JWT_SECRET_KEY = "different-secret"
        with pytest.raises(PermissionDeniedError):
            decode_access_token(token, settings)


class TestRbac:
    def test_scope_team_id(self, user_ctx):
        # admins may query any team; without an explicit request there is no filter
        assert user_ctx.scope_team_id() is None
        from uuid import uuid4

        requested = uuid4()
        assert user_ctx.scope_team_id(requested) == requested

    @pytest.mark.asyncio
    async def test_require_role_allows_and_blocks(self, user_ctx):
        assert (await require_role(ROLE_ADMIN)(user_ctx)) is user_ctx
        with pytest.raises(PermissionDeniedError):
            await require_role("other")(user_ctx)

    def test_ensure_team_access(self, user_ctx, team):
        from uuid import uuid4

        # admins may reach any team
        ensure_team_access(user_ctx, team.id)
        ensure_team_access(user_ctx, uuid4())

        # non-admins are pinned to their own team
        from app.auth.rbac import ROLE_ON_CALL_ENGINEER, UserContext

        oncall = UserContext(
            user_id=user_ctx.user_id,
            email=user_ctx.email,
            name=user_ctx.name,
            role=ROLE_ON_CALL_ENGINEER,
            team_id=team.id,
        )
        ensure_team_access(oncall, team.id)
        with pytest.raises(PermissionDeniedError):
            ensure_team_access(oncall, uuid4())

    @pytest.mark.asyncio
    async def test_get_current_user_resolves_from_db(
        self, db_session, admin_user, settings
    ):
        token = create_access_token(admin_user.external_id, settings)
        resolved = await get_current_user(token=token, db=db_session)
        assert resolved.email == admin_user.email
        assert resolved.role == admin_user.role

    @pytest.mark.asyncio
    async def test_get_current_user_rejects_ghosts(self, db_session):
        with pytest.raises(PermissionDeniedError):
            await get_current_user(token="Bearer x", db=db_session)


class TestSsoProvisioning:
    @pytest.mark.asyncio
    async def test_upsert_creates_and_promotes(self, db_session, settings):
        user = await upsert_user(
            db_session,
            {"sub": "sso-1", "email": "new@example.com", "name": "New"},
            settings,
        )
        assert user.role == settings.DEFAULT_USER_ROLE
        assert user.last_login_at is not None

        settings.OAUTH2_ADMIN_EMAILS = ["new@example.com"]
        promoted = await upsert_user(
            db_session,
            {"sub": "sso-1", "email": "new@example.com", "name": "New"},
            settings,
        )
        assert promoted.id == user.id
        assert promoted.role == ROLE_ADMIN

    @pytest.mark.asyncio
    async def test_upsert_requires_subject(self, db_session, settings):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            await upsert_user(db_session, {}, settings)

    def test_token_response_model(self):
        response = TokenResponse(
            access_token="abc",
            expires_in=1800,
            role="admin",
            team_id=None,
        )
        assert response.token_type == "bearer"
