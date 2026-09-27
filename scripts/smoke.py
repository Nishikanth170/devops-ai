"""End-to-end smoke test against a real uvicorn server.

Boots ``app.main:app`` on a free port with the offline configuration
(fake LLM, in-memory vector store, in-memory queue, SQLite file database),
then exercises health, auth, the incident lifecycle, an ad-hoc agent query, a
pipeline risk assessment and a signed GitHub webhook.

Usage::

    python scripts/smoke.py

Exit code 0 means every step passed.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_DB = Path(tempfile.gettempdir()) / "devops-agent-smoke.db"
GITHUB_SECRET = "smoke-github-secret"

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DEBUG": "false",
        "LLM_PROVIDER": "fake",
        "VECTOR_BACKEND": "memory",
        "QUEUE_BACKEND": "memory",
        "DATABASE_URL": f"sqlite+aiosqlite:///{_DB.as_posix()}",
        "DATABASE_CREATE_SCHEMA": "true",
        "VECTOR_DIMENSIONS": "128",
        "VECTOR_SIMILARITY_THRESHOLD": "0.2",
        "JWT_SECRET_KEY": "smoke-secret",
        "GITHUB_WEBHOOK_SECRET": GITHUB_SECRET,
        "DEFAULT_TEAM_NAME": "smoke-team",
    }
)

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.auth.tokens import create_access_token  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.database import get_session_factory  # noqa: E402
from app.memory.models import Team, User  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def wait_for_server(base_url: str, attempts: int = 80) -> None:
    for _ in range(attempts):
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
                if (await client.get("/health")).status_code == 200:
                    return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.25)
    raise RuntimeError("server did not become ready")


async def seed_admin() -> None:
    """Create the team + admin user through the app's own session factory."""
    async with get_session_factory()() as session:
        team = (
            await session.execute(select(Team).where(Team.name == "smoke-team"))
        ).scalar_one_or_none()
        if team is None:
            team = Team(name="smoke-team")
            session.add(team)
            await session.flush()
        user = (
            await session.execute(
                select(User).where(User.email == "smoke@example.com")
            )
        ).scalar_one_or_none()
        if user is None:
            session.add(
                User(
                    email="smoke@example.com",
                    name="Smoke Tester",
                    role="admin",
                    team_id=team.id,
                    external_id="sso-smoke",
                )
            )
        await session.commit()


async def main() -> int:
    settings = get_settings()
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"

    server = uvicorn.Server(
        uvicorn.Config(
            "app.main:app",
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="on",
        )
    )
    server_task = asyncio.create_task(server.serve())

    try:
        await wait_for_server(base_url)
        await seed_admin()

        token = create_access_token("sso-smoke", settings)
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            ready = await client.get("/health/ready")
            check("health/ready", ready.json().get("status") == "ready", str(ready.json()))

            me = await client.get("/auth/me", headers=headers)
            check("auth/me", me.status_code == 200, me.text[:60])

            unauth = await client.get("/incidents")
            check("auth enforced", unauth.status_code in {401, 403}, str(unauth.status_code))

            created = await client.post(
                "/incidents",
                json={
                    "title": "Connection pool saturated",
                    "description": "pool 98% full during checkout traffic spike",
                    "severity": "SEV2",
                    "service": "payments",
                    "run_agent": True,
                },
                headers=headers,
            )
            check("POST /incidents", created.status_code == 201, created.text[:100])
            incident_id = created.json().get("id") if created.status_code == 201 else None

            if incident_id:
                detail = await client.get(f"/incidents/{incident_id}", headers=headers)
                suggestions = len(detail.json().get("agent_suggestions", []))
                check(
                    "GET /incidents/{id} with agent suggestion",
                    detail.status_code == 200 and suggestions >= 1,
                    f"{suggestions} suggestion(s)",
                )

                resolved = await client.post(
                    f"/incidents/{incident_id}/resolve",
                    json={
                        "root_cause": "connection leak in retry logic",
                        "resolution_steps": "bounded pool size, jittered retry backoff",
                    },
                    headers=headers,
                )
                check(
                    "POST /incidents/{id}/resolve",
                    resolved.status_code == 200
                    and resolved.json()["status"] == "resolved",
                    resolved.text[:80],
                )

            query = await client.post(
                "/agent/query",
                json={"query": "why is the payments connection pool exhausted?"},
                headers=headers,
            )
            check(
                "POST /agent/query",
                query.status_code == 200,
                query.json().get("agent_type", query.text[:80]),
            )

            assess = await client.post(
                "/pipeline/assess",
                json={
                    "service": "payments",
                    "commit_sha": "abc123",
                    "changes": [
                        {
                            "change_type": "database_migration",
                            "description": "add index to orders",
                        }
                    ],
                },
                headers=headers,
            )
            risk = assess.json().get("risk_score") if assess.status_code == 200 else None
            check(
                "POST /pipeline/assess",
                assess.status_code == 200 and risk is not None,
                f"risk={risk} band={assess.json().get('risk_band')}",
            )

            body = json.dumps(
                {
                    "action": "opened",
                    "repository": {"full_name": "acme/payments"},
                    "pull_request": {
                        "number": 7,
                        "title": "Bound the retry loop",
                        "body": "",
                        "user": {"login": "dev"},
                        "diff_url": "",
                        "head": {"sha": "a1"},
                        "base": {"sha": "b2"},
                        "updated_at": "2024-05-01T10:00:00Z",
                    },
                }
            ).encode()
            signature = hmac.new(GITHUB_SECRET.encode(), body, hashlib.sha256).hexdigest()
            webhook = await client.post(
                "/webhooks/github",
                content=body,
                headers={
                    **headers,
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": f"sha256={signature}",
                },
            )
            check(
                "POST /webhooks/github (signed)",
                webhook.status_code == 200,
                f"published={webhook.json().get('events_published')}",
            )

            bad = await client.post(
                "/webhooks/github",
                content=b"{}",
                headers={**headers, "X-Hub-Signature-256": "sha256=invalid"},
            )
            check("webhook signature rejected", bad.status_code == 401, str(bad.status_code))

            stats = await client.get("/dashboard/agents", headers=headers)
            check(
                "GET /dashboard/agents",
                stats.status_code == 200,
                f"llm calls={stats.json().get('llm', {}).get('calls')}"
                if stats.status_code == 200
                else stats.text[:80],
            )
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            server_task.cancel()

    _DB.unlink(missing_ok=True)
    failed = [name for name, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
