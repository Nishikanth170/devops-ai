# DevOps Agent Platform

An AI-powered engineering and DevOps agent platform, built as a **modular monolith** in
Python/FastAPI. Three specialist agents (incident response, code review, pipeline risk) share
a **hybrid memory layer** (relational records + vector embeddings), are driven by inbound
**webhooks** through an **event queue**, and expose a **REST API** for humans and agents.

```
webhooks (GitHub/GitLab/Jenkins/Datadog/PagerDuty/Jira)
        │  verify signature → normalise → enqueue (priority lanes)
        ▼
   EventQueue ──► EventProcessors ──► AgentOrchestrator ──► agents
                                                          │
                                                          ▼
                                              MemoryService (hybrid retrieval)
                                              ├─ relational (PostgreSQL)
                                              └─ vector   (pgvector)
                                                          │
                                                          ▼
                                              AgentSuggestion + human feedback
```

---

## 1. Quick start (fully offline, no API key required)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

# Deterministic offline mode: fake LLM, in-memory vectors, in-memory queue, SQLite
# PowerShell:
$env:LLM_PROVIDER="fake"; $env:VECTOR_BACKEND="memory"; $env:QUEUE_BACKEND="memory"
$env:DATABASE_URL="sqlite+aiosqlite:///./dev.db"; uvicorn app.main:app --reload

# bash/zsh:
# LLM_PROVIDER=fake VECTOR_BACKEND=memory QUEUE_BACKEND=memory \
#   DATABASE_URL=sqlite+aiosqlite:///./dev.db uvicorn app.main:app --reload
```

* API docs: <http://localhost:8000/docs>
* Health: `GET /health`, `GET /health/ready`

Verify the whole stack in one command (boots a real uvicorn server and exercises
health, auth, the incident lifecycle, agent query, risk assessment and a signed webhook):

```bash
python scripts/smoke.py
```

### With the real stack (PostgreSQL + pgvector + Redis + OpenAI)

```bash
cp .env.example .env      # fill in secrets
docker compose up --build # postgres, redis, api, workers (runs alembic upgrade head)
```

---

## 2. Layout

```
app/
├── agents/          # base lifecycle + 3 specialist agents
├── memory/          # relational models, pgvector store, MemoryService
├── llm/             # gateway (retries/costs/structured output), prompts, fake provider
├── integrations/    # GitHub, GitLab, Jenkins, Datadog/PagerDuty, Jira adapters
├── events/          # schemas, queue, webhook ingestion, processors, worker
├── api/             # dependencies, Pydantic schemas, routers
├── auth/            # SSO/OIDC + JWT, RBAC
├── core/            # settings, database, logging, exceptions, utils
├── container.py     # AppServices (lifespan container)
├── orchestrator.py  # routing + agent lifecycle
└── main.py          # FastAPI app factory
alembic/             # migrations (bootstrap revision + env)
scripts/             # smoke.py (end-to-end), inspect_db.py
tests/               # 78 tests, fully offline
```

---

## 3. The agent contract

Every agent implements the same three steps (`app/agents/base.py`):

1. `retrieve_context` – semantic retrieval from the memory layer
2. `generate_response` – LLM call with the retrieved history, structured output
3. `record_outcome` – persist the suggestion so humans can rate it later

`BaseAgent.process()` runs them in order, so the orchestrator, the API and the
queue workers all treat agents identically.

| Agent | Retrieves | Produces | Writes back |
|---|---|---|---|
| `IncidentResponseAgent` | similar incidents + runbooks | root-cause hypothesis, runbook, immediate actions, escalation | incident status, `AgentSuggestion` |
| `CodeReviewAgent` | historical review patterns | inline findings (file/line/severity/category/confidence) | `ReviewComment`, `ReviewPattern` |
| `DevOpsPipelineAgent` | similar deployments | risk score + pipeline gates | `Deployment.risk_score`, `agent_notes` |

**Confidence** is derived, not invented: average retrieval similarity plus a small prior,
capped at 0.95. Low similarity triggers an explicit `escalate` follow-up action instead of a
confident-sounding answer.

**Risk score** blends learned history with intrinsic change-type risk:

```
risk = 0.6 * similarity_weighted_failure_rate + 0.4 * max(change_type_risk)
change_type_risk: database_migration 0.8 | infra 0.6 | dependency 0.5
                  config 0.4 | code 0.3 | feature_flag 0.2
```

---

## 4. Memory: hybrid, not just vectors

`MemoryService` (app/memory/service.py) pairs **semantic recall** with **structured lookup**:

```python
contexts = await memory.find_similar_incidents(
    "database connection pool exhausted", service="payments", top_k=5
)
# each RetrievedContext: similarity + full Incident + its Runbook
```

* **Relational layer** – incidents, runbooks, code reviews, review comments/patterns,
  deployments, pipeline changes, teams, users, agent suggestions.
* **Vector layer** – `vector_memory` table, `pgvector` cosine distance + HNSW index.
  Every row links back to its source record (`source_table` + `source_id` + metadata).
* **Feedback loop** – resolving an incident (`POST /incidents/{id}/resolve`) deletes the
  symptom-only embedding and re-embeds the resolved incident, so later alerts match on the
  *fix*. Every suggestion is stored with its historical matches and a rating
  (`helpful | not_helpful | incorrect`), surfaced in `/dashboard/agents`.

Two backends behind one interface (`VECTOR_BACKEND`):

| Backend | Use | Similarity |
|---|---|---|
| `pgvector` | production | `1 - cosine_distance` in SQL, HNSW-indexed |
| `memory` | dev/test, no pgvector needed | exact cosine in Python |

---

## 5. Event pipeline

1. `POST /webhooks/{source}` verifies the vendor signature (HMAC for GitHub, constant-time
   token compare for GitLab/Jenkins, shared secret for Datadog/Jira).
2. The adapter normalises the payload into `InternalEvent`; the owning **team** is resolved
   from `TEAM_ROUTING` (repo or service → team, created on first use).
3. The event is published to the priority or normal lane and acknowledged immediately –
   agents never block a vendor webhook.
4. Consumers (`EventProcessors`) write memory, run the matching agent and, where the agent
   recommends escalation, notify the monitoring integration.

Handler failures are recorded in a dead-letter list rather than being dropped.

Event sources: `pull_request.opened`, `pull_request.review_submitted`,
`deployment.started|completed|failed|rolled_back`, `build.started|completed|failed`,
`alert.fired|resolved`, `incident.triggered|acknowledged|resolved`, `ticket.*`.

---

## 6. HTTP API (32 paths)

| Area | Endpoints |
|---|---|
| System | `GET /health`, `GET /health/ready`, `GET /admin/llm-stats` (team-lead+) |
| Auth | `GET /auth/login`, `GET /auth/callback`, `POST /auth/token`, `GET /auth/me`, `POST /auth/dev-login`\* |
| Incidents | `GET/POST /incidents`, `GET /incidents/active`, `GET /incidents/{id}`, `GET /incidents/{id}/similar`, `POST /incidents/{id}/resolve`, `POST /incidents/{id}/feedback` |
| Reviews | `GET /reviews`, `GET /reviews/{id}`, `POST /reviews/{id}/analyze`, `POST /reviews/{id}/feedback` |
| Pipeline | `GET /pipeline/history`, `GET /pipeline/active`, `GET /pipeline/stats`, `GET /pipeline/deployments/{id}`, `POST /pipeline/assess`, `POST /pipeline/deployments/{id}/feedback` |
| Agents | `POST /agent/query`, `GET /agent/types`, `GET /agent/suggestions`, `POST /agent/suggestions/{id}/feedback` |
| Dashboard | `GET /dashboard/summary`, `GET /dashboard/agents` (team-lead+) |
| Webhooks | `GET /webhooks/sources`, `POST /webhooks/{source}` |

\* disabled when `ENVIRONMENT=production`.

**Access control**: three roles (`on-call-engineer`, `team-lead`, `admin`) and team scoping –
non-admins only ever see their own team's data.

---

## 7. Configuration

All settings are environment driven (see `.env.example` for the full list). Highlights:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | local PostgreSQL | async SQLAlchemy URL |
| `DATABASE_CREATE_SCHEMA` | `true` | set `false` when Alembic owns the schema |
| `VECTOR_BACKEND` | `pgvector` | or `memory` (dev/test) |
| `LLM_PROVIDER` | `openai` | or `fake` (offline, deterministic) |
| `QUEUE_BACKEND` | `redis` | or `memory` (dev/test, non-durable) |
| `EVENT_CONSUMER_IN_PROCESS` | `true` | set `false` when running dedicated workers |
| `TEAM_ROUTING` | `{}` | `{"acme/payments": "payments"}` |
| `JWT_SECRET_KEY` | dev default | production startup fails if left at the default |

`Settings.assert_production_ready()` refuses to start production with placeholder secrets,
`DEBUG=true`, or a non-durable queue.

---

## 8. Testing

```bash
pytest -q                  # 78 tests, no network, no docker
ruff check app tests scripts
python scripts/smoke.py    # end-to-end against a real server
```

The suite runs against a throwaway SQLite database with the offline gateway and queue, and
covers: agent behaviour (retrieval, confidence, escalation, risk scoring), hybrid memory
(record/retrieve, feedback, stats), integration adapters (signature verification, webhook
parsing for all five vendors), the event queue (priority lanes, dead letters), team routing,
processor flows (alert → incident, PR → findings, deployment lifecycle), auth/RBAC, and the
HTTP API including a signed webhook round-trip.

---

## 9. Notes on this implementation

Deliberate improvements over a literal reading of the original sketch:

* **No process-wide session.** The sketch built a `MemoryService` once at startup and reused
  it forever. Here `AppServices` holds only stateless collaborators, and the
  orchestrator/memory service are built per request (`app/api/deps.py`) or per consumed event
  (`EventProcessors.orchestrator`).
* **`VectorMemory.meta`, not `.metadata`.** `metadata` is reserved by SQLAlchemy's
  declarative API; the column is still named `metadata`
  (`meta: Mapped[dict] = mapped_column("metadata", ...)`).
* **No lazy relationship IO.** Collections such as `deployment.changes` and `review.patterns`
  are queried explicitly; touching them inside async code raises `MissingGreenlet`.
* **Real team resolution.** `"default-team-id"` is not a UUID: teams are resolved from
  `TEAM_ROUTING` and created on demand.
* **Enum-safe queries.** Aggregate filters bind enum *members* (not `.value` strings) so the
  same SQL works against PostgreSQL enums and SQLite.
* **Ports for the offline story.** `LLM_PROVIDER=fake`, `VECTOR_BACKEND=memory` and
  `QUEUE_BACKEND=memory` exist so the platform is runnable, demo-able and CI-testable without
  an OpenAI key, pgvector or Redis.
* **PostgreSQL `UUID` → generic `Uuid`.** Keeps the schema portable for the test database.
* Extended coverage beyond the sketch: GitLab/Jenkins/Jira adapters, SSO login/callback,
  dashboards, pre-flight risk endpoint, idempotent PR re-analysis, dev-login, smoke script.
