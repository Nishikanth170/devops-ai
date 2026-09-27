"""API routers.

``ROUTERS`` is the single list ``app.main`` iterates over, so adding a route
module means exactly one line change here and nothing else.
"""

from app.api.routes import agent_query, dashboard, incidents, pipeline, reviews, webhooks

ROUTERS = [
    incidents.router,
    reviews.router,
    pipeline.router,
    agent_query.router,
    dashboard.router,
    webhooks.router,
]

__all__ = ["ROUTERS", "agent_query", "dashboard", "incidents", "pipeline", "reviews", "webhooks"]
