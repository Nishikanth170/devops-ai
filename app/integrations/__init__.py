"""Integration adapters registry.

``build_adapters`` wires the adapters to application settings; the ingestion
layer looks adapters up by their ``source`` key (the ``{source}`` path parameter
of ``POST /webhooks/{source}``).
"""

from app.core.config import Settings, get_settings
from app.integrations.base import (
    CaseInsensitiveHeaders,
    IntegrationAdapter,
    IntegrationEvent,
    normalize_headers,
)
from app.integrations.datadog import DatadogAdapter
from app.integrations.github import GitHubAdapter
from app.integrations.gitlab import GitLabAdapter
from app.integrations.jenkins import JenkinsAdapter
from app.integrations.jira import JiraAdapter

#: Registered adapter classes keyed by their webhook source name.
ADAPTER_CLASSES: dict[str, type[IntegrationAdapter]] = {
    "github": GitHubAdapter,
    "gitlab": GitLabAdapter,
    "jenkins": JenkinsAdapter,
    "datadog": DatadogAdapter,
    "jira": JiraAdapter,
}


def build_github_config(settings: Settings) -> dict:
    return {
        "webhook_secret": settings.GITHUB_WEBHOOK_SECRET,
        "base_url": settings.GITHUB_API_BASE_URL,
        "token": settings.GITHUB_TOKEN,
        "app_id": settings.GITHUB_APP_ID,
        "private_key": settings.GITHUB_APP_PRIVATE_KEY,
    }


def build_adapters(settings: Settings | None = None) -> dict[str, IntegrationAdapter]:
    """Instantiate every adapter with its configuration from settings."""
    settings = settings or get_settings()
    return {
        "github": GitHubAdapter(build_github_config(settings)),
        "gitlab": GitLabAdapter(
            {
                "webhook_secret": settings.GITLAB_WEBHOOK_SECRET,
                "base_url": settings.GITLAB_API_BASE_URL,
                "token": settings.GITLAB_TOKEN,
            }
        ),
        "jenkins": JenkinsAdapter(
            {
                "base_url": settings.JENKINS_BASE_URL,
                "user": settings.JENKINS_USER,
                "api_token": settings.JENKINS_API_TOKEN,
                "webhook_secret": settings.JENKINS_WEBHOOK_SECRET,
            }
        ),
        "datadog": DatadogAdapter(
            {
                "api_key": settings.DATADOG_API_KEY,
                "app_key": settings.DATADOG_APP_KEY,
                "pagerduty_integration_key": settings.PAGERDUTY_INTEGRATION_KEY,
                "webhook_secret": settings.DATADOG_WEBHOOK_SECRET,
            }
        ),
        "jira": JiraAdapter(
            {
                "base_url": settings.JIRA_BASE_URL,
                "email": settings.JIRA_EMAIL,
                "api_token": settings.JIRA_API_TOKEN,
                "webhook_secret": settings.JIRA_WEBHOOK_SECRET,
            }
        ),
    }


async def close_adapters(adapters: dict[str, IntegrationAdapter]) -> None:
    """Release HTTP clients held by the adapters."""
    for adapter in adapters.values():
        closer = getattr(adapter, "aclose", None)
        if closer is not None:
            await closer()


__all__ = [
    "ADAPTER_CLASSES",
    "CaseInsensitiveHeaders",
    "DatadogAdapter",
    "GitHubAdapter",
    "GitLabAdapter",
    "IntegrationAdapter",
    "IntegrationEvent",
    "JenkinsAdapter",
    "JiraAdapter",
    "build_adapters",
    "close_adapters",
    "normalize_headers",
]
