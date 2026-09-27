"""FastAPI dependency tests: Settings injection only (AppServices covered by API tests)."""

from __future__ import annotations

from app.api import deps as deps_module
from app.core.exceptions import PlatformError


class _AppWithState:
    def __init__(self, services=None):
        self.state = type("State", (), {"services": services})()


class _Request:
    def __init__(self, services=None):
        self.app = _AppWithState(services)


def test_get_settings_dep_returns_settings(settings):
    from app.core.config import get_settings

    assert deps_module.get_settings_dep() is get_settings()


def test_get_services_requires_container():
    import pytest

    with pytest.raises(PlatformError):
        deps_module.get_services(_Request(None))


def test_get_services_returns_container(settings):
    sentinel = object()
    assert deps_module.get_services(_Request(sentinel)) is sentinel
