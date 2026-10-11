"""Regression tests for API-owned arq pool lifecycle."""

from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.core.settings import Settings


class _Pool:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FailingPool(_Pool):
    async def aclose(self) -> None:
        raise ConnectionError("redis went away")


def test_given_redis_pool_when_app_lifespan_runs_then_pool_is_closed(
    monkeypatch,
) -> None:
    pool = _Pool()

    async def create_pool(redis_url: str) -> _Pool:
        assert redis_url == "redis://configured:6379"
        return pool

    monkeypatch.setattr(main, "create_arq_pool", create_pool)
    app = main.create_app(Settings(redis_url="redis://configured:6379"))

    with TestClient(app):
        assert app.state.arq_pool is pool

    assert pool.closed is True


def test_given_pool_close_failure_when_app_lifespan_ends_then_shutdown_completes(
    monkeypatch,
) -> None:
    pool = _FailingPool()

    async def create_pool(redis_url: str) -> _Pool:
        return pool

    monkeypatch.setattr(main, "create_arq_pool", create_pool)
    app = main.create_app(Settings(redis_url="redis://configured:6379"))

    with TestClient(app):
        assert app.state.arq_pool is pool

    assert app.state.arq_pool is None