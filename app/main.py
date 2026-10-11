"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.errors import register_error_handlers
from app.api.routers import config, documents, health, jobs, kbs, query, sources
from app.core.broker import JobQueue, create_arq_pool
from app.core.logging import setup_logging
from app.core.security import build_identity_provider
from app.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info(
        "server starting",
        extra={"version": __version__, "host": settings.server.host, "port": settings.server.port},
    )
    existing_job_queue = getattr(app.state, "job_queue", None)
    existing_arq_pool = getattr(app.state, "arq_pool", None)
    app.state.arq_pool = existing_arq_pool
    app.state.job_queue = existing_job_queue
    if app.state.arq_pool is None:
        try:
            app.state.arq_pool = await create_arq_pool(settings.redis_url)
            app.state.job_queue = JobQueue(app.state.arq_pool)
        except Exception as exc:  # noqa: BLE001 - API remains usable without the broker
            logger.warning("job broker unavailable at startup", extra={"error": str(exc)})
    yield
    pool = getattr(app.state, "arq_pool", None)
    if pool is not None and hasattr(pool, "aclose"):
        try:
            await pool.aclose()
        except Exception as exc:  # noqa: BLE001 - shutdown must not mask app errors
            logger.warning("failed to close job broker", extra={"error": str(exc)})
        finally:
            app.state.arq_pool = None
    logger.info("server stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    setup_logging()

    app = FastAPI(title="Multi-BackendRag", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    # Fail fast at startup if auth is misconfigured (e.g. api_key provider with
    # an unset env var); the provider is then reused for every request.
    app.state.identity_provider = build_identity_provider(settings)

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(kbs.router)
    app.include_router(config.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(query.router)
    app.include_router(sources.router)
    app.include_router(sources.events_router)
    return app
