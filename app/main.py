"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routers import health
from app.core.logging import setup_logging
from app.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info(
        "server starting",
        extra={"version": __version__, "host": settings.server.host, "port": settings.server.port},
    )
    yield
    logger.info("server stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    setup_logging()

    app = FastAPI(title="Multi-BackendRag", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    app.include_router(health.router)
    return app
