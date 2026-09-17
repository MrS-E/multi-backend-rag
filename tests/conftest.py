"""Shared pytest fixtures: an app wired to in-memory SQLite and mocked backends."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.settings import Settings
from app.domain.models import Message
from app.main import create_app
from app.persistence.file_storage import FileStorage
from app.persistence.models import Base
from app.providers.registry import ProviderBundle, ProviderRegistry


class StubEmbedder:
    """Deterministic embedder with a fixed dimension, no network calls."""

    def __init__(self, dimension: int = 8, fail: bool = False) -> None:
        self.dimension = dimension
        self.fail = fail
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        self.calls.append((texts, model))
        if self.fail:
            raise RuntimeError("provider unreachable")
        return [[float(i)] * self.dimension for i, _ in enumerate(texts)]


class StubChat:
    def chat(self, messages: list[Message], model: str, **opts: object) -> str:
        return "stub answer"


class StubVision:
    def caption(self, image: bytes, prompt: str, model: str) -> str:
        return "stub caption"


class StubQdrantStore:
    """Records collection lifecycle calls instead of hitting a Qdrant server."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.created: list[tuple[str, int]] = []
        self.deleted: list[str] = []

    def create_collection(self, kb_id: Any, vector_size: int) -> None:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.created.append((str(kb_id), vector_size))

    def delete_collection(self, kb_id: Any) -> None:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.deleted.append(str(kb_id))

    def collection_exists(self, kb_id: Any) -> bool:
        return any(name == str(kb_id) for name, _ in self.created)


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    """An in-memory SQLite session factory with the schema created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    yield factory
    engine.dispose()


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def qdrant_store() -> StubQdrantStore:
    return StubQdrantStore()


@pytest.fixture
def app(
    session_factory: sessionmaker[Session],
    embedder: StubEmbedder,
    qdrant_store: StubQdrantStore,
    tmp_path: Path,
) -> FastAPI:
    """A FastAPI app with in-memory DB, mocked providers/Qdrant and temp storage."""
    settings = Settings()
    application = create_app(settings)
    application.state.session_factory = session_factory
    application.state.provider_registry = ProviderRegistry(
        {
            "ollama": ProviderBundle(embedder=embedder, chat=StubChat(), vision=StubVision()),
            "openai": ProviderBundle(embedder=embedder, chat=StubChat(), vision=StubVision()),
        }
    )
    application.state.qdrant_store = qdrant_store
    application.state.file_storage = FileStorage(tmp_path)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
