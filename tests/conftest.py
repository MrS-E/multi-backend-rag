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
from app.domain.events import SourceEvent
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


class FakeQdrantStore:
    """In-memory Qdrant stand-in that records points, upserts and deletes.

    Implements the same surface as :class:`app.persistence.qdrant_store.QdrantStore`
    for the ingestion pipeline: collection lifecycle, ``upsert`` and
    ``delete_by_doc_id``. Points are keyed by ``(kb_id, point_id)`` so an upsert
    of the same id replaces the previous vector and payload.
    """

    def __init__(self) -> None:
        self.collections: dict[str, int] = {}
        self.points: dict[tuple[str, str], dict[str, Any]] = {}
        self.upserts: list[tuple[str, list[str], list[list[float]], list[dict[str, Any]]]] = []
        self.deletes: list[tuple[str, str]] = []

    def create_collection(self, kb_id: Any, vector_size: int) -> None:
        self.collections[str(kb_id)] = vector_size

    def delete_collection(self, kb_id: Any) -> None:
        name = str(kb_id)
        self.collections.pop(name, None)
        for key in [key for key in self.points if key[0] == name]:
            del self.points[key]

    def collection_exists(self, kb_id: Any) -> bool:
        return str(kb_id) in self.collections

    def upsert(
        self,
        kb_id: Any,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.upserts.append((str(kb_id), list(ids), list(vectors), list(payloads)))
        for point_id, vector, payload in zip(ids, vectors, payloads, strict=True):
            self.points[(str(kb_id), point_id)] = {"vector": vector, "payload": payload}

    def delete_by_doc_id(self, kb_id: Any, doc_id: str) -> None:
        self.deletes.append((str(kb_id), str(doc_id)))
        for key in [
            key
            for key, point in self.points.items()
            if key[0] == str(kb_id) and point["payload"].get("doc_id") == str(doc_id)
        ]:
            del self.points[key]

    def payloads_for(self, kb_id: Any, doc_id: str) -> list[dict[str, Any]]:
        """Return the payloads currently stored for one document, in insert order."""
        return [
            point["payload"]
            for (name, _), point in self.points.items()
            if name == str(kb_id) and point["payload"].get("doc_id") == str(doc_id)
        ]


class FakeJobQueue:
    """Job-queue stand-in: records enqueued events and returns a deterministic id."""

    def __init__(self, job_id: str = "job-1") -> None:
        self.job_id = job_id
        self.enqueued: list[tuple[Any, list[SourceEvent]]] = []

    async def enqueue_ingest(self, kb_id: Any, events: list[SourceEvent]) -> str:
        self.enqueued.append((kb_id, list(events)))
        return self.job_id


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
