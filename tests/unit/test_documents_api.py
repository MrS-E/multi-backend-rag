"""API-level tests for document upload, list and delete.

The document endpoints only *enqueue* work: upload stores the raw file and a
``documents`` row, delete enqueues a :class:`DeleteEvent`. The arq pool is
replaced by the in-memory :class:`FakeJobQueue` from ``conftest`` so no Redis is
needed; the vector/file cleanup boundary is exercised by draining the recorded
events through the ingestion pipeline with the in-memory stores.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.domain.events import DeleteEvent, SourceEvent, UpsertEvent
from app.persistence.file_storage import FileStorage
from app.persistence.models import Document, KnowledgeBase, RawFileCleanup
from app.rag.pipeline import IngestionPipeline
from app.services.document_service import DocumentService
from tests.conftest import FakeJobQueue, FakeQdrantStore

_MARKDOWN = b"# Title\n\nThe quick brown fox jumps over the lazy dog.\n"


class _FailingJobQueue(FakeJobQueue):
    async def enqueue_ingest(self, kb_id: object, events: list[SourceEvent]) -> str:
        raise ConnectionError("redis went away")


class _CancelledJobQueue(FakeJobQueue):
    async def enqueue_ingest(self, kb_id: object, events: list[SourceEvent]) -> str:
        raise asyncio.CancelledError()


class _BlockingCancelledJobQueue(FakeJobQueue):
    def __init__(self, started: asyncio.Event, continue_event: asyncio.Event) -> None:
        super().__init__()
        self.started = started
        self.continue_event = continue_event

    async def enqueue_ingest(self, kb_id: object, events: list[SourceEvent]) -> str:
        self.started.set()
        await self.continue_event.wait()
        raise asyncio.CancelledError()


class _CleanupFailStorage(FileStorage):
    def delete(self, rel_path: str) -> None:
        raise OSError("disk is unavailable")


@pytest.fixture
def job_queue() -> FakeJobQueue:
    """A job queue that records enqueued events instead of talking to Redis."""
    return FakeJobQueue()


@pytest.fixture
def client(app: FastAPI, job_queue: FakeJobQueue) -> Iterator[TestClient]:
    """The shared test client, wired with the in-memory job queue."""
    app.state.job_queue = job_queue
    with TestClient(app) as test_client:
        yield test_client


def _create_kb(client: TestClient) -> str:
    response = client.post("/kbs", json={"name": "Docs"})
    assert response.status_code == 201
    return response.json()["id"]


def _upload(
    client: TestClient,
    kb_id: str,
    filename: str = "notes.md",
    data: bytes = _MARKDOWN,
) -> Any:
    return client.post(
        f"/kbs/{kb_id}/documents",
        files={"file": (filename, data, "text/markdown")},
    )


def _drain(app: FastAPI, job_queue: FakeJobQueue) -> None:
    """Run every enqueued event through the pipeline, as the arq worker would."""
    session = app.state.session_factory()
    try:
        pipeline = IngestionPipeline(
            session=session,
            settings=app.state.settings,
            registry=app.state.provider_registry,
            qdrant=app.state.qdrant_store,
            storage=app.state.file_storage,
        )
        for kb_id, events in job_queue.enqueued:
            pipeline.process_events(kb_id, events)
    finally:
        session.close()


def test_given_unsupported_extension_when_uploading_then_rejected(
    client: TestClient,
) -> None:
    kb_id = _create_kb(client)

    response = _upload(client, kb_id, filename="malware.exe", data=b"MZ\x90\x00")

    assert response.status_code == 415
    assert 400 <= response.status_code < 500


@pytest.mark.asyncio
async def test_given_broker_error_and_cleanup_error_when_uploading_then_broker_error_is_preserved(
    session_factory: Any, tmp_path: Path
) -> None:
    session = session_factory()
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Docs"))
    session.commit()
    service = DocumentService(session, _CleanupFailStorage(tmp_path), _FailingJobQueue())

    with pytest.raises(ConnectionError, match="redis went away"):
        await service.upload(kb_id, "notes.md", _MARKDOWN, max_upload_bytes=1024 * 1024)

    assert session.scalar(select(Document).where(Document.kb_id == kb_id)) is None
    assert session.scalar(select(RawFileCleanup).where(RawFileCleanup.kb_id == kb_id)) is not None


@pytest.mark.asyncio
async def test_given_cancellation_and_cleanup_error_when_uploading_then_cancellation_is_preserved(
    session_factory: Any, tmp_path: Path
) -> None:
    session = session_factory()
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Docs"))
    session.commit()
    service = DocumentService(session, _CleanupFailStorage(tmp_path), _CancelledJobQueue())

    with pytest.raises(asyncio.CancelledError):
        await service.upload(kb_id, "notes.md", _MARKDOWN, max_upload_bytes=1024 * 1024)

    assert session.scalar(select(RawFileCleanup).where(RawFileCleanup.kb_id == kb_id)) is not None


@pytest.mark.asyncio
async def test_given_enqueue_cancellation_and_shared_raw_path_then_file_is_retained(
    session_factory: Any, tmp_path: Path
) -> None:
    session = session_factory()
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Docs"))
    session.commit()
    storage = FileStorage(tmp_path)
    existing = Document(
        kb_id=kb_id,
        uri="existing.md",
        content_hash="",
        mime="text/markdown",
        raw_path="",
        status="indexed",
    )
    raw_path, digest = storage.save(kb_id, _MARKDOWN)
    existing.content_hash = digest
    existing.raw_path = raw_path
    session.add(existing)
    session.commit()
    service = DocumentService(session, storage, _CancelledJobQueue())

    with pytest.raises(asyncio.CancelledError):
        await service.upload(kb_id, "new.md", _MARKDOWN, max_upload_bytes=1024 * 1024)

    assert storage.read(raw_path) == _MARKDOWN


@pytest.mark.asyncio
async def test_given_upload_in_progress_when_deleting_kb_then_deletion_waits_for_upload(
    session_factory: Any, tmp_path: Path
) -> None:
    from app.core.settings import Settings
    from app.providers.registry import ProviderBundle, ProviderRegistry
    from app.services.kb_service import KBService
    from tests.conftest import StubEmbedder, StubQdrantStore

    session = session_factory()
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Docs"))
    session.commit()
    started = asyncio.Event()
    continue_event = asyncio.Event()
    queue = _BlockingCancelledJobQueue(started, continue_event)
    storage = FileStorage(tmp_path)
    document_service = DocumentService(session, storage, queue)
    registry = ProviderRegistry(
        {
            "ollama": ProviderBundle(StubEmbedder(), None, None),
            "openai": ProviderBundle(StubEmbedder(), None, None),
        }
    )
    kb_service = KBService(
        session, Settings(), registry, StubQdrantStore(), storage, queue
    )
    upload_task = asyncio.create_task(
        document_service.upload(kb_id, "notes.md", _MARKDOWN, max_upload_bytes=1024 * 1024)
    )
    await started.wait()
    delete_task = asyncio.create_task(kb_service.delete_kb(kb_id))
    await asyncio.sleep(0)
    assert not delete_task.done()
    continue_event.set()
    with pytest.raises(asyncio.CancelledError):
        await upload_task
    await delete_task
    assert session.get(KnowledgeBase, kb_id) is None


def test_given_oversize_upload_when_uploading_then_413(
    client: TestClient, app: FastAPI
) -> None:
    app.state.settings.server.max_upload_mb = 0
    kb_id = _create_kb(client)

    response = _upload(client, kb_id, data=b"x" * 1024)

    assert response.status_code == 413


def test_given_over_limit_content_length_when_uploading_then_413(
    client: TestClient, app: FastAPI
) -> None:
    app.state.settings.server.max_upload_mb = 1
    kb_id = _create_kb(client)

    response = client.post(
        f"/kbs/{kb_id}/documents",
        files={"file": ("notes.md", _MARKDOWN, "text/markdown")},
        headers={"Content-Length": str(2 * 1024 * 1024)},
    )

    assert response.status_code == 413
    assert client.get(f"/kbs/{kb_id}/documents").json() == []


def test_given_enqueue_failure_when_uploading_then_pending_row_and_file_are_rolled_back(
    app: FastAPI,
) -> None:
    queue = _FailingJobQueue()
    app.state.job_queue = queue
    with TestClient(app) as client:
        kb_id = _create_kb(client)
        with pytest.raises(ConnectionError, match="redis went away"):
            _upload(client, kb_id)

    session = app.state.session_factory()
    try:
        assert session.scalars(select(Document)).all() == []
    finally:
        session.close()
    kb_storage_path = Path(app.state.file_storage._root) / kb_id
    assert not list(kb_storage_path.iterdir())


def test_given_valid_markdown_when_uploading_then_202_and_enqueued(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)

    response = _upload(client, kb_id)

    assert response.status_code == 202
    assert response.json()["job_id"] == job_queue.job_id

    enqueued_kb, events = job_queue.enqueued[0]
    assert enqueued_kb == uuid.UUID(kb_id)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, UpsertEvent)
    assert event.uri == "notes.md"
    assert event.doc_id == client.get(f"/kbs/{kb_id}/documents").json()[0]["id"]


def test_given_uploaded_document_when_listing_then_returned(client: TestClient) -> None:
    kb_id = _create_kb(client)
    _upload(client, kb_id)

    response = client.get(f"/kbs/{kb_id}/documents")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["uri"] == "notes.md"
    assert body[0]["mime"] == "text/markdown"
    assert body[0]["status"] == "pending"


def test_given_unknown_kb_when_uploading_then_404(client: TestClient) -> None:
    response = _upload(client, str(uuid.uuid4()))

    assert response.status_code == 404


def test_given_pdf_bytes_without_magic_when_uploading_then_422(
    client: TestClient,
) -> None:
    kb_id = _create_kb(client)

    response = _upload(client, kb_id, filename="report.pdf", data=b"not a pdf")

    assert response.status_code == 422


def test_given_document_when_deleting_then_202_and_removed_from_list(
    client: TestClient, app: FastAPI, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    _upload(client, kb_id)
    doc_id = client.get(f"/kbs/{kb_id}/documents").json()[0]["id"]

    session = app.state.session_factory()
    document = session.get(Document, uuid.UUID(doc_id))
    assert document is not None
    raw_path = document.raw_path
    session.close()

    qdrant = FakeQdrantStore()
    app.state.qdrant_store = qdrant

    response = client.delete(f"/kbs/{kb_id}/documents/{doc_id}")

    assert response.status_code == 202
    assert response.json()["job_id"] == job_queue.job_id
    _, events = job_queue.enqueued[-1]
    assert len(events) == 1
    assert isinstance(events[0], DeleteEvent)
    assert events[0].doc_id == doc_id

    _drain(app, job_queue)

    assert client.get(f"/kbs/{kb_id}/documents").json() == []
    assert (kb_id, doc_id) in qdrant.deletes
    with pytest.raises(FileNotFoundError):
        app.state.file_storage.read(raw_path)


def test_given_unknown_document_when_deleting_then_404(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.delete(f"/kbs/{kb_id}/documents/{uuid.uuid4()}")

    assert response.status_code == 404
