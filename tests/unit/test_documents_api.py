"""API-level tests for document upload, list and delete.

The document endpoints only *enqueue* work: upload stores the raw file and a
``documents`` row, delete enqueues a :class:`DeleteEvent`. The arq pool is
replaced by the in-memory :class:`FakeJobQueue` from ``conftest`` so no Redis is
needed; the vector/file cleanup boundary is exercised by draining the recorded
events through the ingestion pipeline with the in-memory stores.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from app.domain.events import DeleteEvent, UpsertEvent
from app.persistence.models import Document
from app.rag.pipeline import IngestionPipeline
from tests.conftest import FakeJobQueue, FakeQdrantStore

_MARKDOWN = b"# Title\n\nThe quick brown fox jumps over the lazy dog.\n"


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
) -> Response:
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


def test_given_oversize_upload_when_uploading_then_413(
    client: TestClient, app: FastAPI
) -> None:
    app.state.settings.server.max_upload_mb = 0
    kb_id = _create_kb(client)

    response = _upload(client, kb_id, data=b"x" * 1024)

    assert response.status_code == 413


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
