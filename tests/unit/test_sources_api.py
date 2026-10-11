"""API-level tests for source registration, event push and manual sync.

The endpoints only *enqueue* work through the broker, so the shared app is wired
with the in-memory :class:`FakeJobQueue` from ``conftest`` (no Redis). The
connector registry is the process-wide default, so ``local_folder`` is the one
registered pull type; event-mode sources accept any type name.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.schemas.sources import SourceEventIn
from app.domain.events import DeleteEvent, UpsertEvent
from app.persistence.models import Document
from tests.conftest import FakeJobQueue


class _UnavailableRedis:
    def lock(self, *_: object, **__: object) -> object:
        raise ConnectionError("redis unavailable")


@pytest.fixture
def job_queue() -> FakeJobQueue:
    """A job queue that records enqueued jobs instead of talking to Redis."""
    return FakeJobQueue()


@pytest.fixture
def client(app: FastAPI, job_queue: FakeJobQueue) -> Iterator[TestClient]:
    """The shared test client, wired with the in-memory job queue."""
    app.state.job_queue = job_queue
    with TestClient(app) as test_client:
        yield test_client


def _create_kb(client: TestClient) -> str:
    response = client.post("/kbs", json={"name": "Sources"})
    assert response.status_code == 201
    return response.json()["id"]


def _create_source(
    client: TestClient,
    kb_id: str,
    type_name: str = "local_folder",
    mode: str = "pull",
    **extra: object,
) -> dict[str, Any]:
    response = client.post(
        f"/kbs/{kb_id}/sources", json={"type": type_name, "mode": mode, **extra}
    )
    assert response.status_code == 201
    return response.json()


def test_given_configured_upload_limit_when_validating_event_then_schema_has_no_static_cap(
) -> None:
    """The runtime settings, not import-time defaults, own the data size limit."""
    assert "maxLength" not in SourceEventIn.model_json_schema()["properties"]["data_b64"]


# --- create ------------------------------------------------------------------


def test_given_registered_pull_type_when_creating_source_then_201_with_defaults(
    client: TestClient,
) -> None:
    kb_id = _create_kb(client)

    body = _create_source(client, kb_id, config={"path": "/tmp"})

    assert body["kb_id"] == kb_id
    assert body["type"] == "local_folder"
    assert body["mode"] == "pull"
    assert body["config"] == {"path": "/tmp"}
    assert body["sync_interval_s"] == 60
    assert body["enabled"] is True
    assert body["cursor"] is None
    assert body["last_sync_at"] is None


def test_given_unknown_pull_type_when_creating_source_then_422(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.post(f"/kbs/{kb_id}/sources", json={"type": "git", "mode": "pull"})

    assert response.status_code == 422
    assert "unknown source type" in response.json()["detail"]


def test_given_arbitrary_type_when_creating_event_source_then_201(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.post(f"/kbs/{kb_id}/sources", json={"type": "webhook", "mode": "event"})

    assert response.status_code == 201
    assert response.json()["mode"] == "event"


def test_given_unknown_kb_when_creating_source_then_404(client: TestClient) -> None:
    response = client.post(
        f"/kbs/{uuid.uuid4()}/sources", json={"type": "local_folder", "mode": "pull"}
    )

    assert response.status_code == 404


def test_given_redis_acquisition_failure_when_creating_source_then_503(
    client: TestClient, app: FastAPI
) -> None:
    kb_id = _create_kb(client)
    app.state.arq_pool = _UnavailableRedis()

    response = client.post(
        f"/kbs/{kb_id}/sources", json={"type": "local_folder", "mode": "pull"}
    )

    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "local_folder"},
        {"mode": "pull"},
        {"type": "local_folder", "mode": "stream"},
    ],
)
def test_given_invalid_create_payload_when_creating_source_then_422(
    client: TestClient, payload: dict[str, object]
) -> None:
    kb_id = _create_kb(client)

    response = client.post(f"/kbs/{kb_id}/sources", json=payload)

    assert response.status_code == 422


def test_given_negative_interval_when_creating_source_then_422(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.post(
        f"/kbs/{kb_id}/sources",
        json={"type": "local_folder", "mode": "pull", "sync_interval_s": -1},
    )

    assert response.status_code == 422


# --- list / get --------------------------------------------------------------


def test_given_registered_sources_when_listing_then_only_that_kbs_returned(
    client: TestClient,
) -> None:
    kb_id = _create_kb(client)
    other_kb = _create_kb(client)
    _create_source(client, kb_id)
    _create_source(client, kb_id, mode="event", type_name="webhook")
    _create_source(client, other_kb)

    response = client.get(f"/kbs/{kb_id}/sources")

    assert response.status_code == 200
    assert {source["mode"] for source in response.json()} == {"pull", "event"}


def test_given_unknown_kb_when_listing_sources_then_404(client: TestClient) -> None:
    response = client.get(f"/kbs/{uuid.uuid4()}/sources")

    assert response.status_code == 404


def test_given_source_when_getting_then_200(client: TestClient) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id)["id"]

    response = client.get(f"/kbs/{kb_id}/sources/{source_id}")

    assert response.status_code == 200
    assert response.json()["id"] == source_id


def test_given_other_kbs_source_when_getting_then_404(client: TestClient) -> None:
    kb_id = _create_kb(client)
    other_kb = _create_kb(client)
    source_id = _create_source(client, other_kb)["id"]

    response = client.get(f"/kbs/{kb_id}/sources/{source_id}")

    assert response.status_code == 404


# --- update ------------------------------------------------------------------


def test_given_pull_source_when_patching_interval_and_enabled_then_updated(
    client: TestClient,
) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id)["id"]

    response = client.patch(
        f"/kbs/{kb_id}/sources/{source_id}",
        json={"sync_interval_s": 120, "enabled": False, "config": {"path": "/data"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sync_interval_s"] == 120
    assert body["enabled"] is False
    assert body["config"] == {"path": "/data"}


def test_given_source_when_patching_to_unknown_pull_type_then_422(client: TestClient) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id)["id"]

    response = client.patch(f"/kbs/{kb_id}/sources/{source_id}", json={"type": "git"})

    assert response.status_code == 422


def test_given_event_source_when_patching_type_then_accepted(client: TestClient) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id, type_name="webhook", mode="event")["id"]

    response = client.patch(f"/kbs/{kb_id}/sources/{source_id}", json={"type": "hook"})

    assert response.status_code == 200
    assert response.json()["type"] == "hook"


def test_given_unknown_source_when_patching_then_404(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.patch(
        f"/kbs/{kb_id}/sources/{uuid.uuid4()}", json={"sync_interval_s": 10}
    )

    assert response.status_code == 404


# --- delete ------------------------------------------------------------------


def test_given_source_when_deleting_then_204_and_gone(client: TestClient) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id)["id"]

    response = client.delete(f"/kbs/{kb_id}/sources/{source_id}")

    assert response.status_code == 204
    assert client.get(f"/kbs/{kb_id}/sources/{source_id}").status_code == 404
    assert client.get(f"/kbs/{kb_id}/sources").json() == []


def test_given_unknown_source_when_deleting_then_404(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.delete(f"/kbs/{kb_id}/sources/{uuid.uuid4()}")

    assert response.status_code == 404


# --- manual sync -------------------------------------------------------------


def test_given_pull_source_when_syncing_then_202_and_enqueued(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id)["id"]

    response = client.post(f"/kbs/{kb_id}/sources/{source_id}/sync")

    assert response.status_code == 202
    assert response.json()["job_id"] == job_queue.job_id
    assert job_queue.synced == [uuid.UUID(str(source_id))]


def test_given_event_source_when_syncing_then_422(client: TestClient) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id, type_name="webhook", mode="event")["id"]

    response = client.post(f"/kbs/{kb_id}/sources/{source_id}/sync")

    assert response.status_code == 422
    assert "pull mode" in response.json()["detail"]


def test_given_unknown_source_when_syncing_then_404(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.post(f"/kbs/{kb_id}/sources/{uuid.uuid4()}/sync")

    assert response.status_code == 404


def test_given_no_broker_when_syncing_then_503(client: TestClient, app: FastAPI) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id)["id"]
    app.state.job_queue = None
    app.state.arq_pool = None

    response = client.post(f"/kbs/{kb_id}/sources/{source_id}/sync")

    assert response.status_code == 503


# --- event push --------------------------------------------------------------


def test_given_single_upsert_when_pushing_event_then_202_and_upsert_enqueued(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    doc_id = str(uuid.uuid4())

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "op": "upsert",
            "doc_id": doc_id,
            "uri": "notes.md",
            "mime": "text/markdown",
            "data_b64": base64.b64encode(b"# notes").decode(),
        },
    )

    assert response.status_code == 202
    assert response.json()["job_id"] == job_queue.job_id
    enqueued_kb, events = job_queue.enqueued[0]
    assert enqueued_kb == uuid.UUID(kb_id)
    assert len(events) == 1
    assert isinstance(events[0], UpsertEvent)
    assert events[0].doc_id == doc_id
    assert events[0].uri == "notes.md"


def test_given_batch_for_event_source_when_pushing_then_source_id_in_metadata(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id, type_name="webhook", mode="event")["id"]

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "source_id": source_id,
            "events": [
                {
                    "op": "upsert",
                    "doc_id": str(uuid.uuid4()),
                    "uri": "a.md",
                    "data_b64": base64.b64encode(b"# a").decode(),
                }
            ],
        },
    )

    assert response.status_code == 202
    _, events = job_queue.enqueued[0]
    assert isinstance(events[0], UpsertEvent)
    assert events[0].metadata == {"source_id": source_id}


def test_given_explicit_source_id_metadata_when_pushing_then_validated_source_is_authoritative(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    other_kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id, type_name="webhook", mode="event")["id"]
    other_source_id = _create_source(
        client, other_kb_id, type_name="webhook", mode="event"
    )["id"]

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "source_id": source_id,
            "events": [
                {
                    "op": "upsert",
                    "doc_id": str(uuid.uuid4()),
                    "uri": "a.md",
                    "data_b64": base64.b64encode(b"# a").decode(),
                    "metadata": {"source_id": other_source_id, "tag": "x"},
                }
            ],
        },
    )

    assert response.status_code == 202
    _, events = job_queue.enqueued[0]
    assert isinstance(events[0], UpsertEvent)
    assert events[0].metadata == {"source_id": source_id, "tag": "x"}


def test_given_pull_source_when_pushing_events_then_422(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id)["id"]

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "source_id": source_id,
            "events": [{"op": "upsert", "doc_id": str(uuid.uuid4()), "uri": "a.md"}],
        },
    )

    assert response.status_code == 422
    assert "event mode" in response.json()["detail"]
    assert job_queue.enqueued == []


def test_given_unknown_source_when_pushing_events_then_404(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "source_id": str(uuid.uuid4()),
            "events": [{"op": "upsert", "doc_id": str(uuid.uuid4()), "uri": "a.md"}],
        },
    )

    assert response.status_code == 404


def test_given_source_in_other_kb_when_pushing_events_then_404(client: TestClient) -> None:
    kb_id = _create_kb(client)
    other_kb = _create_kb(client)
    source_id = _create_source(client, other_kb, type_name="webhook", mode="event")["id"]

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "source_id": source_id,
            "events": [{"op": "upsert", "doc_id": str(uuid.uuid4()), "uri": "a.md"}],
        },
    )

    assert response.status_code == 404


def test_given_empty_batch_when_pushing_events_then_422(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.post(f"/kbs/{kb_id}/events", json={"events": []})

    assert response.status_code == 422


def test_given_delete_op_when_pushing_event_then_delete_enqueued(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    doc_id = str(uuid.uuid4())

    response = client.post(
        f"/kbs/{kb_id}/events", json={"op": "delete", "doc_id": doc_id, "uri": "gone.md"}
    )

    assert response.status_code == 202
    _, events = job_queue.enqueued[0]
    assert events == [DeleteEvent(doc_id=doc_id, uri="gone.md")]


def test_given_base64_data_when_pushing_event_then_bytes_decoded(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    payload = b"# Title\n\nbody\n"

    client.post(
        f"/kbs/{kb_id}/events",
        json={
            "op": "upsert",
            "doc_id": str(uuid.uuid4()),
            "uri": "notes.md",
            "data_b64": base64.b64encode(payload).decode("ascii"),
        },
    )

    _, events = job_queue.enqueued[0]
    assert isinstance(events[0], UpsertEvent)
    assert events[0].data == payload


def test_given_non_uuid_doc_id_when_pushing_event_then_422_before_enqueue(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "op": "upsert",
            "doc_id": "external-42",
            "data_b64": base64.b64encode(b"x").decode(),
        },
    )

    assert response.status_code == 422
    assert job_queue.enqueued == []


def test_given_server_path_when_pushing_public_upsert_then_422(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    kb_id = _create_kb(client)
    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "op": "upsert",
            "doc_id": str(uuid.uuid4()),
            "uri": "notes.md",
            "path": "other-kb/secret",
        },
    )

    assert response.status_code == 422
    assert job_queue.enqueued == []


def test_given_non_ascii_base64_when_pushing_event_then_422(client: TestClient) -> None:
    kb_id = _create_kb(client)
    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "op": "upsert",
            "doc_id": str(uuid.uuid4()),
            "uri": "notes.md",
            "data_b64": "é",
        },
    )

    assert response.status_code == 422


def test_given_existing_document_from_another_kb_when_pushing_then_404(
    client: TestClient, app: FastAPI, job_queue: FakeJobQueue
) -> None:
    first_kb = _create_kb(client)
    second_kb = _create_kb(client)
    session = app.state.session_factory()
    document_id = uuid.uuid4()
    session.add(
        Document(
            id=document_id,
            kb_id=uuid.UUID(first_kb),
            uri="notes.md",
            content_hash="a" * 64,
            mime="text/markdown",
            raw_path=f"{first_kb}/{'a' * 64}",
            status="indexed",
        )
    )
    session.commit()
    session.close()

    response = client.post(
        f"/kbs/{second_kb}/events",
        json={
            "op": "delete",
            "doc_id": str(document_id),
        },
    )

    assert response.status_code == 404
    assert job_queue.enqueued == []


def test_given_unknown_kb_when_pushing_event_then_404(client: TestClient) -> None:
    response = client.post(
        f"/kbs/{uuid.uuid4()}/events",
        json={"op": "upsert", "doc_id": str(uuid.uuid4()), "uri": "a.md"},
    )

    assert response.status_code == 404


def test_given_missing_op_when_pushing_event_then_422(client: TestClient) -> None:
    kb_id = _create_kb(client)

    response = client.post(f"/kbs/{kb_id}/events", json={"doc_id": str(uuid.uuid4())})

    assert response.status_code == 422


def test_given_malformed_base64_when_pushing_event_then_422(
    app: FastAPI, job_queue: FakeJobQueue
) -> None:
    """A malformed ``data_b64`` is client input and must be rejected with ``422``."""
    app.state.job_queue = job_queue
    with TestClient(app, raise_server_exceptions=False) as raw_client:
        kb_id = _create_kb(raw_client)
        response = raw_client.post(
            f"/kbs/{kb_id}/events",
            json={"op": "upsert", "doc_id": str(uuid.uuid4()), "uri": "a.md", "data_b64": "not!"},
        )

    assert response.status_code == 422


def test_given_event_batch_over_limit_when_pushing_then_413(
    client: TestClient, app: FastAPI, job_queue: FakeJobQueue
) -> None:
    app.state.settings.server.max_event_batch = 1
    kb_id = _create_kb(client)
    source_id = _create_source(client, kb_id, type_name="webhook", mode="event")["id"]
    events = [
        {"op": "delete", "doc_id": str(uuid.uuid4())},
        {"op": "delete", "doc_id": str(uuid.uuid4())},
    ]

    response = client.post(
        f"/kbs/{kb_id}/events", json={"source_id": source_id, "events": events}
    )

    assert response.status_code == 413
    assert job_queue.enqueued == []


def test_given_inline_event_data_over_limit_when_pushing_then_413(
    client: TestClient, app: FastAPI, job_queue: FakeJobQueue
) -> None:
    app.state.settings.server.max_upload_mb = 0
    kb_id = _create_kb(client)

    response = client.post(
        f"/kbs/{kb_id}/events",
        json={
            "op": "upsert",
            "doc_id": str(uuid.uuid4()),
            "uri": "notes.md",
            "data_b64": base64.b64encode(b"x").decode("ascii"),
        },
    )

    assert response.status_code == 413
    assert job_queue.enqueued == []
