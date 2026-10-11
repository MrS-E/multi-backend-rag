"""API-level tests for KB CRUD, effective config and collection lifecycle."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.coordination import LeaseLostError, LockLease
from app.domain.errors import ConflictError
from app.persistence.models import KnowledgeBase
from app.services.kb_service import KBService
from tests.conftest import FakeJobQueue, StubEmbedder, StubQdrantStore


def test_given_no_kbs_when_listing_then_empty_list(client: TestClient) -> None:
    response = client.get("/kbs")

    assert response.status_code == 200
    assert response.json() == []


def test_given_valid_payload_when_creating_kb_then_201_and_collection_created(
    client: TestClient, qdrant_store: StubQdrantStore
) -> None:
    response = client.post("/kbs", json={"name": "Docs", "description": "test kb"})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Docs"
    assert body["owner_id"] is None
    assert body["status"] == "active"
    assert qdrant_store.created == [(body["id"], 8)]


def test_given_overrides_when_creating_kb_then_stored_and_probed(
    client: TestClient, embedder: StubEmbedder
) -> None:
    response = client.post(
        "/kbs",
        json={"name": "Docs", "overrides": {"embed_model": "openai/text-embedding-3-large"}},
    )

    assert response.status_code == 201
    assert response.json()["overrides"] == {"embed_model": "openai/text-embedding-3-large"}
    assert embedder.calls[0][1] == "text-embedding-3-large"


def test_given_explicit_nulls_in_overrides_when_creating_then_cleared(
    client: TestClient,
) -> None:
    response = client.post(
        "/kbs",
        json={"name": "Docs", "overrides": {"chat_model": None, "top_k": 3}},
    )

    assert response.status_code == 201
    assert response.json()["overrides"] == {"top_k": 3}


def test_given_provider_unreachable_when_creating_kb_then_row_still_created(
    client: TestClient, embedder: StubEmbedder
) -> None:
    embedder.fail = True

    response = client.post("/kbs", json={"name": "Docs"})

    assert response.status_code == 201
    assert response.json()["status"] == "collection_pending"


def test_given_qdrant_unreachable_when_creating_kb_then_row_still_created(
    client: TestClient, qdrant_store: StubQdrantStore
) -> None:
    qdrant_store.fail = True

    response = client.post("/kbs", json={"name": "Docs"})

    assert response.status_code == 201
    assert response.json()["status"] == "collection_pending"


def test_given_created_kb_when_getting_then_returns_it(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.get(f"/kbs/{kb_id}")

    assert response.status_code == 200
    assert response.json()["id"] == kb_id


def test_given_unknown_kb_when_getting_then_404(client: TestClient) -> None:
    response = client.get(f"/kbs/{uuid.uuid4()}")

    assert response.status_code == 404


def test_given_missing_name_when_creating_then_422(client: TestClient) -> None:
    response = client.post("/kbs", json={})

    assert response.status_code == 422


def test_given_created_kbs_when_listing_then_all_returned(client: TestClient) -> None:
    client.post("/kbs", json={"name": "One"})
    client.post("/kbs", json={"name": "Two"})

    response = client.get("/kbs")

    assert response.status_code == 200
    assert {kb["name"] for kb in response.json()} == {"One", "Two"}


def test_given_kb_when_patching_name_then_updated(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Old"}).json()["id"]

    response = client.patch(f"/kbs/{kb_id}", json={"name": "New"})

    assert response.status_code == 200
    assert response.json()["name"] == "New"


def test_given_kb_when_patching_overrides_then_merged(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs", "overrides": {"top_k": 3}}).json()["id"]

    response = client.patch(f"/kbs/{kb_id}", json={"overrides": {"chat_model": "openai/gpt-4o"}})

    assert response.status_code == 200
    assert response.json()["overrides"] == {"top_k": 3, "chat_model": "openai/gpt-4o"}


def test_given_kb_when_patching_null_override_then_cleared(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs", "overrides": {"top_k": 3}}).json()["id"]

    response = client.patch(f"/kbs/{kb_id}", json={"overrides": {"top_k": None}})

    assert response.status_code == 200
    assert response.json()["overrides"] == {}


def test_given_unknown_provider_prefix_when_patching_then_422(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.patch(f"/kbs/{kb_id}", json={"overrides": {"embed_model": "bogus"}})

    assert response.status_code == 422


def test_given_unprefixed_model_when_patching_then_422(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.patch(f"/kbs/{kb_id}", json={"overrides": {"chat_model": "gpt-4o"}})

    assert response.status_code == 422


def test_given_invalid_vision_model_when_patching_then_422(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.patch(f"/kbs/{kb_id}", json={"overrides": {"vision_model": "anthropic/x"}})

    assert response.status_code == 422


def test_given_valid_model_overrides_when_patching_then_200(
    client: TestClient, app: FastAPI
) -> None:
    app.state.job_queue = FakeJobQueue()
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.patch(
        f"/kbs/{kb_id}",
        json={
            "overrides": {
                "embed_model": "openai/text-embedding-3-large",
                "chat_model": "ollama/llama3.2",
                "vision_model": "ollama/llama3.2-vision",
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["overrides"]["chat_model"] == "ollama/llama3.2"


def test_given_reindex_enqueue_failure_when_patching_then_state_is_not_committed(
    client: TestClient, app: FastAPI
) -> None:
    class _FailingQueue(FakeJobQueue):
        async def enqueue_reindex(self, kb_id: object) -> str:
            raise ConnectionError("redis went away")

    app.state.job_queue = _FailingQueue()
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.patch(
        f"/kbs/{kb_id}",
        json={"overrides": {"embed_model": "openai/text-embedding-3-large"}},
    )

    assert response.status_code == 503
    session = app.state.session_factory()
    try:
        current = session.get(KnowledgeBase, uuid.UUID(kb_id))
        assert current is not None
        assert current.config_version == 0
        assert current.status == "active"
    finally:
        session.close()


@pytest.mark.asyncio
async def test_given_lease_loss_after_reindex_enqueue_then_newer_state_is_not_overwritten(
    app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]
    queue = FakeJobQueue()
    session = app.state.session_factory()
    calls = 0

    def ensure_valid(_lease: LockLease) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            newer_session = app.state.session_factory()
            newer = newer_session.get(KnowledgeBase, uuid.UUID(kb_id))
            assert newer is not None
            newer.status = "reindexing"
            newer.reindex_started_at = datetime.now(UTC)
            newer_session.commit()
            newer_session.close()
            raise LeaseLostError("lease expired after enqueue")

    monkeypatch.setattr(LockLease, "ensure_valid", ensure_valid)
    service = KBService(
        session,
        app.state.settings,
        app.state.provider_registry,
        app.state.qdrant_store,
        app.state.file_storage,
        cast(Any, queue),
    )

    with pytest.raises(ConflictError, match="already running"):
        await service.retry_reindex(uuid.UUID(kb_id))

    check = app.state.session_factory()
    try:
        current = check.get(KnowledgeBase, uuid.UUID(kb_id))
        assert current is not None
        assert current.status == "reindexing"
        assert queue.reindexed == [uuid.UUID(kb_id)]
    finally:
        check.close()
        session.close()


def test_given_invalid_model_override_when_creating_then_422(client: TestClient) -> None:
    response = client.post("/kbs", json={"name": "Docs", "overrides": {"embed_model": "bogus"}})

    assert response.status_code == 422


def test_given_invalid_model_override_when_patching_then_status_unchanged(
    client: TestClient,
) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    client.patch(f"/kbs/{kb_id}", json={"overrides": {"embed_model": "bogus"}})

    assert client.get(f"/kbs/{kb_id}").json()["status"] == "active"


def test_given_kb_when_changing_embed_model_override_then_reindex_required(
    client: TestClient, app: FastAPI,
) -> None:
    queue = FakeJobQueue()
    app.state.job_queue = queue
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.patch(
        f"/kbs/{kb_id}", json={"overrides": {"embed_model": "openai/text-embedding-3-large"}}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "reindex_required"
    assert queue.reindexed == [uuid.UUID(kb_id)]


def test_given_kb_when_changing_only_chat_model_then_status_unchanged(client: TestClient) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.patch(f"/kbs/{kb_id}", json={"overrides": {"chat_model": "openai/gpt-4o"}})

    assert response.status_code == 200
    assert response.json()["status"] == "active"


@pytest.mark.asyncio
async def test_given_kb_update_when_using_local_coordination_then_orm_runs_off_event_loop(
    app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]
    session = app.state.session_factory()
    commit_threads: list[int] = []
    original_commit = session.commit

    def record_commit() -> None:
        commit_threads.append(threading.get_ident())
        original_commit()

    monkeypatch.setattr(session, "commit", record_commit)
    service = KBService(
        session,
        app.state.settings,
        app.state.provider_registry,
        app.state.qdrant_store,
        app.state.file_storage,
        FakeJobQueue(),
        redis=None,
    )
    event_loop_thread = threading.get_ident()
    try:
        updated = await service.update_kb(uuid.UUID(kb_id), "Renamed", None, None)
    finally:
        session.close()

    assert updated.name == "Renamed"
    assert commit_threads
    assert all(thread_id != event_loop_thread for thread_id in commit_threads)


def test_given_kb_when_deleting_then_204_and_collection_removed(
    client: TestClient, qdrant_store: StubQdrantStore
) -> None:
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.delete(f"/kbs/{kb_id}")

    assert response.status_code == 204
    assert qdrant_store.deleted == [kb_id]
    assert client.get(f"/kbs/{kb_id}").status_code == 404


def test_given_unknown_kb_when_deleting_then_404(client: TestClient) -> None:
    response = client.delete(f"/kbs/{uuid.uuid4()}")

    assert response.status_code == 404


def test_given_qdrant_unavailable_when_deleting_then_kb_is_retained(
    client: TestClient, qdrant_store: StubQdrantStore
) -> None:
    qdrant_store.fail = True
    kb_id = client.post("/kbs", json={"name": "Docs"}).json()["id"]

    response = client.delete(f"/kbs/{kb_id}")

    assert response.status_code == 503
    assert client.get(f"/kbs/{kb_id}").status_code == 200


def test_given_kb_with_override_when_getting_effective_config_then_override_wins(
    client: TestClient,
) -> None:
    kb_id = client.post(
        "/kbs", json={"name": "Docs", "overrides": {"chat_model": "openai/gpt-4o"}}
    ).json()["id"]

    response = client.get("/config/effective", params={"kb_id": kb_id})

    assert response.status_code == 200
    body = response.json()
    assert body["chat_model"] == "openai/gpt-4o"
    assert body["embed_model"] == "ollama/bge-m3"


def test_given_no_kb_when_getting_effective_config_then_global_defaults(
    client: TestClient,
) -> None:
    response = client.get("/config/effective")

    assert response.status_code == 200
    body = response.json()
    assert body["embed_model"] == "ollama/bge-m3"
    assert body["chunk_size"] == 700
    assert body["max_ctx_chars"] == 3000


def test_given_unknown_kb_when_getting_effective_config_then_404(client: TestClient) -> None:
    response = client.get("/config/effective", params={"kb_id": str(uuid.uuid4())})

    assert response.status_code == 404


def test_given_running_app_when_getting_health_then_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
