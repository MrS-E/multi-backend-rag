"""Focused tests for the Step 5 worker jobs: pull sync, locking, cursors, reindex.

``sync_source``/``poll_tick``/``reindex_kb`` reach Redis, the DB, Qdrant and the
provider registry through module-level seams; each test swaps those for the
in-memory fakes from ``conftest`` so no Redis, Qdrant or Ollama is needed. The
connector registry is replaced with a recording fake so the port contract
(``pull(source_id, config, cursor)`` → ``PullResult``) is asserted directly.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from arq.worker import Retry
from sqlalchemy.orm import Session, sessionmaker

import app.workers.jobs as jobs
import app.workers.scheduler as scheduler
from app.core.settings import Settings
from app.domain.errors import ConflictError, ProviderError
from app.domain.events import UpsertEvent
from app.interaction.ports import PullResult
from app.interaction.registry import SourceConnectorRegistry
from app.persistence.file_storage import FileStorage
from app.persistence.models import Document, KnowledgeBase, RawFileCleanup, Source, SourceMode
from app.providers.registry import ProviderBundle, ProviderRegistry
from app.services.document_service import DocumentService
from app.services.kb_service import KBService
from tests.conftest import FakeJobQueue, FakeQdrantStore, StubChat, StubEmbedder, StubVision

_MARKDOWN = b"# Title\n\nThe quick brown fox jumps over the lazy dog.\n"


class _RecordingConnector:
    """A connector that records its inputs and returns canned events."""

    type_name = "recording"

    def __init__(self, result: PullResult) -> None:
        self.result = result
        self.calls: list[tuple[uuid.UUID, dict[str, Any], dict[str, Any] | None]] = []

    def pull(
        self,
        source_id: uuid.UUID,
        config: Mapping[str, Any],
        cursor: Mapping[str, Any] | None,
    ) -> PullResult:
        self.calls.append((source_id, dict(config), None if cursor is None else dict(cursor)))
        return self.result


class _FailingEmbedder:
    """An embedder that raises the port's ``ProviderError`` (what real providers do)."""

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise ProviderError("provider unreachable")


class _ProviderFailingConnector:
    type_name = "provider-failing"

    def pull(
        self,
        source_id: uuid.UUID,
        config: Mapping[str, Any],
        cursor: Mapping[str, Any] | None,
    ) -> PullResult:
        raise ProviderError("provider unavailable")


class _BlockingDeleteStorage(FileStorage):
    """Pause cleanup deletion so an upload can prove it waits for the KB lock."""

    def __init__(self, root: Path, started: threading.Event, release: threading.Event) -> None:
        super().__init__(root)
        self.started = started
        self.release = release

    def delete(self, rel_path: str) -> None:
        self.started.set()
        self.release.wait()
        super().delete(rel_path)


class _FakeLock:
    """Non-blocking Redis lock stand-in; ``acquire`` returns a canned answer."""

    def __init__(self, acquired: bool) -> None:
        self.acquired = acquired
        self.released = False

    async def acquire(self) -> bool:
        return self.acquired

    async def release(self) -> None:
        self.released = True


class _FakeRedis:
    """Records locks and enqueued jobs instead of talking to Redis."""

    def __init__(self, acquired: bool = True) -> None:
        self.acquired = acquired
        self.lock_names: list[str] = []
        self.locks: list[_FakeLock] = []
        self.enqueued: list[tuple[str, tuple[Any, ...]]] = []

    def lock(
        self, name: str, timeout: int | None = None, blocking: bool | None = None
    ) -> _FakeLock:
        self.lock_names.append(name)
        lock = _FakeLock(self.acquired)
        self.locks.append(lock)
        return lock

    async def enqueue_job(self, function: str, *args: Any) -> Any:
        self.enqueued.append((function, args))
        return type("_Job", (), {"job_id": "job-1"})()


class _RaceRedis:
    """Redis fake that runs the worker while the scheduler enqueue lock is held."""

    def __init__(self, on_enqueue: Any) -> None:
        self.on_enqueue = on_enqueue
        self.held: set[str] = set()
        self.lock_names: list[str] = []
        self.enqueued: list[tuple[str, tuple[Any, ...]]] = []

    def lock(self, name: str, timeout: int | None = None, blocking: bool | None = None) -> Any:
        self.lock_names.append(name)
        return _RaceLock(self, name)

    async def enqueue_job(self, function: str, *args: Any) -> Any:
        self.enqueued.append((function, args))
        await self.on_enqueue()
        return type("_Job", (), {"job_id": "job-1"})()


class _RaceLock:
    def __init__(self, redis: _RaceRedis, name: str) -> None:
        self.redis = redis
        self.name = name
        self.acquired = False

    async def acquire(self) -> bool:
        if self.name in self.redis.held:
            return False
        self.redis.held.add(self.name)
        self.acquired = True
        return True

    async def release(self) -> None:
        if self.acquired:
            self.redis.held.remove(self.name)


@dataclass
class _Env:
    """The patched worker seams plus the fakes it was wired with."""

    factory: sessionmaker[Session]
    registry: ProviderRegistry
    qdrant: FakeQdrantStore
    storage: FileStorage
    embedder: StubEmbedder
    connectors: SourceConnectorRegistry = field(default_factory=SourceConnectorRegistry)


@pytest.fixture
def env(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> _Env:
    """Patch the worker modules' seams onto in-memory fakes."""
    embedder = StubEmbedder(dimension=8)
    bundle = ProviderBundle(embedder=embedder, chat=StubChat(), vision=StubVision())
    registry = ProviderRegistry({"ollama": bundle, "openai": bundle})
    qdrant = FakeQdrantStore()
    storage = FileStorage(tmp_path)
    connectors = SourceConnectorRegistry()

    monkeypatch.setattr(jobs, "SessionLocal", session_factory)
    monkeypatch.setattr(scheduler, "SessionLocal", session_factory)
    monkeypatch.setattr(jobs, "get_settings", lambda: Settings())
    monkeypatch.setattr(jobs, "get_registry", lambda: registry)
    monkeypatch.setattr(jobs, "QdrantStore", lambda: qdrant)
    monkeypatch.setattr(jobs, "FileStorage", lambda: storage)
    monkeypatch.setattr(jobs, "get_source_registry", lambda: connectors)

    return _Env(
        factory=session_factory,
        registry=registry,
        qdrant=qdrant,
        storage=storage,
        embedder=embedder,
        connectors=connectors,
    )


def _make_kb(session: Session, overrides: dict[str, Any] | None = None) -> uuid.UUID:
    kb_id = uuid.uuid4()
    session.add(KnowledgeBase(id=kb_id, name="Docs", overrides=overrides or {}))
    session.commit()
    return kb_id


def _make_source(
    session: Session,
    kb_id: uuid.UUID,
    *,
    type_name: str = "recording",
    mode: SourceMode = SourceMode.pull,
    config: dict[str, Any] | None = None,
    cursor: dict[str, Any] | None = None,
    interval: int = 60,
    enabled: bool = True,
    last_sync_at: datetime | None = None,
) -> Source:
    source = Source(
        kb_id=kb_id,
        type=type_name,
        mode=mode,
        config=config or {},
        cursor=cursor,
        sync_interval_s=interval,
        enabled=enabled,
        last_sync_at=last_sync_at,
    )
    session.add(source)
    session.commit()
    return source


# --- sync_source: pull + cursor ---------------------------------------------


async def test_given_pull_source_when_syncing_then_events_ingested_and_cursor_persisted(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    doc_id = str(uuid.uuid4())
    connector = _RecordingConnector(
        PullResult(
            events=[
                UpsertEvent(
                    doc_id=doc_id,
                    uri="notes.md",
                    mime="text/markdown",
                    data=_MARKDOWN,
                )
            ],
            cursor={"files": {"notes.md": {"hash": "abc"}}},
        )
    )
    env.connectors.register("recording", connector)
    source = _make_source(session, kb_id, config={"path": "/data"})
    source_id = source.id
    session.close()

    processed = await jobs.sync_source({}, str(source_id))

    assert processed == 1
    refreshed = env.factory().get(Source, source_id)
    assert refreshed.cursor == {"files": {"notes.md": {"hash": "abc"}}}
    assert refreshed.last_sync_at is not None
    document = env.factory().get(Document, uuid.UUID(doc_id))
    assert document is not None
    assert document.status == "indexed"
    assert env.qdrant.payloads_for(kb_id, doc_id)


async def test_given_existing_cursor_when_syncing_then_connector_receives_it(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(PullResult(events=[], cursor={"files": {}}))
    env.connectors.register("recording", connector)
    cursor = {"files": {"notes.md": {"hash": "old", "doc_id": "d"}}}
    source = _make_source(session, kb_id, config={"path": "/data"}, cursor=cursor)
    source_id = source.id
    session.close()

    await jobs.sync_source({}, str(source_id))

    called_source_id, config, passed_cursor = connector.calls[0]
    assert called_source_id == source_id
    assert config == {"path": "/data"}
    assert passed_cursor == cursor


async def test_given_provider_failure_when_syncing_then_retry_is_bounded_and_cursor_is_unchanged(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    previous_cursor = {"files": {"old.md": {"hash": "old"}}}
    source = _make_source(
        session, kb_id, type_name="provider-failing", cursor=previous_cursor
    )
    source_id = source.id
    session.close()
    env.connectors.register("provider-failing", _ProviderFailingConnector())

    with pytest.raises(Retry) as raised:
        await jobs.sync_source({"job_try": 3}, str(source_id))

    assert raised.value.defer_score == jobs.retry_delay(3) * 1000
    refreshed = env.factory().get(Source, source_id)
    assert refreshed.cursor == previous_cursor
    assert refreshed.last_sync_at is None


async def test_given_failed_ingest_when_syncing_then_cursor_and_last_sync_not_persisted(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(
        PullResult(
            events=[
                UpsertEvent(
                    doc_id=str(uuid.uuid4()), uri="notes.md", mime="text/markdown", data=_MARKDOWN
                )
            ],
            cursor={"files": {"notes.md": {"hash": "new"}}},
        )
    )
    env.connectors.register("recording", connector)
    previous_cursor = {"files": {"old.md": {"hash": "old", "doc_id": "old-doc"}}}
    previous_sync = datetime(2026, 1, 1, tzinfo=UTC)
    source = _make_source(session, kb_id, cursor=previous_cursor, last_sync_at=previous_sync)
    source_id = source.id
    session.close()
    env.embedder.fail = True

    with pytest.raises(RuntimeError):
        await jobs.sync_source({}, str(source_id))

    refreshed = env.factory().get(Source, source_id)
    assert refreshed.cursor == previous_cursor
    assert refreshed.last_sync_at == previous_sync.replace(tzinfo=None)


async def test_given_pull_chunks_when_a_later_chunk_fails_then_cursor_is_not_advanced(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    previous_cursor = {"files": {"old.md": {"hash": "old", "doc_id": "old-doc"}}}
    source = _make_source(session, kb_id, cursor=previous_cursor)
    source_id = source.id
    session.close()
    connector = _RecordingConnector(
        PullResult(
            events=[
                UpsertEvent(
                    doc_id=str(uuid.uuid4()), uri="one.md", mime="text/markdown", data=_MARKDOWN
                ),
                UpsertEvent(
                    doc_id=str(uuid.uuid4()), uri="two.md", mime="text/markdown", data=_MARKDOWN
                ),
            ],
            cursor={"files": {"one.md": {"hash": "new"}, "two.md": {"hash": "new"}}},
        )
    )
    env.connectors.register("recording", connector)
    settings = Settings()
    settings.server.max_event_batch = 1
    monkeypatch.setattr(jobs, "get_settings", lambda: settings)

    class _FailingSecondChunkPipeline:
        calls = 0

        def __init__(self, **_: Any) -> None:
            pass

        def process_events(self, _kb_id: uuid.UUID, _events: list[UpsertEvent]) -> None:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second chunk failed")

    monkeypatch.setattr(jobs, "IngestionPipeline", _FailingSecondChunkPipeline)

    with pytest.raises(RuntimeError, match="second chunk"):
        await jobs.sync_source({}, str(source_id))

    refreshed = env.factory().get(Source, source_id)
    assert refreshed.cursor == previous_cursor
    assert refreshed.last_sync_at is None


# --- sync_source: lock semantics --------------------------------------------


async def test_given_source_lock_when_syncing_then_lock_acquired_and_released(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    env.connectors.register("recording", _RecordingConnector(PullResult([], {})))
    source = _make_source(session, kb_id)
    source_id = source.id
    session.close()
    redis = _FakeRedis(acquired=True)

    await jobs.sync_source({"redis": redis}, str(source_id))

    assert redis.lock_names == [f"lock:sync:{source_id}", f"lock:kb:{kb_id}"]
    assert all(lock.released for lock in redis.locks)


async def test_given_lock_held_when_syncing_then_retry_without_pulling(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(PullResult([], {}))
    env.connectors.register("recording", connector)
    previous_cursor = {"files": {"notes.md": {"hash": "old", "doc_id": "doc-1"}}}
    previous_sync = datetime(2026, 1, 2, tzinfo=UTC)
    source = _make_source(session, kb_id, cursor=previous_cursor, last_sync_at=previous_sync)
    source_id = source.id
    session.close()
    redis = _FakeRedis(acquired=False)

    with pytest.raises(Retry):
        await jobs.sync_source({"redis": redis}, str(source_id))

    assert connector.calls == []
    refreshed = env.factory().get(Source, source_id)
    assert refreshed.cursor == previous_cursor
    assert refreshed.last_sync_at == previous_sync.replace(tzinfo=None)
    assert redis.locks[0].released is False


async def test_given_lock_held_and_ingest_fails_when_syncing_then_lock_still_released(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(
        PullResult(
            events=[
                UpsertEvent(
                    doc_id=str(uuid.uuid4()), uri="notes.md", mime="text/markdown", data=_MARKDOWN
                )
            ],
            cursor={"files": {}},
        )
    )
    env.connectors.register("recording", connector)
    source = _make_source(session, kb_id)
    source_id = source.id
    session.close()
    env.embedder.fail = True
    redis = _FakeRedis(acquired=True)

    with pytest.raises(RuntimeError):
        await jobs.sync_source({"redis": redis}, str(source_id))

    assert redis.lock_names == [f"lock:sync:{source_id}", f"lock:kb:{kb_id}"]
    assert all(lock.released for lock in redis.locks)


# --- sync_source: gating -----------------------------------------------------


async def test_given_disabled_source_when_syncing_then_no_events_processed(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(PullResult([], {}))
    env.connectors.register("recording", connector)
    source = _make_source(session, kb_id, enabled=False)
    source_id = source.id
    session.close()

    assert await jobs.sync_source({}, str(source_id)) == 0
    assert connector.calls == []


async def test_given_event_mode_source_when_syncing_then_no_events_processed(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(PullResult([], {}))
    env.connectors.register("recording", connector)
    source = _make_source(session, kb_id, mode=SourceMode.event)
    source_id = source.id
    session.close()

    assert await jobs.sync_source({}, str(source_id)) == 0
    assert connector.calls == []


async def test_given_unknown_source_when_syncing_then_zero(env: _Env) -> None:
    assert await jobs.sync_source({}, str(uuid.uuid4())) == 0


async def test_given_cleared_claim_when_worker_has_stale_token_then_retry_without_pulling(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    source = _make_source(session, kb_id)
    session.close()

    with pytest.raises(Retry):
        await jobs.sync_source({}, str(source.id), "stale-token")

    assert env.factory().get(Source, source.id).sync_claim_token is None


async def test_given_no_redis_when_syncing_then_pull_still_runs(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(PullResult([], {}))
    env.connectors.register("recording", connector)
    source = _make_source(session, kb_id)
    source_id = source.id
    session.close()

    await jobs.sync_source({}, str(source_id))

    assert len(connector.calls) == 1


# --- poll_tick ---------------------------------------------------------------


async def test_given_due_pull_source_when_polling_then_sync_enqueued(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    source = _make_source(session, kb_id, interval=60)
    source_id = source.id
    session.close()
    redis = _FakeRedis()

    enqueued = await scheduler.poll_tick({"redis": redis})

    assert enqueued == 1
    assert redis.enqueued[0][0] == "sync_source"
    assert redis.enqueued[0][1][0] == str(source_id)
    assert len(redis.enqueued[0][1]) == 2


async def test_given_repeated_poll_ticks_when_sync_is_queued_then_only_one_claimed_job(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    source = _make_source(session, kb_id, interval=60)
    source_id = source.id
    session.close()
    redis = _FakeRedis()

    assert await scheduler.poll_tick({"redis": redis}) == 1
    assert await scheduler.poll_tick({"redis": redis}) == 0

    assert redis.enqueued[0][0] == "sync_source"
    assert redis.enqueued[0][1][0] == str(source_id)
    assert len(redis.enqueued[0][1]) == 2


async def test_given_scheduler_claim_when_worker_runs_then_claim_is_released(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(PullResult([], {}))
    env.connectors.register("recording", connector)
    source = _make_source(session, kb_id, type_name="recording", interval=60)
    source_id = source.id
    session.close()
    redis = _FakeRedis()

    assert await scheduler.poll_tick({"redis": redis}) == 1
    await jobs.sync_source({"redis": redis}, str(source_id))

    refreshed = env.factory().get(Source, source_id)
    assert refreshed.sync_claim_token is None


async def test_given_worker_starts_during_scheduler_enqueue_then_sync_is_not_skipped(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    connector = _RecordingConnector(PullResult([], {}))
    env.connectors.register("recording", connector)
    source = _make_source(session, kb_id, type_name="recording", interval=60)
    source_id = source.id
    session.close()

    async def run_worker() -> None:
        assert await jobs.sync_source({"redis": redis}, str(source_id)) == 0

    redis = _RaceRedis(run_worker)
    assert await scheduler.poll_tick({"redis": redis}) == 1

    assert connector.calls == [(source_id, {}, None)]
    assert redis.lock_names[:2] == [
        f"lock:schedule:sync:{source_id}",
        f"lock:sync:{source_id}",
    ]


async def test_given_recently_synced_source_when_polling_then_not_due(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    _make_source(session, kb_id, interval=60, last_sync_at=datetime.now(UTC))
    session.close()
    redis = _FakeRedis()

    assert await scheduler.poll_tick({"redis": redis}) == 0
    assert redis.enqueued == []


async def test_given_elapsed_interval_when_polling_then_due_again(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    _make_source(
        session,
        kb_id,
        interval=60,
        last_sync_at=datetime.now(UTC) - timedelta(seconds=120),
    )
    session.close()

    assert await scheduler.poll_tick({"redis": _FakeRedis()}) == 1


async def test_given_zero_interval_when_polling_then_skipped(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    _make_source(session, kb_id, interval=0)
    session.close()

    assert await scheduler.poll_tick({"redis": _FakeRedis()}) == 0


async def test_given_disabled_or_event_source_when_polling_then_skipped(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    _make_source(session, kb_id, interval=60, enabled=False)
    _make_source(session, kb_id, interval=60, mode=SourceMode.event)
    session.close()

    assert await scheduler.poll_tick({"redis": _FakeRedis()}) == 0


async def test_given_lock_held_when_polling_then_source_skipped(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    _make_source(session, kb_id, interval=60)
    session.close()
    redis = _FakeRedis(acquired=False)

    assert await scheduler.poll_tick({"redis": redis}) == 0
    assert redis.enqueued == []


async def test_given_no_redis_when_polling_then_zero(env: _Env) -> None:
    assert await scheduler.poll_tick({}) == 0


# --- reindex_kb --------------------------------------------------------------


async def test_given_embed_model_change_when_reindexing_then_collection_rebuilt_and_reingested(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session, overrides={"embed_model": "openai/text-embedding-3-large"})
    rel_path, digest = env.storage.save(kb_id, _MARKDOWN)
    doc_id = uuid.uuid4()
    session.add(
        Document(
            id=doc_id,
            kb_id=kb_id,
            uri="notes.md",
            content_hash=digest,
            mime="text/markdown",
            raw_path=rel_path,
            status="indexed",
        )
    )
    session.commit()
    session.close()
    # The pre-change collection has the old dimension.
    env.qdrant.create_collection(kb_id, 4)

    reindexed = await jobs.reindex_kb({}, str(kb_id))

    assert reindexed == 1
    assert env.qdrant.collections[str(kb_id)] == env.embedder.dimension
    assert env.qdrant.payloads_for(kb_id, str(doc_id))
    assert env.factory().get(KnowledgeBase, kb_id).status == "active"


async def test_given_many_documents_when_reindexing_then_documents_are_processed_in_batches(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    for index in range(3):
        rel_path, digest = env.storage.save(kb_id, _MARKDOWN + str(index).encode())
        session.add(
            Document(
                id=uuid.uuid4(),
                kb_id=kb_id,
                uri=f"notes-{index}.md",
                content_hash=digest,
                mime="text/markdown",
                raw_path=rel_path,
                status="indexed",
            )
        )
    session.commit()
    session.close()
    settings = Settings()
    settings.server.max_event_batch = 1
    monkeypatch.setattr(jobs, "get_settings", lambda: settings)
    batches: list[int] = []
    original = jobs.IngestionPipeline.process_events

    def record_batch(
        self, kb_id: uuid.UUID, events: list[UpsertEvent], force: bool = False
    ) -> None:
        batches.append(len(events))
        original(self, kb_id, events, force=force)

    monkeypatch.setattr(jobs.IngestionPipeline, "process_events", record_batch)

    assert await jobs.reindex_kb({}, str(kb_id)) == 3
    assert batches == [1, 1, 1]


async def test_given_reindex_when_reingesting_then_uses_override_embed_model(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session, overrides={"embed_model": "openai/text-embedding-3-large"})
    rel_path, digest = env.storage.save(kb_id, _MARKDOWN)
    session.add(
        Document(
            id=uuid.uuid4(),
            kb_id=kb_id,
            uri="notes.md",
            content_hash=digest,
            mime="text/markdown",
            raw_path=rel_path,
            status="indexed",
        )
    )
    session.commit()
    session.close()

    await jobs.reindex_kb({}, str(kb_id))

    embedded_texts, model = env.embedder.calls[-1]
    assert any("quick brown fox" in text for text in embedded_texts)
    assert model == "text-embedding-3-large"


async def test_given_empty_kb_when_reindexing_then_no_collection_created(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()

    reindexed = await jobs.reindex_kb({}, str(kb_id))

    assert reindexed == 0
    assert str(kb_id) not in env.qdrant.collections


async def test_given_stale_reindexing_state_when_reindexing_then_worker_takes_over(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    kb = session.get(KnowledgeBase, kb_id)
    assert kb is not None
    kb.status = "reindexing"
    kb.reindex_started_at = datetime.now(UTC) - timedelta(hours=2)
    session.commit()
    session.close()

    assert await jobs.reindex_kb({}, str(kb_id)) == 0
    recovered = env.factory().get(KnowledgeBase, kb_id)
    assert recovered is not None
    assert recovered.status == "active"
    assert recovered.reindex_started_at is None
    assert recovered.reindex_generation == 1


async def test_given_cleanup_marker_when_cleanup_worker_runs_then_marker_is_consumed(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    raw_path, _ = env.storage.save(kb_id, _MARKDOWN)
    session.add(RawFileCleanup(kb_id=kb_id, raw_path=raw_path))
    session.commit()
    session.close()

    assert await jobs.retry_raw_file_cleanups({}) == 1
    assert env.factory().query(RawFileCleanup).count() == 0
    with pytest.raises(FileNotFoundError):
        env.storage.read(raw_path)


async def test_given_cleanup_marker_referenced_by_new_document_then_file_is_retained(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    raw_path, digest = env.storage.save(kb_id, _MARKDOWN)
    session.add(RawFileCleanup(kb_id=kb_id, raw_path=raw_path))
    session.add(
        Document(
            id=uuid.uuid4(),
            kb_id=kb_id,
            uri="notes.md",
            content_hash=digest,
            mime="text/markdown",
            raw_path=raw_path,
            status="pending",
        )
    )
    session.commit()
    session.close()

    assert await jobs.retry_raw_file_cleanups({}) == 1
    assert env.storage.read(raw_path) == _MARKDOWN


async def test_given_existing_cleanup_marker_when_deleting_kb_then_marker_is_deduplicated(
    env: _Env,
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    raw_path, digest = env.storage.save(kb_id, _MARKDOWN)
    session.add(
        Document(
            id=uuid.uuid4(),
            kb_id=kb_id,
            uri="notes.md",
            content_hash=digest,
            mime="text/markdown",
            raw_path=raw_path,
            status="pending",
        )
    )
    session.add(RawFileCleanup(kb_id=kb_id, raw_path=raw_path))
    session.commit()
    session.close()

    await KBService(
        env.factory(), Settings(), env.registry, env.qdrant, env.storage
    ).delete_kb(kb_id)

    remaining = env.factory()
    try:
        assert remaining.get(KnowledgeBase, kb_id) is None
        assert (
            remaining.query(RawFileCleanup)
            .filter(RawFileCleanup.raw_path == raw_path)
            .count()
            == 0
        )
    finally:
        remaining.close()


async def test_given_db_failure_after_qdrant_delete_when_retrying_then_marker_recovers_deletion(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    raw_path, digest = env.storage.save(kb_id, _MARKDOWN)
    session.add(
        Document(
            id=uuid.uuid4(),
            kb_id=kb_id,
            uri="notes.md",
            content_hash=digest,
            mime="text/markdown",
            raw_path=raw_path,
            status="pending",
        )
    )
    session.commit()
    original_commit = session.commit
    commit_count = 0

    def fail_after_qdrant() -> None:
        nonlocal commit_count
        commit_count += 1
        if commit_count == 2:
            raise RuntimeError("database commit failed")
        original_commit()

    monkeypatch.setattr(session, "commit", fail_after_qdrant)
    with pytest.raises(RuntimeError, match="database commit failed"):
        await KBService(session, Settings(), env.registry, env.qdrant, env.storage).delete_kb(kb_id)
    session.close()

    recovery_session = env.factory()
    try:
        assert recovery_session.get(KnowledgeBase, kb_id) is not None
        assert recovery_session.query(RawFileCleanup).filter(
            RawFileCleanup.raw_path == raw_path
        ).count() == 1
    finally:
        recovery_session.close()

    await KBService(
        env.factory(), Settings(), env.registry, env.qdrant, env.storage
    ).delete_kb(kb_id)
    final_session = env.factory()
    try:
        assert final_session.get(KnowledgeBase, kb_id) is None
    finally:
        final_session.close()


async def test_given_cleanup_worker_when_uploading_same_kb_then_upload_waits_for_cleanup_lock(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    raw_path, _ = env.storage.save(kb_id, _MARKDOWN)
    session.add(RawFileCleanup(kb_id=kb_id, raw_path=raw_path))
    session.commit()
    session.close()

    started = threading.Event()
    release = threading.Event()
    blocking_storage = _BlockingDeleteStorage(env.storage._root, started, release)
    monkeypatch.setattr(jobs, "FileStorage", lambda: blocking_storage)
    cleanup_task = asyncio.create_task(jobs.retry_raw_file_cleanups({}))
    upload_session = env.factory()
    try:
        assert await asyncio.to_thread(started.wait, 2)
        upload_task = asyncio.create_task(
            DocumentService(upload_session, env.storage, FakeJobQueue()).upload(
                kb_id, "new.md", _MARKDOWN, 1024 * 1024
            )
        )
        await asyncio.sleep(0)
        assert not upload_task.done()

        release.set()
        assert await cleanup_task == 1
        await upload_task
    finally:
        release.set()
        if not cleanup_task.done():
            await cleanup_task
        upload_session.close()


async def test_given_unknown_kb_when_reindexing_then_zero(env: _Env) -> None:
    assert await jobs.reindex_kb({}, str(uuid.uuid4())) == 0


async def test_given_redis_when_reindexing_then_per_kb_lock_is_used(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()
    redis = _FakeRedis()

    assert await jobs.reindex_kb({"redis": redis}, str(kb_id)) == 0

    assert redis.lock_names == [f"lock:kb:{kb_id}"]
    assert redis.locks[0].released is True


async def test_given_reindex_lock_held_then_reindexing_is_deferred(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()
    redis = _FakeRedis(acquired=False)

    with pytest.raises(Retry):
        await jobs.reindex_kb({"redis": redis}, str(kb_id))

    assert redis.lock_names == [f"lock:kb:{kb_id}"]
    assert redis.locks[0].released is False
    assert env.factory().get(KnowledgeBase, kb_id).status == "active"


async def test_given_kb_lock_held_when_deleting_then_delete_is_deferred(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()
    redis = _FakeRedis(acquired=False)
    service = KBService(
        env.factory(), Settings(), env.registry, env.qdrant, env.storage, redis=redis
    )

    with pytest.raises(ConflictError, match="knowledge base is busy"):
        await service.delete_kb(kb_id)

    assert redis.lock_names == [f"lock:kb:{kb_id}"]
    assert env.factory().get(KnowledgeBase, kb_id) is not None


async def test_given_config_changes_during_reindex_then_old_job_cannot_publish_active(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    rel_path, digest = env.storage.save(kb_id, _MARKDOWN)
    session.add(
        Document(
            id=uuid.uuid4(),
            kb_id=kb_id,
            uri="notes.md",
            content_hash=digest,
            mime="text/markdown",
            raw_path=rel_path,
            status="indexed",
        )
    )
    session.commit()
    session.close()
    base_pipeline = jobs.IngestionPipeline

    class _StalePipeline(base_pipeline):
        def process_events(
            self, kb_id: uuid.UUID, events: list[UpsertEvent], force: bool = False
        ) -> None:
            other_session = env.factory()
            current = other_session.get(KnowledgeBase, kb_id)
            assert current is not None
            current.config_version += 1
            current.status = "reindex_required"
            other_session.commit()
            other_session.close()
            super().process_events(kb_id, events, force=force)

    monkeypatch.setattr(jobs, "IngestionPipeline", _StalePipeline)

    with pytest.raises(Retry):
        await jobs.reindex_kb({}, str(kb_id))

    assert env.factory().get(KnowledgeBase, kb_id).status == "reindex_required"


# --- ingest_events retry -----------------------------------------------------


async def test_given_provider_error_when_ingesting_then_retry_with_backoff(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()
    payload = UpsertEvent(
        doc_id=str(uuid.uuid4()), uri="notes.md", mime="text/markdown", data=_MARKDOWN
    ).to_payload()
    failing = ProviderRegistry(
        {
            "ollama": ProviderBundle(
                embedder=_FailingEmbedder(), chat=StubChat(), vision=StubVision()
            )
        }
    )
    monkeypatch.setattr(jobs, "get_registry", lambda: failing)

    with pytest.raises(Retry) as excinfo:
        await jobs.ingest_events({"job_try": 3}, str(kb_id), [payload])

    # arq stores the defer as a millisecond score.
    assert excinfo.value.defer_score == jobs.retry_delay(3) * 1000


async def test_given_non_provider_error_when_ingesting_then_not_retried(env: _Env) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()
    payload = UpsertEvent(
        doc_id="not-a-uuid", uri="notes.md", mime="text/markdown", data=_MARKDOWN
    ).to_payload()

    with pytest.raises(ValueError, match="doc_id"):
        await jobs.ingest_events({"job_try": 1}, str(kb_id), [payload])


async def test_given_event_batch_over_limit_when_ingesting_then_rejected(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()
    settings = Settings()
    settings.server.max_event_batch = 1
    monkeypatch.setattr(jobs, "get_settings", lambda: settings)
    payload = UpsertEvent(
        doc_id=str(uuid.uuid4()), uri="notes.md", mime="text/markdown", data=_MARKDOWN
    ).to_payload()

    with pytest.raises(ValueError, match="event batch exceeds"):
        await jobs.ingest_events({}, str(kb_id), [payload, payload])


async def test_given_inline_event_data_over_limit_when_ingesting_then_rejected(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = env.factory()
    kb_id = _make_kb(session)
    session.close()
    settings = Settings()
    settings.server.max_upload_mb = 0
    monkeypatch.setattr(jobs, "get_settings", lambda: settings)
    payload = UpsertEvent(
        doc_id=str(uuid.uuid4()), uri="notes.md", mime="text/markdown", data=b"x"
    ).to_payload()

    with pytest.raises(ValueError, match="inline event data exceeds"):
        await jobs.ingest_events({}, str(kb_id), [payload])
