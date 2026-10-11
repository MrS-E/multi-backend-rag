"""arq jobs: the RAG-core side of the broker.

``ingest_events`` is the single entry point the API enqueues. It receives
JSON-serializable event payloads (see :mod:`app.domain.events`) plus the target
KB id, then drives the ingestion pipeline with a fresh DB session and the
process-wide provider/Qdrant/storage singletons.

Provider errors are retried with exponential backoff by raising arq's
:class:`~arq.worker.Retry`; any other failure propagates and marks the job
``failed`` in arq's Redis metadata, where ``GET /jobs/{id}`` surfaces it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from arq.worker import Retry
from sqlalchemy import or_, select, update

from app.core.coordination import (
    SOURCE_LOCK_TIMEOUT_S,
    LeaseLostError,
    LockLease,
    acquire_kb_coordination,
    release_kb_coordination,
    release_lock,
)
from app.core.settings import get_settings
from app.domain.config_resolution import resolve_effective_config
from app.domain.errors import ProviderError
from app.domain.events import UpsertEvent, event_from_payload
from app.domain.models import KB_STATUS_ACTIVE, KB_STATUS_REINDEX_REQUIRED, KB_STATUS_REINDEXING
from app.interaction.local_folder import LocalFolderConnector
from app.interaction.registry import get_source_registry
from app.persistence.db import SessionLocal
from app.persistence.file_storage import FileStorage
from app.persistence.models import Document, KnowledgeBase, RawFileCleanup, Source, SourceMode
from app.persistence.qdrant_store import QdrantStore
from app.providers.registry import get_registry, parse_model_str
from app.rag.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

#: Base seconds for the exponential backoff applied to provider errors.
RETRY_BACKOFF_BASE_S = 5

#: Cap on the backoff delay so a long outage still retries promptly.
RETRY_BACKOFF_MAX_S = 300
SOURCE_CLAIM_RENEWAL_S = 300
REINDEX_STALE_AFTER_S = 3600


def retry_delay(job_try: int) -> int:
    """Return the backoff delay (seconds) for the given 1-based attempt number."""
    return min(RETRY_BACKOFF_BASE_S * 2 ** max(job_try - 1, 0), RETRY_BACKOFF_MAX_S)


def _fence(*leases: LockLease | None) -> None:
    """Fence every backend mutation with all leases held by the job."""
    for lease in leases:
        if lease is not None:
            lease.ensure_valid()


async def ingest_events(ctx: dict[str, Any], kb_id: str, events: list[dict[str, Any]]) -> int:
    """Ingest a batch of serialized source events for one knowledge base.

    Returns the number of events processed. A provider outage raises arq's
    ``Retry`` with exponential backoff so the job is re-run once the provider
    recovers instead of failing permanently.
    """
    settings = get_settings()
    if len(events) > settings.server.max_event_batch:
        raise ValueError(f"event batch exceeds the {settings.server.max_event_batch} event limit")
    max_encoded_length = 4 * ((settings.server.max_upload_mb * 1024 * 1024 + 2) // 3)
    for payload in events:
        encoded = payload.get("data_b64")
        if isinstance(encoded, str) and len(encoded) > max_encoded_length:
            raise ValueError("inline event data exceeds the configured upload limit")
    redis = ctx.get("redis")
    coordination_lock = await acquire_kb_coordination(redis, kb_id)
    if redis is not None and coordination_lock is None:
        raise Retry(defer=1)
    job_try = int(ctx.get("job_try", 1))
    lease = LockLease(coordination_lock)
    await lease.__aenter__()
    try:
        _fence(lease)
        return await lease.run_blocking(_ingest_events, settings, kb_id, events, lease)
    except LeaseLostError as exc:
        raise Retry(defer=1) from exc
    except ProviderError as exc:
        delay = retry_delay(job_try)
        logger.warning(
            "provider error during ingestion; retrying",
            extra={"kb_id": kb_id, "job_try": job_try, "defer_s": delay, "error": str(exc)},
        )
        raise Retry(defer=delay) from exc
    finally:
        await lease.__aexit__(None, None, None)
        if coordination_lock is not None:
            await release_kb_coordination(redis, coordination_lock)


def _ingest_events(
    settings: Any,
    kb_id: str,
    events: list[dict[str, Any]],
    lease: LockLease | None = None,
) -> int:
    """Run ingestion while the caller owns the per-KB lease."""
    parsed = [event_from_payload(payload) for payload in events]
    for event in parsed:
        if (
            isinstance(event, UpsertEvent)
            and event.data is not None
            and len(event.data) > settings.server.max_upload_mb * 1024 * 1024
        ):
            raise ValueError("inline event data exceeds the configured upload limit")
    with SessionLocal() as session:
        pipeline = IngestionPipeline(
            session=session,
            settings=settings,
            registry=get_registry(),
            qdrant=QdrantStore(),
            storage=FileStorage(),
            lease_check=lambda: _fence(lease),
        )
        pipeline.process_events(uuid.UUID(kb_id), parsed)
    return len(parsed)


async def _with_source_lock(redis: Any, source_id: str) -> Any:
    """Acquire a non-blocking Redis lock, returning it or ``None``."""
    lock = redis.lock(f"lock:sync:{source_id}", timeout=SOURCE_LOCK_TIMEOUT_S, blocking=False)
    return lock if await lock.acquire() else None


async def _with_reindex_lock(redis: Any, kb_id: str) -> Any:
    """Acquire the shared non-blocking per-KB ingestion lock."""
    return await _with_kb_lock(redis, kb_id)


async def _with_kb_lock(redis: Any, kb_id: str) -> Any:
    """Acquire a non-blocking lock shared by all index mutations for a KB."""
    return await acquire_kb_coordination(redis, kb_id)


async def sync_source(
    ctx: dict[str, Any], source_id: str, claim_token: str | None = None
) -> int:
    """Pull one source, ingest its events, and persist its cursor on success only."""
    redis = ctx.get("redis")
    lock = await _with_source_lock(redis, source_id) if redis is not None else None
    if redis is not None and lock is None:
        raise Retry(defer=1)
    source_lease = LockLease(lock, SOURCE_LOCK_TIMEOUT_S) if lock is not None else None
    if source_lease is not None:
        await source_lease.__aenter__()
    kb_lock = None
    kb_lease: LockLease | None = None
    claim_task: asyncio.Task[None] | None = None
    try:
        with SessionLocal() as session:
            source = session.get(Source, uuid.UUID(source_id))
            if source is None or not source.enabled or source.mode is not SourceMode.pull:
                return 0
            if claim_token is not None and source.sync_claim_token != claim_token:
                raise Retry(defer=1)
            kb_lock = await _with_kb_lock(redis, str(source.kb_id))
            if redis is not None and kb_lock is None:
                raise Retry(defer=1)
            kb_lease = LockLease(kb_lock) if kb_lock is not None else None
            if kb_lease is not None:
                await kb_lease.__aenter__()
            _fence(source_lease, kb_lease)
            existing_claim_token = source.sync_claim_token
            claim_token = claim_token or existing_claim_token or str(uuid.uuid4())
            claim_where = (
                Source.sync_claim_token == claim_token
                if existing_claim_token is not None and claim_token == existing_claim_token
                else Source.sync_claim_token.is_(None)
            )
            claimed = session.execute(
                update(Source)
                .where(Source.id == source.id, claim_where)
                .values(sync_claim_token=claim_token, sync_claimed_at=datetime.now(UTC))
                .execution_options(synchronize_session=False)
            )
            if cast(Any, claimed).rowcount != 1:
                session.rollback()
                raise Retry(defer=1)
            _fence(source_lease, kb_lease)
            session.commit()
            session.refresh(source)
            claim_task = asyncio.create_task(
                _renew_source_claim(source_id, claim_token, source_lease, kb_lease)
            )
            connector = get_source_registry().get(source.type)
            settings = get_settings()
            max_upload_bytes = settings.server.max_upload_mb * 1024 * 1024
            connector_config = dict(source.config)
            if isinstance(connector, LocalFolderConnector):
                connector_config["_max_file_bytes"] = max_upload_bytes
                connector_config["_max_events"] = settings.server.max_event_batch
                if settings.storage.sources_root is not None:
                    connector_config["_allowed_root"] = settings.storage.sources_root
            try:
                pipeline = IngestionPipeline(
                    session=session,
                    settings=settings,
                    registry=get_registry(),
                    qdrant=QdrantStore(),
                    storage=FileStorage(),
                    lease_check=lambda: _fence(source_lease, kb_lease),
                )
                working_cursor = source.cursor
                processed = 0
                while True:
                    if source_lease is not None:
                        result = await source_lease.run_blocking(
                            connector.pull, source.id, connector_config, working_cursor
                        )
                    else:
                        result = await (kb_lease or source_lease or LockLease(None)).run_blocking(
                            connector.pull, source.id, connector_config, working_cursor
                        )
                    _ensure_claim_task(claim_task)
                    for event in result.events:
                        if (
                            isinstance(event, UpsertEvent)
                            and event.data is not None
                            and len(event.data) > max_upload_bytes
                        ):
                            raise ValueError(
                                "pull event data exceeds the configured upload limit"
                            )
                    for offset in range(0, len(result.events), settings.server.max_event_batch):
                        if source_lease is not None:
                            await source_lease.run_blocking(
                                pipeline.process_events,
                                source.kb_id,
                                result.events[offset : offset + settings.server.max_event_batch],
                            )
                        else:
                            await (kb_lease or source_lease or LockLease(None)).run_blocking(
                                pipeline.process_events,
                                source.kb_id,
                                result.events[offset : offset + settings.server.max_event_batch],
                            )
                        _ensure_claim_task(claim_task)
                    processed += len(result.events)
                    continuation_key = getattr(connector, "_CONTINUATION_KEY", None)
                    if continuation_key is None or continuation_key not in result.cursor:
                        _fence(source_lease, kb_lease)
                        final_cursor = result.cursor
                        break
                    working_cursor = result.cursor
                _ensure_claim_task(claim_task)
                _fence(source_lease, kb_lease)
                finalized = session.execute(
                    update(Source)
                    .where(Source.id == source.id, Source.sync_claim_token == claim_token)
                    .values(cursor=final_cursor, last_sync_at=datetime.now(UTC))
                    .execution_options(synchronize_session=False)
                )
                if cast(Any, finalized).rowcount != 1:
                    session.rollback()
                    raise Retry(defer=1)
                _fence(source_lease, kb_lease)
                session.commit()
                return processed
            finally:
                _ensure_claim_task(claim_task)
                _fence(source_lease, kb_lease)
                cleared = session.execute(
                    update(Source)
                    .where(Source.id == source.id, Source.sync_claim_token == claim_token)
                    .values(sync_claim_token=None, sync_claimed_at=None)
                    .execution_options(synchronize_session=False)
                )
                if cast(Any, cleared).rowcount != 1:
                    session.rollback()
                    raise Retry(defer=1)
                _fence(source_lease, kb_lease)
                session.commit()
    except LeaseLostError as exc:
        raise Retry(defer=1) from exc
    except ProviderError as exc:
        job_try = int(ctx.get("job_try", 1))
        delay = retry_delay(job_try)
        logger.warning(
            "provider error during source sync; retrying",
            extra={
                "source_id": source_id,
                "job_try": job_try,
                "defer_s": delay,
                "error": str(exc),
            },
        )
        raise Retry(defer=delay) from exc
    finally:
        if claim_task is not None:
            claim_task.cancel()
            await asyncio.gather(claim_task, return_exceptions=True)
        if kb_lease is not None:
            await kb_lease.__aexit__(None, None, None)
        if kb_lock is not None:
            await release_kb_coordination(redis, kb_lock)
        if source_lease is not None:
            await source_lease.__aexit__(None, None, None)
        if lock is not None:
            await release_lock(lock)


async def _renew_source_claim(
    source_id: str,
    claim_token: str,
    *leases: LockLease | None,
) -> None:
    """Keep a queued/running source claim alive while blocking work is offloaded."""
    while True:
        await asyncio.sleep(SOURCE_CLAIM_RENEWAL_S)
        _fence(*leases)
        with SessionLocal() as session:
            renewed = session.execute(
                update(Source)
                .where(Source.id == uuid.UUID(source_id), Source.sync_claim_token == claim_token)
                .values(sync_claimed_at=datetime.now(UTC))
            )
            if cast(Any, renewed).rowcount != 1:
                session.rollback()
                raise Retry(defer=1)
            _fence(*leases)
            session.commit()


def _ensure_claim_task(claim_task: asyncio.Task[None] | None) -> None:
    """Propagate a failed CAS renewal before another worker mutation runs."""
    if claim_task is not None and claim_task.done():
        claim_task.result()


async def reindex_kb(ctx: dict[str, Any], kb_id: str) -> int:
    """Recreate a KB collection and re-ingest every stored document."""
    redis = ctx.get("redis")
    coordination_lock = await _with_reindex_lock(redis, kb_id)
    if redis is not None and coordination_lock is None:
        raise Retry(defer=1)
    lease = LockLease(coordination_lock)
    await lease.__aenter__()
    try:
        return await lease.run_blocking(_reindex_sync, kb_id, lease)
    except LeaseLostError as exc:
        raise Retry(defer=1) from exc
    except Retry:
        raise
    except Exception as exc:  # noqa: BLE001 - reindex failures must remain recoverable
        job_try = int(ctx.get("job_try", 1))
        logger.warning(
            "reindex failed; keeping the knowledge base retryable",
            extra={"kb_id": kb_id, "job_try": job_try, "error": str(exc)},
        )
        raise Retry(defer=retry_delay(job_try)) from exc
    finally:
        await lease.__aexit__(None, None, None)
        if coordination_lock is not None:
            await release_kb_coordination(redis, coordination_lock)


def _reindex_sync(kb_id: str, lease: LockLease | None = None) -> int:
    """Run blocking reindex work off the event loop so lock renewal can run."""
    with SessionLocal() as session:
        kb = session.get(KnowledgeBase, uuid.UUID(kb_id))
        if kb is None:
            return 0
        config_version = kb.config_version
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=REINDEX_STALE_AFTER_S)
        _fence(lease)
        started = session.execute(
            update(KnowledgeBase)
            .where(
                KnowledgeBase.id == kb.id,
                KnowledgeBase.config_version == config_version,
                KnowledgeBase.status.in_(
                    [KB_STATUS_ACTIVE, KB_STATUS_REINDEX_REQUIRED, KB_STATUS_REINDEXING]
                ),
                or_(
                    KnowledgeBase.status != KB_STATUS_REINDEXING,
                    KnowledgeBase.reindex_started_at.is_(None),
                    KnowledgeBase.reindex_started_at < stale_before,
                ),
            )
            .values(
                status=KB_STATUS_REINDEXING,
                reindex_generation=KnowledgeBase.reindex_generation + 1,
                reindex_started_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        if cast(Any, started).rowcount != 1:
            session.rollback()
            raise Retry(defer=1)
        _fence(lease)
        session.commit()
        session.refresh(kb)
        reindex_generation = kb.reindex_generation
        try:
            settings = get_settings()
            config = resolve_effective_config(settings.defaults, kb.overrides)
            qdrant = QdrantStore()
            if qdrant.collection_exists(kb.id):
                _fence(lease)
                qdrant.delete_collection(kb.id)
            batch_size = settings.server.max_event_batch
            documents = list(
                session.scalars(
                    select(Document)
                    .where(Document.kb_id == kb.id)
                    .order_by(Document.id)
                    .limit(batch_size)
                )
            )
            if documents:
                embedder = get_registry().get_embedder(config.embed_model)
                _, model = parse_model_str(config.embed_model)
                vectors = embedder.embed(["dimension probe"], model)
                _fence(lease)
                qdrant.create_collection(kb.id, len(vectors[0]))
            pipeline = IngestionPipeline(
                session=session,
                settings=settings,
                registry=get_registry(),
                qdrant=qdrant,
                storage=FileStorage(),
                lease_check=lambda: _fence(lease),
            )
            processed = 0
            while documents:
                events = [
                    event_from_payload(
                        {
                            "op": "upsert",
                            "doc_id": str(document.id),
                            "uri": document.uri,
                            "mime": document.mime,
                            "content_hash": document.content_hash,
                            "path": document.raw_path,
                            "metadata": {"source_id": str(document.source_id)}
                            if document.source_id is not None
                            else {},
                        }
                    )
                    for document in documents
                ]
                pipeline.process_events(kb.id, events, force=True)
                processed += len(events)
                last_id = documents[-1].id
                current_version = session.scalar(
                    select(KnowledgeBase.config_version).where(KnowledgeBase.id == kb.id)
                )
                if current_version != config_version:
                    raise Retry(defer=1)
                documents = list(
                    session.scalars(
                        select(Document)
                        .where(Document.kb_id == kb.id, Document.id > last_id)
                        .order_by(Document.id)
                        .limit(batch_size)
                    )
                )
            _fence(lease)
            published = session.execute(
                update(KnowledgeBase)
                .where(
                    KnowledgeBase.id == kb.id,
                    KnowledgeBase.config_version == config_version,
                    KnowledgeBase.status == KB_STATUS_REINDEXING,
                    KnowledgeBase.reindex_generation == reindex_generation,
                )
                .values(status=KB_STATUS_ACTIVE, reindex_started_at=None)
            )
            published_count = cast(Any, published).rowcount
            _fence(lease)
            session.commit()
            if published_count != 1:
                raise Retry(defer=1)
            return processed
        except Exception:
            session.rollback()
            if lease is None or not _lease_lost(lease):
                session.execute(
                    update(KnowledgeBase)
                    .where(
                        KnowledgeBase.id == kb.id,
                        KnowledgeBase.config_version == config_version,
                        KnowledgeBase.status == KB_STATUS_REINDEXING,
                        KnowledgeBase.reindex_generation == reindex_generation,
                    )
                    .values(status=KB_STATUS_REINDEX_REQUIRED, reindex_started_at=None)
                )
                session.commit()
            raise


def _lease_lost(lease: LockLease) -> bool:
    """Return whether a lease has been fenced, without exposing its internals."""
    try:
        lease.ensure_valid()
    except LeaseLostError:
        return True
    return False


def _retry_raw_file_cleanups_sync(
    kb_id: uuid.UUID, lease: LockLease | None = None
) -> int:
    """Delete one KB's durable markers while its coordination lease is held."""
    processed = 0
    with SessionLocal() as session:
        storage = FileStorage()
        markers = list(
            session.scalars(
                select(RawFileCleanup)
                .where(RawFileCleanup.kb_id == kb_id)
                .order_by(RawFileCleanup.created_at)
            )
        )
        for marker in markers:
            try:
                _fence(lease)
                referenced = session.scalar(
                    select(Document.id).where(Document.raw_path == marker.raw_path)
                )
                if referenced is not None:
                    session.delete(marker)
                    _fence(lease)
                    session.commit()
                    processed += 1
                    continue
                _fence(lease)
                storage.delete(marker.raw_path)
                session.delete(marker)
                _fence(lease)
                session.commit()
                processed += 1
            except LeaseLostError:
                session.rollback()
                raise
            except Exception:  # noqa: BLE001 - retain failed markers for a later retry
                session.rollback()
                logger.warning(
                    "failed to retry raw file cleanup",
                    extra={"raw_path": marker.raw_path, "kb_id": str(marker.kb_id)},
                    exc_info=True,
                )
    return processed


async def retry_raw_file_cleanups(ctx: dict[str, Any]) -> int:
    """Consume durable raw-file cleanup markers without blocking the event loop."""
    redis = ctx.get("redis")
    with SessionLocal() as session:
        kb_ids = list(session.scalars(select(RawFileCleanup.kb_id).distinct()))

    processed = 0
    for kb_id in kb_ids:
        lock = await acquire_kb_coordination(redis, str(kb_id))
        if redis is not None and lock is None:
            continue
        lease = LockLease(lock)
        await lease.__aenter__()
        try:
            processed += await lease.run_blocking(_retry_raw_file_cleanups_sync, kb_id, lease)
        except LeaseLostError as exc:
            raise Retry(defer=1) from exc
        finally:
            await lease.__aexit__(None, None, None)
            if lock is not None:
                await release_kb_coordination(redis, lock)
    return processed
