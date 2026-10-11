"""Source CRUD and event-mode application service."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.broker import JobQueue
from app.core.coordination import (
    LeaseLostError,
    LockLease,
    acquire_kb_coordination,
    release_kb_coordination,
)
from app.domain.errors import BrokerUnavailableError, ConflictError, NotFoundError
from app.domain.events import SourceEvent
from app.persistence.models import Document, Source, SourceMode


class SourceService:
    """Manage sources without coupling the API to connector implementations."""

    def __init__(
        self, session: Session, queue: JobQueue | None = None, redis: Any | None = None
    ) -> None:
        self._session = session
        self._queue = queue
        self._redis = redis

    @asynccontextmanager
    async def _kb_lease(self, kb_id: uuid.UUID) -> Any:
        """Serialize source ownership changes with active source workers."""
        try:
            lock = await acquire_kb_coordination(self._redis, str(kb_id))
        except Exception as exc:  # noqa: BLE001 - Redis failures are API 503s
            raise BrokerUnavailableError("job broker unavailable") from exc
        if self._redis is not None and lock is None:
            raise ConflictError("knowledge base is busy")
        lease = LockLease(lock) if lock is not None else None
        if lease is not None:
            await lease.__aenter__()
        try:
            yield lease
        finally:
            if lease is not None:
                await lease.__aexit__(None, None, None)
            if lock is not None:
                await release_kb_coordination(self._redis, lock)

    def list_sources(self, kb_id: uuid.UUID) -> list[Source]:
        """Return sources for a KB in creation order."""
        return list(
            self._session.scalars(
                select(Source).where(Source.kb_id == kb_id).order_by(Source.id)
            )
        )

    def get_source(self, kb_id: uuid.UUID, source_id: uuid.UUID) -> Source:
        """Return a source scoped to a KB."""
        source = self._session.get(Source, source_id)
        if source is None or source.kb_id != kb_id:
            raise NotFoundError(f"source {source_id} not found in knowledge base {kb_id}")
        return source

    def validate_document_ids(self, kb_id: uuid.UUID, doc_ids: list[str]) -> None:
        """Reject event IDs that already belong to another knowledge base."""
        for doc_id in doc_ids:
            try:
                document_uuid = uuid.UUID(doc_id)
            except ValueError:
                continue
            document = self._session.get(Document, document_uuid)
            if document is not None and document.kb_id != kb_id:
                raise NotFoundError(f"document {doc_id} not found in knowledge base {kb_id}")

    async def create_source(
        self,
        kb_id: uuid.UUID,
        type_name: str,
        mode: str,
        config: dict[str, Any],
        sync_interval_s: int,
        enabled: bool,
    ) -> Source:
        """Create and persist a source."""
        try:
            async with self._kb_lease(kb_id) as lease:
                if lease is None:
                    return await asyncio.to_thread(
                        self._create_source_without_lease,
                        kb_id,
                        type_name,
                        mode,
                        config,
                        sync_interval_s,
                        enabled,
                    )
                return await lease.run_blocking(
                    self._create_source_sync,
                    kb_id,
                    type_name,
                    mode,
                    config,
                    sync_interval_s,
                    enabled,
                    lease,
                )
        except LeaseLostError:
            await asyncio.to_thread(self._session.rollback)
            raise

    async def update_source(
        self,
        kb_id: uuid.UUID,
        source_id: uuid.UUID,
        **updates: object,
    ) -> Source:
        """Apply a partial source update."""
        try:
            async with self._kb_lease(kb_id) as lease:
                if lease is None:
                    return await asyncio.to_thread(
                        self._update_source_without_lease, kb_id, source_id, updates
                    )
                return await lease.run_blocking(
                    self._update_source_sync, kb_id, source_id, updates, lease
                )
        except LeaseLostError:
            await asyncio.to_thread(self._session.rollback)
            raise

    async def delete_source(self, kb_id: uuid.UUID, source_id: uuid.UUID) -> None:
        """Delete a source; its documents remain and lose source attribution."""
        try:
            async with self._kb_lease(kb_id) as lease:
                if lease is None:
                    await asyncio.to_thread(self._delete_source_without_lease, kb_id, source_id)
                else:
                    await lease.run_blocking(self._delete_source_sync, kb_id, source_id, lease)
        except LeaseLostError:
            await asyncio.to_thread(self._session.rollback)
            raise

    def _create_source_sync(
        self,
        kb_id: uuid.UUID,
        type_name: str,
        mode: str,
        config: dict[str, Any],
        sync_interval_s: int,
        enabled: bool,
        lease: LockLease,
    ) -> Source:
        source = self._new_source(kb_id, type_name, mode, config, sync_interval_s, enabled)
        lease.ensure_valid()
        self._session.commit()
        self._session.refresh(source)
        return source

    def _create_source_without_lease(
        self,
        kb_id: uuid.UUID,
        type_name: str,
        mode: str,
        config: dict[str, Any],
        sync_interval_s: int,
        enabled: bool,
    ) -> Source:
        source = self._new_source(kb_id, type_name, mode, config, sync_interval_s, enabled)
        self._session.commit()
        self._session.refresh(source)
        return source

    def _new_source(
        self,
        kb_id: uuid.UUID,
        type_name: str,
        mode: str,
        config: dict[str, Any],
        sync_interval_s: int,
        enabled: bool,
    ) -> Source:
        source = Source(
            kb_id=kb_id,
            type=type_name,
            mode=SourceMode(mode),
            config=config,
            sync_interval_s=sync_interval_s,
            enabled=enabled,
        )
        self._session.add(source)
        return source

    def _update_source_sync(
        self,
        kb_id: uuid.UUID,
        source_id: uuid.UUID,
        updates: dict[str, object],
        lease: LockLease,
    ) -> Source:
        source = self._apply_source_updates(kb_id, source_id, updates)
        lease.ensure_valid()
        self._session.commit()
        self._session.refresh(source)
        return source

    def _update_source_without_lease(
        self, kb_id: uuid.UUID, source_id: uuid.UUID, updates: dict[str, object]
    ) -> Source:
        source = self._apply_source_updates(kb_id, source_id, updates)
        self._session.commit()
        self._session.refresh(source)
        return source

    def _apply_source_updates(
        self, kb_id: uuid.UUID, source_id: uuid.UUID, updates: dict[str, object]
    ) -> Source:
        source = self.get_source(kb_id, source_id)
        for key, value in updates.items():
            if value is not None:
                setattr(source, key, SourceMode(cast(str, value)) if key == "mode" else value)
        return source

    def _delete_source_sync(
        self, kb_id: uuid.UUID, source_id: uuid.UUID, lease: LockLease
    ) -> None:
        source = self.get_source(kb_id, source_id)
        self._session.delete(source)
        lease.ensure_valid()
        self._session.commit()

    def _delete_source_without_lease(self, kb_id: uuid.UUID, source_id: uuid.UUID) -> None:
        source = self.get_source(kb_id, source_id)
        self._session.delete(source)
        self._session.commit()


    async def enqueue_events(
        self, kb_id: uuid.UUID, source_id: uuid.UUID | None, events: list[SourceEvent]
    ) -> str:
        """Enqueue event-mode changes through the existing ingestion queue."""
        if self._queue is None:
            raise BrokerUnavailableError("job broker unavailable")
        if source_id is not None:
            source = self.get_source(kb_id, source_id)
            if source.mode is not SourceMode.event:
                raise ValueError("source does not use event mode")
        try:
            return await self._queue.enqueue_ingest(kb_id, events)
        except BrokerUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - enqueue failures are API 503s
            raise BrokerUnavailableError("job broker unavailable") from exc

    async def enqueue_sync(self, kb_id: uuid.UUID, source_id: uuid.UUID) -> str:
        """Validate a pull source and enqueue its synchronization."""
        source = self.get_source(kb_id, source_id)
        if source.mode is not SourceMode.pull:
            raise ValueError("source does not use pull mode")
        if self._queue is None:
            raise BrokerUnavailableError("job broker unavailable")
        try:
            return await self._queue.enqueue_sync(source_id)
        except BrokerUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - enqueue failures are API 503s
            raise BrokerUnavailableError("job broker unavailable") from exc